#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, get_gaussian_model
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ViewerParams
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import viser
    from scene.gaussian_viewer import GaussianViewer
    VISER_FOUND = True
except ImportError:
    VISER_FOUND = False
    print("Viser not found. Live viewer will be disabled.")
    
def create_radial_weight_mask(height, width, center_weight=0.8, edge_weight=5):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_y, center_x = height // 2, width // 2
    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2))
    normalized_distance = distance / max_distance
    weight_mask = center_weight + (edge_weight - center_weight) * normalized_distance
    return weight_mask

def render_wrapper(viewpoint_cam, gaussians, pipe, bg, mode, scaling_modifier=1.0):
    """Wrapper function that handles model-specific rendering.

    All models now have render_tcgs as a class method, so we dispatch to the
    appropriate signature based on mode.

    Args:
        viewpoint_cam: Camera viewpoint
        gaussians: GaussianModel instance
        pipe: Pipeline parameters
        bg: Background color
        mode: Rendering mode ("ddgs", "3dgs", "ubs", "ndgs")
        scaling_modifier: Scaling modifier for rendering
    """
    if mode == "ubs" or mode == "ndgs":
        # UBS/N-DGS mode: use render_tcgs with CUDA-accelerated conditional slicing
        gaussians.background = bg
        return gaussians.render_tcgs(viewpoint_cam, render_mode="RGB", use_tcgs=False, scaling_modifier=scaling_modifier)
    elif "ddgs" in mode or "3dgs" in mode:
        # DDGS/3DGS mode: use model's render_tcgs method
        return gaussians.render_tcgs(viewpoint_cam, pipe, bg, scaling_modifier)
    else:
        raise ValueError(f"Unknown mode: {mode}. All modes should have render_tcgs method.")

def training(dataset, opt, pipe, viewer_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Get the appropriate GaussianModel class based on mode
    mode = dataset.mode
    GaussianModel = get_gaussian_model(mode)

    # Initialize model based on mode
    # For NDGS mode, pass the use_rot_scale_l_triangle flag
    if mode == "ndgs":
        gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                   use_rot_scale_l_triangle=dataset.use_rot_scale_l_triangle)
    else:
        gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    if "nerf_synthetic" in dataset.source_path:
        bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize viser viewer if available and not disabled
    viewer = None
    if VISER_FOUND and not viewer_params.disable_viewer:
        # Compute scene bounds for initial camera setup and cutting plane slider
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        scene_center = xyz.mean(axis=0)
        scene_min = xyz.min(axis=0)
        scene_max = xyz.max(axis=0)
        scene_size = scene_max - scene_min
        scene_radius = float(np.linalg.norm(scene_size))  # Diagonal of bounding box

        # Extract X-axis bounds for cutting plane slider
        x_bounds = (float(scene_min[0]), float(scene_max[0]))

        server = viser.ViserServer(port=viewer_params.port, verbose=False)

        # Set up the scene with proper axes
        server.scene.set_up_direction("+y")

        viewer = GaussianViewer(
            server=server,
            render_fn=lambda camera_state, render_tab_state: gaussians.view_tcgs(
                camera_state, render_tab_state
            ),
            input_dim=getattr(gaussians, 'input_dim', 3),
            mode="training",
            share_url=False,
            scene_bounds=x_bounds,  # Pass X bounds for cutting plane slider
        )

        # Set initial camera via client connection
        @server.on_client_connect
        def on_connect(client):
            camera_distance = scene_radius * 1.2  # Add 20% margin for better view

            # Position camera slightly above and in front of the scene
            # This gives a nice 3/4 view of the object
            initial_camera_position = scene_center + np.array([
                camera_distance * 0.5,   # Slightly to the right
                camera_distance * 0.3,   # Slightly above
                camera_distance          # In front
            ])

            # Set camera position and look_at target
            # Note: set position first, then look_at to avoid unwanted translations
            client.camera.position = tuple(initial_camera_position)
            client.camera.look_at = tuple(scene_center)
            client.camera.up_direction = (0.0, 1.0, 0.0)

        print(f"Viser viewer started on port {viewer_params.port}")
        print(f"Scene center: {scene_center}, radius: {scene_radius:.2f}")
        print(f"X-axis bounds: [{x_bounds[0]:.2f}, {x_bounds[1]:.2f}]")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Handle viewer pause/resume
        if viewer is not None:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        total_loss = 0
        for _ in range(pipe.mv):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Use unified render wrapper (handles all model modes)
            render_pkg = render_wrapper(viewpoint_cam, gaussians, pipe, bg, mode)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            total_loss = total_loss + loss
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_wrapper, (pipe, background, mode))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("\nNumber Gaussian: {}".format(gaussians.get_xyz.shape[0]))
                if mode == "ddgs" or mode == "ddndgs":
                    print("\nNumber Principle: {}".format(gaussians._is_principle.sum()))
                    print("\nNumber Non-Principle: {}".format((~gaussians._is_principle).sum()))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    min_opacity = 0.01 if "ndgs" in mode else 0.005
                    gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, iteration)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Update viewer
            if viewer is not None:
                num_train_rays_per_step = gt_image.numel()
                viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic + 1e-8)
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                viewer.update(iteration, num_train_rays_per_step)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    vp = ViewerParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 2_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 2_000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), vp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
