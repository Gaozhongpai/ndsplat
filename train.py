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
    from scene.beta_viewer import BetaViewer
    VISER_FOUND = True
except ImportError:
    VISER_FOUND = False
    print("Viser not found. Live viewer will be disabled.")

# FastGS multi-view consistent densification utilities
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
    
def create_radial_weight_mask(height, width, center_weight=0.8, edge_weight=5):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_y, center_x = height // 2, width // 2
    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2))
    normalized_distance = distance / max_distance
    weight_mask = center_weight + (edge_weight - center_weight) * normalized_distance
    return weight_mask

def render_wrapper(viewpoint_cam, gaussians, pipe, bg, mode, scaling_modifier=1.0, use_gsplat=False):
    """Wrapper function that handles model-specific rendering.

    All models now have render_tcgs as a class method, so we dispatch to the
    appropriate signature based on mode.

    Args:
        viewpoint_cam: Camera viewpoint
        gaussians: GaussianModel instance
        pipe: Pipeline parameters
        bg: Background color
        mode: Rendering mode ("ddgs", "3dgs", "ubs", "ndgs", "dgs")
        scaling_modifier: Scaling modifier for rendering
        use_gsplat: If True, use gsplat rasterizer instead of TCGS for UBS/DGS modes
    """
    if "ubs" in mode or "ndgs" in mode or "dgs" in mode or "dbs" in mode:
        gaussians.background = bg
        if use_gsplat and hasattr(gaussians, 'render'):
            return gaussians.render(viewpoint_cam, render_mode="RGB")
        return gaussians.render_tcgs(viewpoint_cam, render_mode="RGB", use_tcgs=False, scaling_modifier=scaling_modifier)
    elif "ddgs" in mode or "3dgs" in mode:
        return gaussians.render_tcgs(viewpoint_cam, pipe, bg, scaling_modifier)
    else:
        raise ValueError(f"Unknown mode: {mode}. All modes should have render_tcgs method.")

def training(dataset, opt, pipe, viewer_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer, log_file = prepare_output_and_logger(dataset)

    # Track total training time
    training_start_time = time.time()

    # Get the appropriate GaussianModel class based on mode
    mode = dataset.mode
    GaussianModel = get_gaussian_model(mode)

    # Initialize model based on mode
    # For NDGS mode, pass the use_rot_scale_l_triangle flag
    if "ubs" in mode or "dbs" in mode:
        gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim)
    elif "ndgs" in mode:
        gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                    use_rot_scale_l_triangle=dataset.use_rot_scale_l_triangle,
                                    learnable_lambda_opc=dataset.learnable_lambda_opc,
                                    lambda_opc=dataset.lambda_opc)
    elif "dgs" in mode:
        # DGS mode: Full DGS with configurable view-dependent position
        gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                  use_view_dependent_pos=dataset.use_view_dependent_pos,
                                  use_opacity_pos_decouple=dataset.use_opacity_pos_decouple,
                                  l_22_inv_init_scale=dataset.l_22_inv_init_scale,
                                  lambda_init=dataset.lambda_init)
    else:
        gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, opt_params=opt)
    gaussians.training_setup(opt)
    if checkpoint:
        if checkpoint.endswith('.ply'):
            # Load from PLY file - starts from iteration 0 with fresh optimizer
            print(f"Loading pretrained model from PLY file: {checkpoint}")
            gaussians.load_ply(checkpoint)
            first_iter = 0
            # Reinitialize tracking arrays to match the loaded model size
            num_points = gaussians.get_xyz.shape[0]
            gaussians.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
            gaussians.xyz_gradient_accum_abs = torch.zeros((num_points, 1), device="cuda")  # FastGS: Z gradients
            gaussians.denom = torch.zeros((num_points, 1), device="cuda")
            gaussians.max_radii2D = torch.zeros((num_points,), device="cuda")
            print(f"Note: Loading from PLY resets optimizer state and starts from iteration 0")
            print(f"Loaded {num_points} Gaussians from PLY file")
        else:
            # Load from checkpoint file - resumes training with optimizer state
            print(f"Loading checkpoint from: {checkpoint}")
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            print(f"Resuming training from iteration {first_iter}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
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

        # Use BetaViewer for UBS mode, GaussianViewer for others (including DGS)
        ViewerClass = BetaViewer if "ubs" in mode else GaussianViewer
        viewer = ViewerClass(
            server=server,
            render_fn=lambda camera_state, render_tab_state: gaussians.view_tcgs(
                camera_state, render_tab_state
            ),
            input_dim=getattr(gaussians, 'input_dim', 6),
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

    # Track best test PSNR for saving best checkpoint
    best_psnr_info = {'best_psnr': 0.0, 'best_iteration': 0}

    # Patient-based training for lambda_opc (NDGS mode only, dynamic scenes with input_dim=7)
    # Similar to 4D-GS opacity_scale training strategy
    n_patient = 0
    is_set_patient = False

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

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        total_loss = 0
        # Track visibility across multi-view batch for gradient averaging
        batch_visibility_filters = []
        batch_radii = []
        batch_viewspace_tensors = []  # Store viewspace tensors from each view

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
            render_pkg = render_wrapper(viewpoint_cam, gaussians, pipe, bg, mode, use_gsplat=dataset.use_gsplat)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # Loss
            gt_image = viewpoint_cam.original_image  # Already on CUDA, no need for .cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # MCMC regularization: add opacity and scale regularization during densification phase
            # This matches UBS behavior and helps prevent degenerate solutions
            if opt.densification_strategy == "mcmc":
                if opt.densify_from_iter < iteration < opt.mcmc_densify_until_iter:
                    loss += opt.opacity_reg * torch.abs(gaussians.get_opacity).mean()
                    loss += opt.scale_reg * torch.abs(gaussians.get_scaling[:, :3]).mean()

            # Normalize by number of views (matching 7DGS-ICCV behavior)
            loss = loss / pipe.mv
            total_loss = total_loss + loss

            # Store visibility and viewspace information for gradient averaging
            batch_visibility_filters.append(visibility_filter)
            batch_radii.append(radii)
            batch_viewspace_tensors.append(viewspace_point_tensor)

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Use the last view's data as baseline
            viewspace_point_tensor = batch_viewspace_tensors[-1]
            visibility_filter = batch_visibility_filters[-1]
            radii = batch_radii[-1]

            # Multi-view: aggregate and apply visibility-weighted averaging
            if pipe.mv > 1:
                # Aggregate visibility and radii
                visibility_count = torch.stack(batch_visibility_filters, dim=1).sum(dim=1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii, dim=1).max(dim=1)[0]

                # Aggregate viewspace gradient norms from all views
                batch_point_grad_norms = []
                for vs_tensor in batch_viewspace_tensors:
                    if vs_tensor.grad is not None:
                        batch_point_grad_norms.append(torch.norm(vs_tensor.grad[:, :2], dim=-1))

                # Sum norms and apply visibility weighting
                aggregated_grad_norms = torch.stack(batch_point_grad_norms, dim=1).sum(dim=1)
                aggregated_grad_norms[visibility_filter] *= pipe.mv / visibility_count[visibility_filter]
                viewspace_point_tensor.grad[:, :2] = aggregated_grad_norms.unsqueeze(-1)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            n_patient_change = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_wrapper, (pipe, background, mode, 1.0, dataset.use_gsplat), log_file, best_psnr_info)

            # Patient-based lambda_opc training (NDGS mode with learnable_lambda_opc)
            # Now works for both 6D and 7D (removed input_dim==7 restriction)
            if "ndgs" in mode and hasattr(gaussians, 'learnable_lambda_opc') and gaussians.learnable_lambda_opc:
                if n_patient_change == -1:
                    # Improvement detected - reset patience
                    n_patient = 0
                elif n_patient_change == 1:
                    # No improvement - increment patience
                    n_patient += 1
                    if n_patient == 2 and not is_set_patient:
                        is_set_patient = True
                        # Only print and call if state actually changes (efficiency improvement)
                        if gaussians.enable_lambda_opc_training(opt):
                            print("\n" + "="*50)
                            print("Start training lambda_opc (patience triggered)")
                            print("="*50)

                # Enable lambda_opc training at iteration 14900 (if not already enabled by patience)
                if iteration == 14900:
                    # Only print and call if state actually changes (efficiency improvement)
                    if gaussians.enable_lambda_opc_training(opt):
                        print("\n" + "="*50)
                        print("Start training lambda_opc at iteration 14900")
                        print("="*50)

                # Disable lambda_opc training at iteration 28000 (similar to 4D-GS at 30k-2k)
                if iteration == 28000:
                    # Only print and call if state actually changes (efficiency improvement)
                    if gaussians.disable_lambda_opc_training():
                        print("\n" + "="*50)
                        print("Disable lambda_opc training at iteration 28000")
                        print("="*50)

                # Monitor lambda_opc values after iteration 14900 (or if patient)
                if (iteration > 14900 or is_set_patient) and iteration % 500 == 0:
                    if hasattr(gaussians, '_lambda_opc'):
                        print(f"Mean lambda_opc: {gaussians.get_lambda_opc.mean():.4f}")
                        if gaussians.input_dim == 7:
                            print(f"Mean lambda_opc_time: {gaussians.get_lambda_opc_time.mean():.4f}")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("\nNumber Gaussian: {}".format(gaussians.get_xyz.shape[0]))
                if "ddgs" in mode:
                    print("\nNumber Principle: {}".format(gaussians._is_principle.sum()))
                    print("\nNumber Non-Principle: {}".format((~gaussians._is_principle).sum()))
                scene.save(iteration)

            # Densification
            # Use MCMC-specific densify_until_iter if MCMC strategy is chosen (default 25k vs standard 15k)
            densify_until = opt.mcmc_densify_until_iter if opt.densification_strategy == "mcmc" else opt.densify_until_iter

            if iteration < densify_until:
                if opt.densification_strategy == "standard":
                    # Standard gradient-based densification (clone, split, prune)
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        min_opacity = 0.005 if "3dgs" in mode or "ubs" in mode else 0.01 ## RSNA 0.005, paper 0.01
                        gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, iteration)
                        # Clear CUDA cache after densification to free memory from pruned Gaussians
                        torch.cuda.empty_cache()

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                elif opt.densification_strategy == "mcmc":
                    # MCMC-based densification (relocate dead Gaussians, add new ones)
                    # Only apply MCMC strategy if model supports it (NDGS/UBS have relocate_gs and add_new_gs methods)
                    if hasattr(gaussians, 'relocate_gs') and hasattr(gaussians, 'add_new_gs'):
                        if iteration % opt.mcmc_refine_interval == 0:
                            # Relocate dead Gaussians based on opacity (matching UBS: <= 0.005)
                            dead_mask = (gaussians.get_opacity.squeeze() <= 0.005)
                            gaussians.relocate_gs(dead_mask=dead_mask)

                            # Add new Gaussians up to cap_max
                            num_added = gaussians.add_new_gs(cap_max=opt.mcmc_cap_max)
                            if num_added > 0 and iteration % (opt.mcmc_refine_interval * 10) == 0:
                                print(f"\n[ITER {iteration}] MCMC: Added {num_added} Gaussians, Total: {gaussians.get_xyz.shape[0]}")

                            # Add covariance-weighted noise to Gaussian positions (UBS-style MCMC enhancement)
                            # This helps break symmetry and encourages exploration after MCMC operations
                            if hasattr(gaussians, 'get_xyz_covariance'):
                                xyz_covariance = gaussians.get_xyz_covariance  # [N, 3, 3] spatial covariance

                                # Generate random noise weighted by opacity
                                # High opacity (visible) → near 0 noise, Low opacity (invisible) → more noise
                                noise = (
                                    torch.randn_like(gaussians._xyz)  # [N, 3] random Gaussian noise
                                    * torch.pow(1 - gaussians.get_opacity, 100)  # Weight by (1-opacity)^100
                                    * opt.noise_lr  # Scale by noise_lr hyperparameter
                                    * xyz_lr  # Scale by current xyz learning rate (cached from update_learning_rate)
                                )

                                # Transform noise by spatial covariance to align with Gaussian shape
                                noise = torch.bmm(xyz_covariance, noise.unsqueeze(-1)).squeeze(-1)

                                # Add noise to positions
                                gaussians._xyz.data.add_(noise)
                    else:
                        print(f"\nWarning: MCMC densification requested but model {mode} does not support it. Falling back to standard densification.")
                        # Fallback to standard densification
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            min_opacity = 0.01 if "ndgs" in mode else 0.005
                            gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, iteration)

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

                elif opt.densification_strategy == "fastgs":
                    # FastGS multi-view consistent densification
                    # Adapted from FastGS (arXiv:2511.04283)
                    if hasattr(gaussians, 'densify_and_prune_fastgs') and hasattr(gaussians, 'render_tcgs_with_metric'):
                        # Track gradients for densification (same as standard)
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            # Sample cameras for multi-view scoring
                            my_viewpoint_stack = scene.getTrainCameras().copy()
                            camlist = sampling_cameras(my_viewpoint_stack, num_cams=opt.fastgs_num_sample_cams)

                            # Compute multi-view consistent importance and pruning scores
                            importance_score, pruning_score = compute_gaussian_score_fastgs(
                                camlist, gaussians, background,
                                loss_thresh=opt.fastgs_loss_thresh,
                                DENSIFY=True
                            )

                            # Apply FastGS densification and pruning
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune_fastgs(
                                max_screen_size=size_threshold,
                                min_opacity=0.005,
                                extent=scene.cameras_extent,
                                radii=radii,
                                grad_thresh=opt.fastgs_grad_thresh,
                                grad_abs_thresh=opt.fastgs_grad_abs_thresh,
                                percent_dense=opt.percent_dense,
                                importance_score=importance_score,
                                pruning_score=pruning_score,
                                densify_score_thresh=opt.fastgs_densify_score_thresh,
                                prune_budget_ratio=opt.fastgs_prune_budget_ratio,
                            )

                            # Clear CUDA cache after densification
                            torch.cuda.empty_cache()

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                    else:
                        print(f"\nWarning: FastGS densification requested but model {mode} does not support it. Falling back to standard densification.")
                        # Fallback to standard densification
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            min_opacity = 0.005 if "ndgs" in mode else 0.005
                            gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, iteration)

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

                else:
                    raise ValueError(f"Unknown densification_strategy: {opt.densification_strategy}. Choose 'standard', 'mcmc', or 'fastgs'.")

            # FastGS final pruning phase (after densification ends)
            # This is the multi-view consistent pruning that happens every 3k iterations after 15k
            if (opt.densification_strategy == "fastgs" and
                iteration % opt.fastgs_final_prune_interval == 0 and
                iteration >= opt.fastgs_final_prune_start and
                iteration < opt.fastgs_final_prune_end):
                if hasattr(gaussians, 'final_prune_fastgs') and hasattr(gaussians, 'render_tcgs_with_metric'):
                    # Sample cameras for multi-view scoring
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack, num_cams=opt.fastgs_num_sample_cams)

                    # Compute pruning scores (no DENSIFY since we're only pruning)
                    _, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, background,
                        loss_thresh=opt.fastgs_loss_thresh,
                        DENSIFY=False
                    )

                    # Apply aggressive final pruning
                    gaussians.final_prune_fastgs(
                        min_opacity=0.1,
                        pruning_score=pruning_score,
                        prune_score_thresh=0.9,
                    )

                    print(f"\n[ITER {iteration}] FastGS final prune: {gaussians.get_xyz.shape[0]} Gaussians remaining")
                    torch.cuda.empty_cache()

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

    # Save total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    training_time_path = os.path.join(scene.model_path, "training_time.txt")
    with open(training_time_path, 'w') as f:
        f.write(f"{total_training_time:.2f}")
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    # Write final info to log file and close it
    if log_file:
        log_file.write("=" * 60 + "\n")
        log_file.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        log_file.write(f"Final number of Gaussians: {gaussians.get_xyz.shape[0]}\n")
        log_file.write(f"Best test PSNR: {best_psnr_info['best_psnr']:.4f} at iteration {best_psnr_info['best_iteration']}\n")
        log_file.close()

    # Print best PSNR info
    print(f"\nBest test PSNR: {best_psnr_info['best_psnr']:.4f} at iteration {best_psnr_info['best_iteration']}")

    # Return viewer to keep it alive after training if needed
    return viewer

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

    # Create training log file
    log_path = os.path.join(args.model_path, "training.log")
    log_file = open(log_path, 'w')
    log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Output folder: {args.model_path}\n")
    log_file.write(f"Mode: {args.mode}\n")
    log_file.write("=" * 60 + "\n")
    log_file.flush()

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, log_file

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, log_file=None, best_psnr_info=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        test_psnr_value = None  # Track test PSNR for best checkpoint
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
                log_msg = f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}"
                print("\n" + log_msg)
                # Write to log file
                if log_file:
                    log_file.write(log_msg + "\n")
                    log_file.flush()
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                # Track test PSNR for best checkpoint saving
                if config['name'] == 'test':
                    test_psnr_value = float(psnr_test)

        # Save best checkpoint if this is a new best test PSNR
        # Also track patience for lambda_opc training (return n_patient for caller to track)
        n_patient_increment = 0
        if best_psnr_info is not None and test_psnr_value is not None:
            if test_psnr_value > best_psnr_info['best_psnr']:
                best_psnr_info['best_psnr'] = test_psnr_value
                best_psnr_info['best_iteration'] = iteration
                # Save best point cloud (no need for full checkpoint with optimizer state)
                scene.save(iteration, is_best=True)
                log_msg = f"[ITER {iteration}] New best test PSNR: {test_psnr_value:.4f} - Saving best point cloud"
                print("\n" + log_msg)
                if log_file:
                    log_file.write(log_msg + "\n")
                    log_file.flush()
                # Reset patience counter (improvement detected)
                n_patient_increment = -1  # Signal to reset
            else:
                # No improvement - increment patience
                n_patient_increment = 1

        # Log number of Gaussians
        num_gaussians = scene.gaussians.get_xyz.shape[0]
        if log_file:
            log_file.write(f"[ITER {iteration}] Number of Gaussians: {num_gaussians}\n")
            log_file.flush()

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', num_gaussians, iteration)
        torch.cuda.empty_cache()

        return n_patient_increment  # For patient-based lambda_opc training

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    vp = ViewerParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(500, 30_001, 500)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 2_000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None, help="Path to checkpoint (.pth) or pretrained model (.ply) to load. .pth resumes training with optimizer state, .ply starts fresh from iteration 0")
    parser.add_argument("--keep_viewer", action="store_true", help="Keep the viewer running after training completes")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # For static scenes (input_dim != 7), use less frequent testing and skip best checkpoint
    # Dynamic scenes (input_dim=7) need frequent testing to track best checkpoint
    if args.input_dim != 7:
        # Static scene: only test at key iterations (same as save_iterations)
        args.test_iterations = [500, 2_000, 7_000, 15_000, 30_000]

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    viewer = training(lp.extract(args), op.extract(args), pp.extract(args), vp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    # Keep viewer running if requested
    if viewer is not None and args.keep_viewer:
        print("\nViewer is still running. Press Ctrl+C to exit.")
        try:
            # Keep the main thread alive so viewer server stays up
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
    elif viewer is not None:
        print("\nViewer will shut down. Use --keep_viewer to keep it running after training.")
