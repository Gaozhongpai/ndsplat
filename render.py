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

import json
import os
import time
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, get_gaussian_model
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips, LPIPS


def render_wrapper(view, gaussians, pipeline, background, mode, is_test=False, tight_snugbox=False):
    """Wrapper function that handles model-specific rendering.

    All models now have render_tcgs as a class method, so we dispatch to the
    appropriate signature based on mode.

    Args:
        view: Camera viewpoint
        gaussians: GaussianModel instance
        pipeline: Pipeline parameters
        background: Background color
        mode: Rendering mode ("3dgs", "ndgs", "ubs", "dgs", "dbs")
        is_test: Whether in test mode
        tight_snugbox: Whether to use tight snugbox for faster rendering (FPS measurement)

    Returns:
        Dictionary containing render outputs
    """
    if "ubs" in mode or "ndgs" in mode or "dgs" in mode or "dbs" in mode:
        # UBS/N-DGS/dGS/dBS mode: use render_tcgs with CUDA-accelerated conditional slicing
        gaussians.background = background
        return gaussians.render_tcgs(view, render_mode="RGB", use_tcgs=is_test, tight_snugbox=tight_snugbox)
    elif "3dgs" in mode:
        return gaussians.render_tcgs(view, pipeline, background, is_test=is_test)
    else:
        raise ValueError(f"Unknown mode: {mode}.")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mode, measure_fps=False, lpips_criterion=None):
    """Render a set of views and save results.

    Args:
        model_path: Path to the model
        name: Dataset split name (train/test)
        iteration: Iteration number
        views: List of camera views to render
        gaussians: Gaussian model
        pipeline: Pipeline parameters
        background: Background color
        mode: Rendering mode
        measure_fps: If True, measure FPS on first 20 frames instead of saving images
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # FPS measurement at iteration 30000 (final) or best
    if iteration == 30000 or iteration == "best":
        # Report training time only at iteration 30000
        if iteration == 30000:
            training_time_path = os.path.join(model_path, "training_time.txt")
            if os.path.exists(training_time_path):
                with open(training_time_path, 'r') as f:
                    training_time = float(f.read().strip())
                print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        fpslist = []
        fps_measure_count = min(20, len(views))

        print("Measuring FPS for first 20 frames (tight_snugbox=True)...")
        for idx in range(fps_measure_count):
            view = views[idx]
            num_frames = 500

            # Measure rendering time with tight_snugbox=True
            start_time = time.time()
            for _ in range(num_frames):
                rendering = render_wrapper(view, gaussians, pipeline, background, mode, is_test=True, tight_snugbox=True)["render"]
            end_time = time.time()

            # Calculate FPS
            total_time = end_time - start_time
            fps = num_frames / total_time
            fpslist.append(fps)
            if measure_fps:
                print(f"Frame {idx}: Rendering FPS: {fps:.2f}")

        # Save FPS results
        if fpslist:
            avg_fps = np.array(fpslist).mean()
            print(f"Average Rendering FPS (first {len(fpslist)} frames): {avg_fps:.2f}")

            # Save FPS to file in the iteration directory
            fps_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fps.txt")
            with open(fps_path, 'w') as f:
                f.write(f"{avg_fps:.2f}")

    print("Rendering all frames for saving (use_tcgs=False for quality)...")
    psnrs = []
    ssims = []
    lpipss = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Render with use_tcgs=False for quality-matched evaluation (same as training)
        renderings = render_wrapper(view, gaussians, pipeline, background, mode, is_test=False, tight_snugbox=False)
        rendering = renderings["render"]
        gt = view.original_image[0:3, :, :]

        # Compute metrics on GPU tensors (before PNG save)
        rendering_clamped = torch.clamp(rendering, 0.0, 1.0).unsqueeze(0)
        gt_clamped = torch.clamp(gt, 0.0, 1.0).unsqueeze(0)
        psnrs.append(psnr(rendering_clamped, gt_clamped).item())
        ssims.append(ssim(rendering_clamped, gt_clamped).item())
        if lpips_criterion is not None:
            lpipss.append(lpips(rendering_clamped, gt_clamped, lpips_criterion).item())

        # Save images
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # Save inline metrics (computed on GPU tensors, no PNG round-trip)
    if psnrs:
        metrics = {
            "PSNR": float(np.mean(psnrs)),
            "SSIM": float(np.mean(ssims)),
            "num_views": len(psnrs),
        }
        if lpipss:
            metrics["LPIPS"] = float(np.mean(lpipss))
        print(f"  {name} PSNR: {metrics['PSNR']:.4f}  SSIM: {metrics['SSIM']:.6f}" +
              (f"  LPIPS: {metrics['LPIPS']:.6f}" if lpipss else ""))
        metrics_path = os.path.join(model_path, name, "ours_{}".format(iteration), "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


def render_sets(dataset: ModelParams, iteration, pipeline: PipelineParams, skip_train: bool, skip_test: bool, measure_fps: bool = False):
    """Render train and/or test sets.

    Args:
        dataset: Dataset parameters
        iteration: Iteration to load (-1 for latest)
        pipeline: Pipeline parameters
        skip_train: Skip rendering training views
        skip_test: Skip rendering test views
        measure_fps: If True, measure FPS instead of saving images
    """
    with torch.no_grad():
        # Get the appropriate GaussianModel class based on mode
        mode = dataset.mode
        GaussianModel = get_gaussian_model(mode)
        if "ubs" in mode:
            gaussians = GaussianModel(sh_degree=dataset.sh_degree, input_dim=dataset.input_dim)
        elif "ndgs" in mode:
            gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                        use_rot_scale_l_triangle=dataset.use_rot_scale_l_triangle,
                                        learnable_lambda_opc=dataset.learnable_lambda_opc,
                                        lambda_opc=dataset.lambda_opc)
        elif mode == "dgs":
            # DGS mode: Full DGS with configurable view-dependent position
            gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                      use_view_dependent_pos=dataset.use_view_dependent_pos,
                                      use_opacity_pos_decouple=dataset.use_opacity_pos_decouple,
                                      l_22_inv_init_scale=dataset.l_22_inv_init_scale,
                                      lambda_init=dataset.lambda_init)
        else:
            gaussians = GaussianModel(dataset.sh_degree)

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            load_train_cameras=not skip_train,
            load_test_cameras=not skip_test,
        )

        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Initialize LPIPS criterion once for both splits
        lpips_criterion = LPIPS('vgg', '0.1').to("cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background, mode, measure_fps,
                      lpips_criterion=lpips_criterion)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestCameras(), gaussians, pipeline, background, mode, measure_fps,
                      lpips_criterion=lpips_criterion)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default="-1", type=str, help="Iteration to load (-1 for latest, 'best' for best checkpoint)")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering training views")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test views")
    parser.add_argument("--measure_fps", action="store_true", help="Measure FPS instead of saving images")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    # Training-only parameters (accepted but ignored for convenience in scripts)
    parser.add_argument("--noise_lr", type=float, default=1.0, help="[Training only] Noise learning rate (ignored during rendering)")
    parser.add_argument("--opacity_reg", type=float, default=0.01, help="[Training only] Opacity regularization (ignored during rendering)")
    parser.add_argument("--scale_reg", type=float, default=0.01, help="[Training only] Scale regularization (ignored during rendering)")
    parser.add_argument("--mcmc_cap_max", type=int, default=300000, help="[Training only] MCMC cap max (ignored during rendering)")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.eval = True

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Handle 'best' iteration specially
    if args.iteration == "best":
        iteration = "best"
    else:
        iteration = int(args.iteration)

    render_sets(model.extract(args), iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.measure_fps)
