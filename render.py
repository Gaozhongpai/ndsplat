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


def render_wrapper(view, gaussians, pipeline, background, mode, is_test=False):
    """Wrapper function that handles model-specific rendering.

    All models now have render_tcgs as a class method, so we dispatch to the
    appropriate signature based on mode.

    Args:
        view: Camera viewpoint
        gaussians: GaussianModel instance
        pipeline: Pipeline parameters
        background: Background color
        mode: Rendering mode ("ndgs", "ddgs", "3dgs", "ubs")
        is_test: Whether in test mode

    Returns:
        Dictionary containing render outputs
    """
    if "ubs" in mode or "ndgs" in mode:
        # UBS mode: use render_tcgs with CUDA-accelerated conditional slicing
        gaussians.background = background
        return gaussians.render_tcgs(view, render_mode="RGB", use_tcgs=is_test)
    elif "ddgs" in mode or "3dgs" in mode:
        # DDGS/3DGS mode: use model's render_tcgs method
        return gaussians.render_tcgs(view, pipeline, background, is_test=is_test)
    else:
        raise ValueError(f"Unknown mode: {mode}. All modes should have render_tcgs method.")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mode, measure_fps=False):
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

    # Create principle/non-principle directories only for DDGS mode
    if "ddgs" in mode.lower():
        makedirs(render_path.replace("renders", "renders_principle"), exist_ok=True)
        makedirs(render_path.replace("renders", "renders_non_principle"), exist_ok=True)

    # FPS measurement
    fpslist = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Calculate FPS for first 20 frames
        if measure_fps:
            rendering_principle = None
            num_frames = 500

            # Measure rendering time
            start_time = time.time()
            for _ in range(num_frames):
                rendering = render_wrapper(view, gaussians, pipeline, background, mode, is_test=True)["render"]
            end_time = time.time()

            # Calculate FPS
            total_time = end_time - start_time
            fps = num_frames / total_time
            fpslist.append(fps)
            print(f"Rendering FPS: {fps:.2f}")

            if len(fpslist) == 20:
                print(f"Average Rendering FPS: {np.array(fpslist).mean():.2f}")
        else:
            renderings = render_wrapper(view, gaussians, pipeline, background, mode, is_test=True)
            rendering = renderings["render"]

        # Save rendered images
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        # Save principle/non-principle splits (DDGS mode only)
        if not measure_fps and "ddgs" in mode.lower():
            rendering_principle = renderings.get("render_principle")
            rendering_non_principle = renderings.get("render_non_principle")

            if rendering_principle is not None:
                torchvision.utils.save_image(
                    rendering_principle,
                    os.path.join(render_path.replace("renders", "renders_principle"), '{0:05d}'.format(idx) + ".png")
                )
            if rendering_non_principle is not None:
                torchvision.utils.save_image(
                    rendering_non_principle,
                    os.path.join(render_path.replace("renders", "renders_non_principle"), '{0:05d}'.format(idx) + ".png")
                )

        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, measure_fps: bool = False):
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

        gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        if "nerf_synthetic" in dataset.source_path:
            bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background, mode, measure_fps)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestCameras(), gaussians, pipeline, background, mode, measure_fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load (-1 for latest)")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering training views")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test views")
    parser.add_argument("--measure_fps", action="store_true", help="Measure FPS instead of saving images")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.eval = True

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.measure_fps)
