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
import time
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, get_gaussian_model
from utils.general_utils import safe_state


def render_wrapper(view, gaussians, pipeline, background, mode, is_test=False, tight_snugbox=False):
    """Wrapper function that handles model-specific rendering.

    All models now have render_tcgs as a class method, so we dispatch to the
    appropriate signature based on mode.

    Args:
        view: Camera viewpoint
        gaussians: GaussianModel instance
        pipeline: Pipeline parameters
        background: Background color
        mode: Rendering mode ("ndgs", "ddgs", "3dgs", "ubs", "dgs")
        is_test: Whether in test mode
        tight_snugbox: Whether to use tight snugbox for faster rendering (FPS measurement)

    Returns:
        Dictionary containing render outputs
    """
    if "ubs" in mode or "ndgs" in mode or "dgs" in mode or "dbs" in mode:
        # UBS/N-DGS/dGS/dBS mode: use render_tcgs with CUDA-accelerated conditional slicing
        gaussians.background = background
        return gaussians.render_tcgs(view, render_mode="RGB", use_tcgs=is_test, tight_snugbox=tight_snugbox)
    elif "ddgs" in mode or "3dgs" in mode:
        # DDGS/3DGS mode: use model's render_tcgs method (no tight_snugbox support)
        return gaussians.render_tcgs(view, pipeline, background, is_test=is_test)
    else:
        raise ValueError(f"Unknown mode: {mode}. All modes should have render_tcgs method.")


def measure_fps(views, gaussians, pipeline, background, mode, num_frames=500, num_views=20, is_cuda=True):
    """Measure FPS on a set of views.

    Args:
        views: List of camera views
        gaussians: Gaussian model
        pipeline: Pipeline parameters
        background: Background color
        mode: Rendering mode
        num_frames: Number of frames to render per view for timing
        num_views: Number of views to test
        is_cuda: If True, use CUDA optimizations (is_test=True, tight_snugbox=True).
                 If False, use standard rendering (is_test=False, tight_snugbox=False)

    Returns:
        Average FPS across all tested views
    """
    fpslist = []
    fps_measure_count = min(num_views, len(views))

    render_mode_str = "CUDA optimized (is_test=True, tight_snugbox=True)" if is_cuda else "Standard (is_test=False, tight_snugbox=False)"
    print(f"Measuring FPS for first {fps_measure_count} views ({render_mode_str})...")

    for idx in tqdm(range(fps_measure_count), desc="FPS measurement", unit="view"):
        view = views[idx]

        # Warm-up runs
        for _ in range(10):
            _ = render_wrapper(view, gaussians, pipeline, background, mode, is_test=is_cuda, tight_snugbox=is_cuda)["render"]

        # Measure rendering time
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_frames):
            rendering = render_wrapper(view, gaussians, pipeline, background, mode, is_test=is_cuda, tight_snugbox=is_cuda)["render"]
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate FPS
        total_time = end_time - start_time
        fps = num_frames / total_time
        fpslist.append(fps)
        tqdm.write(f"View {idx}: {fps:.2f} FPS")

    return fpslist


def measure_fps_sets(dataset: ModelParams, iteration, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                     num_frames: int = 500, num_views: int = 20, is_cuda: bool = True):
    """Measure FPS on train and/or test sets.

    Args:
        dataset: Dataset parameters
        iteration: Iteration to load (-1 for latest)
        pipeline: Pipeline parameters
        skip_train: Skip measuring training views
        skip_test: Skip measuring test views
        num_frames: Number of frames to render per view
        num_views: Number of views to test
        is_cuda: If True, use CUDA optimizations (is_test=True, tight_snugbox=True).
                 If False, use standard rendering (is_test=False, tight_snugbox=False)
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
                                        lambda_opc=dataset.lambda_opc)
        elif mode == "dgs":
            # DGS mode: Full DGS with configurable view-dependent position
            gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                      use_view_dependent_pos=dataset.use_view_dependent_pos,
                                      use_opacity_pos_decouple=dataset.use_opacity_pos_decouple)
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

        # Report training time if available (only for best iteration)
        if iteration == "best":
            training_time_path = os.path.join(dataset.model_path, "training_time.txt")
            if os.path.exists(training_time_path):
                with open(training_time_path, 'r') as f:
                    training_time = float(f.read().strip())
                print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        results = {}

        if not skip_train:
            print("\n" + "="*50)
            print("TRAIN SET FPS MEASUREMENT")
            print("="*50)
            train_fps = measure_fps(scene.getTrainCameras(), gaussians, pipeline, background, mode,
                                   num_frames, num_views, is_cuda)
            avg_train_fps = np.array(train_fps).mean()
            results['train'] = {
                'fps_list': train_fps,
                'avg_fps': avg_train_fps
            }
            print(f"\nAverage Train FPS: {avg_train_fps:.2f}")

        if not skip_test:
            print("\n" + "="*50)
            print("TEST SET FPS MEASUREMENT")
            print("="*50)
            test_fps = measure_fps(scene.getTestCameras(), gaussians, pipeline, background, mode,
                                  num_frames, num_views, is_cuda)
            avg_test_fps = np.array(test_fps).mean()
            results['test'] = {
                'fps_list': test_fps,
                'avg_fps': avg_test_fps
            }
            print(f"\nAverage Test FPS: {avg_test_fps:.2f}")

        # Save results to file
        output_dir = os.path.join(dataset.model_path, "fps_results")
        os.makedirs(output_dir, exist_ok=True)

        iteration_str = str(scene.loaded_iter)
        output_file = os.path.join(output_dir, f"fps_iteration_{iteration_str}.txt")

        with open(output_file, 'w') as f:
            f.write(f"FPS Measurement Results\n")
            f.write(f"Model: {dataset.model_path}\n")
            f.write(f"Iteration: {scene.loaded_iter}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"CUDA Optimizations: {'Enabled (is_test=True, tight_snugbox=True)' if is_cuda else 'Disabled (is_test=False, tight_snugbox=False)'}\n")
            f.write(f"Frames per view: {num_frames}\n")
            f.write(f"Number of views tested: {num_views}\n")
            f.write("\n")

            if 'train' in results:
                f.write(f"Train Set Average FPS: {results['train']['avg_fps']:.2f}\n")
                f.write(f"Train Set FPS per view: {', '.join([f'{fps:.2f}' for fps in results['train']['fps_list']])}\n")
                f.write("\n")

            if 'test' in results:
                f.write(f"Test Set Average FPS: {results['test']['avg_fps']:.2f}\n")
                f.write(f"Test Set FPS per view: {', '.join([f'{fps:.2f}' for fps in results['test']['fps_list']])}\n")

        print(f"\nResults saved to: {output_file}")

        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        if 'train' in results:
            print(f"Train Average FPS: {results['train']['avg_fps']:.2f}")
        if 'test' in results:
            print(f"Test Average FPS: {results['test']['avg_fps']:.2f}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="FPS measurement script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default="-1", type=str, help="Iteration to load (-1 for latest, 'best' for best checkpoint)")
    parser.add_argument("--skip_train", action="store_true", help="Skip measuring training views")
    parser.add_argument("--skip_test", action="store_true", help="Skip measuring test views")
    parser.add_argument("--num_frames", default=500, type=int, help="Number of frames to render per view for timing")
    parser.add_argument("--num_views", default=20, type=int, help="Number of views to test")
    parser.add_argument("--is_cuda", action="store_true", default=True, help="Use CUDA optimizations (is_test=True, tight_snugbox=True)")
    parser.add_argument("--no_cuda", dest="is_cuda", action="store_false", help="Disable CUDA optimizations (is_test=False, tight_snugbox=False)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = get_combined_args(parser)
    print("Measuring FPS for " + args.model_path)
    print(f"CUDA optimizations: {'Enabled' if args.is_cuda else 'Disabled'}")
    args.eval = True

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Handle 'best' iteration specially
    if args.iteration == "best":
        iteration = "best"
    else:
        iteration = int(args.iteration)

    measure_fps_sets(model.extract(args), iteration, pipeline.extract(args),
                    args.skip_train, args.skip_test, args.num_frames, args.num_views, args.is_cuda)
