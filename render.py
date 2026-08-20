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

import hashlib
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


def render_wrapper(view, gaussians, pipeline, background, mode, is_test=False, tight_snugbox=False, use_gsplat=False, accutile=True):
    """Wrapper function that handles model-specific rendering.

    Args:
        view: Camera viewpoint
        gaussians: GaussianModel instance
        pipeline: Pipeline parameters
        background: Background color
        mode: Rendering mode ("3dgs", "ndgs", "ubs", "dgs", "dbs")
        is_test: Whether in test mode
        tight_snugbox: Whether to use tight snugbox for faster rendering (FPS measurement)
        use_gsplat: If True, use gsplat rasterizer instead of TCGS

    Returns:
        Dictionary containing render outputs
    """
    if mode == "3dgs" or mode == "clipgs":
        # clipgs (ClipGS baseline reimpl) shares the 3dgs render_tcgs signature.
        return gaussians.render_tcgs(view, pipeline, background, is_test=is_test)
    elif "ubs" in mode or "ndgs" in mode or "dgs" in mode or "dbs" in mode or mode == "factorsplat":
        gaussians.background = background
        if use_gsplat and hasattr(gaussians, 'render'):
            return gaussians.render(view, render_mode="RGB", use_tcgs=is_test, accutile=accutile)
        return gaussians.render_tcgs(view, render_mode="RGB", use_tcgs=is_test, tight_snugbox=tight_snugbox)
    else:
        raise ValueError(f"Unknown mode: {mode}.")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mode, measure_fps=False, use_gsplat=False, bake_appearance=False, source_path=None):
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

        bake = (bake_appearance
                and hasattr(gaussians, "bake_appearance")
                and getattr(views[0], "tf_index", None) is not None)
        if bake:
            print("Measuring FPS with the conditioned appearance BAKED per preset "
                  "(steady-state rate; conditioning amortized per TF switch)")
        else:
            print("Measuring FPS for first 20 frames...")
        for idx, view in enumerate(views[:fps_measure_count]):
            if bake:
                gaussians.bake_appearance(view.tf_index)
            num_frames = 100

            # Warmup
            for _ in range(10):
                render_wrapper(view, gaussians, pipeline, background, mode, is_test=True, tight_snugbox=True, use_gsplat=use_gsplat)

            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(num_frames):
                render_wrapper(view, gaussians, pipeline, background, mode, is_test=True, tight_snugbox=True, use_gsplat=use_gsplat)
            torch.cuda.synchronize()
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
            if bake:
                ms = measure_tf_switch_latency(gaussians, views[0].tf_index)
                print(f"TF-switch latency: {ms:.2f} ms "
                      f"(amortized over the baked steady-state rate)")
                with open(os.path.join(model_path, name,
                                       "ours_{}".format(iteration),
                                       "tf_switch_ms.txt"), 'w') as f:
                    f.write(f"{ms:.3f}")

    if getattr(gaussians, "_tf_baked_state", None) is not None:
        gaussians.unbake()

    # GT frames are identical for every method trained on the same scene/split, but
    # PNG-encoding one 1600^2 image costs ~300 ms, so re-encoding them per run is
    # ~8 min of pure duplication (19 Table-1 runs per scan re-wrote the same GT).
    # Encode once into a per-dataset cache, then hardlink. Byte-identical output,
    # so metrics.py/group_metrics read exactly what they read before.
    gt_cache = os.environ.get("FACTORSPLAT_GT_CACHE")
    cache_dir = None
    if gt_cache and source_path:
        # Key on the FULL resolved dataset path, not its basename. Per-preset
        # oracle datasets share basenames across scans
        # (.../factorsplat_oracle/<scan>/full/test_comp_00_h30_a060), so a
        # basename key made five different anatomies share one GT set and
        # produced 8-15 dB specialist scores against the wrong images.
        real = os.path.realpath(source_path)
        digest = hashlib.sha1(real.encode()).hexdigest()[:12]
        tag = f"{os.path.basename(os.path.dirname(os.path.dirname(real)))}" \
              f"_{os.path.basename(real)}_{digest}"
        cache_dir = os.path.join(gt_cache, tag, f"{name}_{iteration}")
        makedirs(cache_dir, exist_ok=True)

    print("Rendering all frames for saving...")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Render with use_tcgs=False for quality-matched evaluation (same as training)
        renderings = render_wrapper(view, gaussians, pipeline, background, mode, is_test=False, tight_snugbox=False, use_gsplat=use_gsplat)
        rendering = renderings["render"]
        gt = view.original_image[0:3, :, :]

        # Save images
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        gt_out = os.path.join(gts_path, '{0:05d}'.format(idx) + ".png")
        if cache_dir is None:
            torchvision.utils.save_image(gt, gt_out)
        else:
            cached = os.path.join(cache_dir, '{0:05d}'.format(idx) + ".png")
            if not os.path.exists(cached):
                torchvision.utils.save_image(gt, cached)
            if os.path.exists(gt_out):
                os.remove(gt_out)
            try:
                os.link(cached, gt_out)
            except OSError:
                torchvision.utils.save_image(gt, gt_out)


def render_sets(dataset: ModelParams, iteration, pipeline: PipelineParams, skip_train: bool, skip_test: bool, measure_fps: bool = False, bake_appearance: bool = False):
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
        if mode == "3dgs":
            gaussians = GaussianModel(dataset.sh_degree)
        elif mode == "clipgs":
            # ClipGS baseline reimpl; check before the "dgs" in mode branch.
            gaussians = GaussianModel(dataset.sh_degree,
                                      deform_scale=getattr(dataset, "clipgs_deform_scale", False))
        elif "ubs" in mode or "dbs" in mode:
            gaussians = GaussianModel(sh_degree=dataset.sh_degree, input_dim=dataset.input_dim)
        elif "ndgs" in mode:
            gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                        use_rot_scale_l_triangle=dataset.use_rot_scale_l_triangle,
                                        learnable_lambda_opc=dataset.learnable_lambda_opc,
                                        lambda_opc=dataset.lambda_opc)
        elif "dgs" in mode:
            gaussians = GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                                      use_view_dependent_pos=dataset.use_view_dependent_pos,
                                      use_opacity_pos_decouple=dataset.use_opacity_pos_decouple,
                                      l_22_inv_init_scale=dataset.l_22_inv_init_scale,
                                      lambda_init=dataset.lambda_init,
                                      lambda_opc=dataset.lambda_opc,
                                      direct_unrestricted=getattr(dataset, "direct_unrestricted", False))
        elif mode == "factorsplat":
            gaussians = GaussianModel(
                dataset.sh_degree, input_dim=dataset.input_dim,
                use_view_dependent_pos=dataset.use_view_dependent_pos,
                use_opacity_pos_decouple=dataset.use_opacity_pos_decouple,
                l_22_inv_init_scale=dataset.l_22_inv_init_scale,
                lambda_init=dataset.lambda_init, lambda_opc=dataset.lambda_opc,
                direct_unrestricted=getattr(dataset, "direct_unrestricted", False),
                tf_rank=dataset.tf_rank, tf_hidden=dataset.tf_hidden,
                tf_samples=dataset.tf_samples, tf_color_scale=dataset.tf_color_scale,
                tf_opacity_scale=dataset.tf_opacity_scale,
                tf_condition_color=dataset.tf_condition_color,
                tf_condition_opacity=dataset.tf_condition_opacity,
                tf_color_sh_degree=getattr(dataset, "tf_color_sh_degree", 1),
                tf_encoder_type=getattr(dataset, "tf_encoder_type", "functional"),
                tf_embedding_fallback=getattr(dataset, "tf_embedding_fallback", "nearest"),
                tf_aware_prune=getattr(dataset, "tf_aware_prune", True),
                tf_use_lookup=getattr(dataset, "tf_use_lookup", False),
                tf_veg_packed=getattr(dataset, "tf_veg_packed", False),
                tf_lookup_mode=getattr(dataset, "tf_lookup_mode", "joint"),
                tf_lookup_bins=getattr(dataset, "tf_lookup_bins", 64),
                tf_lookup_color_scale=getattr(dataset, "tf_lookup_color_scale", 1.0),
                tf_lookup_opacity_scale=getattr(dataset, "tf_lookup_opacity_scale", 4.0),
                tf_exact_visibility_gate=getattr(dataset, "tf_exact_visibility_gate", False),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            load_train_cameras=not skip_train,
            load_test_cameras=not skip_test,
        )

        # Operator-swap protocol: the render-time clip operator is independent of
        # the trained checkpoint (only train.py used clip_operator before). Setting
        # it here lets a single trained interior be rendered through any clip rule
        # ("analytic"/"moment"/"hardcull"/"rara"), isolating the operator. Default
        # "analytic" reproduces the original render behavior. dgs/ndgs render_tcgs
        # reads self.clip_operator; 3dgs/clipgs ignore it (own paths).
        gaussians.clip_operator = getattr(dataset, "clip_operator", "analytic")

        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        use_gsplat = dataset.use_gsplat

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background, mode, measure_fps,
                      bake_appearance=bake_appearance,
                      use_gsplat=use_gsplat, source_path=dataset.source_path)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestCameras(), gaussians, pipeline, background, mode, measure_fps,
                      bake_appearance=bake_appearance,
                      use_gsplat=use_gsplat, source_path=dataset.source_path)


def measure_tf_switch_latency(gaussians, tf_index, repeats=50):
    """Milliseconds to switch presets: one conditioned-appearance evaluation
    written into the base tensors (the cost the baked FPS amortizes)."""
    import time
    for _ in range(5):
        gaussians.bake_appearance(tf_index)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        gaussians.bake_appearance(tf_index)
    torch.cuda.synchronize()
    return (time.time() - start) / repeats * 1000.0


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
    parser.add_argument("--tf_bake_appearance", action="store_true",
                        help="FactorSplat: bake each preset's conditioned appearance "
                             "into the base SH/opacity before FPS measurement, and "
                             "report TF-switch latency. Steady-state rendering is then "
                             "plain dGS -- the deployable configuration.")

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
                args.skip_train, args.skip_test, args.measure_fps,
                bake_appearance=bool(getattr(args, "tf_bake_appearance", False)))
