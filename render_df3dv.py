#
# DF3DV-41 / DF3DV-1K leaderboard render script.
#
# Produces side-by-side |GT|Rendering| PNGs named after the original eval
# images (extra_*.png) into:
#
#     <scene-All>/MODELS/<method>/renders/extra_<ID>.png
#
# which is exactly what the official benchmark tooling
# (extract_leaderboard_images.py / benchmark_df3dv.py / benchmark_leaderboard.py)
# expects: it loads ground truth from undistortion_images_8/extra_*, and
# extracts the *right half* of each concat image in MODELS/<method>/renders/
# as the predicted rendering.
#
# Usage:
#   python render_df3dv.py -m <model_path> -s <scene-All> --method <method> \
#       --iteration best --mode <dgs|dbs> [--use_gsplat] [--input_dim 6]
#

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, get_gaussian_model
from utils.general_utils import safe_state
from render import render_wrapper


def build_gaussians(dataset):
    mode = dataset.mode
    GaussianModel = get_gaussian_model(mode)
    if mode == "3dgs":
        return GaussianModel(dataset.sh_degree)
    elif "ubs" in mode or "dbs" in mode:
        return GaussianModel(sh_degree=dataset.sh_degree, input_dim=dataset.input_dim)
    elif "ndgs" in mode:
        return GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                             use_rot_scale_l_triangle=dataset.use_rot_scale_l_triangle,
                             learnable_lambda_opc=dataset.learnable_lambda_opc,
                             lambda_opc=dataset.lambda_opc)
    elif "dgs" in mode:
        return GaussianModel(dataset.sh_degree, input_dim=dataset.input_dim,
                             use_view_dependent_pos=dataset.use_view_dependent_pos,
                             use_opacity_pos_decouple=dataset.use_opacity_pos_decouple,
                             l_22_inv_init_scale=dataset.l_22_inv_init_scale,
                             lambda_init=dataset.lambda_init,
                             lambda_opc=dataset.lambda_opc)
    raise ValueError(f"Unknown mode: {mode}")


def render_df3dv(dataset, pipeline, iteration, method, scene_all):
    with torch.no_grad():
        mode = dataset.mode
        gaussians = build_gaussians(dataset)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            load_train_cameras=False,
            load_test_cameras=True,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTestCameras()

        # Output dir consumed by the official leaderboard tooling.
        renders_path = os.path.join(scene_all, "MODELS", method, "renders")
        makedirs(renders_path, exist_ok=True)

        print(f"DF3DV: rendering {len(views)} eval views -> {renders_path}")
        for view in tqdm(views, desc="Rendering DF3DV eval views"):
            out = render_wrapper(view, gaussians, pipeline, background, mode,
                                 is_test=False, tight_snugbox=False,
                                 use_gsplat=dataset.use_gsplat)
            rendering = out["render"].clamp(0.0, 1.0)
            gt = view.original_image[0:3, :, :].clamp(0.0, 1.0)

            # Side-by-side |GT|Rendering| (benchmark extracts the right half).
            concat = torch.cat([gt, rendering], dim=2)  # [3, H, 2W]

            # Name after the original eval image (image_name has no extension).
            out_name = f"{view.image_name}.png"
            torchvision.utils.save_image(concat, os.path.join(renders_path, out_name))


if __name__ == "__main__":
    parser = ArgumentParser(description="DF3DV leaderboard render script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default="best", type=str,
                        help="Iteration to load (-1 latest, 'best' for best checkpoint)")
    parser.add_argument("--method", required=True, type=str,
                        help="Method name for leaderboard folder (MODELS/<method>/renders)")
    parser.add_argument("--quiet", action="store_true")

    # Training-only params accepted-and-ignored for script convenience.
    parser.add_argument("--noise_lr", type=float, default=1.0)
    parser.add_argument("--opacity_reg", type=float, default=0.01)
    parser.add_argument("--scale_reg", type=float, default=0.01)
    parser.add_argument("--mcmc_cap_max", type=int, default=300000)

    args = get_combined_args(parser)
    args.eval = True
    safe_state(args.quiet)

    iteration = "best" if args.iteration == "best" else int(args.iteration)

    # The eval scene root (<scene>-All) is the source_path.
    scene_all = os.path.abspath(args.source_path)
    render_df3dv(model.extract(args), pipeline.extract(args), iteration,
                 args.method, scene_all)
