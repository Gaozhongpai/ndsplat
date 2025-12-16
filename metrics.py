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
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
import trimesh
from lpipsPyTorch import lpips, LPIPS
from PIL import Image
from tqdm import tqdm

from utils.image_utils import psnr
from utils.loss_utils import ssim


def readImages(renders_dir, gt_dir):
    """Read rendered and ground truth images from directories."""
    renders = []
    gts = []
    image_names = []
    png_files = [f for f in os.listdir(renders_dir) if f.endswith('.png')]

    for fname in tqdm(png_files, desc="Loading images"):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

    return renders, gts, image_names


def evaluate(model_paths):
    """
    Evaluate rendered images against ground truth.

    Computes PSNR, SSIM, and LPIPS metrics for each model.

    Args:
        model_paths: List of paths to model output directories
    """
    full_dict = {}
    per_view_dict = {}

    print("")

    # Initialize LPIPS criterion
    net_type = 'vgg'
    version = '0.1'
    criterion = LPIPS(net_type, version).to("cuda")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            # Load training time from training_time.txt if available
            training_time_path = Path(scene_dir) / "training_time.txt"
            training_time = None
            if training_time_path.exists():
                with open(training_time_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only parse if file is not empty
                        training_time = float(content)
                        print(f"  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                # Load images
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # Load point cloud to count Gaussians
                ply_path = str(method_dir).replace("test", "point_cloud").replace("ours_", "iteration_")
                ply_path = os.path.join(ply_path, "point_cloud.ply")
                mesh = trimesh.load(ply_path)
                num_gaussians = mesh.vertices.shape[0]

                full_dict[scene_dir][method].update({"Number": num_gaussians})
                print(f"  Number: {num_gaussians}")

                # Add training time if available
                if training_time is not None:
                    full_dict[scene_dir][method].update({"Training_time": training_time})

                # Load FPS from fps.txt if available
                fps_path = method_dir / "fps.txt"
                if fps_path.exists():
                    with open(fps_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Only parse if file is not empty
                            fps = float(content)
                            full_dict[scene_dir][method].update({"FPS": fps})
                            print(f"  FPS: {fps:.2f}")

                # Compute metrics
                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Computing metrics"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], criterion))

                # Compute averages
                ssim_mean = torch.tensor(ssims).mean().item()
                psnr_mean = torch.tensor(psnrs).mean().item()
                lpips_mean = torch.tensor(lpipss).mean().item()

                print(f"  SSIM : {ssim_mean:>12.7f}")
                print(f"  PSNR : {psnr_mean:>12.7f}")
                print(f"  LPIPS: {lpips_mean:>12.7f}")
                print("")

                # Store results
                full_dict[scene_dir][method].update({
                    "SSIM": ssim_mean,
                    "PSNR": psnr_mean,
                    "LPIPS": lpips_mean
                })

                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: l for l, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                })

            # Save results
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Parse command line arguments
    parser = ArgumentParser(description="Compute metrics for rendered images")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str,
                        help="Paths to model output directories")
    args = parser.parse_args()

    evaluate(args.model_paths)
