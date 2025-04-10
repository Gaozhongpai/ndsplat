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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render
from gaussian_renderer import render_flash as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json, time
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(render_path.replace("renders", "renders_principle"), exist_ok=True)
    makedirs(render_path.replace("renders", "renders_non_principle"), exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    # num_points_list = {"num_points": int(gaussians._is_principle.shape[0]), 
    #                    "num_principle_points": int(gaussians._is_principle.sum()),
    #                    "num_non_principle_points": int((~gaussians._is_principle).sum())}
                
    fpslist = []
    is_fps = True
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # ## to calcuate FPS
        if is_fps:
            rendering_principle = None
            num_frames = 500
            # Measure the start time
            start_time = time.time()
            for _ in range(num_frames):
                rendering = render(view, gaussians, pipeline, background, is_test=True)["render"]
            end_time = time.time()
            # Calculate the total time taken
            total_time = end_time - start_time
            # Calculate FPS
            fps = num_frames / total_time
            fpslist.append(fps)
            print(f"Rendering FPS: {fps:.2f}")
            if len(fpslist) == 20:
                print(f"Average Rendering FPS: {np.array(fpslist).mean():.2f}")
        else:
            renderings = render(view, gaussians, pipeline, background, is_test=True)
            rendering = renderings["render"]
            rendering_principle = renderings["render_principle"]
            rendering_non_principle = renderings["render_non_principle"]
            
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if rendering_principle is not None:
            torchvision.utils.save_image(rendering_principle, os.path.join(render_path.replace("renders", "renders_principle"), '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendering_non_principle, os.path.join(render_path.replace("renders", "renders_non_principle"), '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    # file_path = os.path.join(model_path, name, "ours_{}".format(iteration), "num_points.json")
    # with open(file_path, 'w') as file:
    #     json.dump(num_points_list, file, indent=4)
    # print(num_points_list)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        if "nerf_synthetic" in dataset.source_path:
            bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.eval = True
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)