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
import random
import json
from typing import TYPE_CHECKING
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

if TYPE_CHECKING:
    from scene.gaussian_model_ndgs import GaussianModel


def get_gaussian_model(mode: str):
    """Factory function to get the appropriate GaussianModel class based on mode.

    Args:
        mode: One of "ndgs", "ddgs", "3dgs", "ubs", "ndgs_merged"

    Returns:
        GaussianModel class for the specified mode
    """
    if mode == "ddgs":  ## Neurips 2024
        from scene.gaussian_model_ddgs import GaussianModel
    elif mode == "3dgs":  ## original 3DGS
        from scene.gaussian_model import GaussianModel
    elif mode == "ubs":  ## UBS (ICLR 2026)
        from scene.gaussian_model_ubs import GaussianModel
    elif mode == "ndgs":  ## N-DGS (supports both 6DGS and 7DGS with time, with merged parametrization)
        from scene.gaussian_model_ndgs import GaussianModel
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: ndgs, ddgs, 3dgs, ubs, ndgs_original")
    return GaussianModel


class Scene:

    gaussians : "GaussianModel"

    def __init__(self, args : ModelParams, gaussians : "GaussianModel", load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            # point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(self.loaded_iter))
            # if hasattr(self.gaussians, 'save_ply_decoupled'):
            #     self.gaussians.save_ply_decoupled()
            # self.gaussians.save_ply_decoupled(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:
            if "ddgs" in args.mode:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.source_path)
            elif args.mode == "ubs" or args.mode == "ndgs":
                # UBS and N-DGS models only need spatial_lr_scale (no sh_degree parameter)
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]