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
from concurrent.futures import ThreadPoolExecutor, as_completed

from scene.cameras import Camera
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from tqdm import tqdm
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, *, resolution=None, image_tensor=None, alpha_mask=None):
    if resolution is None:
        resolution = _compute_target_resolution(args, cam_info, resolution_scale)

    if image_tensor is None:
        image_tensor, alpha_mask = _load_image_tensor(
            cam_info.image_path,
            resolution,
            args.white_background
        )

    gt_image = image_tensor[:3, ...]
    loaded_mask = alpha_mask

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  x_threshold=cam_info.x_threshold, color_idx=cam_info.color_idx, label=cam_info.label,
                  data_device=args.data_device, timestamp=cam_info.timestamp)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = [None] * len(cam_infos)
    if not cam_infos:
        return camera_list

    max_workers = min(32, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prepare_camera_payload, args, idx, cam_info, resolution_scale): idx
            for idx, cam_info in enumerate(cam_infos)
        }
        with tqdm(total=len(futures), desc="Preparing cameras") as progress:
            for future in as_completed(futures):
                idx = futures[future]
                resolution, image_tensor, alpha_mask = future.result()
                camera_list[idx] = loadCam(
                    args,
                    idx,
                    cam_infos[idx],
                    resolution_scale,
                    resolution=resolution,
                    image_tensor=image_tensor,
                    alpha_mask=alpha_mask,
                )
                progress.update(1)

    return camera_list


def _load_image_tensor(image_path, resolution, white_background):
    """
    Load an image from disk, optionally composite alpha, and resize using torch-native ops.
    """
    image = read_image(image_path).float() / 255.0  # [C, H, W]

    alpha_mask = None
    if image.shape[0] == 4:
        alpha = image[3:4]
        rgb = image[:3]
        if white_background:
            rgb = rgb * alpha + (1.0 - alpha)
            alpha_mask = None
        else:
            rgb = rgb * alpha
            alpha_mask = alpha
    else:
        rgb = image

    if (resolution[0], resolution[1]) != (rgb.shape[2], rgb.shape[1]):
        size_hw = (resolution[1], resolution[0])
        rgb = F.interpolate(rgb.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)
        if alpha_mask is not None:
            alpha_mask = F.interpolate(alpha_mask.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)

    rgb = rgb.clamp(0.0, 1.0)
    return rgb, alpha_mask


def _compute_target_resolution(args, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        return (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )

    if args.resolution == -1:
        if orig_w > 1600:
            global WARNED
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                      "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    return (int(orig_w / scale), int(orig_h / scale))


def _prepare_camera_payload(args, idx, cam_info, resolution_scale):
    resolution = _compute_target_resolution(args, cam_info, resolution_scale)
    image_tensor, alpha_mask = _load_image_tensor(cam_info.image_path, resolution, args.white_background)
    return resolution, image_tensor, alpha_mask

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
