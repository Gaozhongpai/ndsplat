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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, x_threshold=None, color_idx=None, label=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 timestamp=0.0, compressed_data=None, resolution=None, white_background=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.x_threshold = x_threshold
        self.label = torch.tensor(label).cuda() if label is not None else None
        self.color_idx = color_idx
        self.timestamp = timestamp  # Time dimension for 7DGS

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Support for JPEG compression
        self._compressed_data = compressed_data
        self._resolution = resolution
        self._white_background = white_background
        self._gt_alpha_mask = gt_alpha_mask

        if compressed_data is not None:
            # Lazy loading mode - decompress on first access
            shape = compressed_data['shape']
            self.image_width = shape[2]
            self.image_height = shape[1]
        else:
            # Eager loading mode - store image directly
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 500.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        inverse_transform = self.world_view_transform.inverse()
        self.camera_center = inverse_transform[3, :3]
        self.rotation = inverse_transform[:3, :3].T

    @property
    def original_image(self):
        """Lazy decompression of JPEG-compressed images."""
        if self._compressed_data is not None:
            # Decompress on-demand (no caching to save GPU memory)
            from utils.camera_utils import _decompress_jpeg_to_tensor
            image_tensor = _decompress_jpeg_to_tensor(
                self._compressed_data,
                self._resolution,
                self._white_background
            )
            image = image_tensor.clamp(0.0, 1.0).to(self.data_device)

            # Apply alpha mask if present
            if self._gt_alpha_mask is not None:
                image *= self._gt_alpha_mask.to(self.data_device)
            else:
                image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

            return image
        else:
            # Already loaded in __init__
            return self._original_image

    @original_image.setter
    def original_image(self, value):
        """Setter for eager loading mode."""
        self._original_image = value

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

