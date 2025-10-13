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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
import json
import time
from utils.general_utils import apply_depth_colormap

# Import gsplat functions for rendering and CUDA operations
from gsplat import rasterization
from gsplat import (
    l_triangle_to_rotmat,
    rot_scale_l_triangle_to_covar,
    cond_mean_convariance_opacity,
)

# Import compression utilities
from utils.compress_utils import compress_png, decompress_png, sort_param_dict

# Import viewer utilities
from scene.beta_viewer import BetaRenderTabState

# Import TCGS rasterizer
from tcgs_speedy_rasterizer import (
    GaussianRasterizationSettings as TCGSRasterizationSettings,
    GaussianRasterizer as TCGSRasterizer,
)


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class GaussianModel:
    def setup_functions(self):
        def beta_activation_v1(betas):
            # Stable beta activation with learnable range
            # Instead of pure exponential, use softplus with clamping
            # This prevents gradient explosion while maintaining flexibility
            # Range: softplus(-10) ≈ 0.00005 to softplus(5) ≈ 5.0
            ## clamped = torch.clamp(betas, min=-10.0, max=10.0)
            return 5.77 * F.softplus(betas)

        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        def inverse_softplus(y):
            return y + torch.log(-torch.expm1(-y))

        self.scale_activation = F.softplus
        self.scale_inverse_activation = inverse_softplus

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.beta_activation = beta_activation

        self.l_triangs_activation = lambda x: x
        self.l_triangs_inverse_activation = lambda x: x

    def __init__(self, sh_degree: int = 3, input_dim: int = 6):
        self.input_dim = input_dim

        self._xyz = torch.empty(0)
        self._mean = torch.empty(0)
        self._scale = torch.empty(0)
        self._l_triangle = torch.empty(0)
        self._rgb = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.background = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        # Training attributes for densification (compatibility with 6dgs-iclr train.py)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.percent_dense = 0

        self.setup_functions()

        tril_i, tril_j = torch.tril_indices(input_dim, input_dim, offset=-1)
        # mask out the first 3 skew params (used in get_rotation)
        mask_rest = (tril_i >= 3) | (tril_j >= 3)
        self.rest_i = tril_i[mask_rest].to(torch.int32).to("cuda")
        self.rest_j = tril_j[mask_rest].to(torch.int32).to("cuda")

    def capture(self):
        return (
            self._xyz,
            self._mean,
            self._scale,
            self._l_triangle,
            self._rgb,
            self._opacity,
            self._beta,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._mean,
            self._scale,
            self._l_triangle,
            self._rgb,
            self._opacity,
            self._beta,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scale(self):
        return self.scale_activation(self._scale)

    @property
    def get_l_triangle(self):
        return self.l_triangs_activation(self._l_triangle)

    @property
    def get_mean(self):
        return torch.cat([self._xyz, self._mean], dim=-1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_beta(self):
        return self.beta_activation(self._beta)

    @property
    def get_rotation(self):
        return l_triangle_to_rotmat(self.get_l_triangle[:, :3])

    # @property
    # def get_covariance(self):
    #     d = self.get_scale
    #     R = self.get_rotation
    #     return torch.einsum("nik,nk,njk->nij", R, d**2, R)

    @property
    def get_covariance(self):
        return rot_scale_l_triangle_to_covar(
            self.get_rotation,
            self.get_scale,
            self.get_l_triangle,
            self.rest_i,
            self.rest_j,
        )

    @property
    def get_xyz_covariance(self):
        return rot_scale_l_triangle_to_covar(
            self.get_rotation,
            self.get_scale,
            self.get_l_triangle,
            self.rest_i,
            self.rest_j,
            spatial_block=True,
        )

    def get_cond_mean_convariance_opacity(self, q):
        v = self.get_covariance
        m = self.get_mean
        o = self.get_opacity
        b = self.get_beta[:, 1:]
        return cond_mean_convariance_opacity(m, v, o, b, q)

    @property
    def get_xyz(self):
        """Return spatial positions (for compatibility with 6dgs-iclr train.py)."""
        return self._xyz

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, mcmc_cap_max=None, densification_strategy="standard"):
        """
        Initialize Gaussians from point cloud data.

        Args:
            pcd: Point cloud data with points and colors
            spatial_lr_scale: Spatial learning rate scale
            mcmc_cap_max: Maximum number of Gaussians for MCMC (optional)
            densification_strategy: "standard" or "mcmc" (default: "standard")
        """
        self.spatial_lr_scale = spatial_lr_scale

        # Get point cloud as numpy array
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)

        # Apply random sampling if MCMC is enabled and points exceed cap
        if densification_strategy == "mcmc" and mcmc_cap_max is not None and len(pcd_points) > mcmc_cap_max:
            print(f"\n[MCMC Init] Point cloud has {len(pcd_points)} points, sampling {mcmc_cap_max} for initialization")
            sampled_indices = np.random.choice(len(pcd_points), mcmc_cap_max, replace=False)
            pcd_points = pcd_points[sampled_indices]
            pcd_colors = pcd_colors[sampled_indices]

        fused_point_cloud = torch.tensor(pcd_points).float().cuda()
        fused_color = torch.tensor(pcd_colors).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        xyzs = fused_point_cloud
        means = torch.empty(fused_point_cloud.shape[0], 3, device="cuda").uniform_(
            -1.0, 1.0
        )
        if self.input_dim == 7:
            means_time = torch.empty(
                fused_point_cloud.shape[0], 1, device="cuda"
            ).uniform_(0.0, 1.0)
            means = torch.cat([means, means_time], dim=1)

        # Use the sampled point cloud for KNN distance computation
        dist2 = (
            knn(fused_point_cloud)[:, 1:] ** 2
        ).mean(dim=-1)

        scales = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(
            1, 3
        )
        if self.input_dim > 3:
            scales_rest = self.scale_inverse_activation(
                torch.normal(
                    1,
                    1e-5,
                    size=(fused_point_cloud.shape[0], self.input_dim - 3),
                    device="cuda",
                )
            )
            scales = torch.cat([scales, scales_rest], dim=1)

        l_triangles = self.l_triangs_inverse_activation(
            torch.normal(
                0,
                1e-5,
                size=(
                    fused_point_cloud.shape[0],
                    (self.input_dim**2 + self.input_dim) // 2 - self.input_dim,
                ),
                device="cuda",
            )
        )

        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        betas = torch.zeros(
            (fused_point_cloud.shape[0], self.input_dim - 2),
            dtype=torch.float,
            device="cuda",
        )
        if self.input_dim == 7:
            betas[:, 1:4] -= 3

        self._xyz = nn.Parameter(xyzs.requires_grad_(True))
        self._mean = nn.Parameter(means.requires_grad_(True))
        self._rgb = nn.Parameter(fused_color.requires_grad_(True))
        self._scale = nn.Parameter(scales.requires_grad_(True))
        self._l_triangle = nn.Parameter(l_triangles.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))

    def prune(self, live_mask):
        self._xyz = self._xyz[live_mask]
        self._mean = self._mean[live_mask]
        self._rgb = self._rgb[live_mask]
        self._scale = self._scale[live_mask]
        self._l_triangle = self._l_triangle[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]

    def training_setup(self, training_args):
        # Initialize densification tracking
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._mean],
                "lr": training_args.mean_lr,
                "name": "mean",
            },
            {"params": [self._rgb], "lr": training_args.rgb_lr, "name": "rgb"},
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {"params": [self._beta], "lr": training_args.beta_lr, "name": "beta"},
            {
                "params": [self._scale],
                "lr": training_args.scale_lr,
                "name": "scale",
            },
            {
                "params": [self._l_triangle],
                "lr": training_args.l_triangle_lr,
                "name": "l_triangle",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def oneupSHdegree(self):
        """Dummy method for compatibility with 6dgs-iclr train.py (uses RGB, not SH)."""
        pass

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "red", "green", "blue", "opacity"]
        for i in range(self._beta.shape[1]):
            l.append("beta_{}".format(i))
        for i in range(self.input_dim - 3):
            l.append("mean_{}".format(i))
        for i in range(self._scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._l_triangle.shape[1]):
            l.append("l_triangle_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        mean = self._mean.detach().cpu().numpy()
        rgb = self._rgb.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        l_triangle = self._l_triangle.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, rgb, opacities, betas, mean, scale, l_triangle),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        mean_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("mean_")
        ]
        mean_names = sorted(mean_names, key=lambda x: int(x.split("_")[-1]))
        mean = np.zeros((xyz.shape[0], len(mean_names)))
        for idx, attr_name in enumerate(mean_names):
            mean[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        rgb = np.zeros((xyz.shape[0], 3))
        rgb[:, 0] = np.asarray(plydata.elements[0]["red"])
        rgb[:, 1] = np.asarray(plydata.elements[0]["green"])
        rgb[:, 2] = np.asarray(plydata.elements[0]["blue"])

        beta_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("beta_")
        ]
        beta_names = sorted(beta_names, key=lambda x: int(x.split("_")[-1]))
        betas = np.zeros((xyz.shape[0], len(beta_names)))
        for idx, attr_name in enumerate(beta_names):
            betas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        l_triangle_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("l_triangle")
        ]
        l_triangle_names = sorted(l_triangle_names, key=lambda x: int(x.split("_")[-1]))
        l_triangles = np.zeros((xyz.shape[0], len(l_triangle_names)))
        for idx, attr_name in enumerate(l_triangle_names):
            l_triangles[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rgb = nn.Parameter(
            torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._beta = nn.Parameter(
            torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scale = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._l_triangle = nn.Parameter(
            torch.tensor(l_triangles, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )

    def save_png(self, path):
        path = os.path.join(path, "png")
        mkdir_p(path)
        start_time = time.time()
        opacities = self.get_opacity
        N = opacities.numel()
        n_sidelen = int(N**0.5)
        n_crop = N - n_sidelen**2
        if n_crop:
            index = torch.argsort(opacities.squeeze(), descending=True)
            mask = torch.zeros(N, dtype=torch.bool, device=opacities.device).scatter_(
                0, index[:-n_crop], True
            )
            self.prune(mask.squeeze())
        meta = {}
        param_dict = {
            "xyz": self._xyz,
            "mean": self._mean,
            "rgb": self._rgb,
            "opacity": self._opacity,
            "beta": self._beta,
            "scale": self._scale,
            "l_triangle": self.get_l_triangle,
        }
        param_dict = sort_param_dict(param_dict, n_sidelen)
        for k in param_dict.keys():
            if param_dict[k] is not None and param_dict[k].numel() != 0:
                if k == "xyz":
                    meta[k] = compress_png(path, k, param_dict[k], n_sidelen, bit=32)
                else:
                    meta[k] = compress_png(path, k, param_dict[k], n_sidelen)

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        end_time = time.time()
        print(f"Compression time: {end_time - start_time:.2f} seconds")

    def load_png(self, path):
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        xyz = decompress_png(path, "xyz", meta["xyz"])
        mean = (
            decompress_png(path, "mean", meta["mean"])
            if "mean" in meta
            else np.zeros((xyz.shape[0], self.input_dim - 3), dtype=np.float32)
        )
        rgb = decompress_png(path, "rgb", meta["rgb"])
        opacity = decompress_png(path, "opacity", meta["opacity"])
        beta = decompress_png(path, "beta", meta["beta"])
        scale = decompress_png(path, "scale", meta["scale"])
        l_triangle = decompress_png(path, "l_triangle", meta["l_triangle"])
        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rgb = nn.Parameter(
            torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._beta = nn.Parameter(
            torch.tensor(beta, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scale = nn.Parameter(
            torch.tensor(scale, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._l_triangle = nn.Parameter(
            torch.tensor(l_triangle, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)

                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """Prune Gaussians based on mask (True = keep, False = remove)."""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_mean,
        new_rgb,
        new_opacities,
        new_betas,
        new_scale,
        new_l_triangle,
    ):
        d = {
            "xyz": new_xyz,
            "mean": new_mean,
            "rgb": new_rgb,
            "opacity": new_opacities,
            "beta": new_betas,
            "scale": new_scale,
            "l_triangle": new_l_triangle,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        # Reset densification tracking tensors for all points (following original 3DGS approach)
        # This resets gradient accumulation after densification, which is the standard behavior
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "mean": self._mean,
            "rgb": self._rgb,
            "opacity": self._opacity,
            "beta": self._beta,
            "scale": self._scale,
            "l_triangle": self._l_triangle,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]

            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                # No optimizer state yet, just update the parameter
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        torch.cuda.empty_cache()

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(
            1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1)
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps,
            min=0.005,
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        return (
            self._xyz[idxs],
            self._mean[idxs],
            self._rgb[idxs],
            new_opacity,
            self._beta[idxs],
            self._scale[idxs],
            self._l_triangle[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        # print(f"Relocate: {dead_mask.sum().item()}")
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            relocated_xyz,
            relocated_mean,
            relocated_rgb, 
            relocated_opacity,
            relocated_beta,
            relocated_scale,
            relocated_l_triangle,
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._xyz.index_copy_(0, dead_indices, relocated_xyz)
        self._mean.index_copy_(0, dead_indices, relocated_mean)
        self._rgb.index_copy_(0,  dead_indices, relocated_rgb)
        self._opacity.index_copy_(0, dead_indices, relocated_opacity)
        self._beta.index_copy_(0,  dead_indices, relocated_beta)
        self._scale.index_copy_(0, dead_indices, relocated_scale)
        self._l_triangle.index_copy_(0, dead_indices, relocated_l_triangle)

        self._opacity.index_copy_(0, reinit_idx, self._opacity.index_select(0, dead_indices))

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        # print(f"Add: {num_gs}, Now {target_num}")

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_mean,
            new_rgb,
            new_opacity,
            new_beta,
            new_scale,
            new_l_triangle,
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity

        self.densification_postfix(
            new_xyz,
            new_mean,
            new_rgb,
            new_opacity,
            new_beta,
            new_scale,
            new_l_triangle,
        )
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def get_scaling(self):
        """
        Get 3D scales from ND Gaussian for densification.
        Extracts spatial scales for compatibility with 6dgs-iclr training.
        """
        # Return first 3 dimensions of scales (spatial scales)
        return self.get_scale[:, :3]

    def get_rotation_scale(self):
        """
        Get rotation and scale for densification operations.
        Compatible with 6dgs-iclr densify_and_split.
        """
        rotation = self.get_rotation
        scale = self.get_scale
        return rotation, scale

    @property
    def get_scaling_cond(self):
        """
        Alternative scaling method using conditional covariance from 6D Gaussian.
        Computes scale by slicing the 6D covariance and computing conditional covariance.
        Uses beta parameters to adjust for uncertainty/bandwidth (UBS-specific).
        """
        v = self.get_covariance

        # Slice the 6D covariance matrix
        v_11 = v[:, :3, :3]
        v_12 = v[:, :3, 3:]
        v_21 = v[:, 3:, :3]
        v_22 = v[:, 3:, 3:]

        # Get beta parameters for uncertainty adjustment
        # beta shape: [N, input_dim-2] = [N, C] where C = D-3 (conditional dims)
        # For 6D: [N, 4] (1 spatial + 3 view), For 7D: [N, 5] (1 spatial + 3 view + 1 time)
        beta = self.get_beta[:, 1:]  # [N, C] - exclude first beta (used elsewhere)
        beta_adj = torch.clamp_max(beta / 4.0, 1.0)  # [N, C] - clamp like gsplat
        v_22_inv = torch.linalg.inv(v_22)

        # Compute regression matrix with per-dimension beta adjustment (like gsplat CUDA impl)
        v_regr = torch.bmm(v_12, v_22_inv)  # [N, 3, C]
        v_regr_beta = v_regr * beta_adj.unsqueeze(1)  # [N, 3, C] * [N, 1, C] -> [N, 3, C]

        # Compute beta-adjusted conditional covariance
        v_change = torch.bmm(v_regr_beta, v_21)  # [N, 3, 3]
        v_cond = v_11 - v_change

        U, S, _ = torch.linalg.svd(v_cond)
        scale = torch.sqrt(S)
        return scale

    @property
    def get_rotation_scale_cond(self):
        """
        Alternative rotation/scale method using conditional covariance from 6D Gaussian.
        Computes both rotation and scale from the conditional covariance.
        Uses beta parameters to adjust for uncertainty/bandwidth (UBS-specific).
        """
        v = self.get_covariance

        # Slice the 6D covariance matrix
        v_11 = v[:, :3, :3]
        v_12 = v[:, :3, 3:]
        v_21 = v[:, 3:, :3]
        v_22 = v[:, 3:, 3:]

        # Get beta parameters for uncertainty adjustment
        # beta shape: [N, input_dim-2] = [N, C] where C = D-3 (conditional dims)
        beta = self.get_beta[:, 1:]  # [N, C] - exclude first beta (used elsewhere)
        beta_adj = torch.clamp_max(beta / 4.0, 1.0)  # [N, C] - clamp like gsplat
        v_22_inv = torch.linalg.inv(v_22)

        # Compute regression matrix with per-dimension beta adjustment (like gsplat CUDA impl)
        v_regr = torch.bmm(v_12, v_22_inv)  # [N, 3, C]
        v_regr_beta = v_regr * beta_adj.unsqueeze(1)  # [N, 3, C] * [N, 1, C] -> [N, 3, C]

        # Compute beta-adjusted conditional covariance
        v_change = torch.bmm(v_regr_beta, v_21)  # [N, 3, 3]
        v_cond = v_11 - v_change

        U, S, _ = torch.linalg.svd(v_cond)
        scale = torch.sqrt(S)
        rotation = U

        # Ensure right-handed coordinate system
        det = torch.linalg.det(rotation)
        rotation[:, :, -1] *= det.sign().unsqueeze(-1)
        return rotation, scale

    def reset_opacity(self):
        """Reset opacity for all Gaussians (called during training)."""
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradients and small scales."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Use conditional covariance-based scaling for determining small Gaussians
        scale = self.get_scaling_cond
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale[:, :3], dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_mean = self._mean[selected_pts_mask]
        new_rgb = self._rgb[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_beta = self._beta[selected_pts_mask]
        new_scale = self._scale[selected_pts_mask]
        new_l_triangle = self._l_triangle[selected_pts_mask]

        self.densification_postfix(new_xyz, new_mean, new_rgb, new_opacity, new_beta, new_scale, new_l_triangle)

    def densify_and_split(self, grads, grad_threshold, scene_extent, rotation, scale, N=2):
        """Split large Gaussians with high gradients.

        Note: grads is from BEFORE densify_and_clone, so it only covers the original points.
        We only consider the original points for splitting (not the newly cloned ones).

        Args:
            grads: Gradient accumulation for original points
            grad_threshold: Gradient threshold for splitting
            scene_extent: Scene extent for percent_dense calculation
            rotation: Rotation matrices from conditional covariance [N, 3, 3]
            scale: Scale vectors from conditional covariance [N, 3]
            N: Number of splits per Gaussian (default 2)
        """
        n_original_points = grads.shape[0]
        n_current_points = self.get_xyz.shape[0]

        # Only evaluate gradients for the original points (not newly cloned ones)
        # Create a mask that's the size of CURRENT points, but only considers original points
        grads_squeezed = grads.squeeze()

        # For original points, check gradient threshold
        selected_pts_mask = torch.zeros((n_current_points), device="cuda", dtype=bool)
        selected_pts_mask[:n_original_points] = grads_squeezed >= grad_threshold

        # Also check scale threshold (large Gaussians) - use passed scale from conditional covariance
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale[:, :3], dim=1).values > self.percent_dense*scene_extent)

        # Sample new positions in spatial space (3D) using conditional covariance scale/rotation
        stds_spatial = scale[selected_pts_mask][:, :3].repeat(N, 1)  # Spatial scales (xyz) from conditional cov

        # Sample spatial offsets
        spatial_samples = torch.normal(mean=0, std=stds_spatial)
        # Use rotation from conditional covariance (passed as parameter)
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)

        # Transform spatial samples to world space
        new_xyz = torch.bmm(rots, spatial_samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)

        # Mean parameters: just copy without modification (they're view/appearance parameters)
        new_mean = self._mean[selected_pts_mask].repeat(N, 1)

        new_rgb = self._rgb[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_beta = self._beta[selected_pts_mask].repeat(N, 1)
        new_scale = self._scale[selected_pts_mask]
        # Scale down for split
        new_scale = self.scale_inverse_activation(self.scale_activation(new_scale) * 0.8).repeat(N, 1)
        new_l_triangle = self._l_triangle[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_mean, new_rgb, new_opacity, new_beta, new_scale, new_l_triangle)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        """Main densification and pruning routine (called during training)."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Clone and split (compute scale/rotation using conditional covariance for current state)
        self.densify_and_clone(grads, max_grad, extent)
        rotation, scale = self.get_rotation_scale_cond
        self.densify_and_split(grads, max_grad, extent, rotation, scale)

        # Prune low opacity and large Gaussians
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_cond[:, :3].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Accumulate gradients for densification."""
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render(self, viewpoint_camera, render_mode="RGB", mask=None):
        if mask == None:
            mask = torch.ones_like(self.get_opacity.squeeze()).bool()

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)

        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)

        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        if self.input_dim > 3:
            cam_pos = viewpoint_camera.camera_center
            view_dir = self._xyz - cam_pos.unsqueeze(0)
            view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
            if self.input_dim == 6:
                query = view_dir
            elif self.input_dim == 7:
                timestamp = torch.full(
                    (view_dir.shape[0], 1),
                    viewpoint_camera.timestamp,
                    device=view_dir.device,
                    dtype=view_dir.dtype,
                )
                query = torch.cat([view_dir, timestamp], dim=-1)
            else:
                raise NotImplementedError("Only implemented for 6D or 7D query")
            means, convs, opacities = self.get_cond_mean_convariance_opacity(query)
        else:
            means = self.get_mean
            convs = self.get_covariance
            opacities = self.get_opacity

        rgbs, alphas, meta = rasterization(
            means=means[mask],
            l_triagnles=self.get_l_triangle[mask],
            scales=self.get_scale[mask],
            opacities=opacities.squeeze()[mask],
            betas=self.get_beta[:, :1].squeeze()[mask],
            colors=self._rgb[mask],
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode,
            covars=convs[mask],
        )

        # # Convert from N,H,W,C to N,C,H,W format
        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        return {
            "render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0,
        }

    def render_tcgs(
        self,
        viewpoint_camera,
        render_mode="RGB",
        mask=None,
        use_tcgs=False,
        tight_snugbox=False,
        scaling_modifier=1.0,
    ):
        if render_mode != "RGB":
            raise NotImplementedError("render_tcgs currently supports render_mode='RGB' only.")

        device = self._xyz.device
        if mask is None:
            mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=device)
        else:
            mask = mask.to(dtype=torch.bool, device=device)

        if self.input_dim > 3:
            cam_pos = viewpoint_camera.camera_center
            view_dir = self._xyz - cam_pos.unsqueeze(0)
            view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
            if self.input_dim == 6:
                query = view_dir
            elif self.input_dim == 7:
                timestamp = torch.full(
                    (view_dir.shape[0], 1),
                    viewpoint_camera.timestamp,
                    device=view_dir.device,
                    dtype=view_dir.dtype,
                )
                query = torch.cat([view_dir, timestamp], dim=-1)
            else:
                raise NotImplementedError("Only implemented for 6D or 7D query")
            means, convs, opacities = self.get_cond_mean_convariance_opacity(query)
        else:
            means = self.get_mean
            convs = self.get_covariance
            opacities = self.get_opacity

        means3d = means[mask][..., :3].contiguous()

        # Convert covars from 3x3 symmetric matrix to upper-triangular 6-element vector
        # Upper-tri order: [0,0], [0,1], [0,2], [1,1], [1,2], [2,2] (required by TCGS rasterizer)
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = convs[mask][..., tri_indices[0], tri_indices[1]].contiguous()  # [N, 6]
        rgba = self._rgb[mask].contiguous()  # Should be [N, 3]

        # Extract first beta dimension for TCGS rasterizer and keep as [N, 1]
        betas_full = self.get_beta[mask].contiguous()
        if betas_full.dim() > 1 and betas_full.shape[-1] > 1:
            # Take first beta dimension, keep as [N, 1] for gradient compatibility
            betas = betas_full[:, 0:1].contiguous()
        else:
            betas = betas_full
            # Ensure it's [N, 1] not [N]
            if betas.dim() == 1:
                betas = betas.unsqueeze(-1)

        # Keep opacities as [N, 1] for gradient compatibility with TCGS rasterizer
        opacities = opacities[mask].contiguous()
        if opacities.dim() == 1:
            opacities = opacities.unsqueeze(-1)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        bg_color = (
            self.background.to(device=means3d.device, dtype=means3d.dtype)
            if self.background.numel()
            else torch.tensor([0.0, 0.0, 0.0], device=means3d.device, dtype=means3d.dtype)
        )

        viewmatrix = viewpoint_camera.world_view_transform.to(means3d.device)
        projmatrix = viewpoint_camera.full_proj_transform.to(means3d.device)
        campos = viewpoint_camera.camera_center.to(means3d.device)

        raster_settings = TCGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            x_threshold=viewpoint_camera.x_threshold if viewpoint_camera.x_threshold is not None else float('inf'),
            use_tcgs=use_tcgs,
            tight_snugbox=tight_snugbox,
            debug=False,
        )

        rasterizer = TCGSRasterizer(raster_settings=raster_settings)

        # Create screenspace_points like 6dgs-iclr to maintain proper gradient flow
        # The rasterizer expects means2D to be connected to a [N, 3] gradient graph
        screenspace_points = torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device=means3d.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2d = screenspace_points  # Pass the full [N, 3] tensor, rasterizer will use [:, :2]
        scores = means3d.new_empty(0)

        outputs = rasterizer(
            means3D=means3d,
            means2D=means2d,
            opacities=opacities,
            scores=scores,
            colors_precomp=rgba,
            cov3D_precomp=covars,
            betas=betas,
        )

        if len(outputs) == 3:
            rendered_image, radii, _ = outputs
        else:
            rendered_image, radii = outputs

        if rendered_image.dim() == 4:
            rendered_image = rendered_image[0]
        rgbs = rendered_image.contiguous()

        # Ensure radii is on CUDA device
        if radii.device.type != 'cuda':
            radii = radii.cuda()

        return {
            "render": rgbs,
            "viewspace_points": means2d,
            "visibility_filter": radii > 0,
            "radii": radii,
            "is_used": radii > 0,
        }

    @torch.no_grad()
    def view(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer."""
        assert isinstance(render_tab_state, BetaRenderTabState)

        def quantile_mask(beta, b_xyz=(0, 100), b_view=(0, 100), b_time=(0, 100)):
            """
            beta: [N, 2] -> (x, v) or [N, 3+] -> (x, v, t, ...)
            b_xyz, b_view, b_time: (lo%, hi%) in [0, 100]
            """
            qx_lo, qx_hi = b_xyz[0] / 100, b_xyz[1] / 100
            qv_lo, qv_hi = b_view[0] / 100, b_view[1] / 100

            x = beta[:, 0]
            v = beta[:, 1:4].mean(dim=-1)

            mask = (
                (x >= x.quantile(qx_lo))
                & (x <= x.quantile(qx_hi))
                & (v >= v.quantile(qv_lo))
                & (v <= v.quantile(qv_hi))
            )

            # Optional t-channel (if present)
            if b_time is not None:
                qt_lo, qt_hi = b_time[0] / 100, b_time[1] / 100
                t = beta[:, 4]
                mask = mask & (t >= t.quantile(qt_lo)) & (t <= t.quantile(qt_hi))

            return mask

        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((W, H))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")

        if center:
            self._xyz -= self._xyz.mean(dim=0, keepdim=True)

        if self.input_dim > 3:
            cam_pos = c2w[:3, 3]
            view_dir = self._xyz - cam_pos.unsqueeze(0)
            view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
            if self.input_dim == 6:
                query = view_dir
            elif self.input_dim == 7:
                timestamp = torch.full(
                    (view_dir.shape[0], 1),
                    render_tab_state.timestamp,
                    device=view_dir.device,
                    dtype=view_dir.dtype,
                )
                query = torch.cat([view_dir, timestamp], dim=-1)
            else:
                raise NotImplementedError("Only implemented for 6D or 7D query")
            means, convs, opacities = self.get_cond_mean_convariance_opacity(query)
        else:
            means = self.get_mean
            convs = self.get_covariance
            opacities = self.get_opacity

        render_mode = render_tab_state.render_mode

        mask = quantile_mask(
            self._beta,
            b_xyz=render_tab_state.b_xyz,
            b_view=render_tab_state.b_view,
            b_time=render_tab_state.b_time if self.input_dim == 7 else None,
        )

        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        render_colors, alphas, meta = rasterization(
            means=means[mask],
            l_triagnles=self.get_l_triangle[mask],
            scales=self.get_scale[mask],
            opacities=opacities.squeeze()[mask],
            betas=self.get_beta[:, :1].squeeze()[mask],
            colors=self._rgb[mask],
            viewmats=torch.linalg.inv(c2w).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode if render_mode != "Alpha" else "RGB",
            covars=convs[mask],
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
        )
        render_tab_state.total_count_number = len(self.get_mean)
        render_tab_state.rendered_count_number = (meta["radii"] > 0).sum().item()

        if render_mode == "Alpha":
            render_colors = alphas

        if render_colors.shape[-1] == 1:
            render_colors = apply_depth_colormap(render_colors)

        return render_colors[0].cpu().numpy()

    @torch.no_grad()
    def view_tcgs(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer using TCGS rasterizer.

        This method is identical to view() but uses the TCGS rasterizer (render_tcgs)
        instead of the default gsplat rasterizer (render). This allows for:
        - Comparison between TCGS and gsplat rendering quality
        - Leveraging TCGS-specific features (if use_tcgs=True in settings)
        - Consistent behavior between training and viewer visualization

        Usage:
            # In BetaViewer initialization or config, set:
            viewer = BetaViewer(
                server=server,
                render_fn=lambda cs, rts: beta_model.view_tcgs(cs, rts),
                ...
            )
        """
        # Start timing for FPS calculation
        start_time = time.time()

        if BetaRenderTabState is not None:
            assert isinstance(render_tab_state, BetaRenderTabState)

        def quantile_mask(beta, b_xyz=(0, 100), b_view=(0, 100), b_time=(0, 100)):
            """
            beta: [N, 2] -> (x, v) or [N, 3+] -> (x, v, t, ...)
            b_xyz, b_view, b_time: (lo%, hi%) in [0, 100]
            """
            qx_lo, qx_hi = b_xyz[0] / 100, b_xyz[1] / 100
            qv_lo, qv_hi = b_view[0] / 100, b_view[1] / 100

            x = beta[:, 0]
            v = beta[:, 1:4].mean(dim=-1)

            mask = (
                (x >= x.quantile(qx_lo))
                & (x <= x.quantile(qx_hi))
                & (v >= v.quantile(qv_lo))
                & (v <= v.quantile(qv_hi))
            )

            # Optional t-channel (if present)
            if b_time is not None:
                qt_lo, qt_hi = b_time[0] / 100, b_time[1] / 100
                t = beta[:, 4]
                mask = mask & (t >= t.quantile(qt_lo)) & (t <= t.quantile(qt_hi))

            return mask

        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height

        c2w = camera_state.c2w
        K = camera_state.get_K((W, H))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")

        if center:
            self._xyz -= self._xyz.mean(dim=0, keepdim=True)

        # Build camera for render_tcgs
        from scene.cameras import Camera

        # Extract camera parameters from K matrix
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Compute FoV from focal lengths
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        # Camera class expects R and T in COLMAP convention
        # From getWorld2View2: it does Rt[:3, :3] = R.transpose()
        # This means R should be set such that R.T gives the w2c rotation
        # In COLMAP: R.T @ (world_point - T) = camera_point
        # So R.T is the w2c rotation, and T is the camera center

        # c2w matrix: camera-to-world transform
        # We need to provide R and T such that Camera computes the correct w2c
        w2c = torch.linalg.inv(c2w)

        # COLMAP convention (from readColmapCameras):
        # R should be set such that R.T gives w2c rotation
        # T is the camera translation vector
        R = w2c[:3, :3].cpu().numpy().T  # Transpose of w2c rotation
        T = w2c[:3, 3].cpu().numpy()     # w2c translation

        # Create viewpoint camera
        viewpoint_camera = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=torch.zeros((3, H, W)),
            gt_alpha_mask=None,
            image_name="viewer",
            uid=0,
            x_threshold=render_tab_state.x_threshold,
            data_device="cuda",
            timestamp=render_tab_state.timestamp if self.input_dim == 7 else 0.0
        )

        # Apply beta mask
        mask = quantile_mask(
            self._beta,
            b_xyz=render_tab_state.b_xyz,
            b_view=render_tab_state.b_view,
            b_time=render_tab_state.b_time if self.input_dim == 7 else None,
        )

        # Set background
        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        # Call render_tcgs with mask
        render_output = self.render_tcgs(
            viewpoint_camera,
            render_mode=render_tab_state.render_mode,
            mask=mask,
            use_tcgs=True,
            scaling_modifier=1.0
        )

        render_colors = render_output["render"]

        # Update render stats
        render_tab_state.total_count_number = len(self.get_mean)
        render_tab_state.rendered_count_number = render_output["visibility_filter"].sum().item()

        # Handle alpha mode
        if render_tab_state.render_mode == "Alpha":
            # Extract alpha channel or compute from visibility
            render_colors = render_output["visibility_filter"].float().unsqueeze(0)

        # Handle depth colormap
        if render_colors.shape[0] == 1:
            render_colors = apply_depth_colormap(render_colors)

        # Convert from [C, H, W] to [H, W, C] for viewer
        render_colors = render_colors.permute(1, 2, 0)

        # Calculate and update FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
