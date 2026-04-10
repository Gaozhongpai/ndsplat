#
# dBS-SH: Direct Beta Splatting with Spherical Harmonics
#
# Same as dBS (gaussian_model_dbs.py) but uses SH color representation instead of direct RGB.
# Combines DGS's direct Cholesky precision parameterization with UBS's Beta kernel:
#   d_i = tanh((L^T delta)_i^2 / 2)            # per-dim distance via direct Cholesky precision
#   alpha_cond = alpha * prod((1-d_i)^{beta_i}) # Beta kernel opacity
#   mu_cond = mu_p + V_pq @ diag(beta_q) @ V_qq @ delta  # beta-modulated position shift
#   Sigma_cond = Sigma_pp                        # no covariance correction (use 3DGS spatial cov)
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, apply_depth_colormap
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
from gsplat import (
    rasterization,
    l_triangle_to_rotmat,
    rot_scale_l_triangle_to_covar,
    slice_dbs,
)
import json
import time
from .beta_viewer import BetaRenderTabState
from utils.sh_utils import RGB2SH

try:
    from tcgs_speedy_rasterizer import (
        GaussianRasterizationSettings as TCGSRasterizationSettings,
        GaussianRasterizer as TCGSRasterizer,
    )
    _HAS_TCGS_RASTERIZER = True
except ImportError:
    _HAS_TCGS_RASTERIZER = False


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class GaussianModel:
    """
    dBS-SH: Direct Beta Splatting with Spherical Harmonics.

    Same conditioning as dBS but uses SH features for color instead of direct RGB.

    Parameters (spatial 3DGS):
        _xyz: [N, 3] positions
        _scale: [N, 3] spatial scales (softplus activation)
        _l_triangle: [N, 3] skew-symmetric rotation params (first 3 of lower triangle)
        _features_dc: [N, 1, 3] DC SH coefficients
        _features_rest: [N, (max_sh_degree+1)^2-1, 3] rest SH coefficients
        _opacity: [N, 1] base opacity (sigmoid activation)

    Parameters (conditioning):
        _mean: [N, C] conditioning mean (C=3 view dir, or C=4 view+time)
        _L_22_inv: [N, C*(C+1)/2] Cholesky of precision V_qq = L @ L^T
        _v_12: [N, 3*C] position displacement matrix
        _beta: [N, C] per-dimension beta (activation: 4.0 * exp(beta))
    """

    def setup_functions(self):
        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        def inverse_softplus(y):
            return y + torch.log(-torch.expm1(-y))

        self.scale_activation = F.softplus
        self.scale_inverse_activation = inverse_softplus

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.beta_activation = beta_activation

    def __init__(self, sh_degree: int = 3, input_dim: int = 6):
        self.input_dim = input_dim
        self.cond_dim = input_dim - 3  # C = 3 for 6DGS, 4 for 7DGS
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._mean = torch.empty(0)
        self._scale = torch.empty(0)  # [N, 3] spatial only
        self._l_triangle = torch.empty(0)  # [N, 3] rotation only (skew-symmetric)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)  # [N, C]
        self._L_22_inv = torch.empty(0)  # [N, C*(C+1)/2]
        self._v_12 = torch.empty(0)  # [N, 3*C]
        self.background = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        # Training attributes for densification
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.percent_dense = 0

        self.setup_functions()

        # Indices for spatial covariance construction (3x3 only)
        self.rest_i = torch.zeros(0, dtype=torch.int32, device="cuda")
        self.rest_j = torch.zeros(0, dtype=torch.int32, device="cuda")

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._mean,
            self._scale,
            self._l_triangle,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._beta,
            self._L_22_inv,
            self._v_12,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._mean,
            self._scale,
            self._l_triangle,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._beta,
            self._L_22_inv,
            self._v_12,
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
        return self._l_triangle

    @property
    def get_rotation(self):
        return l_triangle_to_rotmat(self._l_triangle)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_mean(self):
        """Get conditioning mean [N, C]."""
        return self._mean

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_beta(self):
        return self.beta_activation(self._beta)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_covariance(self):
        """Get 3x3 spatial covariance (no correction, Sigma_pp only)."""
        return rot_scale_l_triangle_to_covar(
            self.get_rotation,
            self.get_scale,
            self._l_triangle,
            self.rest_i,
            self.rest_j,
            spatial_block=True,
        )

    @property
    def get_xyz_covariance(self):
        """Alias for spatial covariance."""
        return self.get_covariance

    @property
    def get_v_12(self):
        """Get v_12 with normalization and spatial scaling applied. [N, 3*C]"""
        v_12_dir = F.normalize(self._v_12, dim=1)
        spatial_scale = self.get_scale.mean(dim=1, keepdim=True)
        return v_12_dir * spatial_scale

    def get_cond_mean_opacity(self, query):
        """
        dBS conditional slicing (CUDA-accelerated).

        Args:
            query: [N, C] query (view direction + optional time)

        Returns:
            m_cond: [N, 3] conditional position
            opacity_scale: [N, 1] opacity scaling factor
        """
        return slice_dbs(
            self._xyz,
            self._mean,
            query,
            self.get_v_12,
            self._L_22_inv,
            self.get_beta,
        )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,
                        mcmc_cap_max: int = None, densification_strategy: str = "standard"):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        N = fused_point_cloud.shape[0]
        C = self.cond_dim

        print("Number of points at initialisation : ", N)

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        xyzs = fused_point_cloud

        # Conditioning mean: random unit vectors for view [N, 3] (+ optional time)
        means = torch.empty(N, 3, device="cuda").uniform_(-1.0, 1.0)
        if self.input_dim == 7:
            means_time = torch.empty(N, 1, device="cuda").uniform_(0.0, 1.0)
            means = torch.cat([means, means_time], dim=1)

        # Spatial scales from KNN
        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Rotation (skew-symmetric, 3 params)
        l_triangles = torch.normal(0, 1e-5, size=(N, 3), device="cuda")

        opacities = inverse_sigmoid(
            0.5 * torch.ones((N, 1), dtype=torch.float, device="cuda")
        )

        # Beta: [N, C], initialized to zeros (activated = 4.0)
        betas = torch.zeros((N, C), dtype=torch.float, device="cuda")
        if self.input_dim == 7:
            betas[:, :3] -= 3  # Lower view betas for 7DGS

        # L_22_inv: [N, C*(C+1)/2], Cholesky of precision
        n_L = C * (C + 1) // 2
        L_22_inv = torch.zeros((N, n_L), device="cuda")

        # V_12: [N, 3*C] position displacement, small init
        v_12 = torch.normal(0, 0.01, size=(N, 3 * C), device="cuda")

        self._xyz = nn.Parameter(xyzs.requires_grad_(True))
        self._mean = nn.Parameter(means.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scales.requires_grad_(True))
        self._l_triangle = nn.Parameter(l_triangles.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))
        self._L_22_inv = nn.Parameter(L_22_inv.requires_grad_(True))
        self._v_12 = nn.Parameter(v_12.requires_grad_(True))

    def prune(self, live_mask):
        self._xyz = self._xyz[live_mask]
        self._mean = self._mean[live_mask]
        self._features_dc = self._features_dc[live_mask]
        self._features_rest = self._features_rest[live_mask]
        self._scale = self._scale[live_mask]
        self._l_triangle = self._l_triangle[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]
        self._L_22_inv = self._L_22_inv[live_mask]
        self._v_12 = self._v_12[live_mask]

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._mean], "lr": training_args.mean_lr, "name": "mean"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._beta], "lr": training_args.beta_lr, "name": "beta"},
            {"params": [self._scale], "lr": training_args.scale_lr, "name": "scale"},
            {"params": [self._l_triangle], "lr": training_args.l_triangle_lr, "name": "l_triangle"},
            {"params": [self._L_22_inv], "lr": training_args.l_triangle_lr, "name": "L_22_inv"},
            {"params": [self._v_12], "lr": training_args.mean_lr, "name": "v_12"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._beta.shape[1]):
            l.append("beta_{}".format(i))
        for i in range(self._mean.shape[1]):
            l.append("mean_{}".format(i))
        for i in range(self._scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._l_triangle.shape[1]):
            l.append("l_triangle_{}".format(i))
        for i in range(self._L_22_inv.shape[1]):
            l.append("L_22_inv_{}".format(i))
        for i in range(self._v_12.shape[1]):
            l.append("v_12_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        mean = self._mean.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        l_triangle = self._l_triangle.detach().cpu().numpy()
        L_22_inv = self._L_22_inv.detach().cpu().numpy()
        v_12 = self._v_12.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, f_dc, f_rest, opacities, betas, mean, scale, l_triangle, L_22_inv, v_12),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        size_bytes = os.path.getsize(path) / (1024.0 * 1024.0)
        print(f"Loaded PLY size: {size_bytes:.1f} MB")

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        print(f"Loaded primitive number: {xyz.shape[0]}")

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        def load_named(prefix):
            names = [p.name for p in plydata.elements[0].properties if p.name.startswith(prefix)]
            names = sorted(names, key=lambda x: int(x.split("_")[-1]))
            arr = np.zeros((xyz.shape[0], len(names)))
            for idx, attr_name in enumerate(names):
                arr[:, idx] = np.asarray(plydata.elements[0][attr_name])
            return arr

        mean = load_named("mean_")
        betas = load_named("beta_")
        scales = load_named("scale_")
        l_triangles = load_named("l_triangle_")
        L_22_inv = load_named("L_22_inv_")
        v_12 = load_named("v_12_")

        def to_param(arr, grad=True):
            return nn.Parameter(
                torch.tensor(arr, dtype=torch.float, device="cuda").requires_grad_(grad)
            )

        self._xyz = to_param(xyz)
        self._mean = to_param(mean)
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = to_param(opacities)
        self._beta = to_param(betas)
        self._scale = to_param(scales)
        self._l_triangle = to_param(l_triangles)
        self._L_22_inv = to_param(L_22_inv)
        self._v_12 = to_param(v_12)

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]
        self._v_12 = optimizable_tensors["v_12"]

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
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_betas,
        new_scale,
        new_l_triangle,
        new_L_22_inv,
        new_v_12,
    ):
        d = {
            "xyz": new_xyz,
            "mean": new_mean,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "beta": new_betas,
            "scale": new_scale,
            "l_triangle": new_l_triangle,
            "L_22_inv": new_L_22_inv,
            "v_12": new_v_12,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]
        self._v_12 = optimizable_tensors["v_12"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "mean": self._mean,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "beta": self._beta,
            "scale": self._scale,
            "l_triangle": self._l_triangle,
            "L_22_inv": self._L_22_inv,
            "v_12": self._v_12,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]

            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]
        self._v_12 = optimizable_tensors["v_12"]

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
            self._features_dc[idxs],
            self._features_rest[idxs],
            new_opacity,
            self._beta[idxs],
            self._scale[idxs],
            self._l_triangle[idxs],
            self._L_22_inv[idxs],
            self._v_12[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            relocated_xyz,
            relocated_mean,
            relocated_features_dc,
            relocated_features_rest,
            relocated_opacity,
            relocated_beta,
            relocated_scale,
            relocated_l_triangle,
            relocated_L_22_inv,
            relocated_v_12,
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._xyz.index_copy_(0, dead_indices, relocated_xyz)
        self._mean.index_copy_(0, dead_indices, relocated_mean)
        self._features_dc.index_copy_(0, dead_indices, relocated_features_dc)
        self._features_rest.index_copy_(0, dead_indices, relocated_features_rest)
        self._opacity.index_copy_(0, dead_indices, relocated_opacity)
        self._beta.index_copy_(0, dead_indices, relocated_beta)
        self._scale.index_copy_(0, dead_indices, relocated_scale)
        self._l_triangle.index_copy_(0, dead_indices, relocated_l_triangle)
        self._L_22_inv.index_copy_(0, dead_indices, relocated_L_22_inv)
        self._v_12.index_copy_(0, dead_indices, relocated_v_12)

        self._opacity.index_copy_(
            0, reinit_idx, self._opacity.index_select(0, dead_indices)
        )

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_mean,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_beta,
            new_scale,
            new_l_triangle,
            new_L_22_inv,
            new_v_12,
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity

        self.densification_postfix(
            new_xyz,
            new_mean,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_beta,
            new_scale,
            new_l_triangle,
            new_L_22_inv,
            new_v_12,
        )
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    @property
    def get_scaling(self):
        """Get 3D spatial scales for densification compatibility."""
        return self.get_scale[:, :3]

    def reset_opacity(self):
        """Reset opacity for all Gaussians (called during training)."""
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradients and small scales."""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        scale = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale[:, :3], dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_mean = self._mean[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_beta = self._beta[selected_pts_mask]
        new_scale = self._scale[selected_pts_mask]
        new_l_triangle = self._l_triangle[selected_pts_mask]
        new_L_22_inv = self._L_22_inv[selected_pts_mask]
        new_v_12 = self._v_12[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_mean, new_features_dc, new_features_rest,
            new_opacity, new_beta, new_scale, new_l_triangle,
            new_L_22_inv, new_v_12,
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians with high gradients."""
        n_original_points = grads.shape[0]
        n_current_points = self.get_xyz.shape[0]

        grads_squeezed = grads.squeeze()

        selected_pts_mask = torch.zeros((n_current_points), device="cuda", dtype=bool)
        selected_pts_mask[:n_original_points] = grads_squeezed >= grad_threshold

        scale = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale[:, :3], dim=1).values > self.percent_dense * scene_extent
        )

        # Sample new positions using spatial rotation and scale
        stds_spatial = scale[selected_pts_mask][:, :3].repeat(N, 1)
        spatial_samples = torch.normal(mean=0, std=stds_spatial)
        rots = self.get_rotation[selected_pts_mask].repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, spatial_samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)

        new_mean = self._mean[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_beta = self._beta[selected_pts_mask].repeat(N, 1)
        new_scale = self._scale[selected_pts_mask]
        new_scale = self.scale_inverse_activation(self.scale_activation(new_scale) * 0.8).repeat(N, 1)
        new_l_triangle = self._l_triangle[selected_pts_mask].repeat(N, 1)
        new_L_22_inv = self._L_22_inv[selected_pts_mask].repeat(N, 1)
        new_v_12 = self._v_12[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_mean, new_features_dc, new_features_rest,
            new_opacity, new_beta, new_scale, new_l_triangle,
            new_L_22_inv, new_v_12,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        """Main densification and pruning routine."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling[:, :3].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Accumulate gradients for densification."""
        grad = viewspace_point_tensor.grad
        if grad is None:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render(self, viewpoint_camera, render_mode="RGB", mask=None):
        if mask is None:
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
            means, opacity_scale = self.get_cond_mean_opacity(query)
            opacities = self.get_opacity * opacity_scale
            convs = self.get_covariance
        else:
            means = self._xyz
            convs = self.get_covariance
            opacities = self.get_opacity

        shs = self.get_features[mask]

        rgbs, alphas, meta = rasterization(
            means=means[mask],
            l_triagnles=self.get_l_triangle[mask],
            scales=self.get_scale[mask],
            opacities=opacities.squeeze()[mask],
            betas=self.get_beta[:, :1].squeeze()[mask],
            shs=shs,
            sh_degree=self.active_sh_degree,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode,
            covars=convs[mask],
        )

        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        # gsplat radii/means2d are [C, N, ...], squeeze to [N, ...] for single-camera training
        radii = meta["radii"].squeeze(0)
        # Create a viewspace_points tensor compatible with add_densification_stats.
        # gsplat's means2d is non-leaf; use a backward hook to capture its gradient.
        # gsplat means2d gradients are in normalized coordinates; scale to pixel space
        # to match the gradient magnitude expected by standard densification thresholds.
        means2d = meta["means2d"]  # [1, N, 2]
        N = means2d.shape[1]
        W = viewpoint_camera.image_width
        H = viewpoint_camera.image_height
        viewspace_points = torch.zeros((N, 2), device=means2d.device, requires_grad=True)
        if means2d.requires_grad:
            def _hook(grad, vp=viewspace_points, w=W, h=H):
                # gsplat means2d grads are small due to mean-reduced loss;
                # scale by half-image-size to match 3DGS ndc2Pix gradient magnitude
                g = grad.squeeze(0)
                g = g * torch.tensor([w * 0.5, h * 0.5], device=g.device, dtype=g.dtype)
                vp.grad = g
            means2d.register_hook(_hook)
        return {
            "render": rgbs,
            "viewspace_points": viewspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "is_used": radii > 0,
        }

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", mask=None, use_tcgs=True, scaling_modifier=1.0, **kwargs):
        if not _HAS_TCGS_RASTERIZER:
            raise RuntimeError(
                "tcgs_speedy_rasterizer is not available. Please build/install the extension before calling render_tcgs()."
            )

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
            means, opacity_scale = self.get_cond_mean_opacity(query)
            opacities = self.get_opacity * opacity_scale
            convs = self.get_covariance
        else:
            means = self._xyz
            convs = self.get_covariance
            opacities = self.get_opacity

        means3d = means[mask][..., :3].contiguous()

        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = convs[mask][..., tri_indices[0], tri_indices[1]].contiguous()

        shs = self.get_features[mask].contiguous()

        betas_full = self.get_beta[mask].contiguous()
        if betas_full.dim() > 1 and betas_full.shape[-1] > 1:
            betas = betas_full[:, 0:1].contiguous()
        else:
            betas = betas_full
            if betas.dim() == 1:
                betas = betas.unsqueeze(-1)

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
            use_tcgs=use_tcgs,
            tight_snugbox=use_tcgs,
            debug=False,
        )

        rasterizer = TCGSRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device=means3d.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        scores = means3d.new_empty(0)

        rendered_image, radii, render_time, _ = rasterizer(
            means3D=means3d,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scores=scores,
            cov3D_precomp=covars,
            betas=betas,
        )

        if rendered_image.dim() == 4:
            rendered_image = rendered_image[0]
        rgbs = rendered_image.contiguous()

        if radii.device.type != 'cuda':
            radii = radii.cuda()

        return {
            "render": rgbs,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "is_used": radii > 0,
        }

    @torch.no_grad()
    def view(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer."""
        assert isinstance(render_tab_state, BetaRenderTabState)

        def quantile_mask(beta, b_xyz=(0, 100), b_view=(0, 100), b_time=(0, 100)):
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
            means, opacity_scale = self.get_cond_mean_opacity(query)
            opacities = self.get_opacity * opacity_scale
            convs = self.get_covariance
        else:
            means = self._xyz
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

        shs = self.get_features[mask]

        render_colors, alphas, meta = rasterization(
            means=means[mask],
            l_triagnles=self.get_l_triangle[mask],
            scales=self.get_scale[mask],
            opacities=opacities.squeeze()[mask],
            betas=self.get_beta[:, :1].squeeze()[mask],
            shs=shs,
            sh_degree=self.active_sh_degree,
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
        render_tab_state.total_count_number = len(self._xyz)
        render_tab_state.rendered_count_number = (meta["radii"] > 0).sum().item()

        if render_mode == "Alpha":
            render_colors = alphas

        if render_colors.shape[-1] == 1:
            render_colors = apply_depth_colormap(render_colors)

        return render_colors[0].cpu().numpy()

    @torch.no_grad()
    def view_tcgs(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer using TCGS rasterizer."""
        start_time = time.time()

        if BetaRenderTabState is not None:
            assert isinstance(render_tab_state, BetaRenderTabState)

        def quantile_mask(beta, b_xyz=(0, 100), b_view=(0, 100), b_time=(0, 100)):
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

        from scene.cameras import Camera

        fx = K[0, 0]
        fy = K[1, 1]

        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3].cpu().numpy().T
        T = w2c[:3, 3].cpu().numpy()

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

        mask = quantile_mask(
            self._beta,
            b_xyz=render_tab_state.b_xyz,
            b_view=render_tab_state.b_view,
            b_time=render_tab_state.b_time if self.input_dim == 7 else None,
        )

        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        render_output = self.render_tcgs(
            viewpoint_camera,
            render_mode=render_tab_state.render_mode,
            mask=mask,
            use_tcgs=True,
            scaling_modifier=1.0
        )

        render_colors = render_output["render"]

        render_tab_state.total_count_number = len(self._xyz)
        render_tab_state.rendered_count_number = render_output["visibility_filter"].sum().item()

        if render_tab_state.render_mode == "Alpha":
            render_colors = render_output["visibility_filter"].float().unsqueeze(0)

        if render_colors.shape[0] == 1:
            render_colors = apply_depth_colormap(render_colors)

        render_colors = render_colors.permute(1, 2, 0)

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
