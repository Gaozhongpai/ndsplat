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
# Position-based DGS with simplified v_12/L_22_inv parameterization
# Combines N-DGS position shifting with simplified covariance parameterization
#
# Option 2: Normalized Coupling with Learnable Magnitude
# v_12 is decomposed into direction (unit vectors) and magnitude (bounded by spatial scale)
# This provides structural regularization: small Gaussians can only shift small amounts
#

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# Import CUDA-accelerated slice function (reuse color version for position slicing)
# The math is identical: output = mean + v_12 @ V_22^{-1} @ (query - view_mean)
from gsplat import slice_gaussian_simple


class GaussianModel:
    """
    Position-based DGS with simplified v_12/L_22_inv parameterization.

    This model shifts 3D positions based on view direction (like original DGS/N-DGS)
    but uses the simplified parameterization from Color N-DGS:
    - v_12: [N, 3*C] position-view covariance block (C=3 for view, C=4 for view+time)
    - L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision matrix)

    Key insight: V_22^{-1} = L_22_inv @ L_22_inv^T (no matrix inversion at runtime!)

    Conditional position: x_cond = x + v_12 @ V_22^{-1} @ (v - μ_v)
    Opacity scale: exp(-λ * (v - μ_v)^T @ V_22^{-1} @ (v - μ_v))

    Fixed: lambda_opc = 0.35
    """

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, input_dim: int = 6, learnable_lambda_opc: bool = False):
        """
        Initialize Position-based DGS with bounded v_12 parameterization.

        Args:
            sh_degree: Maximum SH degree for color
            input_dim: 6 for position+view, 7 for position+view+time
            learnable_lambda_opc: If True, make lambda_opc a learnable parameter per Gaussian

        Bounded v_12 parameterization:
            v_12 = normalize(_v_12_direction) * sigmoid(_v_12_scale) * max_pos_shift * spatial_scale

        This provides structural regularization like full N-DGS:
        - Small Gaussians can only shift small amounts
        - Large Gaussians can shift more
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.input_dim = input_dim
        self.cond_dim = input_dim - 3  # C = 3 for view-only, 4 for view+time
        self.learnable_lambda_opc = learnable_lambda_opc

        # Standard 3DGS parameters
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._lambda_opc = torch.empty(0)  # Learnable opacity scaling parameter
        self._label = torch.empty(0)

        # Simplified N-DGS parameters (position+view covariance)
        # View direction mean (learned "canonical" view direction per Gaussian)
        self._view_mean = torch.empty(0)  # [N, C] where C=3 or 4 (with time)

        # Bounded v_12 parameterization:
        # v_12 = normalize(_v_12_direction) * sigmoid(_v_12_scale) * max_pos_shift * spatial_scale
        self._v_12_direction = torch.empty(0)  # [N, 3*C] - will be normalized to unit vectors
        self._v_12_scale = torch.empty(0)  # [N, 1] - sigmoid gives [0,1], then scaled by max_pos_shift * spatial scale

        # L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision)
        self._L_22_inv = torch.empty(0)

        # Hyperparameter for maximum position shift
        self.max_pos_shift = 0.5  # Maximum position shift factor (multiplied by spatial_scale)

        # Auxiliary tensors
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # Background color for rendering (set by render_wrapper)
        self.background = torch.empty(0)

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._view_mean,
            self._v_12_direction,
            self._v_12_scale,
            self._L_22_inv,
            self._opacity,
            self._lambda_opc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._view_mean,
         self._v_12_direction,
         self._v_12_scale,
         self._L_22_inv,
         self._opacity,
         self._lambda_opc,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_view_mean(self):
        """Get normalized view mean direction."""
        view_dir = self._view_mean[:, :3]
        view_dir_normalized = view_dir / (view_dir.norm(dim=1, keepdim=True) + 1e-8)
        if self._view_mean.shape[1] > 3:
            return torch.cat([view_dir_normalized, self._view_mean[:, 3:]], dim=1)
        return view_dir_normalized

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_lambda_opc(self):
        """Get lambda_opc parameter (learnable or fixed).

        Returns:
            If learnable_lambda_opc=True: sigmoid-activated per-Gaussian lambda [N, 1]
            If learnable_lambda_opc=False: fixed value of 0.35 for all Gaussians [N, 1]
        """
        if self.learnable_lambda_opc:
            return self.opacity_activation(self._lambda_opc)
        else:
            # Return fixed value of 0.35 for all Gaussians
            return torch.ones_like(self._opacity) * 0.35

    @property
    def get_v_12(self):
        """
        Get effective v_12 (position-view covariance block) with bounded magnitude.

        Computes: v_12 = normalize(_v_12_direction) * sigmoid(_v_12_scale) * max_pos_shift * spatial_scale

        With global normalization: the entire [3*C] direction vector is normalized to unit norm.
        This allows relative magnitudes between x, y, z axes to be encoded in the direction itself.

        With anisotropic scaling: each of the 3 output dimensions is scaled by its corresponding
        spatial scale, allowing position shifts to be bounded per-axis by Gaussian shape.

        This provides structural regularization like full N-DGS:
        - Small Gaussians (small spatial scale) -> small v_12 -> small position shifts
        - Large Gaussians (large spatial scale) -> can have larger v_12 -> larger shifts allowed

        This mimics the implicit constraint in full N-DGS where V_12 = L_11 @ L_21^T
        """
        # Global normalization: normalize entire [3*C] vector, then reshape
        # This allows relative magnitudes between axes to be encoded in the direction
        C = self._v_12_direction.shape[1] // 3
        v_12_dir = F.normalize(self._v_12_direction, dim=1)  # [N, 3*C] normalized to unit norm
        v_12_dir = v_12_dir.view(-1, 3, C)  # Reshape after normalization

        # Magnitude: sigmoid to [0, 1], then scale by max_pos_shift and anisotropic spatial scale
        # This ensures shift magnitude is bounded by max_pos_shift * Gaussian size per axis
        spatial_scale = self.get_scaling  # [N, 3] - full anisotropic scale
        v_12_magnitude = torch.sigmoid(self._v_12_scale) * self.max_pos_shift  # [N, 1]

        # Apply anisotropic scaling: each row i scaled by spatial_scale[:, i]
        v_12_scaled = v_12_dir * (v_12_magnitude * spatial_scale).unsqueeze(-1)  # [N, 3, 1] * [N, 3, C] = [N, 3, C]

        return v_12_scaled.view(-1, 3 * C)  # [N, 3*C]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def slice_gaussian(self, query, lambda_opc=0.35):
        """
        Perform conditional Gaussian slicing for position using simplified parameterization.
        Uses CUDA-accelerated slice_gaussian_simple (math is identical for position/color).

        Given query view direction (+ optional time), compute:
        - Conditional position mean: x_cond = x + v_12 @ V_22^{-1} @ (v - μ_v)
        - Opacity scale: exp(-λ * (v - μ_v)^T @ V_22^{-1} @ (v - μ_v))

        V_22^{-1} = L_22_inv @ L_22_inv^T (precision from Cholesky, no matrix inversion!)

        Args:
            query: Query direction [N, C] where C=3 (view) or 4 (view+time)
            lambda_opc: Opacity scaling factor (default 0.35, can be [N, 1] per-Gaussian)

        Returns:
            x_cond: Conditional 3D position [N, 3]
            opacity_scale: View-dependent opacity scaling [N, 1]
        """
        # Reuse slice_gaussian_simple - the math is identical:
        # output = mean + v_12 @ V_22^{-1} @ (query - view_mean)
        # For color: color_cond = color_mean + shift
        # For position: x_cond = xyz + shift
        x_cond, opacity_scale = slice_gaussian_simple(
            self._xyz,              # [N, 3] - treat position as "color mean"
            self.get_view_mean,     # [N, C] - view direction mean
            query,                  # [N, C] - query view direction
            self.get_v_12,          # [N, 3*C] - position-view covariance (bounded if use_bounded_v12)
            self._L_22_inv,         # [N, C*(C+1)/2] - Cholesky of precision
            lambda_opc,
        )
        return x_cond, opacity_scale

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussians from point cloud data with simplified parameterization.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        num_gaussians = fused_color.shape[0]
        device = "cuda"
        C = self.cond_dim  # 3 for view-only, 4 for view+time

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        from sklearn.neighbors import NearestNeighbors
        def knn(x, K=4):
            x_np = x.cpu().numpy()
            model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(x)

        # Spatial scales (first 3): from KNN distances
        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        # View direction mean: random unit vectors
        view_dir = torch.randn((num_gaussians, 3), device=device)
        view_mean = (view_dir / view_dir.norm(dim=1, keepdim=True)).float()

        # For input_dim=7, append time dimension
        if self.input_dim == 7:
            mean_time = torch.empty(num_gaussians, 1, device=device).uniform_(0.0, 1.0)
            view_mean = torch.cat([view_mean, mean_time], dim=-1)

        # v_12_direction: [N, 3*C] direction vectors (will be normalized in get_v_12)
        # Initialize with small random values
        v_12_direction = torch.normal(0, 0.01, size=(num_gaussians, 3 * C), device=device)

        # v_12_scale: [N, 1] magnitude scale
        # sigmoid(0) = 0.5, so effective magnitude starts at 0.5 * spatial_scale
        v_12_scale = torch.zeros((num_gaussians, 1), device=device)

        # L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision)
        # Diagonal uses exp() activation, so initialize with log(2.0) ≈ 0.693
        # This gives diagonal=2.0 after exp, so V_22^{-1} has diagonal ~4.0
        # Storage format: [l_00, l_10, l_11, l_20, l_21, l_22, ...] (row-major lower triangular)
        n_L_22_inv = C * (C + 1) // 2
        L_22_inv = torch.zeros(num_gaussians, n_L_22_inv, device=device)
        # Diagonal positions: 0 (i=0), 2 (i=1), 5 (i=2), 9 (i=3)
        for i in range(C):
            diag_idx = i * (i + 1) // 2 + i
            L_22_inv[:, diag_idx] = math.log(2.0)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))
        # Initialize lambda_opc: sigmoid(inverse_sigmoid(0.35)) = 0.35
        lambda_opcs = inverse_sigmoid(0.35 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._label = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.int32).cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._view_mean = nn.Parameter(view_mean.requires_grad_(True))
        self._v_12_direction = nn.Parameter(v_12_direction.requires_grad_(True))
        self._v_12_scale = nn.Parameter(v_12_scale.requires_grad_(True))
        self._L_22_inv = nn.Parameter(L_22_inv.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._lambda_opc = nn.Parameter(lambda_opcs.requires_grad_(self.learnable_lambda_opc))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._view_mean], 'lr': training_args.feature_lr, "name": "view_mean"},
            {'params': [self._v_12_direction], 'lr': training_args.feature_lr, "name": "v_12_direction"},
            {'params': [self._v_12_scale], 'lr': training_args.feature_lr, "name": "v_12_scale"},
            {'params': [self._L_22_inv], 'lr': training_args.rotation_lr, "name": "L_22_inv"},
        ]

        # Add lambda_opc parameter (learnable or not)
        if self.learnable_lambda_opc:
            l.append({'params': [self._lambda_opc], 'lr': training_args.opacity_lr, "name": "lambda_opc"})
        else:
            # Still add to optimizer but with lr=0 for consistency in state management
            l.append({'params': [self._lambda_opc], 'lr': 0.0, "name": "lambda_opc"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('lambda_opc')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Simplified parameterization attributes
        for i in range(self._view_mean.shape[1]):
            l.append('view_mean_{}'.format(i))
        for i in range(self._v_12_direction.shape[1]):
            l.append('v_12_direction_{}'.format(i))
        for i in range(self._v_12_scale.shape[1]):
            l.append('v_12_scale_{}'.format(i))
        for i in range(self._L_22_inv.shape[1]):
            l.append('L_22_inv_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        lambda_opcs = self._lambda_opc.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        view_mean = self._view_mean.detach().cpu().numpy()
        v_12_direction = self._v_12_direction.detach().cpu().numpy()
        v_12_scale = self._v_12_scale.detach().cpu().numpy()
        L_22_inv = self._L_22_inv.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate([
            xyz, f_dc, f_rest, opacities, lambda_opcs, scale, rotation,
            view_mean, v_12_direction, v_12_scale, L_22_inv
        ], axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # Load lambda_opc (with backward compatibility)
        try:
            lambda_opcs = np.asarray(plydata.elements[0]["lambda_opc"])[..., np.newaxis]
        except:
            # Default to inverse_sigmoid(0.35) if not present in file
            lambda_opcs = np.full((xyz.shape[0], 1), inverse_sigmoid(torch.tensor(0.35)).item(), dtype=np.float32)
        lambda_opcs = np.ascontiguousarray(lambda_opcs, dtype=np.float32)

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

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load simplified parameterization
        view_mean_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("view_mean_")]
        view_mean_names = sorted(view_mean_names, key=lambda x: int(x.split('_')[-1]))
        view_mean = np.zeros((xyz.shape[0], len(view_mean_names)))
        for idx, attr_name in enumerate(view_mean_names):
            view_mean[:, idx] = np.asarray(plydata.elements[0][attr_name])

        v_12_direction_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("v_12_direction_")]
        v_12_direction_names = sorted(v_12_direction_names, key=lambda x: int(x.split('_')[-1]))
        v_12_direction = np.zeros((xyz.shape[0], len(v_12_direction_names)))
        for idx, attr_name in enumerate(v_12_direction_names):
            v_12_direction[:, idx] = np.asarray(plydata.elements[0][attr_name])

        v_12_scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("v_12_scale_")]
        v_12_scale_names = sorted(v_12_scale_names, key=lambda x: int(x.split('_')[-1]))
        v_12_scale = np.zeros((xyz.shape[0], len(v_12_scale_names)))
        for idx, attr_name in enumerate(v_12_scale_names):
            v_12_scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

        L_22_inv_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("L_22_inv_")]
        L_22_inv_names = sorted(L_22_inv_names, key=lambda x: int(x.split('_')[-1]))
        L_22_inv = np.zeros((xyz.shape[0], len(L_22_inv_names)))
        for idx, attr_name in enumerate(L_22_inv_names):
            L_22_inv[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._lambda_opc = nn.Parameter(torch.tensor(lambda_opcs, dtype=torch.float, device="cuda").requires_grad_(self.learnable_lambda_opc))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._view_mean = nn.Parameter(torch.tensor(view_mean, dtype=torch.float, device="cuda").requires_grad_(True))
        self._v_12_direction = nn.Parameter(torch.tensor(v_12_direction, dtype=torch.float, device="cuda").requires_grad_(True))
        self._v_12_scale = nn.Parameter(torch.tensor(v_12_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._L_22_inv = nn.Parameter(torch.tensor(L_22_inv, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._view_mean = optimizable_tensors["view_mean"]
        self._v_12_direction = optimizable_tensors["v_12_direction"]
        self._v_12_scale = optimizable_tensors["v_12_scale"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                              new_lambda_opc, new_scaling, new_rotation, new_view_mean, new_v_12_direction, new_v_12_scale, new_L_22_inv):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "lambda_opc": new_lambda_opc,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "view_mean": new_view_mean,
            "v_12_direction": new_v_12_direction,
            "v_12_scale": new_v_12_scale,
            "L_22_inv": new_L_22_inv,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._view_mean = optimizable_tensors["view_mean"]
        self._v_12_direction = optimizable_tensors["v_12_direction"]
        self._v_12_scale = optimizable_tensors["v_12_scale"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians with high gradients."""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_lambda_opc = self._lambda_opc[selected_pts_mask].repeat(N, 1)
        new_view_mean = self._view_mean[selected_pts_mask].repeat(N, 1)
        new_v_12_direction = self._v_12_direction[selected_pts_mask].repeat(N, 1)
        new_v_12_scale = self._v_12_scale[selected_pts_mask].repeat(N, 1)
        new_L_22_inv = self._L_22_inv[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity,
            new_lambda_opc, new_scaling, new_rotation, new_view_mean, new_v_12_direction, new_v_12_scale, new_L_22_inv
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradients and small scales."""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_lambda_opc = self._lambda_opc[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_view_mean = self._view_mean[selected_pts_mask]
        new_v_12_direction = self._v_12_direction[selected_pts_mask]
        new_v_12_scale = self._v_12_scale[selected_pts_mask]
        new_L_22_inv = self._L_22_inv[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities,
            new_lambda_opc, new_scaling, new_rotation, new_view_mean, new_v_12_direction, new_v_12_scale, new_L_22_inv
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", scaling_modifier=1.0, use_tcgs=False, tight_snugbox=False, compact_box_mult=1.0):
        """
        Render using DGS conditional position slicing with diff-gaussian-rasterization.
        This encapsulates the DGS-specific rendering logic with simplified v_12/L_22_inv parameterization.

        Args:
            viewpoint_camera: Camera viewpoint
            render_mode: Rendering mode (RGB, depth, etc.)
            use_tcgs: Whether to use TCGS rasterizer
            scaling_modifier: Scaling factor for Gaussians
            tight_snugbox: Use tight snugbox for TCGS rasterization
            compact_box_mult: FastGS-style compact box multiplier (1.0 = SnugBox, <1.0 = tighter)
        """
        import time

        # Import TCGS rasterizer
        from tcgs_speedy_rasterizer import (
            GaussianRasterizationSettings as TCGSRasterizationSettings,
            GaussianRasterizer as TCGSRasterizer,
        )

        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for conditional slicing
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1))
        view_dir = dir_pp / dir_pp.norm(dim=1, keepdim=True)

        # For 7DGS, append timestamp to query
        if self.input_dim == 7:
            timestamp = torch.full(
                (view_dir.shape[0], 1),
                viewpoint_camera.timestamp if hasattr(viewpoint_camera, 'timestamp') else 0.0,
                device=view_dir.device,
                dtype=view_dir.dtype,
            )
            cond_params = torch.cat([view_dir, timestamp], dim=-1)
        else:
            cond_params = view_dir

        # Use fixed or learnable lambda_opc
        if self.learnable_lambda_opc:
            lambda_opc = self.get_lambda_opc  # [N, 1] per-Gaussian
        else:
            lambda_opc = 0.35  # Fixed scalar

        # Compute conditional position and opacity scale using slice_gaussian
        m_cond, opacity_scale = self.slice_gaussian(cond_params, lambda_opc=lambda_opc)

        # Get SH features for color
        shs = self.get_features

        # Get opacity scaled by view-dependent factor
        # When learnable_lambda_opc=True, opacity_scale already incorporates per-Gaussian lambda
        opacity = self.get_opacity * opacity_scale
        
        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Use background from model if set, otherwise default to black
        bg_color = self.background if hasattr(self, 'background') and self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # Get x_threshold from viewpoint_camera if it exists
        x_threshold = viewpoint_camera.x_threshold if hasattr(viewpoint_camera, 'x_threshold') and viewpoint_camera.x_threshold is not None else float('inf')

        raster_settings = TCGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            x_threshold=x_threshold,
            prefiltered=False,
            use_tcgs=use_tcgs,
            tight_snugbox=tight_snugbox,
            compact_box_mult=compact_box_mult,
            debug=False,
        )

        rasterizer = TCGSRasterizer(raster_settings=raster_settings)

        # Rasterize with conditional positions
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=m_cond,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scores=None,
            scales=self.get_scaling,
            rotations=self.get_rotation,
            cov3D_precomp=None,
        )

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def view_tcgs(self, camera_state, render_tab_state):
        """Callable function for the viewer using TCGS rasterizer.

        This method provides interactive viewing capabilities for the DGS model,
        allowing real-time visualization through the GaussianViewer interface.

        Args:
            camera_state: Camera state from the viewer (contains c2w, K)
            render_tab_state: Render settings from viewer (GaussianRenderTabState)

        Returns:
            numpy array: Rendered image in [H, W, C] format for display
        """
        import time

        # Start timing for FPS calculation
        start_time = time.time()

        from scene.gaussian_viewer import GaussianRenderTabState
        assert isinstance(render_tab_state, GaussianRenderTabState)

        def create_mask(opacity, opacity_threshold, use_percentile=False, percentile=0.0):
            """Create mask based on opacity threshold or percentile."""
            if opacity.dim() > 1:
                opacity_1d = opacity.squeeze(-1)
            else:
                opacity_1d = opacity

            if use_percentile and percentile > 0.0:
                threshold_value = torch.quantile(opacity_1d, percentile / 100.0)
                opacity_mask = opacity_1d > threshold_value
            else:
                opacity_mask = opacity_1d > opacity_threshold

            return opacity_mask

        # Determine render resolution
        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height

        # Extract camera parameters
        c2w = camera_state.c2w
        K = camera_state.get_K((W, H))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")

        # Build camera for render_tcgs
        from scene.cameras import Camera

        # Extract camera parameters from K matrix
        fx = K[0, 0]
        fy = K[1, 1]

        # Compute FoV from focal lengths
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        # Convert c2w to w2c for COLMAP convention
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3].cpu().numpy().T
        T = w2c[:3, 3].cpu().numpy()

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
        )

        # Add timestamp for 7DGS time animation
        if self.input_dim == 7:
            viewpoint_camera.timestamp = render_tab_state.timestamp

        # Apply filtering mask for selective rendering
        opacity = self.get_opacity
        mask = create_mask(
            opacity,
            opacity_threshold=render_tab_state.opacity_threshold,
            use_percentile=render_tab_state.use_opacity_percentile,
            percentile=render_tab_state.opacity_percentile,
        )

        # Update stats
        num_valid = mask.sum().item()
        render_tab_state.total_count_number = len(opacity)
        render_tab_state.rendered_count_number = 0

        # If no Gaussians pass the filter, return a blank image
        if num_valid == 0:
            bg_color = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
            render_colors = bg_color.view(3, 1, 1).expand(3, H, W)
            return render_colors.cpu().numpy().transpose(1, 2, 0)

        # Set background color
        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        # Create masked views WITHOUT modifying self
        _xyz_masked = self._xyz[mask]
        _features_dc_masked = self._features_dc[mask]
        _features_rest_masked = self._features_rest[mask]
        _scaling_masked = self._scaling[mask]
        _rotation_masked = self._rotation[mask]
        _opacity_masked = self._opacity[mask]
        _view_mean_masked = self._view_mean[mask]
        _v_12_direction_masked = self._v_12_direction[mask]
        _v_12_scale_masked = self._v_12_scale[mask]
        _L_22_inv_masked = self._L_22_inv[mask]
        _lambda_opc_masked = self._lambda_opc[mask]

        # Temporarily swap in masked tensors
        orig_xyz, self._xyz = self._xyz, _xyz_masked
        orig_features_dc, self._features_dc = self._features_dc, _features_dc_masked
        orig_features_rest, self._features_rest = self._features_rest, _features_rest_masked
        orig_scaling, self._scaling = self._scaling, _scaling_masked
        orig_rotation, self._rotation = self._rotation, _rotation_masked
        orig_opacity, self._opacity = self._opacity, _opacity_masked
        orig_view_mean, self._view_mean = self._view_mean, _view_mean_masked
        orig_v_12_direction, self._v_12_direction = self._v_12_direction, _v_12_direction_masked
        orig_v_12_scale, self._v_12_scale = self._v_12_scale, _v_12_scale_masked
        orig_L_22_inv, self._L_22_inv = self._L_22_inv, _L_22_inv_masked
        orig_lambda_opc, self._lambda_opc = self._lambda_opc, _lambda_opc_masked

        try:
            # Call render_tcgs with masked Gaussians
            render_output = self.render_tcgs(
                viewpoint_camera,
                render_mode=render_tab_state.render_mode,
                use_tcgs=True,
                scaling_modifier=1.0,
                tight_snugbox=render_tab_state.tight_snugbox,
                compact_box_mult=0.7
            )

            render_colors = render_output["render"]
            render_tab_state.rendered_count_number = render_output["visibility_filter"].sum().item()

            # Handle different render modes
            if render_tab_state.render_mode == "Alpha":
                render_colors = render_output["visibility_filter"].float().unsqueeze(0)

            # Handle depth colormap (if single channel output)
            if render_colors.shape[0] == 1:
                render_colors = render_colors.repeat(3, 1, 1)

        finally:
            # Restore original tensors
            self._xyz = orig_xyz
            self._features_dc = orig_features_dc
            self._features_rest = orig_features_rest
            self._scaling = orig_scaling
            self._rotation = orig_rotation
            self._opacity = orig_opacity
            self._view_mean = orig_view_mean
            self._v_12_direction = orig_v_12_direction
            self._v_12_scale = orig_v_12_scale
            self._L_22_inv = orig_L_22_inv
            self._lambda_opc = orig_lambda_opc

        # Convert from [C, H, W] to [H, W, C] for viewer
        render_colors = render_colors.permute(1, 2, 0)

        # Calculate and update FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()