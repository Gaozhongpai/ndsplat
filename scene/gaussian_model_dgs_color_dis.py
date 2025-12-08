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
# Position + Color DGS with DISENTANGLED key_means for position, color, and opacity.
# Uses slice_gaussian_disentangled with SEPARATE key_means but SHARED v_12:
# - Position: pos_cond = xyz + v_regr @ (query - key_mean_pos)  [learned centering]
# - Color: color_cond = color_mean + v_regr @ (query - key_mean_color)  [key_mean_color=0 for SH-like]
# - Opacity: attention_weight = exp(-λ * (query - key_mean_opacity)^T @ V_22^{-1} @ (query - key_mean_opacity))
#
# Key insight: Shared v_12 defines "sensitivity directions" - how 3D output responds to query.
# Different key_means produce different residuals (query - key_mean), leading to different shifts.
# This saves parameters while maintaining expressiveness.
#
# input_dim=6: view direction only (C=3)
# input_dim=7: view direction + time (C=4)
#

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
import time
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# Import CUDA-accelerated slice function (disentangled version with separate key_means)
from gsplat import slice_gaussian_disentangled


class GaussianModel:
    """
    Position + Color DGS with DISENTANGLED key_means for position, color, and opacity.

    This model shifts 3D positions AND colors based on view direction, with
    SEPARATE key_means but SHARED v_12:
    - Position: key_mean_pos (learned) for full DGS centering
    - Color: key_mean_color=0 (fixed) for SH-like behavior
    - Opacity: key_mean_opacity (learned) for independent opacity control
    - v_12: SHARED value-key covariance for both position and color

    Key insight:
    - Shared v_12 defines "sensitivity directions" - how 3D output responds to query
    - Different key_means produce different residuals (query - key_mean)
    - Same v_12 @ V_22^{-1} @ different_residuals = different shifts
    - This saves 3*C parameters per Gaussian while maintaining expressiveness

    This gives color the same expressiveness as SH degree 1:
        color(v) = color_mean + v_regr @ v  (linear in view direction)

    While position retains the full DGS formulation:
        pos(v) = xyz + v_regr @ (v - key_mean_pos)  (centered at canonical view)

    Parameters:
    - _xyz: [N, 3] base position
    - _color_mean: [N, 3] base RGB color
    - _key_mean_pos: [N, C] learned key mean for POSITION (C=3 or 4)
    - _key_mean_color: [N, C] fixed at ZERO for COLOR (NOT learnable)
    - _key_mean_opacity: [N, C] learned key mean for OPACITY
    - _v_12: [N, 3*C] SHARED value-key covariance block for position and color
    - _L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision, shared)

    At runtime (via slice_gaussian_disentangled):
    - pos_cond = xyz + v_regr @ (query - key_mean_pos)
    - color_cond = color_mean + v_regr @ (query - key_mean_color)  [key_mean_color=0]
    - attention_weight = exp(-λ * (query - key_mean_opacity)^T @ V_22^{-1} @ (query - key_mean_opacity))

    For input_dim=6: view(3) conditioning only
    For input_dim=7: view(3) + time(1) conditioning
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

    def __init__(self, sh_degree: int = 0, input_dim: int = 6):
        """
        Initialize Position + Color DGS model with disentangled key_means.

        Args:
            sh_degree: Ignored (we use conditional color instead of SH)
            input_dim: 6 for view-only (C=3), 7 for view+time (C=4)
        """
        self.active_sh_degree = 0
        self.max_sh_degree = 0  # No SH - we use conditional color
        self.input_dim = input_dim
        self.value_dim = 6  # position(3) + color(3)
        self.cond_dim = input_dim - 3  # C = 3 for view-only, 4 for view+time

        # Standard 3DGS geometry parameters (base, will be shifted)
        self._xyz = torch.empty(0)  # [N, 3] base position
        self._scaling = torch.empty(0)  # [N, 3] 3D scale
        self._rotation = torch.empty(0)  # [N, 4] quaternion rotation
        self._opacity = torch.empty(0)  # [N, 1] base opacity
        self._label = torch.empty(0)

        # Color mean: base RGB color (replaces SH)
        self._color_mean = torch.empty(0)  # [N, 3] RGB base color

        # Disentangled conditional slicing parameters
        # Key mean for POSITION (learned "canonical" view direction per Gaussian)
        self._key_mean_pos = torch.empty(0)  # [N, C] where C=3 or 4 (with time)
        # Key mean for COLOR (fixed at zero, NOT learnable - makes color equivalent to SH deg 1)
        self._key_mean_color = torch.empty(0)  # [N, C] fixed at zero
        # Key mean for OPACITY (learned, can be different from position)
        self._key_mean_opacity = torch.empty(0)  # [N, C] learned
        # v_12: [N, 3*C] SHARED value-key covariance block for position and color
        self._v_12 = torch.empty(0)
        # L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision, shared)
        self._L_22_inv = torch.empty(0)

        # Auxiliary tensors
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # Background color for rendering
        self.background = torch.empty(0)

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._color_mean,
            self._scaling,
            self._rotation,
            self._key_mean_pos,
            self._key_mean_opacity,
            self._v_12,
            self._L_22_inv,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._color_mean,
         self._scaling,
         self._rotation,
         self._key_mean_pos,
         self._key_mean_opacity,
         self._v_12,
         self._L_22_inv,
         self._opacity,
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
    def get_color_mean(self):
        """Get base color mean."""
        return self._color_mean

    @property
    def get_key_mean_pos(self):
        """Get normalized key mean direction for POSITION."""
        view_dir = self._key_mean_pos[:, :3]
        view_dir_normalized = view_dir / (view_dir.norm(dim=1, keepdim=True) + 1e-8)
        if self._key_mean_pos.shape[1] > 3:
            return torch.cat([view_dir_normalized, self._key_mean_pos[:, 3:]], dim=1)
        return view_dir_normalized

    @property
    def get_key_mean_color(self):
        """Get key mean for COLOR (fixed at zero, equivalent to SH deg 1)."""
        return self._key_mean_color

    @property
    def get_key_mean_opacity(self):
        """Get normalized key mean direction for OPACITY."""
        view_dir = self._key_mean_opacity[:, :3]
        view_dir_normalized = view_dir / (view_dir.norm(dim=1, keepdim=True) + 1e-8)
        if self._key_mean_opacity.shape[1] > 3:
            return torch.cat([view_dir_normalized, self._key_mean_opacity[:, 3:]], dim=1)
        return view_dir_normalized

    @property
    def get_features(self):
        """Get color features for compatibility (returns color_mean as [N, 1, 3])."""
        return self._color_mean.unsqueeze(1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        # No-op: we use conditional color instead of SH
        pass

    def slice_gaussian(self, query, lambda_opc=0.35):
        """
        Perform disentangled conditional Gaussian slicing for position, color, and opacity.

        Each component has its own key_mean, but v_12 is SHARED:
        - Position: key_mean_pos (learned) for full DGS centering
        - Color: key_mean_color=0 (fixed) for SH-like behavior
        - Opacity: key_mean_opacity (learned) for independent opacity control
        - v_12: SHARED value-key covariance for position and color

        Given query view direction (+ optional time), compute:
        - Position: pos_cond = xyz + v_regr @ (query - key_mean_pos)
        - Color: color_cond = color_mean + v_regr @ (query - key_mean_color)  [key_mean_color=0]
        - Attention weight: exp(-λ * (query - key_mean_opacity)^T @ V_22^{-1} @ (query - key_mean_opacity))

        V_22^{-1} = L_22_inv @ L_22_inv^T (precision from Cholesky, no matrix inversion!)

        Args:
            query: Query direction [N, C] where C=3 (view) or 4 (view+time)
            lambda_opc: Opacity/attention scaling factor (default 0.35)

        Returns:
            pos_cond: Conditional 3D position [N, 3]
            color_cond: Conditional RGB color [N, 3]
            attention_weight: View-dependent opacity scaling [N, 1]
        """
        # Use disentangled slicing with separate key_means but shared v_12
        pos_cond, color_cond, attention_weight = slice_gaussian_disentangled(
            # Position inputs
            self._xyz,               # [N, 3] - position mean
            self.get_key_mean_pos,   # [N, C] - learned key mean for position
            # Color inputs
            self._color_mean,        # [N, 3] - color mean
            self.get_key_mean_color, # [N, C] - ZERO (no centering, like SH)
            # Opacity key mean
            self.get_key_mean_opacity,  # [N, C] - learned key mean for opacity
            # Shared inputs
            self._v_12,              # [N, 3*C] - SHARED value-key covariance
            query,                   # [N, C] - query view direction (+ time)
            self._L_22_inv,          # [N, C*(C+1)/2] - Cholesky of precision (shared)
            lambda_opc,
        )

        return pos_cond, color_cond, attention_weight

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussians from point cloud data with disentangled key_means.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()  # RGB in [0, 1]

        num_gaussians = fused_color.shape[0]
        device = "cuda"
        C = self.cond_dim  # 3 for view-only, 4 for view+time

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        from sklearn.neighbors import NearestNeighbors
        def knn(x, K=4):
            x_np = x.cpu().numpy()
            model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(x)

        # Spatial scales from KNN distances
        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        # Key mean for POSITION: random unit vectors (learnable)
        view_dir_pos = torch.randn((num_gaussians, 3), device=device)
        key_mean_pos = (view_dir_pos / view_dir_pos.norm(dim=1, keepdim=True)).float()

        # Key mean for OPACITY: random unit vectors (learnable, separate from position)
        view_dir_opc = torch.randn((num_gaussians, 3), device=device)
        key_mean_opacity = (view_dir_opc / view_dir_opc.norm(dim=1, keepdim=True)).float()

        # For input_dim=7, append time dimension
        if self.input_dim == 7:
            mean_time_pos = torch.empty(num_gaussians, 1, device=device).uniform_(0.0, 1.0)
            key_mean_pos = torch.cat([key_mean_pos, mean_time_pos], dim=-1)
            mean_time_opc = torch.empty(num_gaussians, 1, device=device).uniform_(0.0, 1.0)
            key_mean_opacity = torch.cat([key_mean_opacity, mean_time_opc], dim=-1)

        # Key mean for COLOR: fixed at ZERO (NOT learnable)
        # This makes color slicing equivalent to SH degree 1: color = color_mean + linear @ view
        key_mean_color = torch.zeros((num_gaussians, C), device=device)

        # v_12: [N, 3*C] SHARED value-key covariance block for position and color
        v_12 = torch.normal(0, 0.01, size=(num_gaussians, 3 * C), device=device)

        # L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision, shared)
        # Diagonal uses exp() activation, so initialize with log(2.0) ≈ 0.693
        n_L_22_inv = C * (C + 1) // 2
        L_22_inv = torch.zeros(num_gaussians, n_L_22_inv, device=device)
        for i in range(C):
            diag_idx = i * (i + 1) // 2 + i
            L_22_inv[:, diag_idx] = math.log(2.0)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._label = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.int32).cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._color_mean = nn.Parameter(fused_color.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._key_mean_pos = nn.Parameter(key_mean_pos.requires_grad_(True))
        # key_mean_color is NOT a Parameter - fixed at zero (equivalent to SH deg 1)
        self._key_mean_color = key_mean_color  # [N, C] fixed tensor, not nn.Parameter
        self._key_mean_opacity = nn.Parameter(key_mean_opacity.requires_grad_(True))
        self._v_12 = nn.Parameter(v_12.requires_grad_(True))
        self._L_22_inv = nn.Parameter(L_22_inv.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._color_mean], 'lr': training_args.feature_lr, "name": "color_mean"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._key_mean_pos], 'lr': training_args.feature_lr, "name": "key_mean_pos"},
            {'params': [self._key_mean_opacity], 'lr': training_args.feature_lr, "name": "key_mean_opacity"},
            {'params': [self._v_12], 'lr': training_args.feature_lr, "name": "v_12"},
            {'params': [self._L_22_inv], 'lr': training_args.rotation_lr, "name": "L_22_inv"},
        ]

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
        for i in range(3):
            l.append('color_mean_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._key_mean_pos.shape[1]):
            l.append('key_mean_pos_{}'.format(i))
        for i in range(self._key_mean_opacity.shape[1]):
            l.append('key_mean_opacity_{}'.format(i))
        for i in range(self._v_12.shape[1]):
            l.append('v_12_{}'.format(i))
        for i in range(self._L_22_inv.shape[1]):
            l.append('L_22_inv_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        color_mean = self._color_mean.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        key_mean_pos = self._key_mean_pos.detach().cpu().numpy()
        key_mean_opacity = self._key_mean_opacity.detach().cpu().numpy()
        v_12 = self._v_12.detach().cpu().numpy()
        L_22_inv = self._L_22_inv.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate([
            xyz, color_mean, opacities, scale, rotation,
            key_mean_pos, key_mean_opacity, v_12, L_22_inv
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

        color_mean = np.stack([
            np.asarray(plydata.elements[0]["color_mean_{}".format(i)])
            for i in range(3)
        ], axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

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

        key_mean_pos_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("key_mean_pos_")]
        key_mean_pos_names = sorted(key_mean_pos_names, key=lambda x: int(x.split('_')[-1]))
        key_mean_pos = np.zeros((xyz.shape[0], len(key_mean_pos_names)))
        for idx, attr_name in enumerate(key_mean_pos_names):
            key_mean_pos[:, idx] = np.asarray(plydata.elements[0][attr_name])

        key_mean_opacity_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("key_mean_opacity_")]
        key_mean_opacity_names = sorted(key_mean_opacity_names, key=lambda x: int(x.split('_')[-1]))
        key_mean_opacity = np.zeros((xyz.shape[0], len(key_mean_opacity_names)))
        for idx, attr_name in enumerate(key_mean_opacity_names):
            key_mean_opacity[:, idx] = np.asarray(plydata.elements[0][attr_name])

        v_12_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("v_12_")]
        v_12_names = sorted(v_12_names, key=lambda x: int(x.split('_')[-1]))
        v_12 = np.zeros((xyz.shape[0], len(v_12_names)))
        for idx, attr_name in enumerate(v_12_names):
            v_12[:, idx] = np.asarray(plydata.elements[0][attr_name])

        L_22_inv_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("L_22_inv_")]
        L_22_inv_names = sorted(L_22_inv_names, key=lambda x: int(x.split('_')[-1]))
        L_22_inv = np.zeros((xyz.shape[0], len(L_22_inv_names)))
        for idx, attr_name in enumerate(L_22_inv_names):
            L_22_inv[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._color_mean = nn.Parameter(torch.tensor(color_mean, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._key_mean_pos = nn.Parameter(torch.tensor(key_mean_pos, dtype=torch.float, device="cuda").requires_grad_(True))
        # key_mean_color is always zero (not saved/loaded, just recreated)
        C = key_mean_pos.shape[1]  # infer cond_dim from loaded key_mean_pos
        self._key_mean_color = torch.zeros((xyz.shape[0], C), dtype=torch.float, device="cuda")
        self._key_mean_opacity = nn.Parameter(torch.tensor(key_mean_opacity, dtype=torch.float, device="cuda").requires_grad_(True))
        self._v_12 = nn.Parameter(torch.tensor(v_12, dtype=torch.float, device="cuda").requires_grad_(True))
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
        self._color_mean = optimizable_tensors["color_mean"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._key_mean_pos = optimizable_tensors["key_mean_pos"]
        self._key_mean_opacity = optimizable_tensors["key_mean_opacity"]
        self._v_12 = optimizable_tensors["v_12"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]

        # _key_mean_color is not optimized, just slice it directly
        self._key_mean_color = self._key_mean_color[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_color_mean, new_opacities,
                              new_scaling, new_rotation, new_key_mean_pos, new_key_mean_color,
                              new_key_mean_opacity, new_v_12, new_L_22_inv):
        d = {
            "xyz": new_xyz,
            "color_mean": new_color_mean,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "key_mean_pos": new_key_mean_pos,
            "key_mean_opacity": new_key_mean_opacity,
            "v_12": new_v_12,
            "L_22_inv": new_L_22_inv,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._color_mean = optimizable_tensors["color_mean"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._key_mean_pos = optimizable_tensors["key_mean_pos"]
        self._key_mean_opacity = optimizable_tensors["key_mean_opacity"]
        self._v_12 = optimizable_tensors["v_12"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]

        # _key_mean_color is not optimized (fixed at zero), just concatenate directly
        self._key_mean_color = torch.cat([self._key_mean_color, new_key_mean_color], dim=0)

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
        new_color_mean = self._color_mean[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_key_mean_pos = self._key_mean_pos[selected_pts_mask].repeat(N, 1)
        new_key_mean_color = self._key_mean_color[selected_pts_mask].repeat(N, 1)  # Fixed zeros
        new_key_mean_opacity = self._key_mean_opacity[selected_pts_mask].repeat(N, 1)
        new_v_12 = self._v_12[selected_pts_mask].repeat(N, 1)
        new_L_22_inv = self._L_22_inv[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_color_mean, new_opacity,
            new_scaling, new_rotation, new_key_mean_pos, new_key_mean_color,
            new_key_mean_opacity, new_v_12, new_L_22_inv
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
        new_color_mean = self._color_mean[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_key_mean_pos = self._key_mean_pos[selected_pts_mask]
        new_key_mean_color = self._key_mean_color[selected_pts_mask]  # Fixed zeros
        new_key_mean_opacity = self._key_mean_opacity[selected_pts_mask]
        new_v_12 = self._v_12[selected_pts_mask]
        new_L_22_inv = self._L_22_inv[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_color_mean, new_opacities,
            new_scaling, new_rotation, new_key_mean_pos, new_key_mean_color,
            new_key_mean_opacity, new_v_12, new_L_22_inv
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
        Render using Position+Color DGS with joint view-dependent shifting.

        This method:
        1. Computes view direction for each Gaussian
        2. Applies conditional Gaussian slicing to get view-dependent position AND color
        3. Computes view-dependent opacity scaling (attention weight)
        4. Renders using standard 3DGS rasterization with shifted positions and colors
        """
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

        # For 10D (with time), append timestamp to query
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

        # Compute conditional position, color, and attention weight using slice_gaussian
        pos_cond, color_cond, attention_weight = self.slice_gaussian(cond_params, lambda_opc=0.35)

        # Get opacity scaled by attention weight
        opacity = self.get_opacity * attention_weight

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
            sh_degree=0,  # Not using SH, using direct RGB
            campos=viewpoint_camera.camera_center,
            x_threshold=x_threshold,
            prefiltered=False,
            use_tcgs=use_tcgs,
            tight_snugbox=tight_snugbox,
            compact_box_mult=compact_box_mult,
            debug=False,
        )

        rasterizer = TCGSRasterizer(raster_settings=raster_settings)

        # Rasterize with conditional positions and colors
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=pos_cond,  # Use shifted positions
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color_cond,  # Use conditional colors directly
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
        """Callable function for the viewer using TCGS rasterizer."""
        start_time = time.time()

        from scene.gaussian_viewer import GaussianRenderTabState
        assert isinstance(render_tab_state, GaussianRenderTabState)

        def create_mask(opacity, opacity_threshold, use_percentile=False, percentile=0.0):
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
        )

        if self.input_dim == 7:
            viewpoint_camera.timestamp = render_tab_state.timestamp

        opacity = self.get_opacity
        mask = create_mask(
            opacity,
            opacity_threshold=render_tab_state.opacity_threshold,
            use_percentile=render_tab_state.use_opacity_percentile,
            percentile=render_tab_state.opacity_percentile,
        )

        num_valid = mask.sum().item()
        render_tab_state.total_count_number = len(opacity)
        render_tab_state.rendered_count_number = 0

        if num_valid == 0:
            bg_color = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
            render_colors = bg_color.view(3, 1, 1).expand(3, H, W)
            return render_colors.cpu().numpy().transpose(1, 2, 0)

        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        # Create masked views
        orig_xyz, self._xyz = self._xyz, self._xyz[mask]
        orig_color_mean, self._color_mean = self._color_mean, self._color_mean[mask]
        orig_scaling, self._scaling = self._scaling, self._scaling[mask]
        orig_rotation, self._rotation = self._rotation, self._rotation[mask]
        orig_opacity, self._opacity = self._opacity, self._opacity[mask]
        orig_key_mean_pos, self._key_mean_pos = self._key_mean_pos, self._key_mean_pos[mask]
        orig_key_mean_color, self._key_mean_color = self._key_mean_color, self._key_mean_color[mask]
        orig_key_mean_opacity, self._key_mean_opacity = self._key_mean_opacity, self._key_mean_opacity[mask]
        orig_v_12, self._v_12 = self._v_12, self._v_12[mask]
        orig_L_22_inv, self._L_22_inv = self._L_22_inv, self._L_22_inv[mask]

        try:
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

            if render_tab_state.render_mode == "Alpha":
                render_colors = render_output["visibility_filter"].float().unsqueeze(0)

            if render_colors.shape[0] == 1:
                render_colors = render_colors.repeat(3, 1, 1)

        finally:
            self._xyz = orig_xyz
            self._color_mean = orig_color_mean
            self._scaling = orig_scaling
            self._rotation = orig_rotation
            self._opacity = orig_opacity
            self._key_mean_pos = orig_key_mean_pos
            self._key_mean_color = orig_key_mean_color
            self._key_mean_opacity = orig_key_mean_opacity
            self._v_12 = orig_v_12
            self._L_22_inv = orig_L_22_inv

        render_colors = render_colors.permute(1, 2, 0)

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
