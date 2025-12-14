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
# Color-based N-DGS: Conditional color and opacity based on view direction
# Unlike position-based N-DGS, this keeps geometry fixed and only varies appearance
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
import time
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# Import CUDA-accelerated functions from gsplat
from gsplat import slice_gaussian_color, l_triangle_to_covar, slice_gaussian_simple, _slice_gaussian_simple

# Import TCGS rasterizer for rendering
from tcgs_speedy_rasterizer import (
    GaussianRasterizationSettings as TCGSRasterizationSettings,
    GaussianRasterizer as TCGSRasterizer,
)


def randomly_sample_point_cloud(point_cloud, num_samples=15000):
    if len(point_cloud) <= num_samples:
        return point_cloud
    sampled_indices = np.random.choice(len(point_cloud), num_samples, replace=False)
    return point_cloud[sampled_indices]


class GaussianModelColor:
    """
    Color-based N-DGS: Gaussian Splatting with view-dependent color and opacity.

    Unlike position-based N-DGS which shifts 3D positions based on view direction,
    this model keeps geometry fixed and applies conditional Gaussian slicing to
    color space. This is more physically plausible since:
    - View-dependent effects (specularity, reflections) naturally affect appearance
    - Geometry should remain stable across views

    The model parametrizes:
    - Fixed 3D Gaussians: position (xyz), scale, rotation (standard 3DGS)
    - View-dependent color: [base_color | view_direction | (time)] as N-D Gaussian
    - View-dependent opacity: derived from conditional probability

    Key difference from standard 3DGS:
    - Uses conditional Gaussian slicing for view-dependent color instead of Spherical Harmonics
    - Equivalent to SH degree=0 (DC only) but with view-dependent adjustment
    - _color_mean serves as base DC color, adjusted by conditional slicing based on view direction

    For input_dim=6: color(3) + view_direction(3) = 6D
    For input_dim=7: color(3) + view_direction(3) + time(1) = 7D
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

    def __init__(self, sh_degree: int = 0, input_dim: int = 6, use_rot_scale_l_triangle: bool = False,
                 learnable_lambda_opc: bool = False, use_simplified_color: bool = False):
        """
        Initialize Color-based N-DGS model.

        Args:
            sh_degree: Maximum degree of spherical harmonics (fixed at 0 for Color N-DGS,
                       since we use conditional Gaussian slicing for view-dependent color
                       instead of SH. The _color_mean serves as DC color coefficient.)
            input_dim: Dimensionality of color covariance (6 for view-dep, 7 for view-dep + time)
            use_rot_scale_l_triangle: Ignored for Color N-DGS (uses standard 3DGS quaternion+scale).
                                      Kept for API compatibility with other N-DGS models.
            learnable_lambda_opc: If True, make lambda_opc a learnable parameter per Gaussian
            use_simplified_color: If True, use simplified parameterization that directly learns
                                  v_regr [N, 3, C] and L_22 [N, C*(C+1)/2] instead of full covariance.
                                  This reduces parameters from D*(D+1)/2 to 3*C + C*(C+1)/2.
        """
        # Note: use_rot_scale_l_triangle is ignored - Color N-DGS uses standard 3DGS geometry
        _ = use_rot_scale_l_triangle  # Explicitly ignore for API compatibility
        # SH degree is always 0 for Color N-DGS (we use conditional color instead)
        self.active_sh_degree = 0
        self.max_sh_degree = 0  # Force to 0, conditional color replaces SH
        self.input_dim = input_dim  # 6 for color+view, 7 for color+view+time
        self.learnable_lambda_opc = learnable_lambda_opc
        self.use_simplified_color = use_simplified_color
        self.training = True  # Training mode flag (similar to nn.Module)

        # Standard 3DGS geometry parameters (FIXED, not view-dependent)
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)  # 3D scale
        self._rotation = torch.empty(0)  # Quaternion rotation
        self._opacity = torch.empty(0)  # Base opacity

        # Color N-DGS parameters
        # Base color mean: equivalent to SH DC coefficient (degree=0)
        # This is the "average" color that gets adjusted by view-dependent conditioning
        self._color_mean = torch.empty(0)  # [N, 3] RGB base color (replaces SH DC)
        # View direction mean (learned "canonical" view direction per Gaussian)
        self._view_mean = torch.empty(0)  # [N, 3] or [N, 4] with time

        # Conditional dimension size (view direction + optional time)
        self.cond_dim = input_dim - 3  # C = 3 for view-only, 4 for view+time

        if use_simplified_color:
            # Simplified parameterization: learn v_12 and L_22_inv, compute v_regr from them
            # v_12: [N, 3, C] color-view covariance block (flattened to [N, 3*C])
            self._v_12 = torch.empty(0)  # [N, 3*C] -> reshaped to [N, 3, C]
            # L_22_inv: [N, C*(C+1)/2] lower triangular Cholesky factor of V_22^{-1} (precision)
            # V_22^{-1} = L_22_inv^T @ L_22_inv (no matrix inversion needed at runtime!)
            self._L_22_inv = torch.empty(0)  # [N, C*(C+1)/2]
            # Placeholders for compatibility
            self._color_scale = torch.empty(0)
            self._color_l_triangle = torch.empty(0)
        else:
            # Full covariance parameterization (original)
            # Color covariance parameters (diagonal + lower triangular)
            self._color_scale = torch.empty(0)  # [N, input_dim] diagonal of L
            self._color_l_triangle = torch.empty(0)  # [N, input_dim*(input_dim-1)//2]
            # Placeholders for compatibility
            self._v_12 = torch.empty(0)
            self._L_22_inv = torch.empty(0)

        # Lambda for opacity scaling
        self._lambda_opc = torch.empty(0)

        # Auxiliary tensors
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.background = torch.empty(0)

        # Activation functions for color covariance
        self.color_scale_activation = lambda x: torch.exp(x)
        self.color_scale_inverse_activation = lambda x: torch.log(torch.clamp(x, min=1e-6))
        self.color_l_triangle_activation = lambda x: torch.sigmoid(x) * 2.0 - 1.0
        self.color_l_triangle_inverse_activation = lambda x: inverse_sigmoid(
            torch.clamp((x + 1.0) / 2.0, min=1e-6, max=1.0 - 1e-6)
        )

        self.setup_functions()

    def get_color_covariance(self):
        """
        Build full color covariance matrix from diagonal and lower triangular elements.
        Uses CUDA-accelerated l_triangle_to_covar kernel.

        Returns:
            Covariance matrix [N, D, D] where D = input_dim
        """
        diags = self.color_scale_activation(self._color_scale)  # [N, D]
        l_triangs = self.color_l_triangle_activation(self._color_l_triangle)  # [N, D*(D-1)//2]

        # Use CUDA-accelerated l_triangle_to_covar
        V = l_triangle_to_covar(diags, l_triangs)  # [N, D, D]

        return V

    def slice_color_gaussian(self, query, lambda_opc=0.35):
        """
        Perform conditional Gaussian slicing for color using CUDA kernel.

        Given N-D Gaussian over [color(3), view_direction(3), (time(1))]
        and query view direction (+ time), compute:
        - Conditional color mean
        - Conditional opacity scale

        No conditional covariance needed since we output RGB color directly.

        Args:
            query: Query direction (view direction + optional time) [N, C] where C=3 or 4
            lambda_opc: Opacity scaling factor (default 0.35)

        Returns:
            color_cond: Conditional RGB color [N, 3]
            opacity_scale: View-dependent opacity scaling [N, 1]
        """
        # Color mean (base RGB)
        color_mean = self._color_mean  # [N, 3]

        # View direction mean (+ optional time), normalized (avoid inplace ops for autograd)
        view_dir = self._view_mean[:, :3]  # [N, 3]
        view_dir_normalized = view_dir / (view_dir.norm(dim=1, keepdim=True) + 1e-8)
        if self._view_mean.shape[1] > 3:
            # Has time component
            view_mean = torch.cat([view_dir_normalized, self._view_mean[:, 3:]], dim=1)
        else:
            view_mean = view_dir_normalized

        if self.use_simplified_color:
            # Use simplified parameterization
            return self.slice_color_gaussian_simple(query, view_mean, lambda_opc)

        # Full covariance [N, D, D] where D = 3 + C (6 or 7)
        covars = self.get_color_covariance()

        # Use CUDA-accelerated slice_gaussian_color
        color_cond, opacity_scale = slice_gaussian_color(
            color_mean=color_mean,
            view_mean=view_mean,
            query=query,
            covars=covars,
            lambda_opc=lambda_opc,
        )

        return color_cond, opacity_scale

    def slice_color_gaussian_simple(self, query, view_mean, lambda_opc=0.35):
        """
        Perform conditional Gaussian slicing using simplified parameterization.

        This version learns v_12 and L_22_inv (Cholesky of V_22^{-1}), then computes:
        - V_22^{-1} = L_22_inv^T @ L_22_inv (precision matrix, no inversion needed!)
        - v_regr = v_12 @ V_22^{-1} (regression matrix)

        This is mathematically equivalent to the full covariance parameterization
        but uses fewer parameters (no V_11 block needed) and avoids matrix inversion.

        Uses CUDA-accelerated kernel for forward and backward passes.

        Args:
            query: Query direction (view direction + optional time) [N, C] where C=3 or 4
            view_mean: Normalized view mean [N, C]
            lambda_opc: Opacity scaling factor (default 0.35)

        Returns:
            color_cond: Conditional RGB color [N, 3]
            opacity_scale: View-dependent opacity scaling [N, 1]
        """
        # Use CUDA-accelerated slice_gaussian_simple
        # v_12 is stored as [N, 3*C], will be reshaped inside the kernel
        color_cond, opacity_scale = slice_gaussian_simple(
            color_mean=self._color_mean,
            view_mean=view_mean,
            query=query,
            v_12=self._v_12,  # [N, 3*C]
            L_22_inv=self._L_22_inv,  # [N, C*(C+1)/2]
            lambda_opc=lambda_opc,
        )

        return color_cond, opacity_scale

    def slice_color_gaussian_test(self, query, lambda_opc=0.35):
        """
        Optimized version for test/inference using precomputed matrices.

        Args:
            query: Query direction [N, C]
            lambda_opc: Opacity scaling factor

        Returns:
            color_cond: Conditional RGB color [N, 3]
            opacity_scale: View-dependent opacity scaling [N, 1]
        """
        m_1 = self._color_mean
        m_2 = self.view_direction  # Precomputed normalized
        v_22_inv = self.v_22_inv
        v_regr = self.v_regr

        x = query - m_2

        color_cond = m_1 + torch.bmm(v_regr, x.unsqueeze(-1)).squeeze(-1)

        v_22_inv_x = torch.bmm(v_22_inv, x.unsqueeze(-1)).squeeze(-1)
        direction_influence = (x * v_22_inv_x).sum(dim=-1, keepdim=True)
        opacity_scale = torch.exp(-lambda_opc * direction_influence)

        return color_cond, opacity_scale

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._color_mean,
            self._view_mean,
            self._color_scale,
            self._color_l_triangle,
            self._lambda_opc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.use_simplified_color,
            self._v_12,
            self._L_22_inv,
        )

    def restore(self, model_args, training_args):
        # Handle both old format (15 elements) and new format (18 elements)
        if len(model_args) == 15:
            # Old format without simplified parameters
            (self.active_sh_degree,
             self._xyz,
             self._scaling,
             self._rotation,
             self._opacity,
             self._color_mean,
             self._view_mean,
             self._color_scale,
             self._color_l_triangle,
             self._lambda_opc,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
            self.use_simplified_color = False
            self._v_12 = torch.empty(0, device="cuda")
            self._L_22_inv = torch.empty(0, device="cuda")
        else:
            # New format with simplified parameters
            (self.active_sh_degree,
             self._xyz,
             self._scaling,
             self._rotation,
             self._opacity,
             self._color_mean,
             self._view_mean,
             self._color_scale,
             self._color_l_triangle,
             self._lambda_opc,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             self.use_simplified_color,
             self._v_12,
             self._L_22_inv) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_lambda_opc(self):
        if self.learnable_lambda_opc:
            return self.opacity_activation(self._lambda_opc)
        else:
            return torch.ones_like(self._opacity) * 0.35

    def get_covariance(self, scaling_modifier=1):
        """Get 3D spatial covariance (standard 3DGS)."""
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_color_mean(self):
        """Get base color mean (equivalent to SH DC coefficient)."""
        return self._color_mean

    @property
    def get_features(self):
        """
        Get color features for compatibility with SH-based code.
        Returns _color_mean reshaped to [N, 1, 3] to match SH DC format.
        This enables using existing training code that expects SH features.
        """
        # Shape: [N, 1, 3] to match SH format [N, (sh_degree+1)^2, 3] with degree=0
        return self._color_mean.unsqueeze(1)

    @property
    def get_features_dc(self):
        """Get DC color features (same as _color_mean for Color N-DGS)."""
        return self._color_mean.unsqueeze(1)

    def oneupSHdegree(self):
        # No-op for Color N-DGS: we use conditional Gaussian slicing instead of SH
        # The _color_mean acts as the DC color (equivalent to SH degree 0)
        pass

    def train(self):
        """Set model to training mode."""
        self.training = True

    def eval(self):
        """Set model to evaluation mode."""
        self.training = False

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,
                        mcmc_cap_max=None, densification_strategy="standard"):
        """
        Initialize Gaussians from point cloud data.
        """
        self.spatial_lr_scale = spatial_lr_scale

        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)

        if densification_strategy == "mcmc" and mcmc_cap_max is not None and len(pcd_points) > mcmc_cap_max:
            print(f"\n[MCMC Init] Point cloud has {len(pcd_points)} points, sampling {mcmc_cap_max} for initialization")
            sampled_indices = np.random.choice(len(pcd_points), mcmc_cap_max, replace=False)
            pcd_points = pcd_points[sampled_indices]
            pcd_colors = pcd_colors[sampled_indices]

        fused_point_cloud = torch.tensor(pcd_points).float().cuda()
        fused_color = torch.tensor(pcd_colors).float().cuda()

        init_n_gs = fused_color.shape[0]
        device = "cuda"

        print("Number of points at initialization:", init_n_gs)

        # Initialize 3DGS geometry parameters
        from sklearn.neighbors import NearestNeighbors
        def knn(x, K=4):
            x_np = x.cpu().numpy()
            model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(x)

        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3)
        scales = self.scaling_inverse_activation(scales)

        rots = torch.zeros((init_n_gs, 4), device=device)
        rots[:, 0] = 1  # Identity quaternion

        opacities = inverse_sigmoid(0.1 * torch.ones((init_n_gs, 1), dtype=torch.float, device=device))
        lambda_opcs = inverse_sigmoid(0.35 * torch.ones((init_n_gs, 1), dtype=torch.float, device=device))

        # Initialize color N-DGS parameters
        # Color mean: initialized from point cloud colors (RGB in [0, 1])
        color_mean = fused_color.clone()  # [N, 3]

        # View direction mean: random unit vectors
        view_dir = torch.randn((init_n_gs, 3), device=device)
        view_mean = (view_dir / view_dir.norm(dim=1, keepdim=True)).float()

        # For input_dim=7, append time dimension
        if self.input_dim == 7:
            mean_time = torch.empty(init_n_gs, 1, device=device).uniform_(0.0, 1.0)
            view_mean = torch.cat([view_mean, mean_time], dim=-1)

        # Set common parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._color_mean = nn.Parameter(color_mean.requires_grad_(True))
        self._view_mean = nn.Parameter(view_mean.requires_grad_(True))
        self._lambda_opc = nn.Parameter(lambda_opcs.requires_grad_(self.learnable_lambda_opc))

        if self.use_simplified_color:
            # Simplified parameterization initialization
            C = self.cond_dim  # 3 for view-only, 4 for view+time

            # v_12: [N, 3*C] - color-view covariance block, initialized near zero
            v_12 = torch.normal(0, 0.01, size=(init_n_gs, 3 * C), device=device)
            self._v_12 = nn.Parameter(v_12.requires_grad_(True))

            # L_22_inv: [N, C*(C+1)/2] - Cholesky factor of V_22^{-1} (precision)
            # V_22^{-1} = L_22_inv^T @ L_22_inv (no matrix inversion needed at runtime!)
            # Diagonal uses exp() activation, so initialize with log(2.0) ≈ 0.693
            # This gives diagonal=2.0 after exp, so V_22^{-1} has diagonal ~4.0, V_22 has diagonal ~0.25
            # Storage format: [l_00, l_10, l_11, l_20, l_21, l_22, ...] (row-major lower triangular)
            n_L_22_inv = C * (C + 1) // 2
            L_22_inv = torch.zeros(init_n_gs, n_L_22_inv, device=device)
            # Diagonal positions: 0 (for i=0), 2 (for i=1), 5 (for i=2), 9 (for i=3)
            # Formula: diagonal position for row i is i*(i+1)/2 + i
            for i in range(C):
                diag_idx = i * (i + 1) // 2 + i  # Position of L[i,i] in packed format
                L_22_inv[:, diag_idx] = math.log(2.0)  # exp(log(2)) = 2.0
            self._L_22_inv = nn.Parameter(L_22_inv.requires_grad_(True))

            # Placeholder for compatibility
            self._color_scale = nn.Parameter(torch.empty(0, device=device).requires_grad_(False))
            self._color_l_triangle = nn.Parameter(torch.empty(0, device=device).requires_grad_(False))
        else:
            # Full covariance parameterization (original)
            # Color covariance initialization
            # Diagonal: small values for color (tight distribution), larger for view direction
            color_scales_color = self.color_scale_inverse_activation(
                torch.ones(init_n_gs, 3, device=device) * 0.1  # Tight color distribution
            )
            color_scales_view = self.color_scale_inverse_activation(
                torch.ones(init_n_gs, self.input_dim - 3, device=device) * 0.5  # Wider view distribution
            )
            color_scales = torch.cat([color_scales_color, color_scales_view], dim=1)

            # Lower triangular: small random values
            n_l_triangle = self.input_dim * (self.input_dim - 1) // 2
            color_l_triangles = self.color_l_triangle_inverse_activation(
                torch.normal(0, 1e-2, size=(init_n_gs, n_l_triangle), device=device)
            )

            self._color_scale = nn.Parameter(color_scales.requires_grad_(True))
            self._color_l_triangle = nn.Parameter(color_l_triangles.requires_grad_(True))

            # Placeholder for compatibility
            self._v_12 = nn.Parameter(torch.empty(0, device=device).requires_grad_(False))
            self._L_22_inv = nn.Parameter(torch.empty(0, device=device).requires_grad_(False))

        self.max_radii2D = torch.zeros((init_n_gs), device=device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._color_mean], 'lr': training_args.feature_lr, "name": "color_mean"},
            {'params': [self._view_mean], 'lr': training_args.feature_lr, "name": "view_mean"},
        ]

        if self.use_simplified_color:
            # Simplified parameterization
            l.append({'params': [self._v_12], 'lr': training_args.feature_lr, "name": "v_12"})
            l.append({'params': [self._L_22_inv], 'lr': training_args.scale_lr, "name": "L_22_inv"})
            # Placeholders with zero lr
            l.append({'params': [self._color_scale], 'lr': 0.0, "name": "color_scale"})
            l.append({'params': [self._color_l_triangle], 'lr': 0.0, "name": "color_l_triangle"})
        else:
            # Full covariance parameterization
            l.append({'params': [self._color_scale], 'lr': training_args.scale_lr, "name": "color_scale"})
            l.append({'params': [self._color_l_triangle], 'lr': training_args.l_triangle_lr, "name": "color_l_triangle"})
            # Placeholders with zero lr
            l.append({'params': [self._v_12], 'lr': 0.0, "name": "v_12"})
            l.append({'params': [self._L_22_inv], 'lr': 0.0, "name": "L_22_inv"})

        if self.learnable_lambda_opc:
            l.append({'params': [self._lambda_opc], 'lr': training_args.opacity_lr, "name": "lambda_opc"})
        else:
            l.append({'params': [self._lambda_opc], 'lr': 0.0, "name": "lambda_opc"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        l.extend(['scale_{}'.format(i) for i in range(3)])
        l.extend(['rot_{}'.format(i) for i in range(4)])
        l.append('opacity')
        l.append('lambda_opc')
        l.extend(['color_mean_{}'.format(i) for i in range(3)])
        l.extend(['view_mean_{}'.format(i) for i in range(self._view_mean.shape[1])])

        if self.use_simplified_color:
            l.extend(['v_12_{}'.format(i) for i in range(self._v_12.shape[1])])
            l.extend(['L_22_inv_{}'.format(i) for i in range(self._L_22_inv.shape[1])])
        else:
            l.extend(['color_scale_{}'.format(i) for i in range(self._color_scale.shape[1])])
            l.extend(['color_l_triangle_{}'.format(i) for i in range(self._color_l_triangle.shape[1])])
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        lambda_opcs = self._lambda_opc.detach().cpu().numpy()
        color_mean = self._color_mean.detach().cpu().numpy()
        view_mean = self._view_mean.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if self.use_simplified_color:
            v_12 = self._v_12.detach().cpu().numpy()
            L_22_inv = self._L_22_inv.detach().cpu().numpy()
            attributes = np.concatenate([
                xyz, scales, rots, opacities, lambda_opcs,
                color_mean, view_mean, v_12, L_22_inv
            ], axis=1)
        else:
            color_scale = self._color_scale.detach().cpu().numpy()
            color_l_triangle = self._color_l_triangle.detach().cpu().numpy()
            attributes = np.concatenate([
                xyz, scales, rots, opacities, lambda_opcs,
                color_mean, view_mean, color_scale, color_l_triangle
            ], axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)

        scales = np.stack([
            np.asarray(plydata.elements[0]["scale_{}".format(i)])
            for i in range(3)
        ], axis=1)

        rots = np.stack([
            np.asarray(plydata.elements[0]["rot_{}".format(i)])
            for i in range(4)
        ], axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        try:
            lambda_opcs = np.asarray(plydata.elements[0]["lambda_opc"])[..., np.newaxis]
        except:
            lambda_opcs = np.full((xyz.shape[0], 1), inverse_sigmoid(0.35), dtype=np.float32)

        color_mean = np.stack([
            np.asarray(plydata.elements[0]["color_mean_{}".format(i)])
            for i in range(3)
        ], axis=1)

        # Determine view_mean dimension
        view_mean_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("view_mean_")]
        view_mean_dim = len(view_mean_names)
        view_mean = np.stack([
            np.asarray(plydata.elements[0]["view_mean_{}".format(i)])
            for i in range(view_mean_dim)
        ], axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._lambda_opc = nn.Parameter(torch.tensor(lambda_opcs, dtype=torch.float, device="cuda").requires_grad_(self.learnable_lambda_opc))
        self._color_mean = nn.Parameter(torch.tensor(color_mean, dtype=torch.float, device="cuda").requires_grad_(True))
        self._view_mean = nn.Parameter(torch.tensor(view_mean, dtype=torch.float, device="cuda").requires_grad_(True))

        # Check if file contains simplified or full parameterization
        v_12_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("v_12_")]
        L_22_inv_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("L_22_inv_")]

        if len(v_12_names) > 0 and len(L_22_inv_names) > 0:
            # Load simplified parameterization
            self.use_simplified_color = True
            v_12 = np.stack([
                np.asarray(plydata.elements[0]["v_12_{}".format(i)])
                for i in range(len(v_12_names))
            ], axis=1)
            L_22_inv = np.stack([
                np.asarray(plydata.elements[0]["L_22_inv_{}".format(i)])
                for i in range(len(L_22_inv_names))
            ], axis=1)
            self._v_12 = nn.Parameter(torch.tensor(v_12, dtype=torch.float, device="cuda").requires_grad_(True))
            self._L_22_inv = nn.Parameter(torch.tensor(L_22_inv, dtype=torch.float, device="cuda").requires_grad_(True))
            self._color_scale = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._color_l_triangle = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        else:
            # Load full covariance parameterization
            self.use_simplified_color = False
            color_scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color_scale_")]
            color_scale = np.stack([
                np.asarray(plydata.elements[0]["color_scale_{}".format(i)])
                for i in range(len(color_scale_names))
            ], axis=1)
            color_l_triangle_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color_l_triangle_")]
            color_l_triangle = np.stack([
                np.asarray(plydata.elements[0]["color_l_triangle_{}".format(i)])
                for i in range(len(color_l_triangle_names))
            ], axis=1)
            self._color_scale = nn.Parameter(torch.tensor(color_scale, dtype=torch.float, device="cuda").requires_grad_(True))
            self._color_l_triangle = nn.Parameter(torch.tensor(color_l_triangle, dtype=torch.float, device="cuda").requires_grad_(True))
            self._v_12 = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._L_22_inv = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

        # Precompute test-time values (only for full parameterization)
        if not self.use_simplified_color:
            self._precompute_test_values()

        self.active_sh_degree = self.max_sh_degree

    def _precompute_test_values(self):
        """Precompute matrices for fast test-time inference."""
        v = self.get_color_covariance()

        v_12 = v[:, :3, 3:]
        v_22 = v[:, 3:, 3:]

        self.v_22_inv = torch.inverse(v_22)
        self.v_regr = torch.bmm(v_12, self.v_22_inv)

        # Precompute normalized view direction
        view_mean_normalized = self._view_mean.clone()
        view_mean_normalized[:, :3] = self._view_mean[:, :3] / (self._view_mean[:, :3].norm(dim=1, keepdim=True) + 1e-8)
        self.view_direction = view_mean_normalized

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
            # Skip empty placeholder tensors
            if group["params"][0].numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

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
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        self._color_mean = optimizable_tensors["color_mean"]
        self._view_mean = optimizable_tensors["view_mean"]

        if self.use_simplified_color:
            self._v_12 = optimizable_tensors["v_12"]
            self._L_22_inv = optimizable_tensors["L_22_inv"]
        else:
            self._color_scale = optimizable_tensors["color_scale"]
            self._color_l_triangle = optimizable_tensors["color_l_triangle"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
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

    def densification_postfix(self, new_xyz, new_scaling, new_rotation, new_opacity,
                              new_lambda_opc, new_color_mean, new_view_mean,
                              new_color_scale=None, new_color_l_triangle=None,
                              new_v_12=None, new_L_22_inv=None):
        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "lambda_opc": new_lambda_opc,
            "color_mean": new_color_mean,
            "view_mean": new_view_mean,
        }

        if self.use_simplified_color:
            d["v_12"] = new_v_12
            d["L_22_inv"] = new_L_22_inv
            d["color_scale"] = torch.empty(0, device="cuda")
            d["color_l_triangle"] = torch.empty(0, device="cuda")
        else:
            d["color_scale"] = new_color_scale
            d["color_l_triangle"] = new_color_l_triangle
            d["v_12"] = torch.empty(0, device="cuda")
            d["L_22_inv"] = torch.empty(0, device="cuda")

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        self._color_mean = optimizable_tensors["color_mean"]
        self._view_mean = optimizable_tensors["view_mean"]

        if self.use_simplified_color:
            self._v_12 = optimizable_tensors["v_12"]
            self._L_22_inv = optimizable_tensors["L_22_inv"]
        else:
            self._color_scale = optimizable_tensors["color_scale"]
            self._color_l_triangle = optimizable_tensors["color_l_triangle"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians with high gradients."""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        scale = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale, dim=1).values > self.percent_dense * scene_extent
        )

        stds = scale[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) * 0.8)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_lambda_opc = self._lambda_opc[selected_pts_mask].repeat(N, 1)
        new_color_mean = self._color_mean[selected_pts_mask].repeat(N, 1)
        new_view_mean = self._view_mean[selected_pts_mask].repeat(N, 1)

        if self.use_simplified_color:
            new_v_12 = self._v_12[selected_pts_mask].repeat(N, 1)
            new_L_22_inv = self._L_22_inv[selected_pts_mask].repeat(N, 1)
            self.densification_postfix(
                new_xyz, new_scaling, new_rotation, new_opacity, new_lambda_opc,
                new_color_mean, new_view_mean, new_v_12=new_v_12, new_L_22_inv=new_L_22_inv
            )
        else:
            new_color_scale = self._color_scale[selected_pts_mask].repeat(N, 1)
            new_color_l_triangle = self._color_l_triangle[selected_pts_mask].repeat(N, 1)
            self.densification_postfix(
                new_xyz, new_scaling, new_rotation, new_opacity, new_lambda_opc,
                new_color_mean, new_view_mean, new_color_scale=new_color_scale, new_color_l_triangle=new_color_l_triangle
            )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradients and small scales."""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        scale = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_lambda_opc = self._lambda_opc[selected_pts_mask]
        new_color_mean = self._color_mean[selected_pts_mask]
        new_view_mean = self._view_mean[selected_pts_mask]

        if self.use_simplified_color:
            new_v_12 = self._v_12[selected_pts_mask]
            new_L_22_inv = self._L_22_inv[selected_pts_mask]
            self.densification_postfix(
                new_xyz, new_scaling, new_rotation, new_opacity, new_lambda_opc,
                new_color_mean, new_view_mean, new_v_12=new_v_12, new_L_22_inv=new_L_22_inv
            )
        else:
            new_color_scale = self._color_scale[selected_pts_mask]
            new_color_l_triangle = self._color_l_triangle[selected_pts_mask]
            self.densification_postfix(
                new_xyz, new_scaling, new_rotation, new_opacity, new_lambda_opc,
                new_color_mean, new_view_mean, new_color_scale=new_color_scale, new_color_l_triangle=new_color_l_triangle
            )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        """Main densification and pruning routine."""
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
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", scaling_modifier=1.0,
                    use_tcgs=False, tight_snugbox=False, compact_box_mult=1.0):
        """
        Render using Color N-DGS with view-dependent color and opacity.

        This method:
        1. Computes view direction for each Gaussian
        2. Applies conditional Gaussian slicing to get view-dependent color
        3. Computes view-dependent opacity scaling
        4. Renders using standard 3DGS rasterization with computed colors
        """
        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for each Gaussian
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1))
        view_dir = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)

        # For 7D (with time), append timestamp to query
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

        # Get lambda_opc
        if self.learnable_lambda_opc:
            lambda_opc = self.get_lambda_opc.squeeze(-1)
        else:
            lambda_opc = 0.35

        # Compute conditional color and opacity scale
        # Use test path only if precomputed values exist (after load_ply)
        if self.training or not hasattr(self, 'v_22_inv'):
            color_cond, opacity_scale = self.slice_color_gaussian(cond_params, lambda_opc=lambda_opc)
        else:
            color_cond, opacity_scale = self.slice_color_gaussian_test(cond_params, lambda_opc=lambda_opc)

        # Clamp color to valid range [0, 1]
        color_cond = torch.clamp(color_cond, 0.0, 1.0)

        # Compute final opacity
        # Note: lambda_opc is already applied inside the exponential in slice_color_gaussian
        opacity = self.get_opacity * opacity_scale

        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        bg_color = self.background if self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

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

        # Rasterize with scales/rotations (faster CUDA computation) instead of precomputed covariance
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=self.get_xyz,
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

    @torch.no_grad()
    def view_tcgs(self, camera_state, render_tab_state):
        """Callable function for the viewer."""
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

        self.background = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0

        # Create masked views
        orig_xyz, self._xyz = self._xyz, self._xyz[mask]
        orig_scaling, self._scaling = self._scaling, self._scaling[mask]
        orig_rotation, self._rotation = self._rotation, self._rotation[mask]
        orig_opacity, self._opacity = self._opacity, self._opacity[mask]
        orig_lambda_opc, self._lambda_opc = self._lambda_opc, self._lambda_opc[mask]
        orig_color_mean, self._color_mean = self._color_mean, self._color_mean[mask]
        orig_view_mean, self._view_mean = self._view_mean, self._view_mean[mask]

        if self.use_simplified_color:
            orig_v_12, self._v_12 = self._v_12, self._v_12[mask]
            orig_L_22_inv, self._L_22_inv = self._L_22_inv, self._L_22_inv[mask]
        else:
            orig_color_scale, self._color_scale = self._color_scale, self._color_scale[mask]
            orig_color_l_triangle, self._color_l_triangle = self._color_l_triangle, self._color_l_triangle[mask]

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
            self._scaling = orig_scaling
            self._rotation = orig_rotation
            self._opacity = orig_opacity
            self._lambda_opc = orig_lambda_opc
            self._color_mean = orig_color_mean
            self._view_mean = orig_view_mean
            if self.use_simplified_color:
                self._v_12 = orig_v_12
                self._L_22_inv = orig_L_22_inv
            else:
                self._color_scale = orig_color_scale
                self._color_l_triangle = orig_color_l_triangle

        render_colors = render_colors.permute(1, 2, 0)

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
