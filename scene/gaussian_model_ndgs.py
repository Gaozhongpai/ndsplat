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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
import time
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.ndgs_utils import strip_lower_diag

# # Import gsplat functions for N-DGS operations
# from gsplat import (
#     _slice_gaussian_ndgs as slice_gaussian_ndgs,
#     _l_triangle_to_covar as l_triangle_to_covar,
#     _l_triangle_to_rotmat as l_triangle_to_rotmat,
#     _rot_scale_l_triangle_to_covar as rot_scale_l_triangle_to_covar
# )

from gsplat import (
    slice_gaussian_ndgs,
    l_triangle_to_covar,
    l_triangle_to_rotmat,
    rot_scale_l_triangle_to_covar,
    rasterization,
)

# Import TCGS rasterizer
from tcgs_speedy_rasterizer import (
    GaussianRasterizationSettings as TCGSRasterizationSettings,
    GaussianRasterizer as TCGSRasterizer,
)


def randomly_sample_point_cloud(point_cloud, num_samples=15000):
    # Check if the point cloud has fewer points than the number of samples requested
    if len(point_cloud) <= num_samples:
        return point_cloud

    # Randomly select `num_samples` indices
    sampled_indices = np.random.choice(len(point_cloud), num_samples, replace=False)

    # Select the points at these indices
    sampled_points = point_cloud[sampled_indices]

    return sampled_points

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # ti.init(arch=ti.cuda, device_memory_GB=0.1)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        # Time activation: sigmoid maps internal parameter to [0, 1]
        self.time_act = lambda x: torch.sigmoid(x)
        self.time_act_inv = lambda x: inverse_sigmoid(torch.clip(x, min=1e-6, max=1.0 - 1e-6))

    def slice_gaussian(self, q, c_dim=3, lambda_opc=None, lambda_opc_time=None):
        """
        Perform conditional Gaussian slicing for N-DGS.
        Given ND Gaussian with mean [m_1, m_2] and covariance v,
        compute conditional distribution given observation q for m_2.

        Uses CUDA-accelerated gsplat implementation for fast computation.

        Args:
            q: Query direction (view direction + time for 7DGS) [N, C]
            c_dim: Conditional dimension (default 3 for spatial xyz)
            lambda_opc: Opacity scaling factor for view direction. Tensor [N] or scalar.
                       If None, uses self.get_lambda_opc (default behavior)
            lambda_opc_time: Opacity scaling factor for time. Tensor [N], scalar, or None.
                            If None, uses self.get_lambda_opc_time for 7DGS (default behavior)

        Returns:
            m_cond: Conditional mean (3D position) [N, 3]
            cov3D_precomp: Conditional covariance (lower triangular elements) [N, 6]
            scale: Opacity scaling factor based on direction influence [N, 1]
        """
        m_1 = self.get_xyz  # [N, 3]

        # Normalize view direction for consistency
        view_normalized = self._mean_view / self._mean_view.norm(dim=1, keepdim=True)

        # Build m_2 from separate parameters
        if self.input_dim == 7:
            m_2 = torch.cat([view_normalized, self.get_mean_time], dim=-1)  # [N, 4]
        else:
            m_2 = view_normalized  # [N, 3]

        # Build covariance matrix from diagonal and lower triangular elements
        v = self.get_pc_v

        # Use CUDA-accelerated conditional Gaussian slicing
        m_cond, cov3D_precomp, scale = slice_gaussian_ndgs(
            m_1=m_1,
            m_2=m_2,
            query=q,
            covars=v,
            lambda_opc=lambda_opc,
            lambda_opc_time=lambda_opc_time
        )

        return m_cond, cov3D_precomp, scale

    def __init__(self, sh_degree : int, input_dim: int = 6, use_rot_scale_l_triangle: bool = False,
                 learnable_lambda_opc: bool = True, time_duration: list = [0.0, 1.0], lambda_opc: float = 0.35):
        """
        Initialize GaussianModel with flexible covariance parametrization.

        Args:
            sh_degree: Maximum degree of spherical harmonics
            input_dim: Dimensionality (6 for 6DGS, 7 for 7DGS with time)
            use_rot_scale_l_triangle: If True, use rotation-scale-l_triangle parametrization (UBS style).
                                      If False, use direct diagonal-l_triangle parametrization (NDGS style).
            learnable_lambda_opc: If True, make lambda_opc a learnable parameter per Gaussian.
            time_duration: [min, max] time range for 7DGS (default: [0.0, 1.0]).
            lambda_opc: Default lambda_opc value for opacity scaling (default: 0.35).
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.input_dim = input_dim  # 6 for 6DGS, 7 for 7DGS (with time)
        self.use_rot_scale_l_triangle = use_rot_scale_l_triangle

        # Store default lambda_opc value
        self.default_lambda_opc = lambda_opc

        # Track lambda_opc training state to avoid redundant enable/disable calls
        self._lambda_opc_training_enabled = False
        self.learnable_lambda_opc = learnable_lambda_opc
        self.time_duration = time_duration  # Time range for 7DGS

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)  # Single SH
        self._features_rest = torch.empty(0)  # Single SH
        self._mean_view = torch.empty(0)  # View direction mean: [N, 3]
        self._mean_time = torch.empty(0)  # Time mean: [N, 1] (only for input_dim=7)
        self._opacity = torch.empty(0)
        self._lambda_opc = torch.empty(0)  # Learnable opacity scaling parameter (for view)
        self._lambda_opc_time = torch.empty(0)  # Learnable opacity scaling parameter (for time, only for input_dim=7)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)  # FastGS: separate accumulator for split decisions
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.background = torch.empty(0)  # Background color for rendering

        self.n_projection_vecs = 8
        self.color_dim = 8
        self.gs_dim = input_dim  # Use input_dim instead of hardcoded 6
        self.init_color=1.0
        self.cov_bias=1e-1

        # Unified parameters for both parametrizations
        self._scale = torch.empty(0)
        self._l_triangle = torch.empty(0)

        # Activation functions (parametrization-specific)
        if self.use_rot_scale_l_triangle:
            # UBS-style: rotation-scale-l_triangle parametrization
            def inverse_softplus(y):
                return y + torch.log(-torch.expm1(-y))

            self.scale_activation = torch.nn.functional.softplus
            self.scale_inverse_activation = inverse_softplus
            self.l_triangle_activation = lambda x: x
            self.l_triangle_inverse_activation = lambda x: x
        else:
            # NDGS-style: direct diagonal-l_triangle parametrization
            self.scale_activation = lambda x: torch.exp(x)
            self.scale_inverse_activation = lambda x: torch.log(torch.max(x, torch.tensor(1e-6, device=x.device)))
            self.l_triangle_activation = lambda x: torch.sigmoid(x)*2.0-1.0
            self.l_triangle_inverse_activation = lambda x: inverse_sigmoid(torch.clip((x+1.0)/2.0, min=1e-6, max=1.0 - 1e-6))

        self.setup_functions()

        # Compute indices for rest of l_triangle (excluding first 3 spatial rotation params)
        # Only needed for rot_scale_l_triangle parametrization
        if self.use_rot_scale_l_triangle:
            tril_i, tril_j = torch.tril_indices(self.gs_dim, self.gs_dim, offset=-1)
            mask_rest = (tril_i >= 3) | (tril_j >= 3)
            self.rest_i = tril_i[mask_rest].to(torch.int32).to("cuda")
            self.rest_j = tril_j[mask_rest].to(torch.int32).to("cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._mean_view,
            self._mean_time,
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
        self._features_dc,
        self._features_rest,
        self._mean_view,
        self._mean_time,
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
    def get_xyz(self):
        return self._xyz

    @property
    def get_mean(self):
        """Get full mean [xyz, view, time] where view is direction and time is for 7DGS."""
        if self.input_dim == 7:
            return torch.cat([self._xyz, self._mean_view, self.get_mean_time], dim=-1)
        else:
            return torch.cat([self._xyz, self._mean_view], dim=-1)

    @property
    def get_mean_view(self):
        """Get view direction mean [N, 3]."""
        return self._mean_view

    @property
    def get_mean_time(self):
        """Get activated time mean [N, 1] (only valid for input_dim=7)."""
        return self.time_act(self._mean_time)

    @property
    def get_rotation(self):
        """Get 6D rotation matrix from first 3 l_triangle parameters (rot_scale_l_triangle only)."""
        if not self.use_rot_scale_l_triangle:
            raise NotImplementedError("get_rotation only available with rot_scale_l_triangle parametrization")
        return l_triangle_to_rotmat(self.get_l_triangle[:, :3])

    @property
    def get_scale(self):
        """Get activated scale parameters (works for both parametrizations)."""
        return self.scale_activation(self._scale)

    @property
    def get_scaling(self):
        # Return first 3 dimensions of scales (spatial scales)
        return self.get_scale[:, :3]

    @property
    def get_l_triangle(self):
        """
        Get activated l_triangle parameters (works for both parametrizations).

        For input_dim=7, zeros out the view-time cross-terms (indices 18-20) to enforce
        block-diagonal structure between view (3x3) and time (1x1) within the conditioning block.

        L_triangle index layout for 7x7 covariance matrix (lower triangular, excluding diagonal):

               x    y    z   vx   vy   vz    t
          x    -
          y    0    -
          z    1    2    -
         vx    3    4    5    -
         vy    6    7    8    9    -
         vz   10   11   12   13   14    -
          t   15   16   17   18   19   20    -
                          ^^^^^^^^^^^
                          VIEW-TIME cross-terms (zeroed for block-diagonal)

        This makes the view-time conditioning block-diagonal:
        Σ_cond = [Σ_view (3x3),    0        ]
                 [      0,      σ_time (1x1)]
        """
        return self.l_triangle_activation(self._l_triangle)

    @property
    def get_pc_v(self):
        """
        Get full ND covariance matrix using the appropriate parametrization.

        Returns:
            Covariance matrix [N, D, D]
        """
        if self.use_rot_scale_l_triangle:
            # UBS-style: Use CUDA kernel for rotation-scale-l_triangle
            return rot_scale_l_triangle_to_covar(
                self.get_rotation,
                self.get_scale,
                self.get_l_triangle,
                self.rest_i,
                self.rest_j,
            )
        else:
            # NDGS-style: Use CUDA kernel for diagonal-l_triangle
            return l_triangle_to_covar(self.get_scale, self.get_l_triangle)  # [N, D, D] via CUDA

    @property
    def get_features(self):
        # Return single tensor for single SH rendering
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_lambda_opc(self):
        """Get lambda_opc parameter for view (learnable or fixed)."""
        if self.learnable_lambda_opc:
            return self.opacity_activation(self._lambda_opc)
        else:
            # Return fixed default value
            return torch.ones_like(self._opacity) * self.default_lambda_opc

    @property
    def get_lambda_opc_time(self):
        """Get lambda_opc_time parameter for time (learnable or fixed, only for input_dim=7)."""
        if self.input_dim != 7:
            return None
        if self.learnable_lambda_opc:
            return self.opacity_activation(self._lambda_opc_time)
        else:
            # Return fixed default value for time component
            return torch.ones_like(self._opacity) * self.default_lambda_opc

    @property
    def get_scaling_cond(self):
        v = self.get_pc_v

        # Slice the 6D covariance matrix
        v_11 = v[:, :3, :3]
        v_12 = v[:, :3, 3:]
        v_21 = v[:, 3:, :3]
        v_22 = v[:, 3:, 3:]

        # Compute conditional covariance
        v_cond = v_11 - torch.bmm(v_12, torch.linalg.inv(v_22).bmm(v_21))
        U, S, _ = torch.linalg.svd(v_cond)
        scale = torch.sqrt(S)
        return scale
        
    @property
    def get_rotation_scale(self):
        """
        Get rotation and scale from conditional covariance.

        For 7DGS (input_dim=7), also returns sigma_pt (position-time correlation).

        Returns:
            rotation: Rotation matrix [N, 3, 3]
            scale: Scale factors [N, 3]
            sigma_pt: Position-time correlation [N, 3, 1] (only for input_dim=7, else None)
        """
        v = self.get_pc_v

        if self.input_dim == 7:
            # N-DGS 7D: covariance structure is [xyz(3), direction(3), time(1)]
            # This matches the _mean structure: [direction(3), time(1)] appended to xyz
            # Slice the 7D covariance matrix
            v_p = v[:, :3, :3]  # position block (3x3)
            v_pd = v[:, :3, 3:6]  # position-direction block (3x3)
            v_pt = v[:, :3, 6:7]  # position-time block (3x1)
            v_d = v[:, 3:6, 3:6]  # direction block (3x3)
            v_dt = v[:, 3:6, 6:7]  # direction-time block (3x1)
            v_t = v[:, 6:7, 6:7]  # time block (1x1)

            # Build the direction-time block for conditioning [4x4]
            v_dt_block = torch.cat([
                torch.cat([v_d, v_dt], dim=-1),
                torch.cat([v_dt.transpose(-2, -1), v_t], dim=-1)
            ], dim=-2)  # [N, 4, 4]

            v_pdt = torch.cat([v_pd, v_pt], dim=-1)  # [N, 3, 4]

            # Compute conditional covariance (condition on direction and time)
            v_dt_inv = torch.inverse(v_dt_block)
            v_regr = torch.matmul(v_pdt, v_dt_inv)
            v_cond = v_p - torch.matmul(v_regr, v_pdt.transpose(-2, -1))

            # Return position-time correlation for time-based splitting
            sigma_pt = v_pt
        else:
            # 6DGS: covariance structure is [xyz(3), direction(3)]
            v_11 = v[:, :3, :3]
            v_12 = v[:, :3, 3:]
            v_21 = v[:, 3:, :3]
            v_22 = v[:, 3:, 3:]

            # Compute conditional covariance
            v_cond = v_11 - torch.bmm(v_12, torch.linalg.inv(v_22).bmm(v_21))
            sigma_pt = None

        U, S, _ = torch.linalg.svd(v_cond)
        scale = torch.sqrt(S)
        rotation = U

        # Ensure right-handed coordinate system
        det = torch.linalg.det(rotation)
        rotation[:, :, -1] *= det.sign().unsqueeze(-1)
        return rotation, scale, sigma_pt

    @property
    def get_xyz_covariance(self):
        """
        Get 3x3 spatial covariance matrix for MCMC noise injection.

        Returns the conditional covariance of spatial coordinates (xyz)
        given the view-dependent parameters. This is used for MCMC
        densification to add covariance-weighted spatial noise.

        Returns:
            Spatial covariance matrix [N, 3, 3]
        """
        if self.use_rot_scale_l_triangle:
            # Efficient path: use CUDA kernel with spatial_block=True (matches UBS)
            return rot_scale_l_triangle_to_covar(
                self.get_rotation,
                self.get_scale,
                self.get_l_triangle,
                self.rest_i,
                self.rest_j,
                spatial_block=True,
            )
        else:
            # Fallback: compute from full ND covariance
            v = self.get_pc_v

            # Slice the ND covariance matrix
            v_11 = v[:, :3, :3]
            v_12 = v[:, :3, 3:]
            v_21 = v[:, 3:, :3]
            v_22 = v[:, 3:, 3:]

            # Compute conditional covariance (3x3 spatial block)
            v_cond = v_11 - torch.bmm(v_12, torch.linalg.inv(v_22).bmm(v_21))
            return v_cond

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, mcmc_cap_max=None, densification_strategy="standard"):
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
        fused_color = RGB2SH(torch.tensor(pcd_colors).float().cuda())
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        init_n_gs = fused_color.shape[0]
        device = "cuda"
        
        # Initialize view direction mean [N, 3]
        dir = torch.randn((init_n_gs, 3), device=device)
        mean_view = (dir / dir.norm(dim=1, keepdim=True)).float().cuda()  # [N, 3]

        # Initialize time mean [N, 1] for 7DGS (in inverse-sigmoid space)
        # Match 7dgs-iccv: use pcd.time if available, otherwise random in scaled time_duration range
        if self.input_dim == 7:
            pcd_time = getattr(pcd, 'time', None)
            if pcd_time is None:
                # Random times scaled to time_duration range, then mapped to [0, 1] via sigmoid
                fused_times = (torch.rand(init_n_gs, 1, device=device) * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(np.asarray(pcd_time)).cuda().float()
                if fused_times.dim() == 1:
                    fused_times = fused_times.unsqueeze(-1)
            mean_time = self.time_act_inv(fused_times)
        else:
            mean_time = torch.empty(init_n_gs, 0, device=device)  # Empty for 6DGS

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Initialize lambda_opc for view
        lambda_opcs = inverse_sigmoid(self.default_lambda_opc * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Initialize lambda_opc_time for time (only for 7DGS)
        if self.input_dim == 7:
            lambda_opcs_time = inverse_sigmoid(self.default_lambda_opc * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            lambda_opcs_time = torch.empty((fused_point_cloud.shape[0], 0), dtype=torch.float, device="cuda")

        from sklearn.neighbors import NearestNeighbors
        def knn(x, K=4):
            x_np = x.cpu().numpy()
            model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(x)

        # Spatial scales (first 3): from KNN distances
        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales_spatial = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Non-spatial scales: match 7dgs-iccv initialization
        # - Direction scales (3): constant 1.0
        # - Time scale (1): sqrt(duration/10) for 7DGS
        if self.gs_dim > 3:
            if self.input_dim == 7:
                # 7DGS: direction (3) + time (1)
                scales_direction = self.scale_inverse_activation(
                    torch.ones(init_n_gs, 3, device=device) # * 0.1
                )
                # Time scale = sqrt(duration/10)
                dist_t = (self.time_duration[1] - self.time_duration[0]) / 10
                scales_time = self.scale_inverse_activation(
                    torch.ones(init_n_gs, 1, device=device) * math.sqrt(dist_t)
                )
                scales = torch.cat([scales_spatial, scales_direction, scales_time], dim=1)
            else:
                # 6DGS: direction (3) only
                scales_direction = self.scale_inverse_activation(
                    torch.ones(init_n_gs, self.gs_dim - 3, device=device) * 0.1
                )
                scales = torch.cat([scales_spatial, scales_direction], dim=1)
        else:
            scales = scales_spatial

        # L_triangle: [N, gs_dim*(gs_dim-1)//2] initialized to zeros (match 7dgs-iccv)
        l_triangles = self.l_triangle_inverse_activation(
            torch.zeros(init_n_gs, self.gs_dim*(self.gs_dim-1)//2, device=device)
        )

        self._scale = nn.Parameter(scales.requires_grad_(True))
        self._l_triangle = nn.Parameter(l_triangles.requires_grad_(True))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._mean_view = nn.Parameter(mean_view.requires_grad_(True))
        self._mean_time = nn.Parameter(mean_time.requires_grad_(self.input_dim == 7))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # lambda_opc parameters: initially non-trainable, enabled via patient-based training
        # Similar to opacity_scale in 4D-GS: start frozen, enable after patience or iteration > 14900
        self._lambda_opc = nn.Parameter(lambda_opcs)
        self._lambda_opc.requires_grad_(False)  # Start frozen
        self._lambda_opc_time = nn.Parameter(lambda_opcs_time)
        self._lambda_opc_time.requires_grad_(False)  # Start frozen
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # FastGS: for split decisions
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._mean_view], 'lr': training_args.feature_lr, "name": "mean_view"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        ]

        # Add mean_time only for input_dim=7
        if self.input_dim == 7:
            l.append({'params': [self._mean_time], 'lr': training_args.feature_lr, "name": "mean_time"})

        # Support both old and new naming for learning rates
        scale_lr = training_args.scale_lr
        l_triangle_lr = training_args.l_triangle_lr

        l.append({'params': [self._scale], 'lr': scale_lr, "name": "scale"})
        l.append({'params': [self._l_triangle], 'lr': l_triangle_lr, "name": "l_triangle"})

        # Add lambda_opc parameter
        # Patient-based training: start with lr=0, will be updated dynamically during training
        # For learnable mode: training starts after patience trigger or iteration > 14900
        # For non-learnable mode: always lr=0
        if self.learnable_lambda_opc:
            l.append({'params': [self._lambda_opc], 'lr': 0.0, "name": "lambda_opc"})  # Start frozen
        else:
            l.append({'params': [self._lambda_opc], 'lr': 0.0, "name": "lambda_opc"})

        # Add lambda_opc_time parameter (only for input_dim=7)
        if self.input_dim == 7:
            if self.learnable_lambda_opc:
                l.append({'params': [self._lambda_opc_time], 'lr': 0.0, "name": "lambda_opc_time"})  # Start frozen
            else:
                l.append({'params': [self._lambda_opc_time], 'lr': 0.0, "name": "lambda_opc_time"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def enable_lambda_opc_training(self, training_args):
        """
        Enable training for lambda_opc parameters (patient-based training).
        Similar to opacity_scale in 4D-GS: triggered after patience or iteration > 14900.
        Only applies when learnable_lambda_opc=True.

        Returns:
            bool: True if state changed, False if already enabled
        """
        if not self.learnable_lambda_opc:
            return False

        # Check if already enabled to avoid redundant work
        if self._lambda_opc_training_enabled:
            return False

        # Enable gradient computation
        self._lambda_opc.requires_grad_(True)
        if self.input_dim == 7:
            self._lambda_opc_time.requires_grad_(True)

        # Update optimizer learning rates
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "lambda_opc":
                param_group['lr'] = training_args.opacity_lr
            elif param_group["name"] == "lambda_opc_time" and self.input_dim == 7:
                param_group['lr'] = training_args.opacity_lr

        # Mark as enabled
        self._lambda_opc_training_enabled = True
        return True

    def disable_lambda_opc_training(self):
        """
        Disable training for lambda_opc parameters.
        Used during densification (before iteration 14900 and not patient).

        Returns:
            bool: True if state changed, False if already disabled
        """
        if not self.learnable_lambda_opc:
            return False

        # Check if already disabled to avoid redundant work
        if not self._lambda_opc_training_enabled:
            return False

        # Disable gradient computation
        self._lambda_opc.requires_grad_(False)
        if self.input_dim == 7:
            self._lambda_opc_time.requires_grad_(False)

        # Set optimizer learning rates to 0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "lambda_opc":
                param_group['lr'] = 0.0
            elif param_group["name"] == "lambda_opc_time":
                param_group['lr'] = 0.0

        # Mark as disabled
        self._lambda_opc_training_enabled = False
        return True

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # View direction mean attributes [N, 3]
        for i in range(self._mean_view.shape[1]):
            l.append('mean_view_{}'.format(i))
        # Time mean attributes [N, 1] (only for 7DGS)
        for i in range(self._mean_time.shape[1]):
            l.append('mean_time_{}'.format(i))
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('lambda_opc')
        # Add lambda_opc_time only for 7DGS
        if self.input_dim == 7:
            l.append('lambda_opc_time')

        # Use unified naming
        for i in range(self._scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._l_triangle.shape[1]):
            l.append('l_triangle_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        mean_view = self._mean_view.detach().cpu().numpy()
        mean_time = self._mean_time.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        lambda_opcs = self._lambda_opc.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # Build attributes list
        attr_list = [xyz, mean_view, mean_time, f_dc, f_rest, opacities, lambda_opcs]

        # Add lambda_opc_time only for 7DGS
        if self.input_dim == 7:
            lambda_opcs_time = self._lambda_opc_time.detach().cpu().numpy()
            attr_list.append(lambda_opcs_time)

        # Add covariance parameters (unified naming)
        scales = self._scale.detach().cpu().numpy()
        l_triangles = self._l_triangle.detach().cpu().numpy()
        attr_list.extend([scales, l_triangles])

        attributes = np.concatenate(attr_list, axis=1)
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
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        # Load mean_view parameter with backward compatibility
        mean_view_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mean_view_")]
        if mean_view_names:
            # New separate format: mean_view_0, mean_view_1, mean_view_2
            mean_view_names = sorted(mean_view_names, key=lambda x: int(x.split('_')[-1]))
            mean_view = np.zeros((xyz.shape[0], len(mean_view_names)))
            for idx, attr_name in enumerate(mean_view_names):
                mean_view[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            # Try old unified format: mean_0, mean_1, mean_2
            mean_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mean_")]
            if mean_names:
                mean_names = sorted(mean_names, key=lambda x: int(x.split('_')[-1]))
                mean_view = np.zeros((xyz.shape[0], 3))
                for idx in range(3):
                    mean_view[:, idx] = np.asarray(plydata.elements[0][mean_names[idx]])
            else:
                # Old format: convert from nx, ny, nz
                mean_view = np.stack((np.asarray(plydata.elements[0]["nx"]),
                                np.asarray(plydata.elements[0]["ny"]),
                                np.asarray(plydata.elements[0]["nz"])),  axis=1)

        # Load mean_time parameter with backward compatibility (only for 7DGS)
        mean_time_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mean_time_")]
        if mean_time_names:
            # New separate format: mean_time_0
            mean_time_names = sorted(mean_time_names, key=lambda x: int(x.split('_')[-1]))
            mean_time = np.zeros((xyz.shape[0], len(mean_time_names)))
            for idx, attr_name in enumerate(mean_time_names):
                mean_time[:, idx] = np.asarray(plydata.elements[0][attr_name])
        elif self.input_dim == 7:
            # Try old unified format: mean_3
            mean_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mean_")]
            if mean_names and len(mean_names) > 3:
                mean_names = sorted(mean_names, key=lambda x: int(x.split('_')[-1]))
                mean_time = np.asarray(plydata.elements[0][mean_names[3]])[..., np.newaxis]
            else:
                # Old format: try mean_time field
                try:
                    mean_time = np.asarray(plydata.elements[0]["mean_time"])[..., np.newaxis]
                except:
                    mean_time = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        else:
            # 6DGS: no time dimension
            mean_time = np.zeros((xyz.shape[0], 0), dtype=np.float32)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = np.ascontiguousarray(opacities, dtype=np.float32)

        # Load lambda_opc (with backward compatibility)
        try:
            lambda_opcs = np.asarray(plydata.elements[0]["lambda_opc"])[..., np.newaxis]
        except:
            # Default if not present in file
            lambda_opcs = np.full((xyz.shape[0], 1), inverse_sigmoid(self.default_lambda_opc), dtype=np.float32)
        lambda_opcs = np.ascontiguousarray(lambda_opcs, dtype=np.float32)

        # Load lambda_opc_time (only for 7DGS, with backward compatibility)
        if self.input_dim == 7:
            try:
                lambda_opcs_time = np.asarray(plydata.elements[0]["lambda_opc_time"])[..., np.newaxis]
            except:
                # Default if not present in file
                lambda_opcs_time = np.full((xyz.shape[0], 1), inverse_sigmoid(self.default_lambda_opc), dtype=np.float32)
            lambda_opcs_time = np.ascontiguousarray(lambda_opcs_time, dtype=np.float32)
        else:
            lambda_opcs_time = np.empty((xyz.shape[0], 0), dtype=np.float32)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Load covariance parameters (unified naming)
        # Try to load with new unified naming first, fall back to old names for backward compatibility
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        if not scale_names:  # Backward compatibility: try old "diags_" naming
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("diags_")]

        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        l_triangle_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l_triangle_")]
        if not l_triangle_names:  # Backward compatibility: try old "l_triangs_" naming
            l_triangle_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l_triangs_")]

        l_triangle_names = sorted(l_triangle_names, key = lambda x: int(x.split('_')[-1]))
        l_triangles = np.zeros((xyz.shape[0], len(l_triangle_names)))
        for idx, attr_name in enumerate(l_triangle_names):
            l_triangles[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._scale = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._l_triangle = nn.Parameter(torch.tensor(l_triangles, dtype=torch.float, device="cuda").requires_grad_(True))

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._lambda_opc = nn.Parameter(torch.tensor(lambda_opcs, dtype=torch.float, device="cuda").requires_grad_(self.learnable_lambda_opc))
        self._lambda_opc_time = nn.Parameter(torch.tensor(lambda_opcs_time, dtype=torch.float, device="cuda").requires_grad_(self.learnable_lambda_opc and self.input_dim == 7))
        self._mean_view = nn.Parameter(torch.tensor(mean_view, dtype=torch.float, device="cuda").requires_grad_(True))
        self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(self.input_dim == 7))

        ### Precompute test-time values ###
        c_dim = 3
        v = self.get_pc_v  # [N, D, D] via CUDA

        v_11 = v[:, :c_dim, :c_dim]
        v_12 = v[:, :c_dim, c_dim:]
        v_21 = v[:, c_dim:, :c_dim]
        v_22 = v[:, c_dim:, c_dim:]

        self.v_22_inv = torch.inverse(v_22)
        self.v_regr = torch.bmm(v_12, self.v_22_inv)
        v_cond = (v_11 - torch.bmm(self.v_regr, v_21))
        self.cov3D_precomp = strip_lower_diag(v_cond)
        self.shs = self.get_features

        # Precompute direction for test-time slicing
        # Normalize view direction for consistency
        view_normalized = self._mean_view / self._mean_view.norm(dim=1, keepdim=True)
        if self.input_dim == 7:
            self.direction = torch.cat([view_normalized, self.get_mean_time], dim=-1)  # [N, 4]
        else:
            self.direction = view_normalized  # [N, 3]

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
            if group["name"] in ["color_net"]:
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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._mean_view = optimizable_tensors["mean_view"]
        if "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]
        else:
            # For 6DGS (input_dim=6), mean_time is not in optimizer, prune manually
            self._mean_time = self._mean_time[valid_points_mask]
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        if "lambda_opc_time" in optimizable_tensors:
            self._lambda_opc_time = optimizable_tensors["lambda_opc_time"]
        elif self.input_dim == 7:
            self._lambda_opc_time = self._lambda_opc_time[valid_points_mask]
        # Update covariance tensors (unified naming)
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]  # FastGS
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "color_net":
                # Skip the color_net group as it doesn't need concatenation
                continue
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_mean_view, new_mean_time,
                                    new_opacities, new_lambda_opc, new_scale, new_l_triangle, new_lambda_opc_time=None):
        """
        Add new Gaussians to the model after densification.

        Args:
            new_scale: New scale parameters (unified naming)
            new_l_triangle: New l_triangle parameters (unified naming)
            new_lambda_opc: New lambda_opc parameters
            new_lambda_opc_time: New lambda_opc_time parameters (only for 7DGS)
            new_mean_view: New view direction mean parameters [N, 3]
            new_mean_time: New time mean parameters [N, 1] (for 7DGS) or [N, 0] (for 6DGS)
        """
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "mean_view": new_mean_view,
            "mean_time": new_mean_time,  # Always include (empty [N, 0] for 6DGS)
            "opacity": new_opacities,
            "lambda_opc": new_lambda_opc,
            "scale": new_scale,
            "l_triangle": new_l_triangle,
            }

        # Add lambda_opc_time only for 7DGS
        if self.input_dim == 7 and new_lambda_opc_time is not None:
            d["lambda_opc_time"] = new_lambda_opc_time

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._mean_view = optimizable_tensors["mean_view"]
        if "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]
        else:
            # For 6DGS (input_dim=6), mean_time is not in optimizer, concatenate manually
            self._mean_time = torch.cat([self._mean_time, new_mean_time], dim=0)
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        # Update lambda_opc_time for 7DGS
        if self.input_dim == 7:
            if "lambda_opc_time" in optimizable_tensors:
                self._lambda_opc_time = optimizable_tensors["lambda_opc_time"]
            else:
                # Concatenate manually if not in optimizer
                self._lambda_opc_time = torch.cat([self._lambda_opc_time, new_lambda_opc_time], dim=0)
        # Update covariance tensors (unified naming)
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # FastGS
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, rotation, scale, sigma_pt=None, N=2):
        """
        Split large Gaussians with high gradients.

        For 7DGS (input_dim=7), also considers time-based splitting using
        position-time correlation (sigma_pt) similar to 7DGS implementation.

        Args:
            grads: Accumulated gradients
            grad_threshold: Gradient threshold for splitting
            scene_extent: Scene extent for size comparison
            rotation: Rotation matrices [N, 3, 3]
            scale: Scale factors [N, 3]
            sigma_pt: Position-time correlation [N, 3, 1] (only for input_dim=7)
            N: Number of new Gaussians per split
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # Spatial split criterion: large Gaussians
        selected_pts_mask_x = torch.logical_and(selected_pts_mask,
                                              torch.max(scale, dim=1).values > self.percent_dense*scene_extent)

        # Time-based split criterion for 7DGS
        if self.input_dim == 7 and sigma_pt is not None:
            # Compute magnitude of position-time correlation
            # N-DGS 7D structure: [xyz(3), direction(3), time(1)] -> time scale at index 6
            scale_t = self.scale_activation(self._scale[:, 6])
            sigma_pt_magnitude = torch.norm(sigma_pt.squeeze(-1), dim=-1)  # [N, 3, 1] -> [N]

            # Split if high position-time correlation and large time scale
            # Thresholds adapted from 7DGS: sigma_pt_magnitude > 0.15*extent/2, scale_t > 0.25
            selected_pts_mask_t = torch.logical_and(
                sigma_pt_magnitude > 0.15 * scene_extent / 2,
                scale_t > 0.25
            )
            selected_pts_mask_t = torch.logical_and(selected_pts_mask, selected_pts_mask_t)

            # Combine spatial and time-based criteria
            selected_pts_mask = torch.logical_or(selected_pts_mask_x, selected_pts_mask_t)
        else:
            selected_pts_mask = selected_pts_mask_x
            selected_pts_mask_t = None

        stds = scale[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = rotation[selected_pts_mask].repeat(N,1,1) # build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        # For 7DGS: offset split positions along position-time correlation direction
        if self.input_dim == 7 and selected_pts_mask_t is not None and selected_pts_mask_t.sum() > 0:
            scale_t = self.scale_activation(self._scale[:, 6])  # time scale at index 6
            mask_t_in_selected = selected_pts_mask_t[selected_pts_mask]
            # Compute offset: sigma_pt / scale_t * 0.1 (from 7DGS)
            t_offset = sigma_pt[selected_pts_mask_t].squeeze(-1) / scale_t[selected_pts_mask_t].unsqueeze(-1) * 0.1
            # Apply opposite offsets to the two split children
            new_xyz[:selected_pts_mask.sum()][mask_t_in_selected] = new_xyz[:selected_pts_mask.sum()][mask_t_in_selected] + t_offset
            new_xyz[selected_pts_mask.sum():][mask_t_in_selected] = new_xyz[selected_pts_mask.sum():][mask_t_in_selected] - t_offset

        new_mean_view = self._mean_view[selected_pts_mask].repeat(N,1)
        new_mean_time = self._mean_time[selected_pts_mask].repeat(N,1) if self.input_dim == 7 else self._mean_time[selected_pts_mask].repeat(N,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_lambda_opc = self._lambda_opc[selected_pts_mask].repeat(N,1)
        new_lambda_opc_time = self._lambda_opc_time[selected_pts_mask].repeat(N,1) if self.input_dim == 7 else None
        # Scale down for split (applies to all dimensions) - unified naming
        new_scale = self._scale[selected_pts_mask]
        new_scale = self.scale_inverse_activation(self.scale_activation(new_scale) * 0.8).repeat(N, 1)
        new_l_triangle = self._l_triangle[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_mean_view, new_mean_time,
                                   new_opacity, new_lambda_opc, new_scale, new_l_triangle, new_lambda_opc_time)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradients and small scales."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Only clone small Gaussians - compute scale for CURRENT state
        scale = self.get_scaling_cond
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale[:, :3], dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_mean_view = self._mean_view[selected_pts_mask]
        new_mean_time = self._mean_time[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_lambda_opc = self._lambda_opc[selected_pts_mask]
        new_lambda_opc_time = self._lambda_opc_time[selected_pts_mask] if self.input_dim == 7 else None

        # Clone covariance parameters - unified naming
        new_scale = self._scale[selected_pts_mask]
        new_l_triangle = self._l_triangle[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_mean_view, new_mean_time,
                                   new_opacities, new_lambda_opc, new_scale, new_l_triangle, new_lambda_opc_time)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        """Main densification and pruning routine."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Clone and split (compute scale/rotation internally for current state)
        self.densify_and_clone(grads, max_grad, extent)
        rotation, scale, sigma_pt = self.get_rotation_scale
        self.densify_and_split(grads, max_grad, extent, rotation, scale, sigma_pt)

        # Prune low opacity and large Gaussians
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_cond[:, :3].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # FastGS: accumulate XY gradients for cloning, Z gradients for splitting
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # ==================== FastGS-style Densification Methods ====================
    # Adapted from FastGS (arXiv:2511.04283) for N-DGS

    def densify_and_clone_fastgs(self, metric_mask, grad_filter):
        """
        Clone Gaussians that satisfy both metric mask and gradient filter.

        This implements FastGS's multi-view consistent cloning strategy:
        Only clone Gaussians that have high error across multiple views.

        Args:
            metric_mask: Boolean mask from multi-view importance score
            grad_filter: Boolean mask from gradient-based selection
        """
        selected_pts_mask = torch.logical_and(metric_mask, grad_filter)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_mean_view = self._mean_view[selected_pts_mask]
        new_mean_time = self._mean_time[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_lambda_opc = self._lambda_opc[selected_pts_mask]
        new_lambda_opc_time = self._lambda_opc_time[selected_pts_mask] if self.input_dim == 7 else None
        new_scale = self._scale[selected_pts_mask]
        new_l_triangle = self._l_triangle[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_mean_view, new_mean_time,
            new_opacities, new_lambda_opc, new_scale, new_l_triangle, new_lambda_opc_time
        )

    def densify_and_split_fastgs(self, metric_mask, grad_filter, N=2):
        """
        Split Gaussians that satisfy both metric mask and gradient filter.

        This implements FastGS's multi-view consistent splitting strategy:
        Only split large Gaussians that have high error across multiple views.

        Args:
            metric_mask: Boolean mask from multi-view importance score
            grad_filter: Boolean mask from gradient-based selection
            N: Number of new Gaussians per split (default: 2)
        """
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, grad_filter)
        selected_pts_mask[:mask.shape[0]] = mask

        if not selected_pts_mask.any():
            return

        # Get rotation and scale for splitting (sigma_pt returned for 7DGS but not used here)
        rotation, scale, _ = self.get_rotation_scale

        stds = scale[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_mean_view = self._mean_view[selected_pts_mask].repeat(N, 1)
        new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1) if self.input_dim == 7 else self._mean_time[selected_pts_mask].repeat(N, 1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_lambda_opc = self._lambda_opc[selected_pts_mask].repeat(N, 1)
        new_lambda_opc_time = self._lambda_opc_time[selected_pts_mask].repeat(N, 1) if self.input_dim == 7 else None

        # Scale down for split (applies to all dimensions)
        new_scale = self._scale[selected_pts_mask]
        new_scale = self.scale_inverse_activation(self.scale_activation(new_scale) * 0.8).repeat(N, 1)
        new_l_triangle = self._l_triangle[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_mean_view, new_mean_time,
            new_opacity, new_lambda_opc, new_scale, new_l_triangle, new_lambda_opc_time
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune_fastgs(
        self,
        max_screen_size,
        min_opacity: float,
        extent: float,
        radii: torch.Tensor,
        grad_thresh: float = 0.0002,
        grad_abs_thresh: float = 0.0006,
        percent_dense: float = 0.01,
        importance_score: torch.Tensor = None,
        pruning_score: torch.Tensor = None,
        densify_score_thresh: float = 5.0,
        prune_budget_ratio: float = 0.5,
    ):
        """
        FastGS-style densification and pruning for N-DGS.

        This implements the multi-view consistent densification and pruning strategy:
        1. Select candidates based on gradient magnitude (XY for clone, Z for split)
        2. Filter by multi-view importance score (only densify high-error Gaussians)
        3. Clone small Gaussians, split large ones
        4. Prune based on opacity and multi-view pruning score

        Args:
            max_screen_size: Maximum screen size threshold for pruning
            min_opacity: Minimum opacity threshold for pruning
            extent: Scene extent for size-based decisions
            radii: Per-Gaussian radii from rendering
            grad_thresh: Gradient threshold for cloning (XY screen-space gradients)
            grad_abs_thresh: Gradient threshold for splitting (Z depth gradients)
            percent_dense: Percentage of extent for clone/split decision
            importance_score: Per-Gaussian importance score from multi-view
            pruning_score: Per-Gaussian pruning score from multi-view
            densify_score_thresh: Threshold on importance_score for densification
            prune_budget_ratio: Fraction of prunable Gaussians to actually prune
        """
        # Compute gradient-based selection (FastGS uses separate gradients for clone vs split)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        # Update max radii
        self.max_radii2D = torch.max(self.max_radii2D, radii)

        # Gradient-based candidate selection (FastGS key difference: separate thresholds)
        grad_qualifiers = torch.where(torch.norm(grads, dim=-1) >= grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_thresh, True, False)

        # Size-based split/clone decision
        scale = self.get_scaling_cond
        clone_qualifiers = torch.max(scale[:, :3], dim=1).values <= percent_dense * extent
        split_qualifiers = torch.max(scale[:, :3], dim=1).values > percent_dense * extent

        # FastGS: clones use XY gradients, splits use Z (abs) gradients
        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # FastGS key contribution: filter by multi-view importance score
        if importance_score is not None:
            metric_mask = importance_score > densify_score_thresh
        else:
            # If no importance score provided, use gradient-only selection
            metric_mask = torch.ones(self.get_xyz.shape[0], dtype=bool, device="cuda")

        # Clone and split with metric filtering
        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        # Pruning
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_cond[:, :3].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # FastGS pruning strategy: use pruning_score to guide removal
        if pruning_score is not None:
            scores = 1 - pruning_score
            to_remove = torch.sum(prune_mask)
            remove_budget = int(prune_budget_ratio * to_remove)

            # Only prune if there's a budget (matches original FastGS behavior)
            if remove_budget > 0:
                n_points = self.get_xyz.shape[0]
                padded_importance = torch.zeros(n_points, dtype=torch.float32, device="cuda")
                padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())

                selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
                sampled_indices = torch.multinomial(padded_importance, min(remove_budget, n_points), replacement=False)
                selected_pts_mask[sampled_indices] = True
                final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                self.prune_points(final_prune)
            # If remove_budget is 0, don't prune (FastGS behavior)
        else:
            # Fallback: if no pruning_score provided, use standard pruning
            self.prune_points(prune_mask)

        # Reset opacity (clamped at 0.8)
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        torch.cuda.empty_cache()

    def final_prune_fastgs(
        self,
        min_opacity: float = 0.1,
        pruning_score: torch.Tensor = None,
        prune_score_thresh: float = 0.9,
    ):
        """
        Final-stage pruning using FastGS strategy.

        After model convergence, aggressively prune Gaussians based on
        opacity and multi-view consistency score.

        Args:
            min_opacity: Minimum opacity threshold
            pruning_score: Per-Gaussian pruning score from multi-view
            prune_score_thresh: Threshold on pruning_score (higher = more aggressive)
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if pruning_score is not None:
            scores_mask = pruning_score > prune_score_thresh
            final_prune = torch.logical_or(prune_mask, scores_mask)
        else:
            final_prune = prune_mask

        self.prune_points(final_prune)
        torch.cuda.empty_cache()

    # MCMC-based densification methods (adapted from UBS)
    def _update_params(self, idxs, ratio):
        """Update parameters for MCMC sampling with opacity adjustment.

        Args:
            idxs: Indices of Gaussians to sample from
            ratio: Number of times each Gaussian is sampled (for opacity adjustment)

        Returns:
            Tuple of updated parameters
        """
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
            self._features_dc[idxs],
            self._features_rest[idxs],
            self._mean_view[idxs],
            self._mean_time[idxs],  # Always index, even if empty [N, 0] for 6DGS
            new_opacity,
            self._lambda_opc[idxs],
            self._lambda_opc_time[idxs] if self.input_dim == 7 else None,
            self._scale[idxs],
            self._l_triangle[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        """Sample from alive Gaussians with probability weighting.

        Args:
            probs: Probability weights for sampling
            num: Number of samples to draw
            alive_indices: Indices of alive Gaussians (optional)

        Returns:
            sampled_idxs: Indices of sampled Gaussians
            ratio: Count of how many times each index was sampled
        """
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        """Relocate dead Gaussians by sampling from alive ones (MCMC).

        This method implements MCMC-style densification by relocating dead Gaussians
        (low opacity) to positions sampled from alive Gaussians, weighted by opacity.

        Args:
            dead_mask: Boolean mask indicating dead Gaussians to relocate
        """
        if dead_mask is None or dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # Sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            relocated_xyz,
            relocated_features_dc,
            relocated_features_rest,
            relocated_mean_view,
            relocated_mean_time,
            relocated_opacity,
            relocated_lambda_opc,
            relocated_lambda_opc_time,
            relocated_scale,
            relocated_l_triangle,
        ) = self._update_params(reinit_idx, ratio=ratio)

        # Copy relocated parameters to dead Gaussian positions
        self._xyz.index_copy_(0, dead_indices, relocated_xyz)
        self._features_dc.index_copy_(0, dead_indices, relocated_features_dc)
        self._features_rest.index_copy_(0, dead_indices, relocated_features_rest)
        self._mean_view.index_copy_(0, dead_indices, relocated_mean_view)
        # Always copy mean_time (even if empty [N, 0] for 6DGS)
        if self._mean_time.numel() > 0:
            self._mean_time.index_copy_(0, dead_indices, relocated_mean_time)
        self._opacity.index_copy_(0, dead_indices, relocated_opacity)
        self._lambda_opc.index_copy_(0, dead_indices, relocated_lambda_opc)
        if self.input_dim == 7 and relocated_lambda_opc_time is not None:
            self._lambda_opc_time.index_copy_(0, dead_indices, relocated_lambda_opc_time)
        self._scale.index_copy_(0, dead_indices, relocated_scale)
        self._l_triangle.index_copy_(0, dead_indices, relocated_l_triangle)

        # Update opacity at source indices
        self._opacity.index_copy_(0, reinit_idx, self._opacity.index_select(0, dead_indices))

        # Reset optimizer state for updated indices
        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        """Add new Gaussians by sampling from existing ones (MCMC).

        This method implements MCMC-style densification by adding new Gaussians
        sampled from existing ones, weighted by opacity, up to a maximum cap.

        Args:
            cap_max: Maximum number of Gaussians allowed

        Returns:
            Number of Gaussians added
        """
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Sample based on opacity
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_mean_view,
            new_mean_time,
            new_opacity,
            new_lambda_opc,
            new_lambda_opc_time,
            new_scale,
            new_l_triangle,
        ) = self._update_params(add_idx, ratio=ratio)

        # Update opacity at source indices
        self._opacity[add_idx] = new_opacity

        # Add new Gaussians
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_mean_view,
            new_mean_time,
            new_opacity,
            new_lambda_opc,
            new_scale,
            new_l_triangle,
            new_lambda_opc_time,
        )

        # Reset optimizer state for source indices
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def replace_tensors_to_optimizer(self, inds=None):
        """Replace tensors in optimizer, optionally resetting specific indices.

        Args:
            inds: Indices to reset in optimizer state (None = reset all)
        """
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "mean_view": self._mean_view,
            "mean_time": self._mean_time,  # Always include (empty [N, 0] for 6DGS)
            "opacity": self._opacity,
            "lambda_opc": self._lambda_opc,
            "scale": self._scale,
            "l_triangle": self._l_triangle,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "color_net":
                continue
            if group["name"] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]

            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                if inds is not None:
                    # Reset only specified indices
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    # Reset all
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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._mean_view = optimizable_tensors["mean_view"]
        if "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]
        self._opacity = optimizable_tensors["opacity"]
        self._lambda_opc = optimizable_tensors["lambda_opc"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        torch.cuda.empty_cache()

        return optimizable_tensors

    def render(self, viewpoint_camera, render_mode="RGB", mask=None):
        """Render using gsplat rasterization."""
        if mask is None:
            mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)

        K = torch.zeros((3, 3), device=self._xyz.device)
        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        dir_pp = self.get_xyz - viewpoint_camera.camera_center.repeat(self._mean_view.shape[0], 1)
        view_dir = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        if self.input_dim == 7:
            timestamp = torch.full(
                (view_dir.shape[0], 1),
                viewpoint_camera.timestamp if hasattr(viewpoint_camera, 'timestamp') else 0.0,
                device=view_dir.device, dtype=view_dir.dtype,
            )
            cond_params = torch.cat([view_dir, timestamp], dim=-1)
        else:
            cond_params = view_dir

        lambda_opc = self.get_lambda_opc.squeeze(-1)
        lambda_opc_time = self.get_lambda_opc_time.squeeze(-1) if self.input_dim == 7 else None

        shs = self.get_features
        m_cond, cov3D_precomp, pdf_cond = self.slice_gaussian(cond_params, c_dim=3, lambda_opc=lambda_opc, lambda_opc_time=lambda_opc_time)
        opacity = self.get_opacity * pdf_cond

        # Convert upper-tri 6D covariance [N, 6] to 3x3 symmetric matrix [N, 3, 3]
        # Upper-tri order: [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]
        covars = torch.zeros(m_cond.shape[0], 3, 3, device=m_cond.device, dtype=m_cond.dtype)
        covars[:, 0, 0] = cov3D_precomp[:, 0]
        covars[:, 0, 1] = cov3D_precomp[:, 1]
        covars[:, 0, 2] = cov3D_precomp[:, 2]
        covars[:, 1, 0] = cov3D_precomp[:, 1]
        covars[:, 1, 1] = cov3D_precomp[:, 3]
        covars[:, 1, 2] = cov3D_precomp[:, 4]
        covars[:, 2, 0] = cov3D_precomp[:, 2]
        covars[:, 2, 1] = cov3D_precomp[:, 4]
        covars[:, 2, 2] = cov3D_precomp[:, 5]

        # NDGS has no beta parameter, pass ones
        betas = torch.ones(m_cond.shape[0], device=m_cond.device)

        rgbs, alphas, meta = rasterization(
            means=m_cond[mask],
            l_triagnles=None,
            scales=None,
            opacities=opacity.squeeze()[mask],
            betas=betas[mask],
            shs=shs[mask],
            sh_degree=self.active_sh_degree,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode,
            covars=covars[mask],
        )

        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]
        return {
            "render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
        }

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", scaling_modifier=1.0, use_tcgs=False, tight_snugbox=False, compact_box_mult=1.0):
        """
        Render using NDGS conditional slicing with diff-gaussian-rasterization.
        This encapsulates the NDGS-specific rendering logic.

        Args:
            viewpoint_camera: Camera viewpoint
            render_mode: Rendering mode (RGB, depth, etc.)
            use_tcgs: Whether to use TCGS rasterizer
            scaling_modifier: Scaling factor for Gaussians
            tight_snugbox: Use tight snugbox for TCGS rasterization (default: True)
            compact_box_mult: FastGS-style compact box multiplier (1.0 = SnugBox, <1.0 = tighter)
        """
        import math

        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for conditional slicing
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._mean_view.shape[0], 1))
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

        # Get lambda_opc (always tensor [N]) for view and time
        # These are always tensors, whether learnable or fixed (broadcasted to [N])
        lambda_opc = self.get_lambda_opc.squeeze(-1)  # [N]
        lambda_opc_time = self.get_lambda_opc_time.squeeze(-1) if self.input_dim == 7 else None  # [N] or None

        # Training mode: compute conditional slicing
        shs = self.get_features
        # slice_gaussian returns upper-triangular format [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]
        # which is directly compatible with diff_gaussian_rasterization
        m_cond, cov3D_precomp, pdf_cond = self.slice_gaussian(cond_params, c_dim=3, lambda_opc=lambda_opc, lambda_opc_time=lambda_opc_time)

        # Compute opacity with conditional probability
        opacity = self.get_opacity * pdf_cond

        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Use background from model if set, otherwise default to black
        bg_color = self.background if self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # Get x_threshold from viewpoint_camera if it exists, otherwise use infinity
        x_threshold = viewpoint_camera.x_threshold if hasattr(viewpoint_camera, 'x_threshold') and viewpoint_camera.x_threshold is not None else float('inf')

        # No metric_map needed for non-FastGS rendering (backwards compatible)
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

        # Rasterize
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=m_cond,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scores=None,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D_precomp,
        )

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "m_cond": m_cond,  # For position shift regularization
        }

    def render_tcgs_with_metric(self, viewpoint_camera, background=None, scaling_modifier=1.0,
                                 use_tcgs=False, tight_snugbox=False, compact_box_mult=1.0, get_flag=False, metric_map=None):
        """
        Render using NDGS conditional slicing with FastGS-style metric counting support.

        This method extends render_tcgs to support multi-view consistent densification
        by optionally counting per-Gaussian contributions to high-error pixels.

        Args:
            viewpoint_camera: Camera viewpoint
            background: Background color tensor [3] (uses self.background if None)
            scaling_modifier: Scaling factor for Gaussians
            use_tcgs: Whether to use TCGS rasterizer
            tight_snugbox: Use tight snugbox for TCGS rasterization
            compact_box_mult: FastGS-style compact box multiplier (1.0 = SnugBox, <1.0 = tighter)
            get_flag: If True, enable metric counting for FastGS densification
            metric_map: Binary mask [H*W] indicating high-error pixels (required if get_flag=True)

        Returns:
            Dictionary with rendered image, viewspace_points, visibility_filter, radii,
            and accum_metric_counts (if get_flag=True)
        """
        import math

        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for conditional slicing
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._mean_view.shape[0], 1))
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

        # Get lambda_opc (always tensor [N])
        # This is always a tensor, whether learnable or fixed (broadcasted to [N])
        lambda_opc = self.get_lambda_opc.squeeze(-1)  # [N]

        # Training mode: compute conditional slicing
        shs = self.get_features
        m_cond, cov3D_precomp, pdf_cond = self.slice_gaussian(cond_params, c_dim=3, lambda_opc=lambda_opc)

        # Compute opacity with conditional probability
        opacity = self.get_opacity * pdf_cond

        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Use provided background or model's background or default to black
        if background is not None:
            bg_color = background
        elif self.background.numel() > 0:
            bg_color = self.background
        else:
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # Get x_threshold from viewpoint_camera if it exists
        x_threshold = viewpoint_camera.x_threshold if hasattr(viewpoint_camera, 'x_threshold') and viewpoint_camera.x_threshold is not None else float('inf')

        # Create metric_map if None (rasterizer always expects a tensor)
        if metric_map is None:
            metric_map = torch.zeros(
                int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width),
                dtype=torch.int,
                device='cuda'
            )

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
            debug=False,
            get_flag=get_flag,
            metric_map=metric_map,
        )

        rasterizer = TCGSRasterizer(raster_settings=raster_settings)

        # Rasterize
        rendered_image, radii, render_time, accum_metric_counts = rasterizer(
            means3D=m_cond,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scores=None,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D_precomp,
        )

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "accum_metric_counts": accum_metric_counts,
        }

    @torch.no_grad()
    def view_tcgs(self, camera_state, render_tab_state):
        """Callable function for the viewer using TCGS rasterizer.

        This method provides interactive viewing capabilities for the 6DGS model,
        allowing real-time visualization through the GaussianViewer interface.

        Args:
            camera_state: Camera state from the viewer (contains c2w, K)
            render_tab_state: Render settings from viewer (GaussianRenderTabState)

        Returns:
            numpy array: Rendered image in [H, W, C] format for display
        """
        # Start timing for FPS calculation
        start_time = time.time()

        from scene.gaussian_viewer import GaussianRenderTabState
        assert isinstance(render_tab_state, GaussianRenderTabState)

        def create_mask(opacity, opacity_threshold, use_percentile=False, percentile=0.0):
            """
            Create mask based on opacity threshold or percentile for 6DGS.

            Args:
                opacity: [N, 1] Gaussian opacities (after sigmoid activation)
                opacity_threshold: Minimum opacity to render (if not using percentile)
                use_percentile: Use percentile filtering instead of absolute threshold
                percentile: Show top (100-percentile)% most opaque Gaussians

            Returns:
                mask: [N] Boolean mask for valid Gaussians
            """
            # Squeeze opacity to 1D
            if opacity.dim() > 1:
                opacity_1d = opacity.squeeze(-1)
            else:
                opacity_1d = opacity

            if use_percentile and percentile > 0.0:
                # Percentile filtering: show top (100-percentile)% most opaque
                # E.g., percentile=90 means show top 10% (above 90th percentile)
                threshold_value = torch.quantile(opacity_1d, percentile / 100.0)
                opacity_mask = opacity_1d > threshold_value
            else:
                # Absolute threshold filtering
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
        render_tab_state.rendered_count_number = 0  # Will be updated after rendering

        # If no Gaussians pass the filter, return a blank image
        if num_valid == 0:
            bg_color = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
            render_colors = bg_color.view(3, 1, 1).expand(3, H, W)
            return render_colors.cpu().numpy().transpose(1, 2, 0)

        # Set background color
        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        # Create masked views WITHOUT modifying self - much more efficient!
        # These are just tensor views/slices, not copies
        _xyz_masked = self._xyz[mask]
        _features_dc_masked = self._features_dc[mask]
        _features_rest_masked = self._features_rest[mask]
        _mean_view_masked = self._mean_view[mask]
        _mean_time_masked = self._mean_time[mask]
        _opacity_masked = self._opacity[mask]
        _lambda_opc_masked = self._lambda_opc[mask]
        _scale_masked = self._scale[mask]
        _l_triangle_masked = self._l_triangle[mask]

        # Temporarily swap in masked tensors using direct assignment
        # This is still save/restore but at least we're not allocating new storage
        orig_xyz, self._xyz = self._xyz, _xyz_masked
        orig_features_dc, self._features_dc = self._features_dc, _features_dc_masked
        orig_features_rest, self._features_rest = self._features_rest, _features_rest_masked
        orig_mean_view, self._mean_view = self._mean_view, _mean_view_masked
        orig_mean_time, self._mean_time = self._mean_time, _mean_time_masked
        orig_opacity, self._opacity = self._opacity, _opacity_masked
        orig_lambda_opc, self._lambda_opc = self._lambda_opc, _lambda_opc_masked
        orig_scale, self._scale = self._scale, _scale_masked
        orig_l_triangle, self._l_triangle = self._l_triangle, _l_triangle_masked

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
            # Restore original tensors (just pointer swaps, very fast)
            self._xyz = orig_xyz
            self._features_dc = orig_features_dc
            self._features_rest = orig_features_rest
            self._mean_view = orig_mean_view
            self._mean_time = orig_mean_time
            self._opacity = orig_opacity
            self._lambda_opc = orig_lambda_opc
            self._scale = orig_scale
            self._l_triangle = orig_l_triangle

        # Convert from [C, H, W] to [H, W, C] for viewer
        render_colors = render_colors.permute(1, 2, 0)

        # Calculate and update FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
