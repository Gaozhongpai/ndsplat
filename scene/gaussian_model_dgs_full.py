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
# Full DGS with view-dependent position, time-dependent rotation, and opacity
#
# Key features:
# - Position shift: x_cond = x + v_12 @ V_22^{-1} @ (v - μ_v) (additive)
# - Rotation delta: q_cond = q_base ⊗ slerp(I, q_delta, 1 - attention_rot) (only for time dim when input_dim=7)
# - Opacity scale: o_cond = o_base * attention (multiplicative)
#
# Parameterization uses full Cholesky L_22_inv (like dgs.py) for better opacity:
# - Position + Opacity: _view_mean, _v_12, _L_22_inv (full Cholesky of V_22^{-1})
# - Rotation (time-only): _L_22_inv_diag_rot for time attention when input_dim=7
#
# Mathematical justification:
# - Position lives in R³ (unbounded) → additive shift natural
# - Rotation lives on SO(3) manifold → quaternion composition preserves unit norm
# - Opacity lives in [0,1] (bounded) → multiplicative keeps it bounded
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

# Import CUDA-accelerated slice functions
from gsplat import slice_gaussian_full


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (Hamilton product).
    q = [w, x, y, z] convention.

    Args:
        q1: [N, 4] first quaternion
        q2: [N, 4] second quaternion

    Returns:
        [N, 4] product quaternion
    """
    w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
    w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.cat([w, x, y, z], dim=1)


def quaternion_slerp(q0, q1, t):
    """
    Spherical linear interpolation between quaternions.

    Args:
        q0: [N, 4] start quaternion (identity for our use case)
        q1: [N, 4] end quaternion (learned delta)
        t: [N, 1] interpolation factor (0 = q0, 1 = q1)

    Returns:
        [N, 4] interpolated quaternion
    """
    # Normalize inputs
    q0 = F.normalize(q0, dim=1)
    q1 = F.normalize(q1, dim=1)

    # Compute dot product
    dot = (q0 * q1).sum(dim=1, keepdim=True)

    # If dot < 0, negate q1 to take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)

    # Compute angle
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Handle small angles (use linear interpolation)
    small_angle = sin_theta.abs() < 1e-6

    # SLERP formula
    s0 = torch.sin((1 - t) * theta) / (sin_theta + 1e-8)
    s1 = torch.sin(t * theta) / (sin_theta + 1e-8)

    # Fall back to linear interpolation for small angles
    s0 = torch.where(small_angle, 1 - t, s0)
    s1 = torch.where(small_angle, t, s1)

    result = s0 * q0 + s1 * q1
    return F.normalize(result, dim=1)


class GaussianModel:
    """
    Full DGS with view-dependent position, time-dependent rotation, and opacity.

    Uses full Cholesky parameterization (like dgs.py) for better opacity performance:
    - _L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision matrix)

    Mathematical formulation:
    - V_22^{-1} = L_22_inv @ L_22_inv^T (precision from Cholesky, no matrix inversion!)
    - x_cond = x + v_12 @ V_22^{-1} @ (v - μ_v)  (position shift)
    - attention = exp(-λ * (v - μ_v)^T @ V_22^{-1} @ (v - μ_v))  (opacity scale)
    - q_cond = q_base ⊗ slerp(I, q_delta, 1 - attention_time)  (rotation, only for time when input_dim=7)

    Physical interpretation:
    - attention ≈ 1 (canonical view): use base params unchanged
    - attention ≈ 0 (oblique view): apply full position shift and reduce opacity
    - Rotation only varies with time (not view direction) when input_dim=7
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

        # Beta activation: same as UBS - 4.0 * exp(beta)
        self.beta_activation = lambda x: 4.0 * torch.exp(x)

        # Time activation: sigmoid maps internal parameter to [0, 1]
        self.time_act = lambda x: torch.sigmoid(x)
        self.time_act_inv = lambda x: inverse_sigmoid(torch.clip(x, min=1e-6, max=1.0 - 1e-6))

    def __init__(self, sh_degree: int, input_dim: int = 6, use_beta: bool = False,
                 use_view_dependent_pos: bool = True, use_time_dependent_rotation: bool = True,
                 zero_view_time_cross_terms: bool = False, time_duration: list = [0.0, 1.0]):
        """
        Initialize Full DGS with view-dependent position, time-dependent rotation, and opacity.

        Args:
            sh_degree: Maximum SH degree for color
            input_dim: 6 for position+view, 7 for position+view+time
            use_beta: Whether to use spatial beta parameter (default: False)
            use_view_dependent_pos: Enable view-dependent position shift (default: True)
            use_time_dependent_rotation: Enable time-dependent rotation (only effective when input_dim=7)
            zero_view_time_cross_terms: If True, zero out view-time cross-terms to enforce block-diagonal
                                        structure (only effective when input_dim=7). Default: False.
            time_duration: [min, max] time range for 7DGS (default: [0.0, 1.0]).
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.input_dim = input_dim
        self.use_beta = use_beta
        self.use_view_dependent_pos = use_view_dependent_pos
        # Rotation conditioning only makes sense with time dimension
        self.use_time_dependent_rotation = use_time_dependent_rotation and (input_dim == 7)
        self.zero_view_time_cross_terms = zero_view_time_cross_terms
        self.time_duration = time_duration  # Time range for 7DGS
        self.cond_dim = input_dim - 3  # C = 3 for view-only, 4 for view+time

        # Standard 3DGS parameters
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._label = torch.empty(0)

        # === Position + Opacity conditioning (full Cholesky like dgs.py) ===
        # View direction mean (learned "canonical" view direction per Gaussian)
        self._mean_view = torch.empty(0)  # [N, 3] view direction
        self._mean_time = torch.empty(0)  # [N, 1] time mean (only for input_dim=7)

        # Bounded v_12 parameterization for position shift
        self._v_12_direction = torch.empty(0)  # [N, 3*C] - will be normalized
        self._v_12_scale = torch.empty(0)  # [N, 1] - sigmoid gives [0,1]

        # L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision matrix)
        # Uses full Cholesky (not diagonal) for better opacity performance
        self._L_22_inv = torch.empty(0)

        # === Time-dependent Rotation conditioning (only when input_dim=7) ===
        # Only uses the time dimension (index 3) for rotation attention
        # Rotation attention: exp(-||t - μ_t||² * precision_rot)
        # Rotation: q_cond = q_base ⊗ slerp(I, q_delta, 1 - attention_rot)
        self._L_22_inv_diag_rot = torch.empty(0)    # [N, 1] precision for time-based rotation attention
        self._rotation_delta = torch.empty(0)       # [N, 4] quaternion delta

        # Spatial beta: [N, 1] controls spatial uncertainty/bandwidth (optional)
        self._beta = torch.empty(0)

        # Hyperparameter for maximum position shift
        self.max_pos_shift = 1.0

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
        state = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._mean_view,
            self._mean_time,
            self._v_12_direction if self.use_view_dependent_pos else None,
            self._v_12_scale if self.use_view_dependent_pos else None,
            self._L_22_inv,
            self._L_22_inv_diag_rot if self.use_time_dependent_rotation else None,
            self._rotation_delta if self.use_time_dependent_rotation else None,
            self._beta if self.use_beta else None,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        return state

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._mean_view,
         self._mean_time,
         _v_12_direction,
         _v_12_scale,
         self._L_22_inv,
         _L_22_inv_diag_rot,
         _rotation_delta,
         _beta,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        if self.use_view_dependent_pos:
            if _v_12_direction is not None:
                self._v_12_direction = _v_12_direction
            if _v_12_scale is not None:
                self._v_12_scale = _v_12_scale
        if self.use_time_dependent_rotation:
            if _L_22_inv_diag_rot is not None:
                self._L_22_inv_diag_rot = _L_22_inv_diag_rot
            if _rotation_delta is not None:
                self._rotation_delta = _rotation_delta
        if self.use_beta and _beta is not None:
            self._beta = _beta
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
    def get_cond_mean(self):
        """Get conditioning mean [N, C]: normalized view direction concatenated with time (if input_dim=7)."""
        mean_view_normalized = self._mean_view / (self._mean_view.norm(dim=1, keepdim=True) + 1e-8)
        if self.input_dim == 7:
            return torch.cat([mean_view_normalized, self.get_mean_time], dim=1)
        return mean_view_normalized

    @property
    def get_mean_view(self):
        """Get view direction [N, 3]."""
        return self._mean_view

    @property
    def get_mean_time(self):
        """Get activated time mean [N, 1] (only valid for input_dim=7)."""
        return self.time_act(self._mean_time)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_beta(self):
        """Get activated spatial beta."""
        if not self.use_beta:
            return None
        return self.beta_activation(self._beta)

    @property
    def get_v_12(self):
        """
        Get effective v_12 (position-view covariance block) with bounded magnitude.
        Same as dgs_diag.
        """
        if not self.use_view_dependent_pos:
            return None
        C = self._v_12_direction.shape[1] // 3
        v_12_dir = F.normalize(self._v_12_direction, dim=1)
        v_12_dir = v_12_dir.view(-1, 3, C)

        spatial_scale = self.get_scaling.mean(dim=1, keepdim=True)
        v_12_magnitude = torch.sigmoid(self._v_12_scale) * self.max_pos_shift

        v_12_scaled = v_12_dir * (v_12_magnitude * spatial_scale).unsqueeze(-1)

        return v_12_scaled.view(-1, 3 * C)

    @property
    def get_rotation_delta(self):
        """
        Get normalized rotation delta quaternion.
        """
        if not self.use_time_dependent_rotation:
            return None
        return F.normalize(self._rotation_delta, dim=1)

    @property
    def get_L_22_inv(self):
        """
        Get L_22_inv, optionally with block-diagonal structure enforced for input_dim=7.

        For input_dim=7 (C=4), if zero_view_time_cross_terms is True, zeros out the
        view-time cross terms (indices 6, 7, 8) to ensure V_22^{-1} = L @ L^T is
        block-diagonal between view (3x3) and time (1x1).

        L_22_inv index layout for 4x4 precision matrix (lower triangular, row-major):

              vx   vy   vz    t
         vx    0
         vy    1    2
         vz    3    4    5
          t    6    7    8    9
               ^^^^^^^^^^^
               VIEW-TIME cross-terms (zeroed for block-diagonal)

        This makes V_22^{-1} = L @ L^T block-diagonal:
        V_22^{-1} = [Σ_view^{-1} (3x3),       0          ]
                    [        0,         σ_time^{-1} (1x1)]
        """
        if self.input_dim == 7 and self.zero_view_time_cross_terms:
            # Clone to avoid modifying the parameter
            L_22_inv = self._L_22_inv.clone()
            # Zero out cross terms: l_30, l_31, l_32 at indices 6, 7, 8
            L_22_inv[:, 6:9] = 0.0
            return L_22_inv
        return self._L_22_inv

    def compute_time_attention(self, query_time, mean_time, L_22_inv_diag_rot):
        """
        Compute time-dependent attention weight for rotation.

        attention = exp(-(t - μ_t)^2 * exp(d)^2)

        Args:
            query_time: [N, 1] current timestamp
            mean_time: [N, 1] canonical timestamp
            L_22_inv_diag_rot: [N, 1] precision parameter for time

        Returns:
            attention: [N, 1] attention weight in [0, 1]
        """
        # Compute residual
        x = query_time - mean_time  # [N, 1]

        # Compute precision: exp(d)^2
        L_diag = torch.exp(L_22_inv_diag_rot)  # [N, 1]

        # Compute scaled residual and squared norm
        scaled_x = x * L_diag  # [N, 1]
        query_influence = scaled_x * scaled_x  # [N, 1]

        # Attention weight
        attention = torch.exp(-query_influence)  # [N, 1]
        return attention

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def slice_gaussian_full_method(self, query, lambda_opc=None):
        """
        Perform conditional Gaussian slicing for position, rotation, and opacity.

        Uses the CUDA-accelerated slice_gaussian_full function that handles:
        - Position + Opacity: full Cholesky L_22_inv parameterization
        - Rotation: time-only conditioning when input_dim=7

        Given query view direction (+ optional time), compute:
        - Conditional position: x_cond = x + v_12 @ V_22^{-1} @ (v - μ_v)
        - Opacity scale: attention = exp(-λ * (v - μ_v)^T @ V_22^{-1} @ (v - μ_v))
        - Conditional rotation: q_cond = q_base ⊗ slerp(I, q_delta, 1 - attention_time)
          (only for time dimension when input_dim=7)

        Args:
            query: Query direction [N, C] where C=3 (view) or 4 (view+time)
            lambda_opc: Opacity scaling factor (default: 0.125 for input_dim=7, 0.35 for input_dim=6)

        Returns:
            x_cond: Conditional 3D position [N, 3]
            rotation_cond: Conditional rotation quaternion [N, 4]
            opacity_scale: View-dependent opacity scaling [N, 1]
        """
        # Set default lambda_opc based on input_dim
        if lambda_opc is None:
            lambda_opc = 0.125 if self.input_dim == 7 else 0.35

        # Use slice_gaussian_full CUDA kernel
        # Use get_L_22_inv to enforce block-diagonal structure (zero view-time cross terms)
        x_cond, rotation_cond, opacity_scale = slice_gaussian_full(
            xyz=self._xyz,                   # [N, 3]
            view_mean=self.get_cond_mean,    # [N, C]
            query=query,                     # [N, C]
            v_12=self.get_v_12 if self.use_view_dependent_pos else None,  # [N, 3*C] or None
            L_22_inv=self.get_L_22_inv,      # [N, C*(C+1)/2] full Cholesky (block-diagonal for C=4)
            rotation=self.get_rotation,      # [N, 4]
            rotation_delta=self.get_rotation_delta if self.use_time_dependent_rotation else None,  # [N, 4] or None
            L_22_inv_diag_rot=self._L_22_inv_diag_rot if self.use_time_dependent_rotation else None,  # [N, 1] or None
            lambda_opc=lambda_opc,
        )

        return x_cond, rotation_cond, opacity_scale

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussians from point cloud data with full view-dependent parameterization.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        num_gaussians = fused_color.shape[0]
        device = "cuda"
        C = self.cond_dim

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

        # Spatial scales from KNN distances
        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        # View direction mean: random unit vectors [N, 3]
        mean_view = torch.randn((num_gaussians, 3), device=device)
        mean_view = (mean_view / mean_view.norm(dim=1, keepdim=True)).float()

        # Time mean [N, 1] (only for input_dim=7, in inverse-sigmoid space)
        # Match 7dgs-iccv: use pcd.time if available, otherwise random in scaled time_duration range
        if self.input_dim == 7:
            pcd_time = getattr(pcd, 'time', None)
            if pcd_time is None:
                # Random times scaled to time_duration range, then mapped to [0, 1] via sigmoid
                fused_times = (torch.rand(num_gaussians, 1, device=device) * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(np.asarray(pcd_time)).cuda().float()
                if fused_times.dim() == 1:
                    fused_times = fused_times.unsqueeze(-1)
            mean_time = self.time_act_inv(fused_times)
        else:
            mean_time = torch.empty(num_gaussians, 0, device=device)

        # Position shift parameters
        if self.use_view_dependent_pos:
            v_12_direction = torch.normal(0, 0.01, size=(num_gaussians, 3 * C), device=device)
            v_12_scale = torch.zeros((num_gaussians, 1), device=device)
        else:
            v_12_direction = torch.empty(0, device=device)
            v_12_scale = torch.empty(0, device=device)

        # L_22_inv: [N, C*(C+1)/2] Cholesky of V_22^{-1} (precision)
        # Diagonal uses exp() activation, so initialize with log(scale)
        # Storage format: [l_00, l_10, l_11, l_20, l_21, l_22, ...] (row-major lower triangular)
        #
        # For input_dim=7 (C=4), we want BLOCK-DIAGONAL structure (no view-time cross terms):
        # L = [l_00,    0,    0,    0  ]   indices: 0
        #     [l_10, l_11,    0,    0  ]   indices: 1, 2
        #     [l_20, l_21, l_22,    0  ]   indices: 3, 4, 5
        #     [  0,    0,    0,  l_33  ]   indices: 6, 7, 8, 9 (set 6,7,8 to 0)
        # This makes V_22^{-1} = L @ L^T block-diagonal between view (3x3) and time (1x1)
        n_L_22_inv = C * (C + 1) // 2
        L_22_inv = torch.zeros(num_gaussians, n_L_22_inv, device=device)
        # Diagonal positions: 0 (i=0), 2 (i=1), 5 (i=2), 9 (i=3)
        # Match 7dgs-iccv: view directions use 1.0, time uses sqrt(duration/10)
        for i in range(C):
            diag_idx = i * (i + 1) // 2 + i
            if self.input_dim == 7 and i == C - 1:
                # Time precision: L_22_inv is the Cholesky of precision (V_22^{-1} = L @ L^T)
                # Covariance scale in 7dgs: sqrt(duration/10)
                # Precision = 1/variance = 1/(sqrt(duration/10))^2 = 10/duration
                # L_diag for precision Cholesky: sqrt(precision) = sqrt(10/duration) = 1/sqrt(duration/10)
                dist_t = (self.time_duration[1] - self.time_duration[0]) / 10
                L_22_inv[:, diag_idx] = math.log(1.0 / math.sqrt(dist_t))
            else:
                # View direction precision: 1.0 (variance=1, precision=1)
                L_22_inv[:, diag_idx] = math.log(1.0)  # log(1.0) = 0

        # Time-dependent rotation conditioning (only when input_dim=7)
        # Uses only time dimension (index 3) for rotation attention
        if self.use_time_dependent_rotation:
            # Only 1 dimension for time-based precision
            L_22_inv_diag_rot = torch.full((num_gaussians, 1), 0.347, device=device)
            rotation_delta = torch.zeros((num_gaussians, 4), device=device)
            rotation_delta[:, 0] = 1.0  # w = 1, x = y = z = 0
            # Add small noise to break symmetry
            rotation_delta[:, 1:] = torch.normal(0, 0.01, size=(num_gaussians, 3), device=device)
            rotation_delta = F.normalize(rotation_delta, dim=1)
        else:
            L_22_inv_diag_rot = torch.empty(0, device=device)
            rotation_delta = torch.empty(0, device=device)

        # Spatial beta (optional)
        if self.use_beta:
            betas = torch.zeros((num_gaussians, 1), dtype=torch.float, device=device)
        else:
            betas = torch.empty(0, device=device)

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._label = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.int32).cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._mean_view = nn.Parameter(mean_view.requires_grad_(True))
        self._mean_time = nn.Parameter(mean_time.requires_grad_(self.input_dim == 7))
        self._L_22_inv = nn.Parameter(L_22_inv.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        if self.use_view_dependent_pos:
            self._v_12_direction = nn.Parameter(v_12_direction.requires_grad_(True))
            self._v_12_scale = nn.Parameter(v_12_scale.requires_grad_(True))
        if self.use_time_dependent_rotation:
            self._L_22_inv_diag_rot = nn.Parameter(L_22_inv_diag_rot.requires_grad_(True))
            self._rotation_delta = nn.Parameter(rotation_delta.requires_grad_(True))
        if self.use_beta:
            self._beta = nn.Parameter(betas.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Optionally register gradient hook to zero out view-time cross terms for input_dim=7
        if self.input_dim == 7 and self.zero_view_time_cross_terms:
            def zero_cross_term_grad(grad):
                # Zero out indices 6, 7, 8 (l_30, l_31, l_32) to enforce block-diagonal
                grad[:, 6:9] = 0.0
                return grad
            self._L_22_inv.register_hook(zero_cross_term_grad)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._mean_view], 'lr': training_args.feature_lr, "name": "mean_view"},
            {'params': [self._L_22_inv], 'lr': training_args.rotation_lr, "name": "L_22_inv"},
        ]

        # Add mean_time only for input_dim=7
        if self.input_dim == 7:
            l.append({'params': [self._mean_time], 'lr': training_args.feature_lr, "name": "mean_time"})

        # Add view-dependent position parameters
        if self.use_view_dependent_pos:
            l.append({'params': [self._v_12_direction], 'lr': training_args.feature_lr, "name": "v_12_direction"})
            l.append({'params': [self._v_12_scale], 'lr': training_args.feature_lr, "name": "v_12_scale"})

        # Add time-dependent rotation parameters (precision + delta)
        if self.use_time_dependent_rotation:
            l.append({'params': [self._L_22_inv_diag_rot], 'lr': training_args.rotation_lr, "name": "L_22_inv_diag_rot"})
            l.append({'params': [self._rotation_delta], 'lr': training_args.rotation_lr, "name": "rotation_delta"})

        # Add beta if enabled
        if self.use_beta:
            beta_lr = getattr(training_args, 'beta_lr', training_args.rotation_lr)
            l.append({'params': [self._beta], 'lr': beta_lr, "name": "beta"})

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
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._mean_view.shape[1]):
            l.append('mean_view_{}'.format(i))
        for i in range(self._mean_time.shape[1]):
            l.append('mean_time_{}'.format(i))
        for i in range(self._L_22_inv.shape[1]):
            l.append('L_22_inv_{}'.format(i))
        if self.use_view_dependent_pos:
            for i in range(self._v_12_direction.shape[1]):
                l.append('v_12_direction_{}'.format(i))
            for i in range(self._v_12_scale.shape[1]):
                l.append('v_12_scale_{}'.format(i))
        if self.use_time_dependent_rotation:
            for i in range(self._L_22_inv_diag_rot.shape[1]):
                l.append('L_22_inv_diag_rot_{}'.format(i))
            for i in range(self._rotation_delta.shape[1]):
                l.append('rotation_delta_{}'.format(i))
        if self.use_beta:
            for i in range(self._beta.shape[1]):
                l.append('beta_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        mean_view = self._mean_view.detach().cpu().numpy()
        mean_time = self._mean_time.detach().cpu().numpy()
        L_22_inv = self._L_22_inv.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attrs_list = [xyz, f_dc, f_rest, opacities, scale, rotation,
                      mean_view, mean_time, L_22_inv]

        if self.use_view_dependent_pos:
            v_12_direction = self._v_12_direction.detach().cpu().numpy()
            v_12_scale = self._v_12_scale.detach().cpu().numpy()
            attrs_list.append(v_12_direction)
            attrs_list.append(v_12_scale)
        if self.use_time_dependent_rotation:
            L_22_inv_diag_rot = self._L_22_inv_diag_rot.detach().cpu().numpy()
            rotation_delta = self._rotation_delta.detach().cpu().numpy()
            attrs_list.append(L_22_inv_diag_rot)
            attrs_list.append(rotation_delta)
        if self.use_beta:
            betas = self._beta.detach().cpu().numpy()
            attrs_list.append(betas)

        attributes = np.concatenate(attrs_list, axis=1)
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

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load view direction [N, 3]
        mean_view_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mean_view_")]
        if mean_view_names:
            mean_view_names = sorted(mean_view_names, key=lambda x: int(x.split('_')[-1]))
            mean_view = np.zeros((xyz.shape[0], len(mean_view_names)))
            for idx, attr_name in enumerate(mean_view_names):
                mean_view[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            # Backward compatibility: try old view_mean format
            view_mean_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("view_mean_")]
            view_mean_names = sorted(view_mean_names, key=lambda x: int(x.split('_')[-1]))
            mean_view = np.zeros((xyz.shape[0], 3))
            for idx in range(3):
                mean_view[:, idx] = np.asarray(plydata.elements[0][view_mean_names[idx]])

        # Load view time [N, 1] (only for input_dim=7)
        mean_time_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mean_time_")]
        if mean_time_names:
            mean_time_names = sorted(mean_time_names, key=lambda x: int(x.split('_')[-1]))
            mean_time = np.zeros((xyz.shape[0], len(mean_time_names)))
            for idx, attr_name in enumerate(mean_time_names):
                mean_time[:, idx] = np.asarray(plydata.elements[0][attr_name])
        elif self.input_dim == 7:
            # Backward compatibility: try old view_mean format (index 3)
            view_mean_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("view_mean_")]
            if view_mean_names and len(view_mean_names) > 3:
                view_mean_names = sorted(view_mean_names, key=lambda x: int(x.split('_')[-1]))
                mean_time = np.asarray(plydata.elements[0][view_mean_names[3]])[..., np.newaxis]
            else:
                mean_time = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        else:
            mean_time = np.zeros((xyz.shape[0], 0), dtype=np.float32)

        # Load position conditioning parameters if enabled
        if self.use_view_dependent_pos:
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

        # Load L_22_inv (full Cholesky of precision matrix)
        L_22_inv_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("L_22_inv_")]
        L_22_inv_names = sorted(L_22_inv_names, key=lambda x: int(x.split('_')[-1]))
        L_22_inv = np.zeros((xyz.shape[0], len(L_22_inv_names)))
        for idx, attr_name in enumerate(L_22_inv_names):
            L_22_inv[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load time-dependent rotation conditioning parameters if enabled
        if self.use_time_dependent_rotation:
            # L_22_inv_diag_rot (time-specific precision, single dimension)
            L_22_rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("L_22_inv_diag_rot_")]
            L_22_rot_names = sorted(L_22_rot_names, key=lambda x: int(x.split('_')[-1]))
            L_22_inv_diag_rot = np.zeros((xyz.shape[0], len(L_22_rot_names)))
            for idx, attr_name in enumerate(L_22_rot_names):
                L_22_inv_diag_rot[:, idx] = np.asarray(plydata.elements[0][attr_name])

            # rotation_delta
            rot_delta_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rotation_delta_")]
            rot_delta_names = sorted(rot_delta_names, key=lambda x: int(x.split('_')[-1]))
            rotation_delta = np.zeros((xyz.shape[0], len(rot_delta_names)))
            for idx, attr_name in enumerate(rot_delta_names):
                rotation_delta[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load beta if enabled
        if self.use_beta:
            beta_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("beta_")]
            beta_names = sorted(beta_names, key=lambda x: int(x.split('_')[-1]))
            betas = np.zeros((xyz.shape[0], len(beta_names)))
            for idx, attr_name in enumerate(beta_names):
                betas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._mean_view = nn.Parameter(torch.tensor(mean_view, dtype=torch.float, device="cuda").requires_grad_(True))
        self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(self.input_dim == 7))
        self._L_22_inv = nn.Parameter(torch.tensor(L_22_inv, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.use_view_dependent_pos:
            self._v_12_direction = nn.Parameter(torch.tensor(v_12_direction, dtype=torch.float, device="cuda").requires_grad_(True))
            self._v_12_scale = nn.Parameter(torch.tensor(v_12_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_time_dependent_rotation:
            self._L_22_inv_diag_rot = nn.Parameter(torch.tensor(L_22_inv_diag_rot, dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation_delta = nn.Parameter(torch.tensor(rotation_delta, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_beta:
            self._beta = nn.Parameter(torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True))

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
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mean_view = optimizable_tensors["mean_view"]
        if self.input_dim == 7:
            self._mean_time = optimizable_tensors["mean_time"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]
        if self.use_view_dependent_pos:
            self._v_12_direction = optimizable_tensors["v_12_direction"]
            self._v_12_scale = optimizable_tensors["v_12_scale"]
        if self.use_time_dependent_rotation:
            self._L_22_inv_diag_rot = optimizable_tensors["L_22_inv_diag_rot"]
            self._rotation_delta = optimizable_tensors["rotation_delta"]
        if self.use_beta:
            self._beta = optimizable_tensors["beta"]

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
                              new_scaling, new_rotation, new_mean_view, new_mean_time,
                              new_L_22_inv, new_v_12_direction=None, new_v_12_scale=None,
                              new_L_22_inv_diag_rot=None, new_rotation_delta=None, new_beta=None):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "mean_view": new_mean_view,
            "L_22_inv": new_L_22_inv,
        }
        if self.input_dim == 7:
            d["mean_time"] = new_mean_time
        if self.use_view_dependent_pos:
            d["v_12_direction"] = new_v_12_direction
            d["v_12_scale"] = new_v_12_scale
        if self.use_time_dependent_rotation:
            d["L_22_inv_diag_rot"] = new_L_22_inv_diag_rot
            d["rotation_delta"] = new_rotation_delta
        if self.use_beta:
            d["beta"] = new_beta

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mean_view = optimizable_tensors["mean_view"]
        if self.input_dim == 7:
            self._mean_time = optimizable_tensors["mean_time"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]
        if self.use_view_dependent_pos:
            self._v_12_direction = optimizable_tensors["v_12_direction"]
            self._v_12_scale = optimizable_tensors["v_12_scale"]
        if self.use_time_dependent_rotation:
            self._L_22_inv_diag_rot = optimizable_tensors["L_22_inv_diag_rot"]
            self._rotation_delta = optimizable_tensors["rotation_delta"]
        if self.use_beta:
            self._beta = optimizable_tensors["beta"]

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
        new_mean_view = self._mean_view[selected_pts_mask].repeat(N, 1)
        new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1) if self.input_dim == 7 else self._mean_time[selected_pts_mask].repeat(N, 1)
        new_L_22_inv = self._L_22_inv[selected_pts_mask].repeat(N, 1)
        new_v_12_direction = self._v_12_direction[selected_pts_mask].repeat(N, 1) if self.use_view_dependent_pos else None
        new_v_12_scale = self._v_12_scale[selected_pts_mask].repeat(N, 1) if self.use_view_dependent_pos else None
        new_L_22_inv_diag_rot = self._L_22_inv_diag_rot[selected_pts_mask].repeat(N, 1) if self.use_time_dependent_rotation else None
        new_rotation_delta = self._rotation_delta[selected_pts_mask].repeat(N, 1) if self.use_time_dependent_rotation else None
        new_beta = self._beta[selected_pts_mask].repeat(N, 1) if self.use_beta else None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity,
            new_scaling, new_rotation, new_mean_view, new_mean_time, new_L_22_inv,
            new_v_12_direction, new_v_12_scale,
            new_L_22_inv_diag_rot, new_rotation_delta, new_beta
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
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_mean_view = self._mean_view[selected_pts_mask]
        new_mean_time = self._mean_time[selected_pts_mask]
        new_L_22_inv = self._L_22_inv[selected_pts_mask]
        new_v_12_direction = self._v_12_direction[selected_pts_mask] if self.use_view_dependent_pos else None
        new_v_12_scale = self._v_12_scale[selected_pts_mask] if self.use_view_dependent_pos else None
        new_L_22_inv_diag_rot = self._L_22_inv_diag_rot[selected_pts_mask] if self.use_time_dependent_rotation else None
        new_rotation_delta = self._rotation_delta[selected_pts_mask] if self.use_time_dependent_rotation else None
        new_beta = self._beta[selected_pts_mask] if self.use_beta else None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities,
            new_scaling, new_rotation, new_mean_view, new_mean_time, new_L_22_inv,
            new_v_12_direction, new_v_12_scale,
            new_L_22_inv_diag_rot, new_rotation_delta, new_beta
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
        Render using Full DGS with view-dependent position, scale, rotation, and opacity.
        """
        import time

        from tcgs_speedy_rasterizer import (
            GaussianRasterizationSettings as TCGSRasterizationSettings,
            GaussianRasterizer as TCGSRasterizer,
        )

        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for conditional slicing
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1))
        mean_view = dir_pp / dir_pp.norm(dim=1, keepdim=True)

        # For 7DGS, append timestamp
        if self.input_dim == 7:
            timestamp = torch.full(
                (mean_view.shape[0], 1),
                viewpoint_camera.timestamp if hasattr(viewpoint_camera, 'timestamp') else 0.0,
                device=mean_view.device,
                dtype=mean_view.dtype,
            )
            cond_params = torch.cat([mean_view, timestamp], dim=-1)
        else:
            cond_params = mean_view

        # Get all conditional parameters using CUDA-accelerated fused kernel
        # (scale is NOT conditioned, use get_scaling directly)
        m_cond, rotation_cond, opacity_scale = self.slice_gaussian_full_method(cond_params)

        # Get SH features for color
        shs = self.get_features

        # Get opacity scaled by view-dependent factor
        opacity = self.get_opacity * opacity_scale

        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        bg_color = self.background if hasattr(self, 'background') and self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
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

        # Rasterize with conditional position and rotation (scale is NOT conditioned)
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=m_cond,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scores=None,
            scales=self.get_scaling,    # Scale is NOT view-dependent
            rotations=rotation_cond,    # View/time-dependent rotation
            cov3D_precomp=None,
            betas=self.get_beta,
        )

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def view_tcgs(self, camera_state, render_tab_state):
        """Callable function for the viewer using TCGS rasterizer."""
        import time

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
        _xyz_masked = self._xyz[mask]
        _features_dc_masked = self._features_dc[mask]
        _features_rest_masked = self._features_rest[mask]
        _scaling_masked = self._scaling[mask]
        _rotation_masked = self._rotation[mask]
        _opacity_masked = self._opacity[mask]
        _mean_view_masked = self._mean_view[mask]
        _mean_time_masked = self._mean_time[mask]
        _L_22_inv_masked = self._L_22_inv[mask]
        if self.use_view_dependent_pos:
            _v_12_direction_masked = self._v_12_direction[mask]
            _v_12_scale_masked = self._v_12_scale[mask]
        if self.use_time_dependent_rotation:
            _L_22_inv_diag_rot_masked = self._L_22_inv_diag_rot[mask]
            _rotation_delta_masked = self._rotation_delta[mask]
        if self.use_beta:
            _beta_masked = self._beta[mask]

        # Temporarily swap in masked tensors
        orig_xyz, self._xyz = self._xyz, _xyz_masked
        orig_features_dc, self._features_dc = self._features_dc, _features_dc_masked
        orig_features_rest, self._features_rest = self._features_rest, _features_rest_masked
        orig_scaling, self._scaling = self._scaling, _scaling_masked
        orig_rotation, self._rotation = self._rotation, _rotation_masked
        orig_opacity, self._opacity = self._opacity, _opacity_masked
        orig_mean_view, self._mean_view = self._mean_view, _mean_view_masked
        orig_mean_time, self._mean_time = self._mean_time, _mean_time_masked
        orig_L_22_inv, self._L_22_inv = self._L_22_inv, _L_22_inv_masked
        if self.use_view_dependent_pos:
            orig_v_12_direction, self._v_12_direction = self._v_12_direction, _v_12_direction_masked
            orig_v_12_scale, self._v_12_scale = self._v_12_scale, _v_12_scale_masked
        if self.use_time_dependent_rotation:
            orig_L_22_inv_diag_rot, self._L_22_inv_diag_rot = self._L_22_inv_diag_rot, _L_22_inv_diag_rot_masked
            orig_rotation_delta, self._rotation_delta = self._rotation_delta, _rotation_delta_masked
        if self.use_beta:
            orig_beta, self._beta = self._beta, _beta_masked

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
            # Restore original tensors
            self._xyz = orig_xyz
            self._features_dc = orig_features_dc
            self._features_rest = orig_features_rest
            self._scaling = orig_scaling
            self._rotation = orig_rotation
            self._opacity = orig_opacity
            self._mean_view = orig_mean_view
            self._mean_time = orig_mean_time
            self._L_22_inv = orig_L_22_inv
            if self.use_view_dependent_pos:
                self._v_12_direction = orig_v_12_direction
                self._v_12_scale = orig_v_12_scale
            if self.use_time_dependent_rotation:
                self._L_22_inv_diag_rot = orig_L_22_inv_diag_rot
                self._rotation_delta = orig_rotation_delta
            if self.use_beta:
                self._beta = orig_beta

        render_colors = render_colors.permute(1, 2, 0)

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
