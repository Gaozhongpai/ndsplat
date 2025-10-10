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
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.ndgs_utils import strip_lower_diag

# Import gsplat functions for N-DGS operations
from gsplat import (
    _slice_gaussian_ndgs_test as slice_gaussian_ndgs_test,
    _slice_gaussian_ndgs as slice_gaussian_ndgs,
    _l_triangle_to_rotmat as l_triangle_to_rotmat,
    _rot_scale_l_triangle_to_covar as rot_scale_l_triangle_to_covar
)

from gsplat import (
    slice_gaussian_ndgs, 
    slice_gaussian_ndgs_test,
    l_triangle_to_rotmat,
    rot_scale_l_triangle_to_covar
)

# Import TCGS rasterizer
from tcgs_speedy_rasterizer import (
    GaussianRasterizationSettings as TCGSRasterizationSettings,
    GaussianRasterizer as TCGSRasterizer,
)

# from utils.ndgs_utils import project_vectors_gaussians
# from evaluators.lsh_evaluator import EvaluatorLSH
# import taichi as ti


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

    def slice_gaussian(self, q, c_dim=3, lambda_opc=0.35):
        """
        Perform conditional Gaussian slicing for N-DGS.
        Given ND Gaussian with mean [m_1, m_2] and covariance v,
        compute conditional distribution given observation q for m_2.

        Uses gsplat CUDA implementation for efficiency.

        Args:
            q: Query direction (view direction + time for 7DGS) [N, C]
            c_dim: Conditional dimension (default 3 for spatial xyz)
            lambda_opc: Opacity scaling factor (default 0.35)

        Returns:
            m_cond: Conditional mean (3D position) [N, 3]
            cov3D_precomp: Conditional covariance (lower triangular elements) [N, 6]
            scale: Opacity scaling factor based on direction influence [N, 1]
        """
        m_1 = self.get_xyz  # [N, 3]

        # For 7DGS, m_2 includes both normal (3D) and time (1D)
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            normal_normalized = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3]
            m_2 = torch.cat([normal_normalized, self._mean_time], dim=-1)  # [N, 4]
        else:
            m_2 = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3]

        # Build covariance matrix from diagonal and lower triangular elements
        v = self.get_pc_v

        # Use gsplat CUDA implementation for conditional Gaussian slicing
        m_cond, cov3D_precomp, scale = slice_gaussian_ndgs(
            m_1=m_1,
            m_2=m_2,
            query=q,
            covars=v,
            lambda_opc=lambda_opc
        )

        return m_cond, cov3D_precomp, scale

    def slice_gaussian_test(self, q, lambda_opc=0.35):
        """
        Optimized version of slice_gaussian for test/inference time.
        Uses precomputed self.v_22_inv and self.v_regr to avoid redundant computation.
        Now accelerated with custom CUDA kernel.

        Args:
            q: Query direction (view direction)
            lambda_opc: Opacity scaling factor (default 0.35)

        Returns:
            m_cond: Conditional mean (3D position)
            scale: Opacity scaling factor based on direction influence
        """
        m_1 = self.get_xyz
        m_2 = self.direction
        v_22_inv = self.v_22_inv
        v_regr = self.v_regr

        # Use optimized CUDA kernel for inference
        m_cond, scale = slice_gaussian_ndgs_test(
            m_1=m_1,
            m_2=m_2,
            v_22_inv=v_22_inv,
            v_regr=v_regr,
            query=q,
            lambda_opc=lambda_opc
        )
        return m_cond, scale


    def __init__(self, sh_degree : int, input_dim: int = 6):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.input_dim = input_dim  # 6 for 6DGS, 7 for 7DGS (with time)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._normal = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
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
        
        # self.color_net = nn.Sequential(
        #                     nn.Linear(6, 64),
        #                     nn.ReLU(),
        #                     nn.Linear(64, 3),
        #                     nn.Sigmoid()  # Output in [0, 1] range
        #                 ).to("cuda")
        
        # Use rotation-scale-l_triangle parametrization (like UBS model)
        self._scale = torch.empty(0)
        self._l_triangle = torch.empty(0)

        # Activation functions for scale and l_triangle
        def inverse_softplus(y):
            return y + torch.log(-torch.expm1(-y))

        self.scale_activation = torch.nn.functional.softplus
        self.scale_inverse_activation = inverse_softplus
        self.l_triangs_activation = lambda x: x
        self.l_triangs_inverse_activation = lambda x: x

        self.mean_scale = 1.0
        self.setup_functions()

        # Compute indices for rest of l_triangle (excluding first 3 spatial rotation params)
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
            self._normal,
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
        self._normal, 
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
    def get_normal(self):
        return self._normal
    
    @property
    def get_xyz_normal(self):
        # For 6DGS: [xyz, normal_xyz] -> 6D
        # For 7DGS: [xyz, normal_xyz, time] -> 7D (if _mean_time exists)
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            return torch.cat([self._xyz, self._normal, self._mean_time], dim=-1)
        return torch.cat([self._xyz, self._normal], dim=-1)
    
    # def cull(self, q, total_m, total_v):
    #     q_projections, _ = project_vectors_gaussians(vecs=q, projection_vecs=self.projection_vecs, n_hashes=self.n_projection_vecs)
    #     m_projections, m_projections_range = project_vectors_gaussians(vecs=total_m, projection_vecs=self.projection_vecs, cov=total_v, n_hashes=self.n_projection_vecs)
    #     mask = EvaluatorLSH.cull(q_projections.contiguous(), m_projections.contiguous(), m_projections_range.contiguous())
    #     return mask
    
    @property
    def get_rotation(self):
        """Get 6D rotation matrix from first 3 l_triangle parameters."""
        return l_triangle_to_rotmat(self.get_l_triangle[:, :3])

    @property
    def get_scale(self):
        """Get activated scale parameters."""
        return self.scale_activation(self._scale)

    @property
    def get_l_triangle(self):
        """Get activated l_triangle parameters."""
        return self.l_triangs_activation(self._l_triangle)

    @property
    def get_pc_v(self):
        """Use CUDA kernel for 6D covariance matrix construction with rotation-scale-l_triangle."""
        return rot_scale_l_triangle_to_covar(
            self.get_rotation,
            self.get_scale,
            self.get_l_triangle,
            self.rest_i,
            self.rest_j,
        )
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    # def get_color(self, cond_params):
    #     color = self.color_net(torch.cat([self._features_dc, cond_params, cond_params - self.get_normal], dim=-1))
    #     return color
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_scaling(self):
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
        v = self.get_pc_v     
        
        # Slice the 6D covariance matrix
        v_11 = v[:, :3, :3]
        v_12 = v[:, :3, 3:]
        v_21 = v[:, 3:, :3]
        v_22 = v[:, 3:, 3:]

        # Compute conditional covariance
        v_cond = v_11 - torch.bmm(v_12, torch.linalg.inv(v_22).bmm(v_21))

        ## Perform eigendecomposition
        # eigenvalues, eigenvectors = torch.linalg.eigh(v_cond)
        ## Extract scale and rotation
        # scale = torch.sqrt(torch.abs(eigenvalues))
        # rotation = eigenvectors
        
        U, S, _ = torch.linalg.svd(v_cond)
        scale = torch.sqrt(S)
        rotation = U

        # Ensure right-handed coordinate system
        det = torch.linalg.det(rotation)
        rotation[:, :, -1] *= det.sign().unsqueeze(-1)
        return rotation, scale

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        from scene.dataset_readers import fetchPly

        # pcd_sparse = fetchPly(os.path.join(path, "points3d_sparse.ply"))
        # scatter_point_clcoud = torch.tensor(np.asarray(pcd_sparse.points)).float().cuda()
        # scatter_point_clcoud = scatter_point_clcoud + torch.randn_like(scatter_point_clcoud)*10
        # fused_point_cloud = torch.cat([fused_point_cloud, scatter_point_clcoud], dim=0)
        # fused_color = RGB2SH(torch.tensor(
        #                         np.concatenate([np.asarray(pcd.colors), np.asarray(pcd_sparse.colors)])
        #                     ).float().cuda())
        
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        init_n_gs = fused_color.shape[0]
        device = "cuda"
        
        dir = torch.randn((init_n_gs, 3), device=device)
        normal = (dir / dir.norm(dim=1, keepdim=True)).float().cuda()

        # For 7DGS, initialize time dimension
        if self.input_dim == 7:
            mean_time = torch.empty(init_n_gs, 1, device=device).uniform_(0.0, 1.0)
            self._mean_time = nn.Parameter(mean_time.requires_grad_(True))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        # self._scale = torch.nn.Parameter(self.scale_activation_inv(torch.cat([scales, scales], dim=-1)))

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Initialize scales using KNN distances (matching UBS model)
        from sklearn.neighbors import NearestNeighbors
        def knn(x, K=4):
            x_np = x.cpu().numpy()
            model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(x)

        # Spatial scales (first 3): from KNN distances
        dist2 = (knn(fused_point_cloud)[:, 1:] ** 2).mean(dim=-1)
        scales_spatial = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Non-spatial scales (remaining dimensions): small random values
        if self.gs_dim > 3:
            scales_rest = self.scale_inverse_activation(
                torch.normal(1, 1e-5, size=(init_n_gs, self.gs_dim - 3), device=device)
            )
            scales = torch.cat([scales_spatial, scales_rest], dim=1)
        else:
            scales = scales_spatial

        # L_triangle: [N, gs_dim*(gs_dim-1)//2] initialized to small noise
        l_triangles = self.l_triangs_inverse_activation(
            torch.normal(0, 1e-5, size=(init_n_gs, self.gs_dim*(self.gs_dim-1)//2), device=device)
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._scale = nn.Parameter(scales.requires_grad_(True))
        self._l_triangle = nn.Parameter(l_triangles.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._normal], 'lr': training_args.feature_lr, "name": "normal"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scale], 'lr': training_args.scale_lr if hasattr(training_args, 'scale_lr') else training_args.diags_lr, "name": "scale"},
            {'params': [self._l_triangle], 'lr': training_args.l_triangle_lr if hasattr(training_args, 'l_triangle_lr') else training_args.l_triangs_lr, "name": "l_triangle"},
        ]

        # Add time parameter for 7DGS
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            l.append({'params': [self._mean_time], 'lr': training_args.feature_lr, "name": "mean_time"})

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

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # Add time for 7DGS
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            l.append('mean_time')
        for i in range(self._scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._l_triangle.shape[1]):
            l.append('l_triangle_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scale.detach().cpu().numpy()
        l_triangles = self._l_triangle.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # Build attributes list, including time for 7DGS
        attr_list = [xyz, normals, f_dc, f_rest, opacities]
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            mean_time = self._mean_time.detach().cpu().numpy()
            attr_list.append(mean_time)
        attr_list.extend([scales, l_triangles])

        attributes = np.concatenate(attr_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        # # Save color_net
        # torch.save(self.color_net.state_dict(), path.replace(".ply", "_color_net.pth"))
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = np.ascontiguousarray(opacities, dtype=np.float32)

        # Load time dimension for 7DGS
        mean_time = None
        if self.input_dim == 7:
            try:
                mean_time = np.asarray(plydata.elements[0]["mean_time"])[..., np.newaxis]
            except:
                # Initialize with default values if not present
                mean_time = np.zeros((xyz.shape[0], 1), dtype=np.float32)

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

        # projection_vecs = np.load(path.replace("ply", "npy"))
        # self.projection_vecs = torch.from_numpy(projection_vecs).float().cuda()

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        l_triangle_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l_triangle_")]
        l_triangle_names = sorted(l_triangle_names, key = lambda x: int(x.split('_')[-1]))
        l_triangles = np.zeros((xyz.shape[0], len(l_triangle_names)))
        for idx, attr_name in enumerate(l_triangle_names):
            l_triangles[:, idx] = np.asarray(plydata.elements[0][attr_name])


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scale = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._l_triangle = nn.Parameter(torch.tensor(l_triangles, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load time parameter for 7DGS
        if self.input_dim == 7 and mean_time is not None:
            self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(True))
    
        ### test ####
        c_dim = 3
        # Use CUDA kernel
        v = self.get_pc_v  # [N, D, D] using CUDA
        
        v_11 = v[:, :c_dim, :c_dim]
        v_12 = v[:, :c_dim, c_dim:]
        v_21 = v[:, c_dim:, :c_dim]
        v_22 = v[:, c_dim:, c_dim:]

        self.v_22_inv = torch.inverse(v_22)
        self.v_regr = torch.bmm(v_12, self.v_22_inv)
        v_cond = (v_11 - torch.bmm(self.v_regr, v_21)) # * scale.unsqueeze(-1)
        self.cov3D_precomp = strip_lower_diag(v_cond)
        self.shs = self.get_features 
        self.direction = self.get_normal/self.get_normal.norm(dim=1, keepdim=True)
    
        # # Load color_net
        # color_net_path = path.replace(".ply", "_color_net.pth")
        # if os.path.exists(color_net_path):
        #     self.color_net.load_state_dict(torch.load(color_net_path))
        # else:
        #     print(f"Warning: Color net state dict not found at {color_net_path}")

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
            if group["name"] == "color_net":
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
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        # Handle time parameter for 7DGS
        if self.input_dim == 7 and "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_normal, \
                                    new_opacities, new_scale, new_l_triangle, new_mean_time=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "normal": new_normal,
            "opacity": new_opacities,
            "scale" : new_scale,
            "l_triangle" : new_l_triangle,
            }

        # Add time parameter for 7DGS
        if self.input_dim == 7 and new_mean_time is not None:
            d["mean_time"] = new_mean_time

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        # Handle time parameter for 7DGS
        if self.input_dim == 7 and "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densify_and_clone_split(self, grads, grad_threshold, scene_extent, rotation, scale, N=3):
        device = self.get_xyz.device
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # Split and Clone Masks
        split_threshold = self.percent_dense * scene_extent
        max_scale = torch.max(scale, dim=1).values
        selected_pts_mask_split = torch.logical_and(selected_pts_mask, max_scale > split_threshold)
        # selected_pts_mask_clone = torch.logical_and(selected_pts_mask, max_scale <= split_threshold)
    
        scales = scale[selected_pts_mask]
        means = torch.zeros((scales.size(0), 3), device=device)        
        samples = torch.normal(mean=scales, std=scales)
        samples = torch.cat([samples, means, -samples], dim=0)
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_scale = self._scale[selected_pts_mask]
        split_diags = self.scale_inverse_activation(self.scale_activation(self._scale[selected_pts_mask_split]) * 0.8)
        new_scale[selected_pts_mask_split[selected_pts_mask]] = split_diags
        new_scale = new_scale.repeat(N, 1)

        new_l_triangle = self._l_triangle[selected_pts_mask].repeat(N, 1)

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal,
                                new_opacity, new_scale, new_l_triangle, new_mean_time)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=torch.bool)))
        self.prune_points(prune_filter)    

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians with high gradients (matching UBS model).

        Note: grads is from BEFORE densify_and_clone, so it only covers the original points.
        We only consider the original points for splitting (not the newly cloned ones).
        """
        n_original_points = grads.shape[0]
        n_current_points = self.get_xyz.shape[0]

        # Only evaluate gradients for the original points (not newly cloned ones)
        grads_squeezed = grads.squeeze()

        # For original points, check gradient threshold
        selected_pts_mask = torch.zeros((n_current_points), device="cuda", dtype=bool)
        selected_pts_mask[:n_original_points] = grads_squeezed >= grad_threshold

        # Compute scale and rotation for CURRENT state (after cloning)
        rotation, scale = self.get_rotation_scale

        # Also check scale threshold (large Gaussians)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale[:, :3], dim=1).values > self.percent_dense * scene_extent
        )

        # Sample new positions - only use spatial scales (first 3 dims)
        stds = scale[selected_pts_mask][:, :3].repeat(N, 1)
        samples = torch.normal(mean=0, std=stds)
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)

        # Transform samples to world space
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # Scale down for split (applies to all dimensions)
        new_scale = self._scale[selected_pts_mask]
        new_scale = self.scale_inverse_activation(self.scale_activation(new_scale) * 0.8).repeat(N, 1)
        new_l_triangle = self._l_triangle[selected_pts_mask].repeat(N, 1)

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal,
                                   new_opacity, new_scale, new_l_triangle, new_mean_time)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradients and small scales (matching UBS model)."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Only clone small Gaussians - compute scale for CURRENT state
        scale = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scale[:, :3], dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scale = self._scale[selected_pts_mask]
        new_l_triangle = self._l_triangle[selected_pts_mask]

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, new_opacities, new_scale, new_l_triangle, new_mean_time)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        """Main densification and pruning routine (matching UBS model)."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Clone and split (compute scale/rotation internally for current state)
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune low opacity and large Gaussians
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling[:, :3].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", use_tcgs=False, is_test=False, scaling_modifier=1.0):
        """
        Render using 6DGS conditional slicing with diff-gaussian-rasterization.
        This encapsulates the 6DGS-specific rendering logic.
        """
        import math

        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for conditional slicing
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_normal.shape[0], 1))
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

        lambda_opc = 0.35

        if is_test:
            # Test mode: use precomputed values
            m_cond, pdf_cond = self.slice_gaussian_test(cond_params, lambda_opc=lambda_opc)
            shs = self.shs
            cov3D_precomp = self.cov3D_precomp
        else:
            # Training mode: compute conditional slicing
            shs = self.get_features
            m_cond, cov3D_precomp, pdf_cond = self.slice_gaussian(cond_params, c_dim=3, lambda_opc=lambda_opc)

        # Compute opacity with conditional probability
        opacity = self.get_opacity * pdf_cond

        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Use background from model if set, otherwise default to black
        bg_color = self.background if self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

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
            prefiltered=False,
            use_tcgs=use_tcgs,
            tight_snugbox=True,
            debug=False
        )

        rasterizer = TCGSRasterizer(raster_settings=raster_settings)
        
        # Rasterize
        rendered_image, radii, render_time = rasterizer(
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
        }