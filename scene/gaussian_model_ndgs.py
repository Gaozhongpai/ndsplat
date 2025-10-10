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
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.ndgs_utils import strip_lower_diag

from gsplat import (
    _slice_gaussian_ndgs_test as slice_gaussian_ndgs_test,
    _slice_gaussian_ndgs as slice_gaussian_ndgs,
    _l_triangle_to_covar as l_triangle_to_covar
)

# Import CUDA-accelerated gsplat operations for N-DGS (verified with unit tests)
from gsplat import (
    slice_gaussian_ndgs_test,
    slice_gaussian_ndgs,
    l_triangle_to_covar,
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

        Uses CUDA-accelerated gsplat implementation for fast computation.

        Args:
            q: Query direction (view direction) [N, C]
            c_dim: Conditional dimension (default 3 for spatial xyz)
            lambda_opc: Opacity scaling factor (default 0.35)

        Returns:
            m_cond: Conditional mean (3D position) [N, 3]
            cov3D_precomp: Conditional covariance (lower triangular elements) [N, 6]
            scale: Opacity scaling factor based on direction influence [N, 1]
        """
        m_1 = self.get_xyz  # [N, 3]
        m_2 = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3] normalized

        # Build covariance matrix from diagonal and lower triangular elements
        v = self.get_pc_v

        # Use CUDA-accelerated conditional Gaussian slicing
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
        CUDA-accelerated for fast inference.

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

        # Use CUDA-accelerated inference
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
        
        # self.opacity_act_inv=lambda x: inverse_sigmoid(x)
        self.diags_act = lambda x: torch.exp(x)
        self.diags_act_inv = lambda x: torch.log(torch.max(x, torch.tensor(1e-6, device=x.device)))
        self.l_triangs_act = lambda x: torch.sigmoid(x)*2.0-1.0
        self.l_triangs_act_inv = lambda x: inverse_sigmoid(torch.clip((x+1.0)/2.0, min=1e-6, max=1.0 - 1e-6))
        self.mean_scale = 1.0
        self.setup_functions()

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
    def get_pc_v(self):
        # Use CUDA-accelerated covariance computation
        diag = self.diags_act(self.diags)
        l_triang = self.l_triangs_act(self.l_triangs)
        return l_triangle_to_covar(diag, l_triang)  # [N, D, D] via CUDA
    
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
        # self.diags = torch.nn.Parameter(self.diags_act_inv(torch.cat([scales, scales], dim=-1)))

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self.diags = torch.nn.Parameter(self.diags_act_inv(torch.ones([init_n_gs, self.gs_dim], device=device) * self.cov_bias))
        self.l_triangs = torch.nn.Parameter(self.l_triangs_act_inv(torch.zeros([init_n_gs, self.gs_dim*(self.gs_dim-1)//2], device=device)))
        # self.color = torch.nn.Parameter(torch.ones([init_n_gs, self.color_dim], device=device) * self.init_color)
        
        # # LSH Culling params
        # self.projection_vecs = torch.randn(self.n_projection_vecs, self.gs_dim, device=device)
        # self.projection_vecs /= self.projection_vecs.norm(dim=-1, keepdim=True)
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
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
            {'params': [self.diags], 'lr': training_args.diags_lr, "name": "diags"},
            {'params': [self.l_triangs], 'lr': training_args.l_triangs_lr, "name": "l_triangs"},
            # {'params': self.color_net.parameters(), 'lr': training_args.color_lr, "name": "color_net"}
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
        for i in range(self.diags.shape[1]):
            l.append('diags_{}'.format(i))
        for i in range(self.l_triangs.shape[1]):
            l.append('l_triangs_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # projection_vecs = self.projection_vecs.cpu().numpy()
        # np.save(path.replace("ply", "npy"), projection_vecs)
        opacities = self._opacity.detach().cpu().numpy()
        diags = self.diags.detach().cpu().numpy()
        l_triangs = self.l_triangs.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # Build attributes list, including time for 7DGS
        attr_list = [xyz, normals, f_dc, f_rest, opacities]
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            mean_time = self._mean_time.detach().cpu().numpy()
            attr_list.append(mean_time)
        attr_list.extend([diags, l_triangs])

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

        diags_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("diags_")]
        diags_names = sorted(diags_names, key = lambda x: int(x.split('_')[-1]))
        diags = np.zeros((xyz.shape[0], len(diags_names)))
        for idx, attr_name in enumerate(diags_names):
            diags[:, idx] = np.asarray(plydata.elements[0][attr_name])

        l_triangs_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l_triangs_")]
        l_triangs_names = sorted(l_triangs_names, key = lambda x: int(x.split('_')[-1]))
        l_triangs = np.zeros((xyz.shape[0], len(l_triangs_names)))
        for idx, attr_name in enumerate(l_triangs_names):
            l_triangs[:, idx] = np.asarray(plydata.elements[0][attr_name])
            

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self.diags = nn.Parameter(torch.tensor(diags, dtype=torch.float, device="cuda").requires_grad_(True))
        self.l_triangs = nn.Parameter(torch.tensor(l_triangs, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load time parameter for 7DGS
        if self.input_dim == 7 and mean_time is not None:
            self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(True))
    
        ### test ####
        c_dim = 3
        # Use CUDA-accelerated covariance
        v = self.get_pc_v  # [N, D, D] via CUDA
        
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
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self.diags = optimizable_tensors["diags"]
        self.l_triangs = optimizable_tensors["l_triangs"]

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
                                    new_opacities, new_diags, new_l_triangs, new_mean_time=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "normal": new_normal,
            "opacity": new_opacities,
            "diags" : new_diags,
            "l_triangs" : new_l_triangs,
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
        self.diags = optimizable_tensors["diags"]
        self.l_triangs = optimizable_tensors["l_triangs"]

        # Handle time parameter for 7DGS
        if self.input_dim == 7 and "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, rotation, scale, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale, dim=1).values > self.percent_dense*scene_extent)

        stds = scale[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = rotation[selected_pts_mask].repeat(N,1,1) # build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_diags = self.diags_act_inv(self.diags_act(self.diags[selected_pts_mask]) * 0.8).repeat(N, 1)
        # new_diags = self.diags[selected_pts_mask].repeat(N, 1)
        new_l_triangs = self.l_triangs[selected_pts_mask].repeat(N, 1)

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, \
                                   new_opacity, new_diags, new_l_triangs, new_mean_time)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, scale):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale[:, :3], dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_diags = self.diags[selected_pts_mask]
        new_l_triangs = self.l_triangs[selected_pts_mask]

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, new_opacities, new_diags, new_l_triangs, new_mean_time)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        scale = self.get_scaling
        self.densify_and_clone(grads, max_grad, extent, scale)
        rotation, scale = self.get_rotation_scale
        self.densify_and_split(grads, max_grad, extent, rotation, scale)
        scale = self.get_scaling
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
            
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = scale[:, :3].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", use_tcgs=False, is_test=False, scaling_modifier=1.0, tight_snugbox=True):
        """
        Render using 6DGS conditional slicing with diff-gaussian-rasterization.
        This encapsulates the 6DGS-specific rendering logic.

        Args:
            viewpoint_camera: Camera viewpoint
            render_mode: Rendering mode (RGB, depth, etc.)
            use_tcgs: Whether to use TCGS rasterizer
            is_test: Whether in test mode
            scaling_modifier: Scaling factor for Gaussians
            tight_snugbox: Use tight snugbox for TCGS rasterization (default: True)
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
            # slice_gaussian returns upper-triangular format [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]
            # which is directly compatible with diff_gaussian_rasterization
            m_cond, cov3D_precomp, pdf_cond = self.slice_gaussian(cond_params, c_dim=3, lambda_opc=lambda_opc)

        # Compute opacity with conditional probability
        opacity = self.get_opacity * pdf_cond

        # Set up rasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Use background from model if set, otherwise default to black
        bg_color = self.background if self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # Get x_threshold from viewpoint_camera if it exists, otherwise use infinity
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

    @torch.no_grad()
    def view_tcgs(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer using TCGS rasterizer.

        This method provides interactive viewing capabilities for the 6DGS model,
        allowing real-time visualization through the GaussianViewer interface.

        Args:
            camera_state: Camera state from the viewer (contains c2w, K)
            render_tab_state: Render settings from viewer (GaussianRenderTabState)
            center: Optional centering of the scene

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

        # Optional centering
        if center:
            self._xyz -= self._xyz.mean(dim=0, keepdim=True)

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

        # Apply filtering mask for selective rendering
        opacity = self.get_opacity
        mask = create_mask(
            opacity,
            opacity_threshold=render_tab_state.opacity_threshold,
            use_percentile=render_tab_state.use_opacity_percentile,
            percentile=render_tab_state.opacity_percentile,
        )

        # Debug: Print mask statistics (uncomment if you want to see opacity info)
        # print(f"Opacity range: [{opacity.min().item():.4f}, {opacity.max().item():.4f}]")
        # print(f"Opacity threshold: {render_tab_state.opacity_threshold:.4f}")
        # print(f"Mask: {mask.sum().item()} / {mask.shape[0]} Gaussians passed")

        # Set background color
        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        # Temporarily filter Gaussians based on mask
        original_xyz = self._xyz
        original_features_dc = self._features_dc
        original_features_rest = self._features_rest
        original_normal = self._normal
        original_opacity = self._opacity
        original_diags = self.diags
        original_l_triangs = self.l_triangs

        # Check if mask has any valid Gaussians
        num_valid = mask.sum().item()

        # Update stats
        render_tab_state.total_count_number = len(original_xyz)
        render_tab_state.rendered_count_number = 0  # Will be updated after rendering

        # If no Gaussians pass the filter, return a blank image
        if num_valid == 0:
            # Return background color
            bg_color = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
            render_colors = bg_color.view(3, 1, 1).expand(3, H, W)
            return render_colors.cpu().numpy().transpose(1, 2, 0)

        # Apply mask
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._normal = self._normal[mask]
        self._opacity = self._opacity[mask]
        self.diags = self.diags[mask]
        self.l_triangs = self.l_triangs[mask]

        try:
            # Call render_tcgs
            render_output = self.render_tcgs(
                viewpoint_camera,
                render_mode=render_tab_state.render_mode,
                use_tcgs=True,
                is_test=False,
                scaling_modifier=1.0,
                tight_snugbox=render_tab_state.tight_snugbox
            )

            render_colors = render_output["render"]

            # Update render stats with actual rendered count
            render_tab_state.rendered_count_number = render_output["visibility_filter"].sum().item()

            # Handle different render modes
            if render_tab_state.render_mode == "Alpha":
                # For alpha mode, show opacity
                render_colors = render_output["visibility_filter"].float().unsqueeze(0)

            # Handle depth colormap (if single channel output)
            if render_colors.shape[0] == 1:
                # Simple grayscale to RGB conversion for depth
                render_colors = render_colors.repeat(3, 1, 1)

        finally:
            # Restore original tensors
            self._xyz = original_xyz
            self._features_dc = original_features_dc
            self._features_rest = original_features_rest
            self._normal = original_normal
            self._opacity = original_opacity
            self.diags = original_diags
            self.l_triangs = original_l_triangs

        # Convert from [C, H, W] to [H, W, C] for viewer
        render_colors = render_colors.permute(1, 2, 0)

        # Calculate and update FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            render_tab_state.fps = 1.0 / elapsed_time
        else:
            render_tab_state.fps = 0.0

        return render_colors.cpu().numpy()
