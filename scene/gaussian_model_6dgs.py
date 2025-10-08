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
from utils.ndgs_utils import create_cholesky, create_cholesky_v2, strip_lower_diag
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


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
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
        self.gs_dim = 6
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
        return torch.cat([self._xyz, self._normal], dim=-1)
    
    # def cull(self, q, total_m, total_v):
    #     q_projections, _ = project_vectors_gaussians(vecs=q, projection_vecs=self.projection_vecs, n_hashes=self.n_projection_vecs)
    #     m_projections, m_projections_range = project_vectors_gaussians(vecs=total_m, projection_vecs=self.projection_vecs, cov=total_v, n_hashes=self.n_projection_vecs)
    #     mask = EvaluatorLSH.cull(q_projections.contiguous(), m_projections.contiguous(), m_projections_range.contiguous())
    #     return mask
    
    @property
    def get_pc_v(self):
        return create_cholesky(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))
    
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
        # v = self.get_pc_v
        v = create_cholesky_v2(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))       
        
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
        # v = self.get_pc_v
        v = create_cholesky_v2(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))       
        
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
    
    def get_nd_scale_rotation(self, eps=1e-10, is_rotate=True):
        # Compute full 7x7 covariance matrix
        Sigma = create_cholesky_v2(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))
        
        # Add small diagonal for numerical stability
        Sigma = Sigma + eps * torch.eye(self.gs_dim, device=Sigma.device)
    
        # Fallback to eigendecomposition for numerical stability
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        s = torch.sqrt(torch.clamp(eigvals, min=eps))
        R = eigvecs
        
        if is_rotate:
            # Ensure proper rotation
            det = torch.linalg.det(R)
            R[..., :, -1] *= torch.sign(det).unsqueeze(-1)
        return R, s

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
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, diags, l_triangs), axis=1)
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
    
        ### test ####
        c_dim = 3
        v = create_cholesky_v2(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))
        
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
        self.diags = optimizable_tensors["diags"]
        self.l_triangs = optimizable_tensors["l_triangs"]

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
                                    new_opacities, new_diags, new_l_triangs):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "normal": new_normal,
            "opacity": new_opacities,
            "diags" : new_diags,
            "l_triangs" : new_l_triangs,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self.diags = optimizable_tensors["diags"]
        self.l_triangs = optimizable_tensors["l_triangs"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_clone_split_bk(self, grads, grad_threshold, scene_extent, rotation, scale, N=3):
        device = self.get_xyz.device  # Ensure device consistency
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Split and Clone Masks
        split_threshold = self.percent_dense * scene_extent
        selected_pts_mask_split = selected_pts_mask & (torch.max(scale, dim=1).values > split_threshold)
        selected_pts_mask_clone = selected_pts_mask & (torch.max(scale, dim=1).values <= split_threshold)

        # Number of points to process
        num_split = selected_pts_mask_split.sum()
        num_clone = selected_pts_mask_clone.sum()

        # Handle Split Points
        if num_split > 0:
            stds_split = scale[selected_pts_mask_split]
            means_split = torch.zeros((num_split, 3), device=device)
            samples_split = torch.cat([
                means_split,
                means_split + stds_split,  # Ensure proper broadcasting
                means_split - stds_split
            ], dim=0)  # Shape: (num_split * 3, 3)

            rots_split = rotation[selected_pts_mask_split].repeat(N, 1, 1)  # Shape: (num_split * N, 3, 3)

            new_xyz_split = torch.bmm(rots_split, samples_split.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask_split].repeat(N, 1)
            new_normal_split = self._normal[selected_pts_mask_split].repeat(N, 1)
            new_features_dc_split = self._features_dc[selected_pts_mask_split].repeat(N, 1, 1)
            new_features_rest_split = self._features_rest[selected_pts_mask_split].repeat(N, 1, 1)
            new_opacity_split = self._opacity[selected_pts_mask_split].repeat(N, 1)
            new_diags_split = self.diags_act_inv(
                                        self.diags_act(self.diags[selected_pts_mask_split]) * 0.8
                                        ).repeat(N, 1)
            new_l_triangs_split = self.l_triangs[selected_pts_mask_split].repeat(N, 1)
        else:
            new_xyz_split = torch.empty((0, 3), device=device)
            new_normal_split = torch.empty((0, 3), device=device)
            new_features_dc_split = torch.empty((0, *self._features_dc.shape[1:]), device=device)
            new_features_rest_split = torch.empty((0, *self._features_rest.shape[1:]), device=device)
            new_opacity_split = torch.empty((0, *self._opacity.shape[1:]), device=device)
            new_diags_split = torch.empty((0, self.diags.shape[1]), device=device)
            new_l_triangs_split = torch.empty((0, self.l_triangs.shape[1]), device=device)

        # Handle Clone Points
        if num_clone > 0:
            stds_clone = scale[selected_pts_mask_clone]
            means_clone = torch.zeros((num_clone, 3), device=device)
            samples_clone = torch.cat([
                means_clone,
                means_clone + stds_clone,
                means_clone - stds_clone
            ], dim=0)  # Shape: (num_clone * 3, 3)

            rots_clone = rotation[selected_pts_mask_clone].repeat(N, 1, 1)  # Shape: (num_clone * N, 3, 3)

            new_xyz_clone = torch.bmm(rots_clone, samples_clone.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask_clone].repeat(N, 1)
            new_normal_clone = self._normal[selected_pts_mask_clone].repeat(N, 1)
            new_features_dc_clone = self._features_dc[selected_pts_mask_clone].repeat(N, 1, 1)
            new_features_rest_clone = self._features_rest[selected_pts_mask_clone].repeat(N, 1, 1)
            new_opacity_clone = self._opacity[selected_pts_mask_clone].repeat(N, 1)
            new_diags_clone = self.diags[selected_pts_mask_clone].repeat(N, 1)
            new_l_triangs_clone = self.l_triangs[selected_pts_mask_clone].repeat(N, 1)
        else:
            new_xyz_clone = torch.empty((0, 3), device=device)
            new_normal_clone = torch.empty((0, 3), device=device)
            new_features_dc_clone = torch.empty((0, *self._features_dc.shape[1:]), device=device)
            new_features_rest_clone = torch.empty((0, *self._features_rest.shape[1:]), device=device)
            new_opacity_clone = torch.empty((0, *self._opacity.shape[1:]), device=device)
            new_diags_clone = torch.empty((0, self.diags.shape[1]), device=device)
            new_l_triangs_clone = torch.empty((0, self.l_triangs.shape[1]), device=device)

        # Concatenate Split and Clone Results
        new_xyz = torch.cat([new_xyz_split, new_xyz_clone], dim=0)
        new_normal = torch.cat([new_normal_split, new_normal_clone], dim=0)
        new_features_dc = torch.cat([new_features_dc_split, new_features_dc_clone], dim=0)
        new_features_rest = torch.cat([new_features_rest_split, new_features_rest_clone], dim=0)
        new_opacity = torch.cat([new_opacity_split, new_opacity_clone], dim=0)
        new_diags = torch.cat([new_diags_split, new_diags_clone], dim=0)
        new_l_triangs = torch.cat([new_l_triangs_split, new_l_triangs_clone], dim=0)

        # Post-processing
        self.densification_postfix(
            new_xyz, 
            new_features_dc, 
            new_features_rest, 
            new_normal, 
            new_opacity, 
            new_diags, 
            new_l_triangs
        )

        # Construct Prune Filter
        # Only the original selected points are to be pruned; the newly added points should remain
        prune_filter = torch.cat((
            selected_pts_mask, 
            torch.zeros(new_xyz.shape[0], device=device, dtype=torch.bool)
        ))
        self.prune_points(prune_filter)
    
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

        new_diags = self.diags[selected_pts_mask]
        split_diags = self.diags_act_inv(self.diags_act(self.diags[selected_pts_mask_split]) * 0.8)
        new_diags[selected_pts_mask_split[selected_pts_mask]] = split_diags
        new_diags = new_diags.repeat(N, 1)    

        new_l_triangs = self.l_triangs[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal,
                                new_opacity, new_diags, new_l_triangs)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=torch.bool)))
        self.prune_points(prune_filter)    
    
    def densify_and_clone_split_v2(self, grads, grad_threshold, scene_extent, rotation, scale, N=3):
        device = self.get_xyz.device
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Split and Clone Masks
        # split_threshold = self.percent_dense * scene_extent
        # max_scale = torch.max(scale, dim=1).values
        # selected_pts_mask_split = torch.logical_and(selected_pts_mask, max_scale > split_threshold)
        # selected_pts_mask_clone = torch.logical_and(selected_pts_mask, max_scale <= split_threshold)

        # Common operations for both split and clone
        stds = scale[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.cat([means, means + stds, means - stds], dim=0)
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # Merging new_diags_split and new_diags_clone
        new_diags = self.diags[selected_pts_mask].repeat(N, 1)        
        # split_diags = self.diags_act_inv(self.diags_act(self.diags[selected_pts_mask_split]) * 2/3)
        # new_diags[selected_pts_mask_split[selected_pts_mask]] = split_diags
        # new_diags = new_diags.repeat(N, 1)

        new_l_triangs = self.l_triangs[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal,
                                new_opacity, new_diags, new_l_triangs)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=torch.bool)))
        self.prune_points(prune_filter)
        
    def densify_and_split_6d(self, grads, grad_threshold, scene_extent, rotation, scale, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale[:, :3], dim=1).values > self.percent_dense*scene_extent)

        stds = scale[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), self.gs_dim),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = rotation[selected_pts_mask].repeat(N,1,1) # build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz_normal = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz_normal[selected_pts_mask].repeat(N, 1)
        new_xyz = new_xyz_normal[:, :3]
        new_normal = new_xyz_normal[:, 3:6]
        # new_normal = self._normal[selected_pts_mask].repeat(N,1)
        
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_diags = self.diags_act_inv(self.diags_act(self.diags[selected_pts_mask]) * 0.8).repeat(N, 1)
        # new_diags = self.diags[selected_pts_mask].repeat(N, 1)
        new_l_triangs = self.l_triangs[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, \
                                   new_opacity, new_diags, new_l_triangs)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

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

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, \
                                   new_opacity, new_diags, new_l_triangs)

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

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, new_opacities, new_diags, new_l_triangs)

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

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", use_tcgs=False, is_test=False, scaling_modifier=1.0):
        """
        Render using 6DGS conditional slicing with diff-gaussian-rasterization.
        This encapsulates the 6DGS-specific rendering logic.
        """
        from utils.ndgs_utils import create_cholesky_v2, slice_gaussian, slice_gaussian_test
        import math

        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Compute view direction for conditional slicing
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_normal.shape[0], 1))
        cond_params = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        lambda_opc = 0.35

        if is_test:
            # Test mode: use precomputed values
            m_cond, pdf_cond = slice_gaussian_test(
                self.get_xyz, self.direction, cond_params,
                self.v_22_inv, self.v_regr, lambda_opc=lambda_opc
            )
            shs = self.shs
            cov3D_precomp = self.cov3D_precomp
        else:
            # Training mode: compute conditional slicing
            direction = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)
            shs = self.get_features

            v = create_cholesky_v2(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))
            m_cond, cov3D_precomp, pdf_cond = slice_gaussian(
                self.get_xyz, direction, cond_params, v, c_dim=3, lambda_opc=lambda_opc
            )

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