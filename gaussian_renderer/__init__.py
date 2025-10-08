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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel
import scene
from utils.sh_utils import eval_sh
from utils.ndgs_utils import slice_gaussian_test  # Used in render_flash
from utils.ndgs_utils import Rasterizer  # Used in render_flash


def render_flash(viewpoint_camera, pc : GaussianModel, pipe, 
           bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, is_test = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    num_vertex = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Cull based on view direction
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_normal.shape[0], 1))
    cond_params = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    m_cond, pdf_cond = slice_gaussian_test(pc.get_xyz, pc.direction, cond_params, pc.v_22_inv, pc.v_regr)
    opacity = pc.get_opacity * pdf_cond
    
    rasterizer = Rasterizer(num_vertex, device)

    rendered_image = rasterizer.forward(m_cond, pc.shs, opacity, pc.cov3D_precomp, viewpoint_camera, bg_color)    
    return {"render": rendered_image, 
        "render_principle": None,
        "render_non_principle": None,
        "viewspace_points": screenspace_points,
        "visibility_filter" : None,
        "radii": None}


def render(viewpoint_camera, pc : GaussianModel, pipe, 
           bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, is_test = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        # x_threshold=float('inf'),
        use_tcgs=False,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if scene.MODE != "3dgs":
        override_color = pc.get_color(viewpoint_camera, is_test)
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            sh2rgb = sh2rgb.mean(dim=1, keepdim=True).repeat(1, 3)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if scene.MODE == "ddgs":
                shs = pc._features_dc[pc.num_principle:] @ pc._lambda[pc.num_fprinciple:].view(-1, 3)
                colors_precomp = pc._color
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image_principle = None
    rendered_image_non_principle = None
    
    is_render_principle = False ##### settings
    if "ddgs" in scene.MODE and is_render_principle:
        rendered_image_non_principle, radii_principle = rasterizer(
                means3D = means3D[pc.num_principle:],
                means2D = means2D[pc.num_principle:],
                shs = shs,
                colors_precomp = colors_precomp[pc.num_principle:],
                opacities = opacity[pc.num_principle:],
                scales = scales[pc.num_principle:],
                rotations = rotations[pc.num_principle:],
                cov3D_precomp = cov3D_precomp)

        rendered_image_principle, radii_principle = rasterizer(
                means3D = means3D[:pc.num_principle],
                means2D = means2D[:pc.num_principle],
                shs = shs,
                colors_precomp = colors_precomp[:pc.num_principle],
                opacities = opacity[:pc.num_principle],
                scales = scales[:pc.num_principle],
                rotations = rotations[:pc.num_principle],
                cov3D_precomp = cov3D_precomp)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # N = int(pc.num_principle) if is_cuda else pc.get_features.shape[0]
    rendered_image, radii = rasterizer(
        # N, # int(pc.num_principle), pc.get_features.shape[0]
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, 
            "render_principle": rendered_image_principle,
            "render_non_principle": rendered_image_non_principle,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
