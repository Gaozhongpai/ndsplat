#
# Residual Gabor extension of the Full DGS model (dgs base + additive Gabor band).
#
# Design (mirrors the Gabor Fields paper: low-pass Gaussian base + residual
# Gabor kernels):
#   - The dGS base (scene/gaussian_model_dgs.py) is the Gaussian base. Its
#     parameters, save/load and forward behaviour are UNCHANGED.
#   - On top of the base each Gaussian carries a residual Gabor band: a screen
#     space cosine modulation of its Gaussian footprint,
#         weight_gabor = weight * (1 + amp * cos(omega . d_screen + phase))
#     where d_screen = (splat_center_px - pixel) is the same offset the base
#     footprint uses, omega is a per-Gaussian screen-space frequency (carried as
#     a 3-vector _gabor_omega, its first two components used as the screen
#     frequency), and phase is _gabor_phase.
#   - amp = _gabor_amp (raw, no activation). It is ZERO-initialised, so a fresh
#     model renders BYTE-IDENTICALLY to plain dGS (the modulation factor is 1),
#     and the CUDA forward is gated so a null gabor buffer is the exact dGS path.
#
# Exact vs approximate: the forward Gabor factor is exact for the *unclipped*
# footprint. The half-space clip of a Gabor atom is the complex-error-function
# (Faddeeva) generalisation of the real-erf clipPhi; here we reuse the base's
# real-erf clip on the Gaussian envelope and leave the cosine unclipped. That
# is APPROXIMATE when a clip plane crosses a high-frequency atom. For the heart
# fit no clip plane is active, so the residual is exact there.

import torch
import numpy as np
from torch import nn

from scene.gaussian_model_dgs import GaussianModel as DGSGaussianModel


class GaussianModel(DGSGaussianModel):
    """dGS base + additive residual Gabor band.

    Adds three per-Gaussian tensors on top of the dGS parameter set:
      _gabor_omega  [N, 3]  screen/world frequency (first 2 comps used on screen)
      _gabor_phase  [N, 1]  phase
      _gabor_amp    [N, 1]  residual amplitude (0 => no residual => == dGS)
    All are zero-initialised so a fresh model is identical to plain dGS.
    """

    # Names of the extra Gabor optimizer groups / PLY attribute prefixes, so the
    # densification / save / load / prune plumbing can iterate them generically.
    _GABOR_SPECS = (
        ("_gabor_omega", "gabor_omega", 3),
        ("_gabor_phase", "gabor_phase", 1),
        ("_gabor_amp", "gabor_amp", 1),
    )

    def __init__(self, *args, gabor_omega_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gabor_omega_init = gabor_omega_init
        self._gabor_omega = torch.empty(0)
        self._gabor_phase = torch.empty(0)
        self._gabor_amp = torch.empty(0)
        # Enabled by default; can be forced off to recover the exact dGS path.
        self.use_gabor = True

    # ---- accessors -------------------------------------------------------
    @property
    def get_gabor_omega(self):
        return self._gabor_omega

    @property
    def get_gabor_phase(self):
        return self._gabor_phase

    @property
    def get_gabor_amp(self):
        return self._gabor_amp

    def _init_gabor_params(self, num_gaussians, device="cuda"):
        """Zero-init the residual Gabor band (== dGS at init)."""
        omega = torch.full((num_gaussians, 3), float(self.gabor_omega_init), device=device)
        phase = torch.zeros((num_gaussians, 1), device=device)
        amp = torch.zeros((num_gaussians, 1), device=device)
        self._gabor_omega = nn.Parameter(omega.requires_grad_(True))
        self._gabor_phase = nn.Parameter(phase.requires_grad_(True))
        self._gabor_amp = nn.Parameter(amp.requires_grad_(True))

    # ---- creation --------------------------------------------------------
    def create_from_pcd(self, pcd, spatial_lr_scale, mcmc_cap_max=None, densification_strategy="standard"):
        super().create_from_pcd(pcd, spatial_lr_scale, mcmc_cap_max, densification_strategy)
        self._init_gabor_params(self.get_xyz.shape[0], device=self.get_xyz.device)

    # ---- optimizer -------------------------------------------------------
    def training_setup(self, training_args):
        super().training_setup(training_args)
        # If the Gabor params were never created (e.g. before load_ply/create),
        # create them zero so the optimizer has something to hold.
        if self._gabor_amp.numel() == 0:
            self._init_gabor_params(self.get_xyz.shape[0], device=self.get_xyz.device)
        # Learning rates: reuse rotation_lr for omega/phase, feature_lr for amp.
        lr_omega = getattr(training_args, "gabor_omega_lr", training_args.rotation_lr)
        lr_phase = getattr(training_args, "gabor_phase_lr", training_args.rotation_lr)
        lr_amp = getattr(training_args, "gabor_amp_lr", training_args.feature_lr)
        self.optimizer.add_param_group({'params': [self._gabor_omega], 'lr': lr_omega, "name": "gabor_omega"})
        self.optimizer.add_param_group({'params': [self._gabor_phase], 'lr': lr_phase, "name": "gabor_phase"})
        self.optimizer.add_param_group({'params': [self._gabor_amp], 'lr': lr_amp, "name": "gabor_amp"})

    # ---- capture/restore -------------------------------------------------
    def capture(self):
        return super().capture() + (self._gabor_omega, self._gabor_phase, self._gabor_amp)

    def restore(self, model_args, training_args):
        # The last 3 entries are the Gabor tensors we appended in capture().
        gabor = model_args[-3:]
        base_args = model_args[:-3]
        # Set the Gabor tensors first so training_setup (called inside the base
        # restore) can register their optimizer groups.
        self._gabor_omega, self._gabor_phase, self._gabor_amp = (
            nn.Parameter(t.requires_grad_(True)) for t in gabor
        )
        super().restore(base_args, training_args)

    # ---- PLY attributes --------------------------------------------------
    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        # Insert the gabor columns right before the mip attributes, which the
        # base appends last. Simpler: append at the very end and make save_ply
        # append the corresponding column arrays in the same order.
        gabor_cols = []
        for _, prefix, dim in self._GABOR_SPECS:
            for i in range(dim):
                gabor_cols.append(f"{prefix}_{i}")
        # Base already appended mip columns; keep gabor after everything so the
        # header order == save_ply's attrs_list order (see save_ply).
        return l + gabor_cols

    def save_ply(self, path):
        # Reuse the base save_ply but append the gabor columns. The base builds
        # attrs_list in construct_list_of_attributes order and writes; since we
        # extended construct_list_of_attributes we must extend the data too. The
        # cleanest robust path: replicate the base write with the extra columns.
        import os
        from utils.system_utils import mkdir_p
        from plyfile import PlyData, PlyElement

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
        attrs_list = [xyz, f_dc, f_rest, opacities, scale, rotation, mean_view, mean_time, L_22_inv]
        if self.use_view_dependent_pos:
            attrs_list.append(self._v_12_direction.detach().cpu().numpy())
            attrs_list.append(self._lambda_view.detach().cpu().numpy()[:, np.newaxis])
            if self.input_dim == 7:
                attrs_list.append(self._lambda_time.detach().cpu().numpy()[:, np.newaxis])
        if self._label.numel() == xyz.shape[0]:
            label = self._label.detach().to(torch.float32).cpu().numpy()
            if label.ndim == 1:
                label = label[:, np.newaxis]
            attrs_list.append(label)
        attrs_list.extend(self.mip_ply_columns())
        # Gabor columns (order matches construct_list_of_attributes tail)
        attrs_list.append(self._gabor_omega.detach().cpu().numpy())
        attrs_list.append(self._gabor_phase.detach().cpu().numpy())
        attrs_list.append(self._gabor_amp.detach().cpu().numpy())

        attributes = np.concatenate(attrs_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        from plyfile import PlyData
        super().load_ply(path)
        plydata = PlyData.read(path)
        names = {p.name for p in plydata.elements[0].properties}
        n = self._xyz.shape[0]
        device = self._xyz.device

        def _read(prefix, dim, fill):
            cols = [p for p in names if p.startswith(prefix + "_")]
            if len(cols) >= dim:
                cols = sorted([c for c in cols], key=lambda x: int(x.split('_')[-1]))[:dim]
                arr = np.zeros((n, dim), dtype=np.float32)
                for i, c in enumerate(cols):
                    arr[:, i] = np.asarray(plydata.elements[0][c])
                return arr
            return np.full((n, dim), fill, dtype=np.float32)

        omega = _read("gabor_omega", 3, self.gabor_omega_init)
        phase = _read("gabor_phase", 1, 0.0)
        amp = _read("gabor_amp", 1, 0.0)  # absent (plain dGS ckpt) => zero => == dGS
        self._gabor_omega = nn.Parameter(torch.tensor(omega, dtype=torch.float, device=device).requires_grad_(True))
        self._gabor_phase = nn.Parameter(torch.tensor(phase, dtype=torch.float, device=device).requires_grad_(True))
        self._gabor_amp = nn.Parameter(torch.tensor(amp, dtype=torch.float, device=device).requires_grad_(True))

    # ---- densification plumbing -----------------------------------------
    def _prune_optimizer(self, mask):
        optimizable_tensors = super()._prune_optimizer(mask)
        for attr, name, _ in self._GABOR_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def prune_points(self, mask):
        super().prune_points(mask)
        # base.prune_points calls _prune_optimizer (overridden) which already
        # reassigned the gabor tensors, so nothing extra needed here.

    def _gabor_slice(self, mask_or_idx):
        return (
            self._gabor_omega[mask_or_idx],
            self._gabor_phase[mask_or_idx],
            self._gabor_amp[mask_or_idx],
        )

    def densification_postfix(self, *args, **kwargs):
        # Base builds the cat dict from its own param names. We need to feed the
        # gabor extension tensors too. The base signature does not know about
        # gabor, so we pop them from kwargs and add after.
        new_gabor = kwargs.pop("new_gabor", None)
        # Temporarily stash so cat_tensors_to_optimizer picks them up.
        self._pending_new_gabor = new_gabor
        super().densification_postfix(*args, **kwargs)
        self._pending_new_gabor = None

    def cat_tensors_to_optimizer(self, tensors_dict):
        # Inject the gabor extension tensors into the dict the base assembled.
        new_gabor = getattr(self, "_pending_new_gabor", None)
        if new_gabor is not None:
            omega, phase, amp = new_gabor
            tensors_dict = dict(tensors_dict)
            tensors_dict["gabor_omega"] = omega
            tensors_dict["gabor_phase"] = phase
            tensors_dict["gabor_amp"] = amp
        optimizable_tensors = super().cat_tensors_to_optimizer(tensors_dict)
        for attr, name, _ in self._GABOR_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # Recompute the same selection mask the base uses, gather new gabor rows,
        # and route them through densification_postfix via _pending_new_gabor.
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        new_gabor = (
            self._gabor_omega[selected_pts_mask].repeat(N, 1),
            self._gabor_phase[selected_pts_mask].repeat(N, 1),
            self._gabor_amp[selected_pts_mask].repeat(N, 1),
        )
        self._pending_new_gabor = new_gabor
        super().densify_and_split(grads, grad_threshold, scene_extent, N)
        self._pending_new_gabor = None

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        ).to(self._xyz.device)
        new_gabor = (
            self._gabor_omega[selected_pts_mask],
            self._gabor_phase[selected_pts_mask],
            self._gabor_amp[selected_pts_mask],
        )
        self._pending_new_gabor = new_gabor
        super().densify_and_clone(grads, grad_threshold, scene_extent)
        self._pending_new_gabor = None

    # ---- MCMC plumbing ---------------------------------------------------
    def _update_params(self, idxs, ratio):
        result = super()._update_params(idxs, ratio)
        result["gabor_omega"] = self._gabor_omega[idxs]
        result["gabor_phase"] = self._gabor_phase[idxs]
        result["gabor_amp"] = self._gabor_amp[idxs]
        return result

    def relocate_gs(self, dead_mask=None):
        if dead_mask is None or dead_mask.sum() == 0:
            return
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        super().relocate_gs(dead_mask)
        # relocate_gs samples reinit_idx internally; re-sample not exposed. The
        # base already index_copy_'d the shared params. For gabor we mirror by
        # copying from the same alive distribution is not directly available, so
        # we leave gabor residual at the dead slots reset to zero (safe: zero
        # residual == pure dGS for those, they will re-learn).
        with torch.no_grad():
            self._gabor_omega[dead_indices] = float(self.gabor_omega_init)
            self._gabor_phase[dead_indices] = 0.0
            self._gabor_amp[dead_indices] = 0.0

    def replace_tensors_to_optimizer(self, inds=None):
        optimizable_tensors = super().replace_tensors_to_optimizer(inds)
        for attr, name, _ in self._GABOR_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def add_new_gs(self, cap_max):
        # add_new_gs builds params via _update_params (overridden) and passes to
        # densification_postfix positionally; route gabor via _pending_new_gabor.
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        params = self._update_params(add_idx, ratio=ratio)
        self._opacity[add_idx] = params['opacity']
        self._pending_new_gabor = (params['gabor_omega'], params['gabor_phase'], params['gabor_amp'])
        self.densification_postfix(
            params['xyz'], params['features_dc'], params['features_rest'], params['opacity'],
            params['scaling'], params['rotation'], params['mean_view'], params['mean_time'],
            params['L_22_inv'], params.get('v_12_direction'),
            params.get('lambda_view'), params.get('lambda_time'),
        )
        self._pending_new_gabor = None
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    # ---- rendering -------------------------------------------------------
    def _gabor_tensors_for_raster(self):
        """Pack the gabor band into the [N, 5] tensor the CUDA forward expects:
        columns = (omega_x_screen, omega_y_screen, phase, amp, unused).
        Returns None when the residual is globally disabled so the CUDA path is
        byte-identical to dGS.
        """
        if not getattr(self, "use_gabor", True):
            return None
        if self._gabor_amp.numel() == 0:
            return None
        omega = self._gabor_omega
        # Use the first two components as the screen-space frequency (per-pixel
        # d is in screen/pixel units). This keeps the residual analytic and
        # cheap; a full world->screen frequency projection is left as future
        # work (noted approximate).
        gabor = torch.cat([
            omega[:, 0:2],
            self._gabor_phase,
            self._gabor_amp,
        ], dim=1).contiguous()
        return gabor

    def render_tcgs(self, viewpoint_camera, render_mode="RGB", scaling_modifier=1.0,
                    use_tcgs=False, tight_snugbox=False, compact_box_mult=1.0):
        # Reuse the base render_tcgs but inject the gabor buffer. The base builds
        # the raster settings + call itself; rather than duplicate ~100 lines we
        # temporarily monkey-set an attribute the base does not read, so instead
        # we replicate the minimal call by delegating to base and relying on the
        # base already forwarding a `gabor` kwarg. Since the base does NOT know
        # about gabor, we override the rasterizer call here fully.
        import math
        from tcgs_speedy_rasterizer import (
            GaussianRasterizationSettings as TCGSRasterizationSettings,
            GaussianRasterizer as TCGSRasterizer,
        )

        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1))
        mean_view = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        if self.input_dim == 7:
            timestamp = torch.full((mean_view.shape[0], 1),
                                   viewpoint_camera.timestamp if hasattr(viewpoint_camera, 'timestamp') else 0.0,
                                   device=mean_view.device, dtype=mean_view.dtype)
            cond_params = torch.cat([mean_view, timestamp], dim=-1)
        else:
            cond_params = mean_view

        m_cond, opacity_scale = self.slice_gaussian_full_method(cond_params)
        shs = self.get_features
        opacity = self.get_opacity * opacity_scale
        scales, opacity, antialiasing = self.mip_filtered(opacity, scales=self.get_scaling)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        bg_color = self.background if hasattr(self, 'background') and self.background.numel() > 0 else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        x_threshold = viewpoint_camera.x_threshold if hasattr(viewpoint_camera, 'x_threshold') and viewpoint_camera.x_threshold is not None else float('inf')
        clip_plane = getattr(viewpoint_camera, 'clip_plane', None)
        clip_plane_tensor = (torch.tensor(clip_plane, dtype=torch.float32, device="cuda") if clip_plane is not None else None)

        clip_operator = getattr(self, "clip_operator", "analytic")
        analytic_clip = (clip_operator != "moment")
        if clip_operator == "hardcull" and clip_plane_tensor is not None:
            from tcgs_speedy_rasterizer import hard_clip_mask
            keep = hard_clip_mask(m_cond, clip_plane=clip_plane_tensor)
            opacity = opacity * keep.to(opacity.dtype).view(-1, 1)
            clip_plane_tensor = None

        gabor = self._gabor_tensors_for_raster()

        raster_settings = TCGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            x_threshold=x_threshold, clip_plane=clip_plane_tensor,
            analytic_clip=analytic_clip, prefiltered=False,
            use_tcgs=use_tcgs, tight_snugbox=tight_snugbox,
            compact_box_mult=compact_box_mult, debug=False,
            antialiasing=antialiasing,
        )
        rasterizer = TCGSRasterizer(raster_settings=raster_settings)
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=m_cond, means2D=screenspace_points, shs=shs,
            colors_precomp=None, opacities=opacity, scores=None,
            scales=scales, rotations=self.get_rotation,
            cov3D_precomp=None, betas=None, gabor=gabor,
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
