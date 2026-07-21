#
# NASG-Gabor color model on the Full DGS base (`dgs-nasg`): REPLACES the SH
# view-dependent color with a few Normalized Anisotropic Spherical Gabor lobes,
# per "Beyond Spherical Harmonics: Rethinking Appearance Models for Radiance
# Reconstruction" (Miazga, Condor, Didyk; reference implementation
# github.com/ewaMiazga/NASGabor).
#
# Color of a primitive for view direction v (their spherical_nasg_gabor.cuh,
# ported verbatim to torch):
#     C(v) = c0 + sum_j pdf_j(v) * w_j,           w_j = (r,g,b) per lobe
#     pdf   = e^{2 lam (E*Kb - 1)} * E * (1 + cos(k*vx))/2 * inv_norm
#     Kb    = (vz+1)/2,  E = Kb^{eps + a*vx^2/(1-vz^2)}
#     inv_norm = lam*sqrt(1+a) / (2*pi*(1+eps_n-e^{-2 lam}))
# with per-lobe params {cos_t, cos_p, cos_u (orientation frame), lam_raw,
# a_raw (exp-activated, clamped 1e4), k_raw (k=(tanh+1)*20), r, g, b}:
# 3 + 9L color scalars per primitive vs 48 for SH degree 3.
#
# Integration choices:
#   - Colors are evaluated PER VIEW in Python (differentiable) and passed to
#     the rasterizer as colors_precomp — ZERO CUDA changes (the same pattern
#     as the projected-Gabor band study; see GABOR_HANDOFF.md on
#     feat/gabor-residual for the plumbing lessons reused here).
#   - The base's _features_dc doubles as c0 storage via the SH DC convention
#     c0 = SH_C0 * f_dc + 0.5, so with 0 active lobes the render is EXACTLY an
#     SH-degree-0 render of the same checkpoint (parity gate), and
#     warm-starting from an SH checkpoint needs no conversion. _features_rest
#     is loaded (checkpoint compat) but permanently FROZEN and never evaluated.
#   - Progressive lobe activation as upstream: active_lobes starts at 0 and
#     steps up where SH degree would (train.py's oneupSHdegree every 1000 it).
#     DEVIATION from upstream: lobe weights are ZERO-init (upstream: 0.5) so a
#     newly activated lobe changes nothing and bootstraps from its gradient —
#     the staged-activation analog of the Gabor band's amp bootstrap.
#

import math

import numpy as np
import torch
from torch import nn

from scene.gaussian_model_dgs import GaussianModel as DGSGaussianModel

SH_C0 = 0.28209479177387814
TWO_PI = 6.283185307179586


def eval_nasg_gabor(c0, pos, shape, weight, dirs, active_lobes):
    """Torch port of nasg_gabor_fwd (spherical_nasg_gabor.cuh).

    c0 [N,3] base color; pos/shape/weight [N,L,3] lobe params
    (pos = cos_t, cos_p, cos_u; shape = lam_raw, a_raw, k_raw; weight = rgb);
    dirs [N,3] view directions (normalized inside); returns colors [N,3],
    clamped >= 0 like the SH path.
    """
    colors = c0
    if active_lobes > 0:
        v = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        P = pos[:, :active_lobes]                       # [N,l,3]
        ct = P[..., 0].clamp(-0.999999, 0.999999)
        cp = P[..., 1].clamp(-0.999999, 0.999999)
        cu = P[..., 2].clamp(-0.999999, 0.999999)
        st = torch.sqrt(1.0 - ct * ct)
        sp = torch.sqrt(1.0 - cp * cp)
        su = torch.sqrt(1.0 - cu * cu)
        # frame axes (their unusual parameterization, kept verbatim)
        x0 = ct * cp * cu - st * su
        x1 = st * cp * cu + ct * su
        x2 = -sp * cu
        z0 = ct * sp
        z1 = st * sp
        z2 = cp

        S = shape[:, :active_lobes]
        lam = torch.exp(S[..., 0]).clamp_max(1e4)
        a = torch.exp(S[..., 1]).clamp_max(1e4)
        k = (torch.tanh(S[..., 2]) + 1.0) * 20.0

        vx = v[:, None, 0] * x0 + v[:, None, 1] * x1 + v[:, None, 2] * x2
        vz = v[:, None, 0] * z0 + v[:, None, 1] * z1 + v[:, None, 2] * z2

        mask_one = vz >= 1.0 - 1e-7
        mask_zero = vz <= -1.0 + 1e-7
        valid = ~(mask_one | mask_zero)
        vz_s = torch.where(valid, vz, torch.zeros_like(vz))

        K_base = (vz_s + 1.0) * 0.5
        K_exp = 5e-6 + a * vx * vx / (1.0 - vz_s * vz_s)
        exp_val = torch.pow(K_base, K_exp)
        inv_norm = lam * torch.sqrt(1.0 + a) \
            / (TWO_PI * (1.0 + 1e-8 - torch.exp(-2.0 * lam)))
        gabor_term = (1.0 + torch.cos(k * vx)) * 0.5
        pdf = torch.exp(2.0 * lam * (exp_val * K_base - 1.0)) \
            * exp_val * gabor_term * inv_norm
        pdf = torch.where(valid, pdf, torch.zeros_like(pdf))
        pdf = torch.where(mask_one, torch.ones_like(pdf), pdf)

        colors = colors + (pdf.unsqueeze(-1) * weight[:, :active_lobes]).sum(dim=1)
    return colors.clamp_min(0.0)


class GaussianModel(DGSGaussianModel):
    """dGS base with NASG-Gabor color replacing the SH evaluation.

    Extra per-Gaussian tensors (L = lobe_number):
      _nasg_pos    [N, L*3]  cos_t, cos_p, cos_u per lobe
      _nasg_shape  [N, L*3]  lam_raw, a_raw, k_raw per lobe
      _nasg_weight [N, L*3]  lobe RGB (zero-init => pure c0 render)
    _features_dc stores c0 (SH DC convention); _features_rest is frozen.
    """

    _NASG_SPECS = (
        ("_nasg_pos", "nasg_pos"),
        ("_nasg_shape", "nasg_shape"),
        ("_nasg_weight", "nasg_weight"),
    )

    def __init__(self, *args, lobe_number: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lobe_number = int(lobe_number)
        self.active_lobes = 0
        self._nasg_pos = torch.empty(0)
        self._nasg_shape = torch.empty(0)
        self._nasg_weight = torch.empty(0)

    # ---- lobe params -------------------------------------------------------
    def _init_nasg_params(self, n, device="cuda"):
        """Upstream init: frame cosines 0, lam_raw = a_raw = log(0.5),
        k_raw = -1.6 (k ~ 1.56); weights ZERO (see header)."""
        L = self.lobe_number
        pos = torch.zeros((n, L * 3), device=device)
        shape = torch.zeros((n, L * 3), device=device)
        shape[:, 0::3] = math.log(0.5)
        shape[:, 1::3] = math.log(0.5)
        shape[:, 2::3] = -1.6
        weight = torch.zeros((n, L * 3), device=device)
        self._nasg_pos = nn.Parameter(pos.requires_grad_(True))
        self._nasg_shape = nn.Parameter(shape.requires_grad_(True))
        self._nasg_weight = nn.Parameter(weight.requires_grad_(True))

    def oneupSHdegree(self):
        # Progressive lobe activation in place of SH degree stepping.
        if self.active_lobes < self.lobe_number:
            self.active_lobes += 1

    def eval_colors(self, dirs):
        """NASG-Gabor color per primitive for the given view directions."""
        c0 = SH_C0 * self._features_dc.reshape(-1, 3) + 0.5
        n = c0.shape[0]
        return eval_nasg_gabor(
            c0,
            self._nasg_pos.reshape(n, self.lobe_number, 3),
            self._nasg_shape.reshape(n, self.lobe_number, 3),
            self._nasg_weight.reshape(n, self.lobe_number, 3),
            dirs, self.active_lobes)

    # ---- creation / optimizer ----------------------------------------------
    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._init_nasg_params(self.get_xyz.shape[0], device=self.get_xyz.device)

    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self._nasg_weight.numel() == 0:
            self._init_nasg_params(self.get_xyz.shape[0], device=self.get_xyz.device)
        # Upstream LRs: base color 2.5e-4, lobe params 2.5e-3.
        lr_lobe = getattr(training_args, "nasg_features_lr", 0.0025)
        lr_c0 = getattr(training_args, "nasg_c0_lr", 0.00025)
        for group in self.optimizer.param_groups:
            if group.get("name") == "f_dc":
                group["lr"] = lr_c0
            elif group.get("name") == "f_rest":
                # SH rest coefficients are dead weight kept only for
                # checkpoint compatibility; never trained, never evaluated.
                group["lr"] = 0.0
                for p in group["params"]:
                    p.requires_grad_(False)
        for attr, name in self._NASG_SPECS:
            self.optimizer.add_param_group(
                {"params": [getattr(self, attr)], "lr": lr_lobe, "name": name})
        # Color-only fit: freeze the entire geometry/opacity/conditioning base
        # so ONLY c0 + the lobes train (pair with --densify_until_iter 0).
        # Staged fitting was load-bearing in the Gabor band study.
        self._nasg_color_only = bool(getattr(training_args, "nasg_color_only", False))
        if self._nasg_color_only:
            color_groups = {"f_dc", "nasg_pos", "nasg_shape", "nasg_weight"}
            for group in self.optimizer.param_groups:
                if group.get("name") not in color_groups:
                    group["lr"] = 0.0
                    for p in group["params"]:
                        p.requires_grad_(False)

    def update_learning_rate(self, iteration):
        # The base scheduler would re-raise the frozen xyz LR every iteration.
        if getattr(self, "_nasg_color_only", False):
            return 0.0
        return super().update_learning_rate(iteration)

    # ---- capture/restore ----------------------------------------------------
    def capture(self):
        return super().capture() + (self._nasg_pos, self._nasg_shape, self._nasg_weight)

    def restore(self, model_args, training_args):
        nasg = model_args[-3:]
        self._nasg_pos, self._nasg_shape, self._nasg_weight = (
            nn.Parameter(t.requires_grad_(True)) for t in nasg)
        super().restore(model_args[:-3], training_args)

    # ---- PLY ----------------------------------------------------------------
    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        for _, prefix in self._NASG_SPECS:
            for i in range(self.lobe_number * 3):
                l.append(f"{prefix}_{i}")
        return l

    def save_ply(self, path):
        # Replicate the base write with the NASG columns appended (same
        # approach as the gabor band model; the base builds a fixed attrs
        # list). f_rest columns are saved but zero-information (frozen).
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
        attrs_list.append(self._nasg_pos.detach().cpu().numpy())
        attrs_list.append(self._nasg_shape.detach().cpu().numpy())
        attrs_list.append(self._nasg_weight.detach().cpu().numpy())

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

        def _read(prefix, dim):
            cols = sorted((p for p in names if p.startswith(prefix + "_")),
                          key=lambda x: int(x.split('_')[-1]))[:dim]
            arr = np.zeros((n, dim), dtype=np.float32)
            for i, c in enumerate(cols):
                arr[:, i] = np.asarray(plydata.elements[0][c])
            return torch.tensor(arr, dtype=torch.float, device=device)

        if any(p.startswith("nasg_pos_") for p in names):
            # Resuming a trained NASG model: keep its lobes, all active.
            L3 = self.lobe_number * 3
            self._nasg_pos = nn.Parameter(_read("nasg_pos", L3).requires_grad_(True))
            self._nasg_shape = nn.Parameter(_read("nasg_shape", L3).requires_grad_(True))
            self._nasg_weight = nn.Parameter(_read("nasg_weight", L3).requires_grad_(True))
            self.active_lobes = self.lobe_number
        else:
            # Warm-starting an SH checkpoint: c0 comes free via f_dc; lobes
            # fresh and inactive (activated by the oneupSHdegree schedule).
            self._init_nasg_params(n, device=device)
            self.active_lobes = 0

    # ---- densification plumbing ---------------------------------------------
    # Same battle-tested pattern as the gabor models: growth ops stash the new
    # rows in _pending_new_nasg BEFORE the base dispatches back into
    # densification_postfix; only touch the stash when the kwarg is present.
    def _prune_optimizer(self, mask):
        optimizable_tensors = super()._prune_optimizer(mask)
        for attr, name in self._NASG_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        pending = getattr(self, "_pending_new_nasg", None)
        if pending is not None:
            tensors_dict = dict(tensors_dict)
            for (attr, name), t in zip(self._NASG_SPECS, pending):
                tensors_dict[name] = t
        optimizable_tensors = super().cat_tensors_to_optimizer(tensors_dict)
        for attr, name in self._NASG_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def densification_postfix(self, *args, **kwargs):
        if "new_nasg" in kwargs:
            self._pending_new_nasg = kwargs.pop("new_nasg")
        super().densification_postfix(*args, **kwargs)
        self._pending_new_nasg = None

    def _nasg_rows(self, mask_or_idx, repeat_n=1):
        rows = tuple(getattr(self, attr)[mask_or_idx] for attr, _ in self._NASG_SPECS)
        if repeat_n > 1:
            rows = tuple(r.repeat(repeat_n, 1) for r in rows)
        return rows

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = padded_grad >= grad_threshold
        selected = torch.logical_and(
            selected,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        self._pending_new_nasg = self._nasg_rows(selected, repeat_n=N)
        super().densify_and_split(grads, grad_threshold, scene_extent, N)
        self._pending_new_nasg = None

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = torch.logical_and(
            selected,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        ).to(self._xyz.device)
        self._pending_new_nasg = self._nasg_rows(selected)
        super().densify_and_clone(grads, grad_threshold, scene_extent)
        self._pending_new_nasg = None

    # ---- MCMC plumbing -------------------------------------------------------
    def _update_params(self, idxs, ratio):
        result = super()._update_params(idxs, ratio)
        for attr, name in self._NASG_SPECS:
            result[name] = getattr(self, attr)[idxs]
        return result

    def relocate_gs(self, dead_mask=None):
        if dead_mask is None or dead_mask.sum() == 0:
            return
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        super().relocate_gs(dead_mask)
        with torch.no_grad():
            n = dead_indices.shape[0]
            fresh_shape = torch.zeros((n, self.lobe_number * 3), device="cuda")
            fresh_shape[:, 0::3] = math.log(0.5)
            fresh_shape[:, 1::3] = math.log(0.5)
            fresh_shape[:, 2::3] = -1.6
            self._nasg_pos[dead_indices] = 0.0
            self._nasg_shape[dead_indices] = fresh_shape
            self._nasg_weight[dead_indices] = 0.0
            for group in self.optimizer.param_groups:
                if group.get("name") in {n for _, n in self._NASG_SPECS}:
                    state = self.optimizer.state.get(group["params"][0])
                    if state:
                        state["exp_avg"][dead_indices] = 0
                        state["exp_avg_sq"][dead_indices] = 0

    def replace_tensors_to_optimizer(self, inds=None):
        optimizable_tensors = super().replace_tensors_to_optimizer(inds)
        for attr, name in self._NASG_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        params = self._update_params(add_idx, ratio=ratio)
        self._opacity[add_idx] = params['opacity']
        self._pending_new_nasg = tuple(params[n] for _, n in self._NASG_SPECS)
        self.densification_postfix(
            params['xyz'], params['features_dc'], params['features_rest'], params['opacity'],
            params['scaling'], params['rotation'], params['mean_view'], params['mean_time'],
            params['L_22_inv'], params.get('v_12_direction'),
            params.get('lambda_view'), params.get('lambda_time'),
        )
        self._pending_new_nasg = None
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    # ---- rendering ------------------------------------------------------------
    def render_tcgs(self, viewpoint_camera, render_mode="RGB", scaling_modifier=1.0,
                    use_tcgs=False, tight_snugbox=False, compact_box_mult=1.0):
        # dGS render path with the SH evaluation replaced by per-view NASG
        # colors passed as colors_precomp.
        from tcgs_speedy_rasterizer import (
            GaussianRasterizationSettings as TCGSRasterizationSettings,
            GaussianRasterizer as TCGSRasterizer,
        )

        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype,
                                              requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self._xyz.shape[0], 1))
        view_dirs = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        if self.input_dim == 7:
            timestamp = torch.full((view_dirs.shape[0], 1),
                                   viewpoint_camera.timestamp if hasattr(viewpoint_camera, 'timestamp') else 0.0,
                                   device=view_dirs.device, dtype=view_dirs.dtype)
            cond_params = torch.cat([view_dirs, timestamp], dim=-1)
        else:
            cond_params = view_dirs

        m_cond, opacity_scale = self.slice_gaussian_full_method(cond_params)
        colors = self.eval_colors(view_dirs)
        opacity = self.get_opacity * opacity_scale
        scales, opacity, antialiasing = self.mip_filtered(opacity, scales=self.get_scaling)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        bg_color = self.background if hasattr(self, 'background') and self.background.numel() > 0 \
            else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        x_threshold = viewpoint_camera.x_threshold if hasattr(viewpoint_camera, 'x_threshold') \
            and viewpoint_camera.x_threshold is not None else float('inf')
        clip_plane = getattr(viewpoint_camera, 'clip_plane', None)
        clip_plane_tensor = (torch.tensor(clip_plane, dtype=torch.float32, device="cuda")
                             if clip_plane is not None else None)

        clip_operator = getattr(self, "clip_operator", "analytic")
        analytic_clip = (clip_operator != "moment")
        if clip_operator == "hardcull" and clip_plane_tensor is not None:
            from tcgs_speedy_rasterizer import hard_clip_mask
            keep = hard_clip_mask(m_cond, clip_plane=clip_plane_tensor)
            opacity = opacity * keep.to(opacity.dtype).view(-1, 1)
            clip_plane_tensor = None

        raster_settings = TCGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0,
            campos=viewpoint_camera.camera_center,
            x_threshold=x_threshold, clip_plane=clip_plane_tensor,
            analytic_clip=analytic_clip, prefiltered=False,
            use_tcgs=use_tcgs, tight_snugbox=tight_snugbox,
            compact_box_mult=compact_box_mult, debug=False,
            antialiasing=antialiasing,
        )
        rasterizer = TCGSRasterizer(raster_settings=raster_settings)
        rendered_image, radii, render_time, _ = rasterizer(
            means3D=m_cond, means2D=screenspace_points, shs=None,
            colors_precomp=colors, opacities=opacity, scores=None,
            scales=scales, rotations=self.get_rotation,
            cov3D_precomp=None, betas=None,
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
