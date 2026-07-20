#
# Residual Gabor extension of the dBS model (Beta-kernel base + additive Gabor
# band): `dbs-gabor`. The dBS base (scene/gaussian_model_dbs_sh.py) is
# UNCHANGED; each primitive additionally carries a residual Gabor band that
# modulates its Beta footprint:
#     weight_gabor = (1 - sigma)^beta * (1 + amp * cos(omega_2d . d + phase))
# The CUDA rasterizer already composes beta and gabor (verified by the float64
# "gabor + beta kernel" config in test_cutting_plane.py); this class provides
# the model-side plumbing that was previously dGS-only.
#
# The wave vector, init, bounds, tanh amp and per-view projection follow
# scene/gaussian_model_gabor.py (see project_gabor_band there for the math and
# its verification). Two dBS-specific adaptations:
#   - Whitening uses a Cholesky factor of get_covariance (the dBS spatial
#     covariance comes from the l-triangle construction, not quat/scale, and
#     get_covariance is EXACTLY the cov3D_precomp the renderer consumes).
#     With Sigma = L L^T: whitened wave = L^T k, so init k = L^-T u * w_u and
#     the frequency clamp bounds ||L^T k||.
#   - The Gaussian-conditioning projection (omega_2d + along-ray attenuation)
#     is an APPROXIMATION under a Beta envelope — the same approximation the
#     analytic clip already makes for the beta kernel in this rasterizer.
#
# amp is zero-initialised, so a fresh dbs-gabor model renders byte-identically
# to plain dbs (the CUDA forward gates on amp != 0 per splat and on a null
# buffer globally).

import torch
import numpy as np
from torch import nn

from scene.gaussian_model_dbs_sh import GaussianModel as DBSGaussianModel
from scene.gaussian_model_gabor import project_gabor_band


class GaussianModel(DBSGaussianModel):
    """dBS base + additive residual Gabor band.

    Adds three per-Gaussian tensors on top of the dBS parameter set:
      _gabor_omega  [N, 3]  world-space wave vector k (projected per view)
      _gabor_phase  [N, 1]  phase
      _gabor_amp    [N, 1]  raw residual amplitude; tanh-activated at render
    amp is zero-initialised so a fresh model is identical to plain dBS.
    """

    _GABOR_SPECS = (
        ("_gabor_omega", "gabor_omega", 3),
        ("_gabor_phase", "gabor_phase", 1),
        ("_gabor_amp", "gabor_amp", 1),
    )

    # Whitened-frame frequency bounds (rad per envelope sigma) — same as the
    # dGS gabor model / the Gabor Fields reference.
    WHITENED_OMEGA_LO = 0.5
    WHITENED_OMEGA_HI = 3.0
    WHITENED_OMEGA_INIT_LO = 0.7
    WHITENED_OMEGA_INIT_HI = 1.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gabor_omega = torch.empty(0)
        self._gabor_phase = torch.empty(0)
        self._gabor_amp = torch.empty(0)
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

    def _sigma_chol(self, idx=None):
        """Cholesky factor L of the spatial covariance (Sigma = L L^T), with a
        tiny diagonal jitter for numerical safety."""
        Sigma = self.get_covariance if idx is None else self.get_covariance[idx]
        n = Sigma.shape[0]
        eye = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype).expand(n, 3, 3)
        return torch.linalg.cholesky(Sigma + 1e-10 * eye)

    @torch.no_grad()
    def _whitened_omega(self, idx=None, device="cuda"):
        """Random wave vectors ~ one oscillation per footprint: k = L^-T u w_u,
        u uniform on S^2, w_u ~ U(0.7, 1.5) rad/sigma (then ||L^T k|| = w_u)."""
        L = self._sigma_chol(idx)
        n = L.shape[0]
        u = torch.randn(n, 3, 1, device=device)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        w_u = torch.rand(n, 1, 1, device=device) \
            * (self.WHITENED_OMEGA_INIT_HI - self.WHITENED_OMEGA_INIT_LO) \
            + self.WHITENED_OMEGA_INIT_LO
        k = torch.linalg.solve_triangular(
            L.transpose(1, 2), u * w_u, upper=True)
        return k.squeeze(-1)

    def _init_gabor_params(self, num_gaussians, device="cuda"):
        omega = self._whitened_omega(device=device)
        assert omega.shape[0] == num_gaussians, \
            f"gabor init needs base params: {omega.shape[0]} vs {num_gaussians}"
        phase = torch.zeros((num_gaussians, 1), device=device)
        amp = torch.zeros((num_gaussians, 1), device=device)
        self._gabor_omega = nn.Parameter(omega.requires_grad_(True))
        self._gabor_phase = nn.Parameter(phase.requires_grad_(True))
        self._gabor_amp = nn.Parameter(amp.requires_grad_(True))

    @torch.no_grad()
    def gabor_whitened_magnitude(self):
        """Whitened frequency magnitude ||L^T k|| per primitive [N, 1]
        (rad per envelope sigma)."""
        L = self._sigma_chol()
        white = torch.einsum('nji,nj->ni', L, self._gabor_omega)  # (L^T k)_i
        return white.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def clamp_gabor_frequency(self):
        """Keep the whitened frequency ||L^T k|| in [0.5, 3.0] by rescaling k;
        call after each optimizer step (train.py does for 'gabor' modes)."""
        if self._gabor_omega.numel() == 0:
            return
        m = self.gabor_whitened_magnitude()
        factor = m.clamp(self.WHITENED_OMEGA_LO, self.WHITENED_OMEGA_HI) \
            / m.clamp_min(1e-12)
        self._gabor_omega.mul_(factor)

    # ---- creation / optimizer --------------------------------------------
    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._init_gabor_params(self.get_xyz.shape[0], device=self.get_xyz.device)

    def training_setup(self, training_args):
        super().training_setup(training_args)
        if self._gabor_amp.numel() == 0:
            self._init_gabor_params(self.get_xyz.shape[0], device=self.get_xyz.device)
        lr_omega = getattr(training_args, "gabor_omega_lr", training_args.l_triangle_lr)
        lr_phase = getattr(training_args, "gabor_phase_lr", training_args.l_triangle_lr)
        lr_amp = getattr(training_args, "gabor_amp_lr", training_args.feature_lr)
        self.optimizer.add_param_group({'params': [self._gabor_omega], 'lr': lr_omega, "name": "gabor_omega"})
        self.optimizer.add_param_group({'params': [self._gabor_phase], 'lr': lr_phase, "name": "gabor_phase"})
        self.optimizer.add_param_group({'params': [self._gabor_amp], 'lr': lr_amp, "name": "gabor_amp"})
        # Residual-only fit: freeze the entire dBS base so ONLY the gabor band
        # trains (pair with --densify_until_iter 0).
        self._gabor_residual_only = bool(getattr(training_args, "gabor_residual_only", False))
        if self._gabor_residual_only:
            gabor_groups = {"gabor_omega", "gabor_phase", "gabor_amp"}
            for group in self.optimizer.param_groups:
                if group.get("name") not in gabor_groups:
                    group["lr"] = 0.0
                    for p in group["params"]:
                        p.requires_grad_(False)

    def update_learning_rate(self, iteration):
        # The base scheduler would re-raise the frozen xyz LR every iteration.
        if getattr(self, "_gabor_residual_only", False):
            return 0.0
        return super().update_learning_rate(iteration)

    # ---- capture/restore -------------------------------------------------
    def capture(self):
        return super().capture() + (self._gabor_omega, self._gabor_phase, self._gabor_amp)

    def restore(self, model_args, training_args):
        gabor = model_args[-3:]
        base_args = model_args[:-3]
        self._gabor_omega, self._gabor_phase, self._gabor_amp = (
            nn.Parameter(t.requires_grad_(True)) for t in gabor
        )
        super().restore(base_args, training_args)

    # ---- PLY attributes --------------------------------------------------
    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        for _, prefix, dim in self._GABOR_SPECS:
            for i in range(dim):
                l.append(f"{prefix}_{i}")
        return l

    def save_ply(self, path):
        # Replicate the base write with the gabor columns appended (the base
        # concatenates a fixed tuple, so we cannot reuse it directly).
        import os
        from utils.system_utils import mkdir_p
        from plyfile import PlyData, PlyElement

        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        mean = self._mean.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        l_triangle = self._l_triangle.detach().cpu().numpy()
        L_22_inv = self._L_22_inv.detach().cpu().numpy()
        v_12 = self._v_12.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, f_dc, f_rest, opacities, betas, mean, scale, l_triangle, L_22_inv, v_12,
             self._gabor_omega.detach().cpu().numpy(),
             self._gabor_phase.detach().cpu().numpy(),
             self._gabor_amp.detach().cpu().numpy()),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
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
                cols = sorted(cols, key=lambda x: int(x.split('_')[-1]))[:dim]
                arr = np.zeros((n, dim), dtype=np.float32)
                for i, c in enumerate(cols):
                    arr[:, i] = np.asarray(plydata.elements[0][c])
                return arr
            return np.full((n, dim), fill, dtype=np.float32)

        phase = _read("gabor_phase", 1, 0.0)
        amp = _read("gabor_amp", 1, 0.0)  # absent (plain dbs ckpt) => zero => == dBS
        if any(p.startswith("gabor_omega_") for p in names):
            omega_t = torch.tensor(_read("gabor_omega", 3, 0.0),
                                   dtype=torch.float, device=device)
        else:
            # Warm-starting a plain dBS checkpoint: whitened-frame random init.
            omega_t = self._whitened_omega(device=device)
        self._gabor_omega = nn.Parameter(omega_t.requires_grad_(True))
        self._gabor_phase = nn.Parameter(torch.tensor(phase, dtype=torch.float, device=device).requires_grad_(True))
        self._gabor_amp = nn.Parameter(torch.tensor(amp, dtype=torch.float, device=device).requires_grad_(True))

    # ---- densification plumbing ------------------------------------------
    # The base cat_tensors_to_optimizer indexes tensors_dict[name] for EVERY
    # optimizer group, so once the gabor groups exist every densify call must
    # supply gabor rows; the overrides below stash them in _pending_new_gabor.
    def _prune_optimizer(self, mask):
        optimizable_tensors = super()._prune_optimizer(mask)
        for attr, name, _ in self._GABOR_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
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

    def _gabor_rows(self, mask_or_idx, repeat_n=1):
        rows = (
            self._gabor_omega[mask_or_idx],
            self._gabor_phase[mask_or_idx],
            self._gabor_amp[mask_or_idx],
        )
        if repeat_n > 1:
            rows = tuple(r.repeat(repeat_n, 1) for r in rows)
        return rows

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = torch.logical_and(
            selected,
            torch.max(self.get_scaling[:, :3], dim=1).values <= self.percent_dense * scene_extent)
        self._pending_new_gabor = self._gabor_rows(selected)
        super().densify_and_clone(grads, grad_threshold, scene_extent)
        self._pending_new_gabor = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_current = self.get_xyz.shape[0]
        selected = torch.zeros((n_current), device="cuda", dtype=bool)
        selected[:grads.shape[0]] = grads.squeeze() >= grad_threshold
        selected = torch.logical_and(
            selected,
            torch.max(self.get_scaling[:, :3], dim=1).values > self.percent_dense * scene_extent)
        self._pending_new_gabor = self._gabor_rows(selected, repeat_n=N)
        super().densify_and_split(grads, grad_threshold, scene_extent, N)
        self._pending_new_gabor = None

    # ---- MCMC plumbing ---------------------------------------------------
    def relocate_gs(self, dead_mask=None):
        if dead_mask is None or dead_mask.sum() == 0:
            return
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        super().relocate_gs(dead_mask)
        # Fresh whitened residual at the relocated slots (zero amp == pure dBS
        # there; they re-learn).
        with torch.no_grad():
            self._gabor_omega[dead_indices] = self._whitened_omega(dead_indices)
            self._gabor_phase[dead_indices] = 0.0
            self._gabor_amp[dead_indices] = 0.0
            # Also reset the gabor Adam moments at the relocated slots.
            for group in self.optimizer.param_groups:
                if group.get("name") in {"gabor_omega", "gabor_phase", "gabor_amp"}:
                    state = self.optimizer.state.get(group["params"][0])
                    if state:
                        state["exp_avg"][dead_indices] = 0
                        state["exp_avg_sq"][dead_indices] = 0

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        (new_xyz, new_mean, new_features_dc, new_features_rest, new_opacity,
         new_beta, new_scale, new_l_triangle, new_L_22_inv, new_v_12) = \
            self._update_params(add_idx, ratio=ratio)
        self._opacity[add_idx] = new_opacity
        self._pending_new_gabor = self._gabor_rows(add_idx)
        self.densification_postfix(
            new_xyz, new_mean, new_features_dc, new_features_rest, new_opacity,
            new_beta, new_scale, new_l_triangle, new_L_22_inv, new_v_12)
        self._pending_new_gabor = None
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    def replace_tensors_to_optimizer(self, inds=None):
        # The dbs base builds its tensors_dict from a fixed attribute list and
        # then indexes tensors_dict[group.name] for EVERY optimizer group with
        # no membership guard — with the gabor groups registered that is a
        # KeyError: 'gabor_omega'. Reimplemented here with the gabor entries
        # included (found by external review, 2026-07-20).
        tensors_dict = {
            "xyz": self._xyz,
            "mean": self._mean,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "beta": self._beta,
            "scale": self._scale,
            "l_triangle": self._l_triangle,
            "L_22_inv": self._L_22_inv,
            "v_12": self._v_12,
            "gabor_omega": self._gabor_omega,
            "gabor_phase": self._gabor_phase,
            "gabor_amp": self._gabor_amp,
        }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            tensor = tensors_dict[group["name"]]
            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]
        self._L_22_inv = optimizable_tensors["L_22_inv"]
        if "v_12" in optimizable_tensors:
            self._v_12 = optimizable_tensors["v_12"]
        for attr, name, _ in self._GABOR_SPECS:
            if name in optimizable_tensors:
                setattr(self, attr, optimizable_tensors[name])
        torch.cuda.empty_cache()
        return optimizable_tensors

    # ---- rendering -------------------------------------------------------
    def render_tcgs(self, viewpoint_camera, render_mode="RGB", mask=None,
                    use_tcgs=False, scaling_modifier=1.0, **kwargs):
        # Replicate the base render_tcgs with the gabor buffer injected (the
        # base does not know the rasterizer's `gabor` kwarg).
        import math
        from tcgs_speedy_rasterizer import (
            GaussianRasterizationSettings as TCGSRasterizationSettings,
            GaussianRasterizer as TCGSRasterizer,
        )
        if render_mode != "RGB":
            raise NotImplementedError("render_tcgs currently supports render_mode='RGB' only.")
        device = self._xyz.device
        if mask is None:
            mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=device)
        else:
            mask = mask.to(dtype=torch.bool, device=device)

        if self.input_dim > 3:
            cam_pos = viewpoint_camera.camera_center
            view_dir = self._xyz - cam_pos.unsqueeze(0)
            view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
            if self.input_dim == 6:
                query = view_dir
            elif self.input_dim == 7:
                timestamp = torch.full((view_dir.shape[0], 1), viewpoint_camera.timestamp,
                                       device=view_dir.device, dtype=view_dir.dtype)
                query = torch.cat([view_dir, timestamp], dim=-1)
            else:
                raise NotImplementedError("Only implemented for 6D or 7D query")
            means, opacity_scale = self.get_cond_mean_opacity(query)
            opacities = self.get_opacity * opacity_scale
            convs = self.get_covariance
        else:
            means = self._xyz
            convs = self.get_covariance
            opacities = self.get_opacity

        means3d = means[mask][..., :3].contiguous()
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = convs[mask][..., tri_indices[0], tri_indices[1]].contiguous()
        shs = self.get_features[mask].contiguous()
        betas_full = self.get_beta[mask].contiguous()
        if betas_full.dim() > 1 and betas_full.shape[-1] > 1:
            betas = betas_full[:, 0:1].contiguous()
        else:
            betas = betas_full if betas_full.dim() > 1 else betas_full.unsqueeze(-1)
        opacities = opacities[mask].contiguous()
        if opacities.dim() == 1:
            opacities = opacities.unsqueeze(-1)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Residual Gabor band: project the (masked) world wave vectors for this
        # view. Sigma is EXACTLY the cov3D_precomp the renderer consumes. The
        # dBS path does not use mip antialiasing => the renderer dilates the 2D
        # covariance by 0.3, so antialiasing=False here matches it.
        gabor = None
        if getattr(self, "use_gabor", True) and self._gabor_amp.numel() > 0:
            gabor = project_gabor_band(
                self._gabor_omega[mask], self._gabor_phase[mask],
                self._gabor_amp[mask], viewpoint_camera, means3d,
                convs[mask], antialiasing=False,
                tanfovx=tanfovx, tanfovy=tanfovy)

        bg_color = (self.background.to(device=means3d.device, dtype=means3d.dtype)
                    if self.background.numel()
                    else torch.tensor([0.0, 0.0, 0.0], device=means3d.device, dtype=means3d.dtype))
        raster_settings = TCGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(means3d.device),
            projmatrix=viewpoint_camera.full_proj_transform.to(means3d.device),
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center.to(means3d.device),
            x_threshold=(viewpoint_camera.x_threshold
                         if getattr(viewpoint_camera, 'x_threshold', None) is not None else float('inf')),
            clip_plane=(torch.tensor(viewpoint_camera.clip_plane, dtype=torch.float32, device="cuda")
                        if getattr(viewpoint_camera, 'clip_plane', None) is not None else None),
            prefiltered=False, use_tcgs=use_tcgs, tight_snugbox=use_tcgs,
            debug=False,
        )
        rasterizer = TCGSRasterizer(raster_settings=raster_settings)
        screenspace_points = torch.zeros_like(means3d, dtype=means3d.dtype,
                                              requires_grad=True, device=means3d.device) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        rendered_image, radii, render_time, _ = rasterizer(
            means3D=means3d, means2D=screenspace_points, shs=shs,
            colors_precomp=None, opacities=opacities,
            scores=means3d.new_empty(0), cov3D_precomp=covars,
            betas=betas, gabor=gabor,
        )
        if rendered_image.dim() == 4:
            rendered_image = rendered_image[0]
        if radii.device.type != 'cuda':
            radii = radii.cuda()
        return {
            "render": rendered_image.contiguous(),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "is_used": radii > 0,
        }
