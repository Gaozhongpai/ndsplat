#
# ClipGS baseline (our re-implementation) for the XClipGS comparison.
#
# This is OUR faithful re-implementation of the clip mechanism of ClipGS
# (Li et al., MICCAI 2025, arXiv:2507.06647), NOT their released code. It is
# built on the identical 3DGS backbone / renderer / data as our analytic
# operator so the comparison is apples-to-apples (same backbone, same clip
# planes, same supervision), isolating the operator.
#
# ClipGS's clip = two pieces, both reproduced here:
#   1. a per-primitive HARD keep/drop against the plane (the HC operator class),
#      made trainable through a STRAIGHT-THROUGH ESTIMATOR (STE): the forward
#      pass is the binary cut, the backward pass passes gradients through as if
#      the cut were the identity, so the (otherwise non-differentiable) hard cull
#      can be supervised end-to-end; and
#   2. a small LEARNED per-primitive DEFORMATION MLP that predicts position (and
#      scale) offsets to mask the popping/quantization the binary cut produces.
#
# Contrast with our operator (gaussian_model.py + analytic_clip): we render the
# exact truncated density in closed form (a per-pixel CDF factor), with no STE
# and no auxiliary network. This file exists only to benchmark ClipGS's approach
# under our controlled setup.

import os

import torch
from torch import nn

from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from tcgs_speedy_rasterizer import hard_clip_mask


class _STEHardCull(torch.autograd.Function):
    """Straight-through hard cull. Forward: opacity * keep (binary keep from the
    center-vs-plane test). Backward: gradient flows to opacity unchanged (as if
    keep were 1 everywhere), so the binary decision is trainable. This is exactly
    ClipGS's trick for making its keep/drop clip differentiable."""

    @staticmethod
    def forward(ctx, opacity, keep):
        return opacity * keep.to(opacity.dtype).view(-1, 1)

    @staticmethod
    def backward(ctx, grad_out):
        # straight-through: pass the gradient to opacity as-is, none to `keep`
        return grad_out, None


def ste_hard_cull(opacity, keep):
    return _STEHardCull.apply(opacity, keep)


class ClipDeformMLP(nn.Module):
    """Small per-primitive deformation network (ClipGS's discontinuity-masking
    MLP). Input: the Gaussian center (3) plus the clip-plane signed distance (1)
    so the deformation is plane-aware; output: a position offset (3) and a log-
    scale offset (3). Kept tiny (2 hidden layers, width 64) as in the ClipGS
    spirit of a lightweight corrector, not a heavy renderer."""

    def __init__(self, width=64, deform_scale=False, out_scale=1e-3):
        super().__init__()
        self.deform_scale = deform_scale
        self.out_scale = out_scale     # near-identity at init WITHOUT killing gradients
        out = 6 if deform_scale else 3
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.ReLU(inplace=True),
            nn.Linear(width, width), nn.ReLU(inplace=True),
            nn.Linear(width, out),
        )
        # Near-identity at init, but KEEP the last-layer weights nonzero so
        # gradients flow to the earlier layers. (Zeroing the last-layer WEIGHT
        # would make d(out)/d(prev) = 0, freezing the MLP at zero forever -- the
        # dead-network trap.) Small random weights + zero bias + out_scale gives a
        # tiny initial deformation while remaining trainable.
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xyz, signed_dist):
        # xyz: [N,3] centers; signed_dist: [N,1] n.x - tau (plane-relative)
        inp = torch.cat([xyz, signed_dist], dim=1)
        out = self.net(inp) * self.out_scale
        d_xyz = out[:, :3]
        d_logscale = out[:, 3:6] if self.deform_scale else None
        return d_xyz, d_logscale


class GaussianModel(GaussianModel3DGS):
    """3DGS backbone + ClipGS clip (STE hard-cull + deformation MLP)."""

    def __init__(self, sh_degree: int, deform_scale: bool = False):
        super().__init__(sh_degree)
        self._clipgs_deform_scale = deform_scale
        self.deform_mlp = None  # created in training_setup (needs a device)

    # --- the deformation MLP joins the optimizer -----------------------------
    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.deform_mlp = ClipDeformMLP(deform_scale=self._clipgs_deform_scale).cuda()
        deform_lr = getattr(training_args, "clipgs_deform_lr", 1e-4)
        self.optimizer.add_param_group(
            {"params": list(self.deform_mlp.parameters()),
             "lr": deform_lr, "name": "clipgs_deform"})

    # --- densification must skip the deform-MLP param group -----------------
    # The base _prune_optimizer / cat_tensors_to_optimizer index EVERY optimizer
    # group by the per-Gaussian mask; the ClipGS deform-MLP weights are NOT
    # per-Gaussian, so we exclude that group from both. (Its params are updated
    # by Adam normally; densification just must not slice them.)
    _NON_POINT_GROUPS = ("clipgs_deform",)

    def _prune_optimizer(self, mask):
        deform = [g for g in self.optimizer.param_groups
                  if g["name"] in self._NON_POINT_GROUPS]
        self.optimizer.param_groups = [g for g in self.optimizer.param_groups
                                       if g["name"] not in self._NON_POINT_GROUPS]
        out = super()._prune_optimizer(mask)
        self.optimizer.param_groups += deform
        return out

    def cat_tensors_to_optimizer(self, tensors_dict):
        deform = [g for g in self.optimizer.param_groups
                  if g["name"] in self._NON_POINT_GROUPS]
        self.optimizer.param_groups = [g for g in self.optimizer.param_groups
                                       if g["name"] not in self._NON_POINT_GROUPS]
        out = super().cat_tensors_to_optimizer(tensors_dict)
        self.optimizer.param_groups += deform
        return out

    # --- MLP persistence: save_ply/load_ply write/read a sibling deform_mlp.pt --
    # The base save_ply/load_ply only handle the Gaussians; the trained ClipGS
    # deformation MLP must be persisted alongside or the rendered/eval'd cut uses
    # an untrained (near-identity) MLP -> not a faithful ClipGS.
    def _mlp_path(self, ply_path):
        return os.path.join(os.path.dirname(ply_path), "deform_mlp.pt")

    def save_ply(self, path):
        super().save_ply(path)
        if self.deform_mlp is not None:
            torch.save({"state_dict": self.deform_mlp.state_dict(),
                        "deform_scale": self._clipgs_deform_scale},
                       self._mlp_path(path))

    def load_ply(self, path):
        super().load_ply(path)
        mp = self._mlp_path(path)
        if os.path.isfile(mp):
            ckpt = torch.load(mp, map_location="cuda")
            self._clipgs_deform_scale = ckpt.get("deform_scale", self._clipgs_deform_scale)
            if self.deform_mlp is None:
                self.deform_mlp = ClipDeformMLP(deform_scale=self._clipgs_deform_scale).cuda()
            self.deform_mlp.load_state_dict(ckpt["state_dict"])

    def _apply_deform(self, xyz, scales, clip_plane_tensor):
        """Apply the ClipGS deformation MLP. Returns (xyz', scales'). No-op if the
        MLP is absent (e.g. at inference before training_setup) or no plane."""
        if self.deform_mlp is None or clip_plane_tensor is None:
            return xyz, scales
        n = clip_plane_tensor[:3]
        tau = clip_plane_tensor[3]
        signed = (xyz @ n - tau).unsqueeze(1)         # [N,1] n.x - tau
        d_xyz, d_logscale = self.deform_mlp(xyz, signed)
        xyz = xyz + d_xyz
        if d_logscale is not None and scales is not None:
            scales = scales * torch.exp(d_logscale)
        return xyz, scales

    # --- render: 3DGS via the clip-capable tcgs rasterizer, ClipGS operator ---
    def render_tcgs(self, viewpoint_camera, pipe, background,
                    scaling_modifier=1.0, override_color=None, is_test=False):
        import math
        from tcgs_speedy_rasterizer import (
            GaussianRasterizationSettings as TCGSRasterizationSettings,
            GaussianRasterizer as TCGSRasterizer,
        )
        from utils.sh_utils import eval_sh

        screenspace_points = torch.zeros_like(
            self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        clip_plane = getattr(viewpoint_camera, "clip_plane", None)
        clip_plane_tensor = (
            torch.tensor(clip_plane, dtype=torch.float32, device="cuda")
            if clip_plane is not None else None)

        means3D = self.get_xyz
        opacity = self.get_opacity
        scales = self.get_scaling
        rotations = self.get_rotation

        # ClipGS deformation MLP (plane-aware position/scale correction).
        means3D, scales = self._apply_deform(means3D, scales, clip_plane_tensor)

        # ClipGS operator: HARD keep/drop, trainable via straight-through.
        # We do NOT hand the plane to the rasterizer (no analytic factor); the cut
        # is entirely the binary center test, made differentiable by the STE.
        if clip_plane_tensor is not None:
            keep = hard_clip_mask(means3D, clip_plane=clip_plane_tensor)
            opacity = ste_hard_cull(opacity, keep)

        raster_settings = TCGSRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy,
            bg=background, scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            x_threshold=float("inf"),   # cut is applied by us (STE), not the kernel
            clip_plane=None,            # <- no analytic clip; ClipGS is binary
            analytic_clip=False,
            prefiltered=False,
            # use_tcgs=False -> the standard DIFFERENTIABLE backward. The tcgs
            # "speedy" path (use_tcgs=True, the default) does not backprop to the
            # primitive params, so training silently gets zero gradients. This is
            # the same setting the working dGS sweep uses.
            use_tcgs=False,
            tight_snugbox=False,
            compact_box_mult=1.0,
            antialiasing=False,
            debug=getattr(pipe, "debug", False),
        )
        rasterizer = TCGSRasterizer(raster_settings=raster_settings)

        shs = None
        colors_precomp = None
        if override_color is None:
            if getattr(pipe, "convert_SHs_python", False):
                shs_view = self.get_features.transpose(1, 2).view(
                    -1, 3, (self.max_sh_degree + 1) ** 2)
                dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.get_features
        else:
            colors_precomp = override_color

        # tcgs rasterizer signature: needs scores/betas kwargs, returns 4 values.
        rendered_image, radii, _render_time, _ = rasterizer(
            means3D=means3D, means2D=screenspace_points,
            shs=shs, colors_precomp=colors_precomp,
            opacities=opacity, scores=None,
            scales=scales, rotations=rotations,
            cov3D_precomp=None, betas=None)

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    # --- interactive viewer hook (view.py / GaussianViewer) -------------------
    def view_tcgs(self, camera_state, render_tab_state):
        """Drive the viser viewer. Mirrors the dgs model's view_tcgs but calls this
        class's render_tcgs (which applies the ClipGS deformation MLP + STE cull).
        The viewer's x_threshold slider becomes an x-axis clip plane [1,0,0,tau];
        tau=inf (slider at max) means no cut."""
        import math
        import time
        import numpy as np
        from types import SimpleNamespace
        from scene.cameras import Camera
        from scene.gaussian_viewer import GaussianRenderTabState
        assert isinstance(render_tab_state, GaussianRenderTabState)
        start = time.time()

        W = (render_tab_state.render_width if render_tab_state.preview_render
             else render_tab_state.viewer_width)
        H = (render_tab_state.render_height if render_tab_state.preview_render
             else render_tab_state.viewer_height)

        c2w = torch.from_numpy(camera_state.c2w).float().cuda()
        K = torch.from_numpy(camera_state.get_K((W, H))).float().cuda()
        fx, fy = K[0, 0], K[1, 1]
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3].cpu().numpy().T
        T = w2c[:3, 3].cpu().numpy()

        xt = render_tab_state.x_threshold
        cam = Camera(colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                     image=torch.zeros((3, H, W)), gt_alpha_mask=None,
                     image_name="viewer", uid=0, x_threshold=xt, data_device="cuda")
        # x_threshold -> general clip plane; inf/None = no cut.
        cam.clip_plane = None if (xt is None or not math.isfinite(xt)) else [1.0, 0.0, 0.0, float(xt)]

        self.background = torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False)
        out = self.render_tcgs(cam, pipe, self.background, is_test=True)

        img = out["render"]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        render_tab_state.total_count_number = int(self.get_xyz.shape[0])
        render_tab_state.rendered_count_number = int(out["visibility_filter"].sum().item())
        dt = time.time() - start
        render_tab_state.fps = (1.0 / dt) if dt > 0 else 0.0
        return img.permute(1, 2, 0).detach().cpu().numpy()
