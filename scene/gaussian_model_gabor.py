#
# Residual Gabor extension of the Full DGS model (dgs base + additive Gabor band).
#
# Design (a PROJECTED GABOR RESIDUAL, inspired by the Gabor Fields paper's
# low-pass Gaussian base + residual Gabor kernels — NOT a port: upstream uses
# independent residual primitives with their own centers/envelopes/signed
# opacities and staged pyramid training; here one co-located modulation rides
# each existing dGS primitive and shares its envelope, opacity, color, center):
#   - The dGS base (scene/gaussian_model_dgs.py) is the Gaussian base. Its
#     parameters, save/load and forward behaviour are UNCHANGED.
#   - On top of the base each Gaussian carries a residual Gabor band: a cosine
#     modulation of its Gaussian footprint,
#         weight_gabor = weight * (1 + amp * cos(omega_2d . d_screen + phase))
#     where d_screen = (splat_center_px - pixel) is the same offset the base
#     footprint uses. _gabor_omega [N,3] is a WORLD-SPACE wave vector k tied to
#     the primitive; per view it is projected to the screen frequency omega_2d
#     by conditioning the 3D modulated Gaussian on the pixel ray (see
#     _gabor_tensors_for_raster). This makes the stripes view-consistent: an
#     oblique view sees the frequency foreshortened, and a wave vector aligned
#     with the viewing ray washes out (Gaussian-attenuated amplitude) instead of
#     painting stripes with a wrong orientation.
#   - amp = tanh(_gabor_amp) in (-1, 1), so the CUDA-side clamp of the
#     modulation factor at zero (its zero-gradient dead zone) is unreachable.
#     _gabor_amp is ZERO-initialised (tanh(0) = 0), so a fresh model renders
#     BYTE-IDENTICALLY to plain dGS (the modulation factor is 1), and the CUDA
#     forward is gated so a null gabor buffer is the exact dGS path.
#   - Frequency lives in the WHITENED frame of the base primitive (the Gabor
#     Fields convention): magnitude ||S R^T k|| is rad per envelope sigma. Init
#     draws it ~ U(0.7, 1.5) along a random whitened direction (about one
#     oscillation across the footprint) and clamp_gabor_frequency() keeps it in
#     [0.5, 3.0] after each optimizer step — the lower bound keeps the band
#     genuinely oscillatory (omega -> 0 with amp != 0 is redundant with base
#     opacity), the upper bound blocks super-Nyquist atoms that alias.
#
# Exact vs approximate: the forward Gabor factor is exact for the *unclipped*
# footprint. The half-space clip of a Gabor atom is the complex-error-function
# (Faddeeva) generalisation of the real-erf clipPhi; here we reuse the base's
# real-erf clip on the Gaussian envelope and leave the cosine unclipped. That
# is APPROXIMATE when a clip plane crosses an active atom. NOTE the heart_900
# set is HALF clipped views (train and test), so this approximation IS active
# in the fits and the metrics: the v2 residual gains +0.199 dB on the intact
# test half but only +0.081 dB on the clipped half.

import torch
import numpy as np
from torch import nn

from scene.gaussian_model_dgs import GaussianModel as DGSGaussianModel
from utils.general_utils import build_rotation, build_scaling_rotation


def project_gabor_band(k, phase, raw_amp, viewpoint_camera, means3D, Sigma,
                       antialiasing, tanfovx, tanfovy, clip_plane=None):
    """World-space wave vector -> per-view screen float4 for the CUDA forward.

    When `clip_plane` = (nx, ny, nz, tau) is given (analytic half-space clip
    n.x <= tau active this view), additionally returns the per-splat scalar
        b = Cov(clip coordinate, wave phase | pixel) / s
          = (n^T Sigma k - q_n^T Sigma2d^-1 q_g) / sqrt(n^T Sigma n - q_n^T Sigma2d^-1 q_n)
    that the CUDA kernel needs for the EXACT clipped-Gabor factor
        Phi(l) + amp_eff * Re{ e^{i m} * 0.5 erfc(-(l - i b)/sqrt(2)) }
    (complex-erf generalisation of clipPhi; validated against numerical line
    integration in scripts/tests/gabor_projection_check.py, check 3). By
    Cauchy-Schwarz |b| <= sigma_xi <= the whitened frequency bound (3.0), which
    keeps the CUDA evaluation numerically safe. Returns (gabor4, gabor_b) —
    gabor_b is None when no plane is active.

    The 3D atom is G_3d(x) * (1 + amp * cos(k . (x - mu) + phase)); conditioning
    the world offset delta on the screen offset Delta = P delta (P = the same
    affine world->pixel map the covariance projection uses,
    Sigma_2d = P Sigma P^T) gives E[cos(k.delta + phase) | Delta]
        = exp(-0.5 * Var(k.delta | Delta)) * cos((M^T k) . Delta + phase),
    M = Sigma P^T Sigma_2d^-1. So per view:
        omega_2d = -Sigma_2d^-1 P Sigma k      (CUDA d = center - pixel = -Delta)
        amp_eff  = tanh(raw_amp) * exp(-0.5 * (k^T Sigma k - q^T Sigma_2d^-1 q)),
                   q = P Sigma k.
    The attenuation kills atoms whose wave points along the viewing ray (they
    would otherwise paint stripes with an arbitrary orientation), and the
    frequency foreshortens correctly with view — this is what makes the band
    consistent across training views. Everything here is autograd-
    differentiable, so grad flows from the CUDA grad_gabor back into k.
    Sigma_2d uses the same dilation as the renderer (0.1 antialiased /
    0.3 classic), which also guarantees invertibility. Exact for a Gaussian
    envelope; for the beta kernel the same conditioning is an approximation
    (the same one the analytic clip already makes for beta).

    Verified against finite-difference Jacobians and numerical line-integral
    conditional expectations in scripts/tests/gabor_projection_check.py.

    Args: k [N,3], phase [N,1], raw_amp [N,1] (tanh applied here),
          means3D [N,3] rendered centers, Sigma [N,3,3] world covariance.
    Returns the packed [N,4] {omega_x, omega_y, phase, amp_eff} tensor.
    """
    W_mat = viewpoint_camera.world_view_transform             # [4,4], row-vector conv
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)

    # View-space centers, with the same frustum clamp computeCov2D applies.
    t = means3D @ W_mat[:3, :3] + W_mat[3, :3]                # [N,3]
    tz = t[:, 2].clamp_min(0.2)
    tx = (t[:, 0] / tz).clamp(-1.3 * tanfovx, 1.3 * tanfovx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tanfovy, 1.3 * tanfovy) * tz

    # T = Wm @ J (float64-reference convention): screen offset
    # Delta = T[:, :2]^T . delta_world.
    n = k.shape[0]
    J = torch.zeros(n, 3, 2, device=k.device, dtype=k.dtype)
    J[:, 0, 0] = focal_x / tz
    J[:, 1, 1] = focal_y / tz
    J[:, 2, 0] = -focal_x * tx / (tz * tz)
    J[:, 2, 1] = -focal_y * ty / (tz * tz)
    T = W_mat[:3, :3].unsqueeze(0) @ J                        # [N,3,2]

    ST = Sigma @ T                                            # [N,3,2]
    Sigma2d = T.transpose(1, 2) @ ST                          # [N,2,2]
    ks = 0.1 if antialiasing else 0.3
    a = Sigma2d[:, 0, 0] + ks
    b = Sigma2d[:, 0, 1]
    c = Sigma2d[:, 1, 1] + ks
    det = (a * c - b * b).clamp_min(1e-12)

    q = torch.einsum('nij,ni->nj', ST, k)                     # [N,2] = T^T Sigma k
    # omega_2d = -Sigma_2d^-1 q (closed-form 2x2 inverse)
    wx = -(c * q[:, 0] - b * q[:, 1]) / det
    wy = -(a * q[:, 1] - b * q[:, 0]) / det
    # Var(k.delta | Delta) = k^T Sigma k - q^T Sigma_2d^-1 q; note
    # Sigma_2d^-1 q = -omega_2d.
    kSk = torch.einsum('ni,nij,nj->n', k, Sigma, k)
    var_ray = (kSk + q[:, 0] * wx + q[:, 1] * wy).clamp_min(0.0)
    atten = torch.exp(-0.5 * var_ray).unsqueeze(-1)           # [N,1]

    amp_eff = torch.tanh(raw_amp) * atten
    gabor4 = torch.cat([
        wx.unsqueeze(-1),
        wy.unsqueeze(-1),
        phase,
        amp_eff,
    ], dim=1).contiguous()

    gabor_b = None
    if clip_plane is not None:
        nrm = clip_plane[:3].to(dtype=k.dtype, device=k.device)
        Sn = Sigma @ nrm                                      # [N,3]
        q_n = torch.einsum('nij,ni->nj', T, Sn)               # [N,2] = T^T Sigma n
        # Sigma2d^-1 q_n via the same closed-form 2x2 inverse
        inx = (c * q_n[:, 0] - b * q_n[:, 1]) / det
        iny = (a * q_n[:, 1] - b * q_n[:, 0]) / det
        s2 = (Sn @ nrm - (q_n[:, 0] * inx + q_n[:, 1] * iny)).clamp_min(1e-12)
        cov = torch.einsum('ni,ni->n', Sn, k) \
            - (q[:, 0] * inx + q[:, 1] * iny)
        gabor_b = (cov / torch.sqrt(s2)).unsqueeze(-1).contiguous()
    return gabor4, gabor_b


class GaussianModel(DGSGaussianModel):
    """dGS base + additive residual Gabor band.

    Adds three per-Gaussian tensors on top of the dGS parameter set:
      _gabor_omega  [N, 3]  world-space wave vector k (projected per view)
      _gabor_phase  [N, 1]  phase
      _gabor_amp    [N, 1]  raw residual amplitude; tanh-activated at render
    amp is zero-initialised so a fresh model is identical to plain dGS.
    """

    # Names of the extra Gabor optimizer groups / PLY attribute prefixes, so the
    # densification / save / load / prune plumbing can iterate them generically.
    _GABOR_SPECS = (
        ("_gabor_omega", "gabor_omega", 3),
        ("_gabor_phase", "gabor_phase", 1),
        ("_gabor_amp", "gabor_amp", 1),
    )

    # Whitened-frame frequency bounds (rad per envelope sigma), following the
    # Gabor Fields reference (BoundedAdam omega bounds [0.5, 3.0]). Init draws
    # from [WHITENED_OMEGA_INIT_LO, _HI], inside the clamp range.
    WHITENED_OMEGA_LO = 0.5
    WHITENED_OMEGA_HI = 3.0
    WHITENED_OMEGA_INIT_LO = 0.7
    WHITENED_OMEGA_INIT_HI = 1.5

    def __init__(self, *args, gabor_omega_init: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        # amp is ALWAYS zero-init (so the forward stays byte-identical to dGS),
        # but omega must start NONZERO so that once amp bootstraps off zero the
        # cosine already carries a real spatial frequency and can become
        # oscillatory. With omega == 0 the residual degenerates to a uniform
        # footprint rescale and never learns any Gabor structure (its
        # omega/phase gradients stay ~0). The init lives in the WHITENED frame
        # of each base primitive: k = R S^-1 u * w_u with u a random unit
        # direction and w_u ~ U(0.7, 1.5) rad/sigma, i.e. roughly one
        # oscillation across the footprint regardless of the primitive's size
        # or anisotropy (gabor_omega_init is kept for CLI compatibility but no
        # longer used).
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

    @torch.no_grad()
    def _whitened_omega(self, idx=None, device="cuda"):
        """Random world-space wave vectors in the whitened frame of the base:
        k = R S^-1 u * w_u, u ~ uniform on S^2, w_u ~ U(0.7, 1.5) rad/sigma.
        The whitened wave cos(w_u * u . p), p = S^-1 R^T (x - mu), then has
        about one oscillation across the footprint for every primitive,
        independent of its size/anisotropy. Requires the base params to exist.
        idx selects a subset of primitives (None = all)."""
        rot = self._rotation if idx is None else self._rotation[idx]
        s = self.get_scaling if idx is None else self.get_scaling[idx]
        n = rot.shape[0]
        R = build_rotation(rot)                                   # [n,3,3]
        u = torch.randn(n, 3, device=device)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        w_u = torch.rand(n, 1, device=device) \
            * (self.WHITENED_OMEGA_INIT_HI - self.WHITENED_OMEGA_INIT_LO) \
            + self.WHITENED_OMEGA_INIT_LO
        k = torch.einsum('nij,nj->ni', R, u / s.clamp_min(1e-8)) * w_u
        return k

    def _init_gabor_params(self, num_gaussians, device="cuda"):
        """Init the residual Gabor band. amp is ZERO (=> forward == dGS), omega
        is a whitened-frame random wave vector (see _whitened_omega) so the
        cosine carries a real frequency once amp bootstraps off zero."""
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
        """Whitened frequency magnitude ||S R^T k|| per primitive [N, 1]
        (rad per envelope sigma)."""
        k = self._gabor_omega
        R = build_rotation(self._rotation)
        # S R^T k, per-axis: (R^T k)_i * s_i
        white = torch.einsum('nji,nj->ni', R, k) * self.get_scaling
        return white.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def clamp_gabor_frequency(self):
        """Keep the whitened frequency magnitude ||S R^T k|| inside
        [WHITENED_OMEGA_LO, WHITENED_OMEGA_HI] by rescaling k. Call after each
        optimizer step (the Gabor Fields reference enforces the same bounds via
        BoundedAdam). Lower bound: a near-DC residual is redundant with base
        opacity. Upper bound: super-Nyquist atoms alias and train poorly."""
        if self._gabor_omega.numel() == 0:
            return
        m = self.gabor_whitened_magnitude()
        factor = m.clamp(self.WHITENED_OMEGA_LO, self.WHITENED_OMEGA_HI) \
            / m.clamp_min(1e-12)
        self._gabor_omega.mul_(factor)

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
        # Residual-only fit: freeze the entire dGS base (zero LR + no grads) so
        # ONLY the gabor band trains. The base stays byte-identical to the
        # warm-start checkpoint (pair with --densify_until_iter 0).
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

        # A plain-dGS checkpoint trained with use_view_dependent_pos=False has no
        # v_12_direction / lambda columns, so the base load_ply leaves them at
        # width 0. When we then fit with use_view_dependent_pos=True the dGS
        # slice backward returns a [N, 3*C] v_12 gradient into a [N, 0] leaf and
        # crashes. Re-initialise the view-dependent params to zeros of the right
        # shape here (zero shift == identical initial render), so warm-starting a
        # non-view-dependent checkpoint into the gabor fit is well-formed. This
        # only touches the gabor subclass, never the base dGS load path.
        if self.use_view_dependent_pos:
            C = self.input_dim - 3
            if self._v_12_direction.numel() == 0 or self._v_12_direction.shape[1] != 3 * C:
                self._v_12_direction = nn.Parameter(
                    torch.zeros(n, 3 * C, device=device).requires_grad_(True))
            if self._lambda_view.numel() != n:
                self._lambda_view = nn.Parameter(
                    torch.full((n,), float(self.lambda_init), device=device).requires_grad_(
                        not self.use_opacity_pos_decouple))
            if self.input_dim == 7 and self._lambda_time.numel() != n:
                self._lambda_time = nn.Parameter(
                    torch.full((n,), float(self.lambda_init), device=device).requires_grad_(
                        not self.use_opacity_pos_decouple))

        def _read(prefix, dim, fill):
            cols = [p for p in names if p.startswith(prefix + "_")]
            if len(cols) >= dim:
                cols = sorted([c for c in cols], key=lambda x: int(x.split('_')[-1]))[:dim]
                arr = np.zeros((n, dim), dtype=np.float32)
                for i, c in enumerate(cols):
                    arr[:, i] = np.asarray(plydata.elements[0][c])
                return arr
            return np.full((n, dim), fill, dtype=np.float32)

        phase = _read("gabor_phase", 1, 0.0)
        amp = _read("gabor_amp", 1, 0.0)  # absent (plain dGS ckpt) => zero => == dGS
        if any(p.startswith("gabor_omega_") for p in names):
            # Resuming a gabor fit: keep its trained wave vectors.
            omega_t = torch.tensor(_read("gabor_omega", 3, 0.0),
                                   dtype=torch.float, device=device)
        else:
            # Warm-starting a plain dGS checkpoint: whitened-frame random init.
            # (The old code filled a CONSTANT (v,v,v) here, which seeded every
            # atom with the same stripe direction — a genuine init bug.)
            omega_t = self._whitened_omega(device=device)
        self._gabor_omega = nn.Parameter(omega_t.requires_grad_(True))
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
        # Base builds the cat dict from its own param names; the gabor rows
        # reach cat_tensors_to_optimizer via _pending_new_gabor. Callers stash
        # them either through the new_gabor kwarg or by pre-setting
        # _pending_new_gabor: the overridden densify_and_clone/split/add_new_gs
        # do the latter BEFORE the base method dynamically dispatches back into
        # this override, so only touch the stash when the kwarg is explicitly
        # present — unconditionally popping with a None default clobbered the
        # pre-set stash and every growth path crashed with
        # KeyError: 'gabor_omega' (found by external review, 2026-07-20).
        if "new_gabor" in kwargs:
            self._pending_new_gabor = kwargs.pop("new_gabor")
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
            self._gabor_omega[dead_indices] = self._whitened_omega(dead_indices)
            self._gabor_phase[dead_indices] = 0.0
            self._gabor_amp[dead_indices] = 0.0
            # The base replace_tensors_to_optimizer skips groups it does not
            # know, so reset the gabor Adam moments at the relocated slots here.
            for group in self.optimizer.param_groups:
                if group.get("name") in {"gabor_omega", "gabor_phase", "gabor_amp"}:
                    state = self.optimizer.state.get(group["params"][0])
                    if state:
                        state["exp_avg"][dead_indices] = 0
                        state["exp_avg_sq"][dead_indices] = 0

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
    def _gabor_tensors_for_raster(self, viewpoint_camera, means3D, scales,
                                  rotations, antialiasing, tanfovx, tanfovy,
                                  scaling_modifier=1.0, clip_plane=None):
        """Pack the gabor band into the [N, 4] tensor the CUDA forward expects
        plus the per-splat exact-clip scalar b (projection math and
        conventions: see project_gabor_band above). Returns (None, None) when
        the residual is globally disabled so the CUDA path is byte-identical
        to dGS."""
        if not getattr(self, "use_gabor", True):
            return None, None
        if self._gabor_amp.numel() == 0:
            return None, None
        # 3D world covariance of the base (same construction as the renderer).
        L = build_scaling_rotation(scaling_modifier * scales, rotations)
        Sigma = L @ L.transpose(1, 2)                             # [N,3,3]
        return project_gabor_band(
            self._gabor_omega, self._gabor_phase, self._gabor_amp,
            viewpoint_camera, means3D, Sigma, antialiasing, tanfovx, tanfovy,
            clip_plane=clip_plane)

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

        # Effective analytic-clip plane this view (mirrors the wrapper's
        # _extract_clip_plane: an explicit plane wins, else legacy x_threshold
        # maps to (1,0,0,tau)). Drives the exact clipped-Gabor scalar b.
        gabor_plane = None
        if analytic_clip:
            if clip_plane_tensor is not None:
                gabor_plane = clip_plane_tensor
            elif math.isfinite(x_threshold):
                gabor_plane = torch.tensor([1.0, 0.0, 0.0, x_threshold],
                                           dtype=torch.float32, device="cuda")
        gabor, gabor_b = self._gabor_tensors_for_raster(
            viewpoint_camera, means3D=m_cond, scales=scales,
            rotations=self.get_rotation, antialiasing=antialiasing,
            tanfovx=tanfovx, tanfovy=tanfovy, scaling_modifier=scaling_modifier,
            clip_plane=gabor_plane)

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
            cov3D_precomp=None, betas=None, gabor=gabor, gabor_b=gabor_b,
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
