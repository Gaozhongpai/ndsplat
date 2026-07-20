"""Numerical verification of the world->screen Gabor wave-vector projection
in GaussianModel._gabor_tensors_for_raster (scene/gaussian_model_gabor.py).

Two independent checks, both convention-sensitive (transposes, signs, ndc2Pix),
neither reusing the closed form under test:

1. JACOBIAN: the analytic screen-gradient matrix T (columns = d(px,py)/d(world))
   built from W_mat[:3,:3] @ J must match a finite-difference Jacobian of the
   REAL pixel projection pipeline (world -> full_proj -> ndc -> ndc2Pix), the
   same pipeline the rasterizer uses to place splat centers.

2. CONDITIONAL EXPECTATION: for pixels around the projected center, the value
   atten * cos(omega_2d . d + phase) predicted by the closed form must match
   the numerically line-integrated
       E[cos(k . delta + phase) | P delta = Delta]
   under delta ~ N(0, Sigma), computed by quadrature along the line
   {delta : P delta = Delta} (P from finite differences, independent of the
   closed form). Run in float64.

3. EXACT CLIPPED GABOR (complex-erf operator): with a half-space clip
   n.x <= tau active, the exact per-pixel factor is
       E[(1 + a cos xi) 1[omega <= tau] | Delta]
         = Phi(l) + a * exp(-sigma_xi^2/2) * Re{ e^{i m} * Phi_c(l, b) },
   Phi_c(l, b) = 0.5 * erfc(-(l - i b)/sqrt(2)),
   b = Cov(omega, xi | Delta)/s   (one new per-splat scalar),
   which must match numerical line integration of the modulated Gaussian with
   the indicator (integrated piecewise up to the exact plane crossing). The
   b = 0 special case reduces to the currently-shipped approximation
   Phi(l) * (1 + a_eff cos m).

CPU-only, no rasterizer needed:
    python scripts/tests/gabor_projection_check.py
"""
import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Camera with the exact 3DGS conventions (transposed storage, ndc2Pix).
def make_camera(width=64, height=48, fovx=0.8, fovy=0.6, seed=3):
    rng = np.random.default_rng(seed)
    # random look-at pose
    cam_pos = np.array([0.4, -0.3, -4.0]) + rng.normal(0, 0.2, 3)
    fwd = np.array([0.05, -0.08, 1.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    up2 = np.cross(fwd, right)
    R = np.stack([right, up2, fwd], axis=0)          # world->view rows
    t = -R @ cam_pos
    w2v = np.eye(4); w2v[:3, :3] = R; w2v[:3, 3] = t
    world_view = torch.tensor(w2v, dtype=torch.float64).transpose(0, 1)

    znear, zfar = 0.01, 100.0
    tanfovx, tanfovy = math.tan(fovx / 2), math.tan(fovy / 2)
    P = np.zeros((4, 4))
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    proj = torch.tensor(P, dtype=torch.float64).transpose(0, 1)
    full_proj = world_view @ proj
    return world_view, full_proj, tanfovx, tanfovy, width, height


def project_pixel(x_world, full_proj, W, H):
    """world point -> (px, py) exactly as the rasterizer does (ndc2Pix)."""
    hom = torch.cat([x_world, torch.ones_like(x_world[..., :1])], -1) @ full_proj
    ndc = hom[..., :3] / (hom[..., 3:4] + 1e-9)
    px = ((ndc[..., 0] + 1.0) * W - 1.0) * 0.5
    py = ((ndc[..., 1] + 1.0) * H - 1.0) * 0.5
    return torch.stack([px, py], -1)


def analytic_T(mu, world_view, tanfovx, tanfovy, W, H):
    """Mirror of the T construction in _gabor_tensors_for_raster (float64)."""
    focal_x = W / (2.0 * tanfovx)
    focal_y = H / (2.0 * tanfovy)
    t = mu @ world_view[:3, :3] + world_view[3, :3]
    tz = t[2].clamp_min(0.2)
    tx = (t[0] / tz).clamp(-1.3 * tanfovx, 1.3 * tanfovx) * tz
    ty = (t[1] / tz).clamp(-1.3 * tanfovy, 1.3 * tanfovy) * tz
    J = torch.zeros(3, 2, dtype=torch.float64)
    J[0, 0] = focal_x / tz
    J[1, 1] = focal_y / tz
    J[2, 0] = -focal_x * tx / (tz * tz)
    J[2, 1] = -focal_y * ty / (tz * tz)
    return world_view[:3, :3] @ J                     # [3,2]


def closed_form(mu, Sigma, k, phase, world_view, tanfovx, tanfovy, W, H, ks=0.0):
    """Mirror of the omega_2d / attenuation closed form (float64, 1 splat)."""
    T = analytic_T(mu, world_view, tanfovx, tanfovy, W, H)
    ST = Sigma @ T
    S2 = T.transpose(0, 1) @ ST
    a = S2[0, 0] + ks; b = S2[0, 1]; c = S2[1, 1] + ks
    det = a * c - b * b
    q = ST.transpose(0, 1) @ k
    wx = -(c * q[0] - b * q[1]) / det
    wy = -(a * q[1] - b * q[0]) / det
    kSk = k @ Sigma @ k
    var_ray = (kSk + q[0] * wx + q[1] * wy).clamp_min(0.0)
    atten = torch.exp(-0.5 * var_ray)
    return torch.stack([wx, wy]), atten, T


def main():
    torch.set_default_dtype(torch.float64)
    world_view, full_proj, tanfovx, tanfovy, W, H = make_camera()
    rng = np.random.default_rng(11)
    n_fail = 0

    for trial in range(8):
        # Sample splats well inside the frustum-clamp limits: at the periphery
        # computeCov2D clamps tx/ty (deliberately distorting the Jacobian), so
        # the analytic T — which mirrors that clamp — no longer matches an
        # unclamped finite-difference Jacobian. That is expected, not a bug.
        while True:
            mu = torch.tensor(rng.normal(0, 0.6, 3))
            tv = mu @ world_view[:3, :3] + world_view[3, :3]
            if (abs(tv[0] / tv[2]) < 0.8 * 1.3 * tanfovx
                    and abs(tv[1] / tv[2]) < 0.8 * 1.3 * tanfovy):
                break
        # random anisotropic covariance
        A = torch.tensor(rng.normal(0, 1, (3, 3)))
        s = torch.tensor(np.diag(rng.uniform(0.05, 0.35, 3)))
        Q, _ = torch.linalg.qr(A)
        L = Q @ s
        Sigma = L @ L.T
        # random wave vector, roughly 0.7..1.5 rad per whitened sigma
        u = torch.tensor(rng.normal(0, 1, 3)); u /= u.norm()
        w_u = float(rng.uniform(0.7, 1.5))
        k = torch.linalg.solve(L.T, u) * w_u          # k = L^-T u * w  => ||L^T k|| = w
        phase = float(rng.uniform(0, 2 * np.pi))

        # ---- check 1: T vs finite differences of the real projection ------
        T = analytic_T(mu, world_view, tanfovx, tanfovy, W, H)
        eps = 1e-6
        T_fd = torch.zeros(3, 2)
        for i in range(3):
            e = torch.zeros(3); e[i] = eps
            pp = project_pixel(mu + e, full_proj, W, H)
            pm = project_pixel(mu - e, full_proj, W, H)
            T_fd[i] = (pp - pm) / (2 * eps)
        err_T = (T - T_fd).abs().max() / T_fd.abs().max()
        ok_T = err_T < 1e-5

        # ---- check 2: closed form vs numerical conditional expectation ----
        omega2d, atten, _ = closed_form(mu, Sigma, k, phase, world_view,
                                        tanfovx, tanfovy, W, H, ks=0.0)
        P = T_fd.transpose(0, 1)                      # [2,3], independent of T
        center = project_pixel(mu, full_proj, W, H)
        # basis of the null space of P (the "ray" direction in delta space)
        _, _, Vh = torch.linalg.svd(P)
        nvec = Vh[2]                                  # P @ nvec ~ 0
        # min-norm particular solution delta0 with P delta0 = Delta
        Pinv = torch.linalg.pinv(P)
        Sinv = torch.linalg.inv(Sigma)

        max_err = 0.0
        for _ in range(12):
            d_pix = torch.tensor(rng.uniform(-4, 4, 2))     # pixel offset d = center - pixel
            Delta = -d_pix                                   # Delta = pixel - center
            delta0 = Pinv @ Delta
            ss = torch.linspace(-60, 60, 60001)
            pts = delta0.unsqueeze(0) + ss.unsqueeze(1) * nvec.unsqueeze(0)
            g = torch.exp(-0.5 * torch.einsum('ni,ij,nj->n', pts, Sinv, pts))
            cosv = torch.cos(pts @ k + phase)
            num = torch.trapz(g * cosv, ss)
            den = torch.trapz(g, ss)
            e_cos_num = num / den
            e_cos_cf = atten * torch.cos(omega2d @ d_pix + phase)
            max_err = max(max_err, float((e_cos_num - e_cos_cf).abs()))
        ok_E = max_err < 5e-9  # trapezoid quadrature floor at 60001 samples

        # ---- check 3: exact clipped Gabor factor (complex-erf operator) ----
        # Random plane meaningfully crossing the splat.
        from scipy.special import wofz, erfc as r_erfc
        nrm = torch.tensor(rng.normal(0, 1, 3)); nrm /= nrm.norm()
        sig_n = float(torch.sqrt(nrm @ Sigma @ nrm))
        tau = float(nrm @ mu) + float(rng.uniform(-1.0, 1.0)) * sig_n
        a_res = 0.5                                   # residual amplitude
        # Per-splat scalars (dilation ks=0 to match the FD/quadrature setup)
        S2 = P @ Sigma.numpy() @ P.T if False else None  # (kept torch below)
        T2 = T_fd                                     # [3,2], independent of closed form
        S2d = T2.transpose(0, 1) @ Sigma @ T2
        S2inv = torch.linalg.inv(S2d)
        q_g = T2.transpose(0, 1) @ (Sigma @ k)
        q_n = T2.transpose(0, 1) @ (Sigma @ nrm)
        sig_xi2 = float(k @ Sigma @ k - q_g @ S2inv @ q_g)
        s2 = float(nrm @ Sigma @ nrm - q_n @ S2inv @ q_n)
        s_c = math.sqrt(max(s2, 1e-30))
        c_ov = float(nrm @ Sigma @ k - q_n @ S2inv @ q_g)
        b_c = c_ov / s_c

        def phi_c(l, b):
            # 0.5*erfc(-(l-ib)/sqrt2), stable via Faddeeva w in the upper half
            zeta = -(l - 1j * b) / math.sqrt(2.0)
            if l <= 0:
                return 0.5 * np.exp(-zeta * zeta) * wofz(1j * zeta)
            return 1.0 - 0.5 * np.exp(-zeta * zeta) * wofz(-1j * zeta)

        max_err_c = 0.0
        for _ in range(12):
            d_pix = torch.tensor(rng.uniform(-4, 4, 2))
            Delta = -d_pix
            delta0 = Pinv @ Delta
            # closed form
            m_ph = float(omega2d @ d_pix + phase)
            e_omega = float(nrm @ mu + q_n @ S2inv @ Delta)
            l_c = (tau - e_omega) / s_c
            F_cf = 0.5 * r_erfc(-l_c / math.sqrt(2.0)) \
                + a_res * math.exp(-0.5 * sig_xi2) \
                * float(np.real(np.exp(1j * m_ph) * phi_c(l_c, b_c)))
            # numerical: adaptive line integral, with the plane crossing and
            # the Gaussian peak as explicit breakpoints (plain trapezoid loses
            # its spectral accuracy at the clipped endpoint).
            from scipy.integrate import quad
            Sinv_np = Sinv.numpy()
            d0_np = delta0.numpy(); v_np = nvec.numpy()
            k_np = k.numpy(); n_np = nrm.numpy(); mu_np = mu.numpy()

            def g_np(s):
                p = d0_np + s * v_np
                return np.exp(-0.5 * p @ Sinv_np @ p)

            def f_np(s):
                p = d0_np + s * v_np
                return g_np(s) * (1.0 + a_res * np.cos(p @ k_np + phase))

            vSv = v_np @ Sinv_np @ v_np
            s_peak = -(v_np @ Sinv_np @ d0_np) / vSv
            sig_line = 1.0 / math.sqrt(vSv)
            lo, hi = s_peak - 45 * sig_line, s_peak + 45 * sig_line
            n_dot_dir = float(n_np @ v_np)
            if abs(n_dot_dir) < 1e-12:
                segs = [(lo, hi)] if float(n_np @ (mu_np + d0_np)) <= tau else []
            else:
                s_star = (tau - float(n_np @ mu_np) - float(n_np @ d0_np)) / n_dot_dir
                s_star = min(max(s_star, lo), hi)
                segs = [(lo, s_star)] if n_dot_dir > 0 else [(s_star, hi)]
            num = sum(quad(f_np, aa, bb, points=[min(max(s_peak, aa), bb)],
                           epsabs=1e-14, epsrel=1e-12, limit=300)[0]
                      for aa, bb in segs if bb - aa > 1e-14)
            den = quad(g_np, lo, hi, points=[s_peak],
                       epsabs=1e-14, epsrel=1e-12, limit=300)[0]
            F_num = num / den
            max_err_c = max(max_err_c, abs(F_num - F_cf))
        ok_C = max_err_c < 5e-8

        print(f"[trial {trial}] |T-T_fd|rel={err_T:.2e} {'PASS' if ok_T else 'FAIL'}   "
              f"max|E_num-E_closed|={max_err:.2e} {'PASS' if ok_E else 'FAIL'}   "
              f"clipgabor|F_num-F_cf|={max_err_c:.2e} {'PASS' if ok_C else 'FAIL'}   "
              f"(b={b_c:+.3f}, atten={float(atten):.3f})")
        n_fail += (not ok_T) + (not ok_E) + (not ok_C)

    print("ALL PASS" if n_fail == 0 else f"{n_fail} FAILURES")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
