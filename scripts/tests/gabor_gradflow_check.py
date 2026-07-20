#!/usr/bin/env python3
"""Gradient-flow + view-consistency checks for the projected Gabor band.

Works for both gabor model families:
    dgs-gabor  on a plain dgs checkpoint   (default)
    dbs-gabor  on a plain dbs-sh checkpoint  (GABOR_MODE=dbs-gabor GABOR_PLY=...)

With the world-space wave vector k projected per view in Python, the CUDA
grad_gabor must chain back through the projection into the k/phase/amp leaves:

  1. amp=0 (fresh warm start): dL/d(raw amp) nonzero (bootstrap through
     tanh' * attenuation), dL/dk and dL/d(phase) exactly zero (carry factor amp).
  2. amp=0.1: dL/dk and dL/d(phase) nonzero.
  3. The packed float4 buffer differs between two views for the same k
     (view-dependent omega_2d / attenuation).
  4. Whitened init magnitudes in [0.7, 1.5]; clamp_gabor_frequency keeps
     the whitened magnitude in [0.5, 3.0] after perturbation.

Run inside the ndgs container (heart data mounted at /data):
    python scripts/tests/gabor_gradflow_check.py
    GABOR_MODE=dbs-gabor GABOR_PLY=/data/.../point_cloud.ply \
        python scripts/tests/gabor_gradflow_check.py
"""
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import torch

DATA = "/data/nerf_dataset/heart_900"
MODE = os.environ.get("GABOR_MODE", "dgs-gabor")
if MODE == "dgs-gabor":
    PLY = os.environ.get(
        "GABOR_PLY",
        "/data/output/xclipgs/ours/heart_900/point_cloud/iteration_30000/point_cloud.ply")
    MODEL_KW = dict(input_dim=6, use_view_dependent_pos=False,
                    use_opacity_pos_decouple=False, l_22_inv_init_scale=2.0,
                    lambda_init=-1.2, lambda_opc=0.35)
else:
    PLY = os.environ["GABOR_PLY"]
    MODEL_KW = dict(input_dim=6, l_22_inv_init_scale=2.0)


def load_cameras():
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    args = SimpleNamespace(resolution=-1, data_device="cuda",
                           white_background=False, use_jpeg_compression=False)
    cam_infos = readCamerasFromTransforms(DATA, "transforms_test.json", False)
    return [loadCam(args, i, cam_infos[i], 1.0) for i in (0, 5)]


def packed_buffer(m, cam):
    """Build the per-view float4 buffer the way each model's render_tcgs does."""
    from scene.gaussian_model_gabor import project_gabor_band
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    if MODE == "dgs-gabor":
        dir_pp = m.get_xyz - cam.camera_center.repeat(m._xyz.shape[0], 1)
        cond = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        m_cond, _ = m.slice_gaussian_full_method(cond)
        scales, _, antialiasing = m.mip_filtered(m.get_opacity, scales=m.get_scaling)
        return m._gabor_tensors_for_raster(
            cam, means3D=m_cond, scales=scales, rotations=m.get_rotation,
            antialiasing=antialiasing, tanfovx=tanfovx, tanfovy=tanfovy)
    dir_pp = m._xyz - cam.camera_center.unsqueeze(0)
    query = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
    means, _ = m.get_cond_mean_opacity(query)
    return project_gabor_band(
        m._gabor_omega, m._gabor_phase, m._gabor_amp, cam,
        means[..., :3].contiguous(), m.get_covariance, antialiasing=False,
        tanfovx=tanfovx, tanfovy=tanfovy)


def main():
    from scene import get_gaussian_model

    cams = load_cameras()
    m = get_gaussian_model(MODE)(3, **MODEL_KW)
    m.load_ply(PLY)
    m.active_sh_degree = m.max_sh_degree
    m.background = torch.zeros(3, dtype=torch.float32, device="cuda")
    n_fail = 0
    print(f"[mode] {MODE}  N={m.get_xyz.shape[0]}  ply={PLY}")

    # -- 4a. whitened init range --------------------------------------------
    w = m.gabor_whitened_magnitude()
    ok = (w.min() >= 0.69) and (w.max() <= 1.51)
    print(f"[init] whitened |omega| in [{w.min():.3f}, {w.max():.3f}]  "
          f"{'PASS' if ok else 'FAIL'} (want [0.7, 1.5])")
    n_fail += not ok

    # -- 1. bootstrap at amp=0 ----------------------------------------------
    img = m.render_tcgs(cams[0], use_tcgs=False)["render"]
    img.mean().backward()
    g_amp = m._gabor_amp.grad.abs()
    g_k = m._gabor_omega.grad.abs().max().item() if m._gabor_omega.grad is not None else 0.0
    g_ph = m._gabor_phase.grad.abs().max().item() if m._gabor_phase.grad is not None else 0.0
    nz = int((g_amp > 0).sum())
    ok = nz > 0.5 * g_amp.numel() and g_k == 0.0 and g_ph == 0.0
    print(f"[grad@amp=0] dL/d(amp) nonzero on {nz}/{g_amp.numel()}  "
          f"max|dL/dk|={g_k:.1e} max|dL/dphase|={g_ph:.1e}  "
          f"{'PASS' if ok else 'FAIL'} (want many, 0, 0)")
    n_fail += not ok
    for p in (m._gabor_amp, m._gabor_omega, m._gabor_phase):
        p.grad = None

    # -- 2. omega/phase grads alive at amp=0.1 ------------------------------
    with torch.no_grad():
        m._gabor_amp.fill_(0.1)
    img = m.render_tcgs(cams[0], use_tcgs=False)["render"]
    img.mean().backward()
    g_k = m._gabor_omega.grad
    g_ph = m._gabor_phase.grad
    nk = int((g_k.abs().sum(1) > 0).sum())
    nph = int((g_ph.abs() > 0).sum())
    ok = nk > 0.5 * g_k.shape[0] and nph > 0.5 * g_ph.numel()
    print(f"[grad@amp=0.1] dL/dk nonzero rows {nk}/{g_k.shape[0]}, "
          f"dL/dphase nonzero {nph}/{g_ph.numel()}  {'PASS' if ok else 'FAIL'}")
    n_fail += not ok

    # -- 3. view dependence of the packed buffer ----------------------------
    with torch.no_grad():
        buf0 = packed_buffer(m, cams[0])
        buf1 = packed_buffer(m, cams[1])
    d_omega = (buf0[:, :2] - buf1[:, :2]).abs().max().item()
    d_amp = (buf0[:, 3] - buf1[:, 3]).abs().max().item()
    ok = d_omega > 1e-3 and d_amp > 1e-5
    print(f"[view-dep] max|omega2d(v0)-omega2d(v5)|={d_omega:.3f}  "
          f"max|amp_eff(v0)-amp_eff(v5)|={d_amp:.4f}  {'PASS' if ok else 'FAIL'}")
    n_fail += not ok

    # -- 4b. frequency clamp -------------------------------------------------
    with torch.no_grad():
        m._gabor_omega.mul_(10.0)  # push out of bounds
    m.clamp_gabor_frequency()
    w = m.gabor_whitened_magnitude()
    ok = (w.min() >= 0.499) and (w.max() <= 3.001)
    print(f"[clamp] whitened |omega| in [{w.min():.3f}, {w.max():.3f}]  "
          f"{'PASS' if ok else 'FAIL'} (want [0.5, 3.0])")
    n_fail += not ok

    print("ALL PASS" if n_fail == 0 else f"{n_fail} FAILURES")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
