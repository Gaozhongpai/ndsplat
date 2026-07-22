#!/usr/bin/env python3
"""§4 Coupling-invariant + shift-magnitude diagnostics for the dGS variants.

Since the identifiability probe showed full observability, the quality-null
must come from expressive equivalence + a SMALL learned view-shift. This
quantifies the shift on each trained checkpoint:

  Coupling invariant (dimensionless, coordinate-invariant):
      K = S^{-1/2} M P^{1/2},   ||K||_F^2 = E_{delta~N(0,P^-1)} ||S^{-1/2} M delta||^2
      (expected squared displacement measured in the primitive's own spatial
       std units). For dGS, M = v_12 D_Lambda P, so K = S^{-1/2} v_12 D_Lambda P^{1/2}.

  Physical shift over the actual training views:
      per primitive, RMS ||M delta_i|| (world units) and its ratio to the
      primitive's mean spatial std sqrt(mean(scale^2)). If this ratio is <<1
      the view shift is sub-footprint -> reparameterizing it cannot move PSNR.

Compares baseline vs coupling(eta=0.01) vs whitened checkpoints, so it also
answers "how much coupling energy did the regularizer actually remove, and at
what PSNR cost".

Run inside the ndgs container:
    python scripts/tests/dgs_coupling_diag.py
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import numpy as np
import torch

DATA = "/data/nerf_dataset/heart_900"
ROOT = "/data/output/xclipgs/dgscoord"
RUNS = ["baseline", "coupling", "whitened"]
MODEL_KW = dict(input_dim=6, use_view_dependent_pos=True,
                use_opacity_pos_decouple=False, l_22_inv_init_scale=2.0,
                lambda_init=-1.2, lambda_opc=0.35)


def load_train_cams():
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    args = SimpleNamespace(resolution=-1, data_device="cuda",
                           white_background=False, use_jpeg_compression=False)
    infos = readCamerasFromTransforms(DATA, "transforms_train.json", False)
    return [loadCam(args, i, infos[i], 1.0) for i in range(0, len(infos), 8)]  # subsample cams for speed


def analyze(mode, name, cams):
    from scene import get_gaussian_model
    from scene.gaussian_model_dgs_whitened import unpack_L
    m = get_gaussian_model(mode)(3, **MODEL_KW)
    m.load_ply(f"{ROOT}/heart_900_{name}/point_cloud/iteration_30000/point_cloud.ply")
    with torch.no_grad():
        L = unpack_L(m.get_L_22_inv)                       # [N,3,3]
        P = L @ L.transpose(1, 2)
        S = m.get_scaling                                  # [N,3]
        S_ihalf = S.rsqrt()                                # diag S^{-1/2}
        lam = m.lambda_activation(m._lambda_view)          # [N]
        v12 = m.get_v_12.reshape(-1, 3, 3)                 # already S-scaled+normalized
        # M = lam * v12 @ P   (the effective regression matrix)
        M = lam[:, None, None] * torch.einsum('nij,njk->nik', v12, P)
        # coupling invariant K = S^{-1/2} M P^{1/2}; ||K||_F^2 = tr(S^-1 M P M^T)
        Pinv = torch.linalg.inv(P + 1e-8 * torch.eye(3, device="cuda"))
        # E||S^-1/2 M delta||^2, delta~N(0,P^-1): = tr(S^-1 M P^-1 M^T)
        MSinv = S_ihalf[:, :, None] * M                    # S^{-1/2} M  [N,3,3]
        Kfro2 = torch.einsum('nij,njk,nik->n', MSinv, Pinv, MSinv)
        Kfro = Kfro2.clamp_min(0).sqrt()

        # physical RMS shift over actual training views, ratio to footprint
        mu_q = m.get_cond_mean
        xyz = m.get_xyz
        sq_shift = torch.zeros(xyz.shape[0], device="cuda")
        nv = 0
        for cam in cams:
            cpos = cam.camera_center.to("cuda")
            d = xyz - cpos.unsqueeze(0)
            q = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)
            delta = q - mu_q
            shift = torch.einsum('nij,nj->ni', M, delta)
            sq_shift += (shift * shift).sum(1)
            nv += 1
        rms_shift = (sq_shift / nv).sqrt()                 # world units
        foot = (S ** 2).mean(1).sqrt()                     # mean spatial std
        ratio = rms_shift / foot.clamp_min(1e-8)

    q = lambda t, p: float(t.quantile(p))
    return dict(
        Kfro_med=q(Kfro, 0.5), Kfro_p90=q(Kfro, 0.9),
        shift_ratio_med=q(ratio, 0.5), shift_ratio_p90=q(ratio, 0.9),
        lam_med=q(lam, 0.5), lam_p90=q(lam, 0.9),
        n=xyz.shape[0])


def main():
    cams = load_train_cams()
    print(f"[setup] {len(cams)} sampled training cameras\n")
    modes = {"baseline": "dgs", "coupling": "dgs", "whitened": "dgs-white"}
    print(f"{'run':10s} {'||K||_F med':>12s} {'p90':>8s} "
          f"{'shift/foot med':>15s} {'p90':>8s} {'lam med':>9s}")
    rows = {}
    for name in RUNS:
        r = analyze(modes[name], name, cams)
        rows[name] = r
        print(f"{name:10s} {r['Kfro_med']:12.4f} {r['Kfro_p90']:8.4f} "
              f"{r['shift_ratio_med']:15.4f} {r['shift_ratio_p90']:8.4f} "
              f"{r['lam_med']:9.4f}")
    b, c = rows["baseline"], rows["coupling"]
    print(f"\n[coupling regularizer effect] median ||K||_F: "
          f"{b['Kfro_med']:.4f} -> {c['Kfro_med']:.4f} "
          f"({100*(c['Kfro_med']/b['Kfro_med']-1):+.1f}%)")
    print(f"[interpretation] shift/footprint median ~{b['shift_ratio_med']:.3f}: "
          f"the view-dependent position move is this fraction of a splat's own "
          f"size. If <<1, reparameterizing it cannot move PSNR (the real reason "
          f"for the quality-null, given full observability).")


if __name__ == "__main__":
    main()
