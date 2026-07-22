#!/usr/bin/env python3
"""§1 Local-identifiability of dGS conditional coordinates (empirical).

The view-dependent maps are, per primitive, with delta_i = q_i - mu_q over the
TRAINING cameras that see it:
    mu_i        = mu_p + M delta_i               (linear in M [3xC])
    log(a_i/a)  = -lambda_o delta_i^T P delta_i  (linear in P in S^C)

Theorem (Jacobian-rank):
    M observable  <=>  span{delta_i}          = R^C
    P observable  <=>  span{delta_i delta_i^T} = S^C  (dim C(C+1)/2)

This script measures, on the real heart checkpoint + its 90 training cameras,
what fraction of primitives actually satisfy each condition — i.e. how much of
M and P is unobservable and can drift without changing training outputs. That
directly explains why the whitened / coupling / cca reparameterizations were
quality-null: they only reshape parameters, and the unobservable directions do
not render.

Coverage matrices per primitive:
    G1 = sum_i delta_i delta_i^T          [C,C]      (M observability: rank C)
    G2 = sum_i vech(dd^T) vech(dd^T)^T    [D,D], D=C(C+1)/2  (P observability)
Report the eigenvalue spectra and the effective rank (eigs > tol * max).

Run inside the ndgs container (heart data + checkpoint mounted at /data):
    python scripts/tests/dgs_identifiability.py
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import numpy as np
import torch

DATA = "/data/nerf_dataset/heart_900"
PLY = "/data/output/xclipgs/dgscoord/heart_900_baseline/point_cloud/iteration_30000/point_cloud.ply"
MODEL_KW = dict(input_dim=6, use_view_dependent_pos=True,
                use_opacity_pos_decouple=False, l_22_inv_init_scale=2.0,
                lambda_init=-1.2, lambda_opc=0.35)


def load_train_cams():
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    args = SimpleNamespace(resolution=-1, data_device="cuda",
                           white_background=False, use_jpeg_compression=False)
    infos = readCamerasFromTransforms(DATA, "transforms_train.json", False)
    return [loadCam(args, i, infos[i], 1.0) for i in range(len(infos))]


def main():
    from scene import get_gaussian_model
    m = get_gaussian_model("dgs")(3, **MODEL_KW)
    m.load_ply(PLY)
    xyz = m.get_xyz                                    # [N,3]
    mu_q = m.get_cond_mean                             # [N,3] normalized view mean
    N = xyz.shape[0]
    cams = load_train_cams()
    print(f"[setup] N={N} primitives, {len(cams)} training cameras")

    # Per-view query direction to every splat, then delta = q - mu_q.
    # Accumulate G1 = sum dd^T [N,3,3] and the P-feature Gram G2 [N,6,6].
    # vech order for symmetric 3x3: (00,01,02,11,12,22) with sqrt(2) on
    # off-diagonals so <P, dd^T>_F = vech(P).vech(dd^T).
    s2 = np.sqrt(2.0)
    G1 = torch.zeros(N, 3, 3, device="cuda")
    G2 = torch.zeros(N, 6, 6, device="cuda")
    n_seen = torch.zeros(N, device="cuda")
    # "sees it": splat in front of camera (positive depth) — a light visibility
    # proxy; the identifiability argument only needs the queries actually used.
    with torch.no_grad():
        for cam in cams:
            cpos = cam.camera_center.to("cuda")
            d = xyz - cpos.unsqueeze(0)
            q = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)
            # depth along view axis (world_view_transform is 3DGS-transposed:
            # column 2 of the rotation is the view direction)
            fwd = cam.world_view_transform[:3, 2].to("cuda")
            depth = d @ fwd
            vis = (depth > 0).float()
            delta = q - mu_q                            # [N,3]
            dd = torch.einsum('ni,nj->nij', delta, delta)   # [N,3,3]
            G1 += vis[:, None, None] * dd
            vech = torch.stack([dd[:, 0, 0], s2 * dd[:, 0, 1], s2 * dd[:, 0, 2],
                                dd[:, 1, 1], s2 * dd[:, 1, 2], dd[:, 2, 2]], dim=1)
            G2 += vis[:, None, None] * torch.einsum('ni,nj->nij', vech, vech)
            n_seen += vis

    def eff_rank(G, tol=1e-6):
        # Chunk the batched eigensolve (cuSOLVER batched syevd caps batch size).
        evs = []
        for c in torch.split(G, 20000, dim=0):
            evs.append(torch.linalg.eigvalsh(c.double()).float())
        ev = torch.cat(evs, dim=0).clamp_min(0)         # ascending
        mx = ev[:, -1:].clamp_min(1e-30)
        return (ev > tol * mx).sum(dim=1), ev

    r1, ev1 = eff_rank(G1)
    r2, ev2 = eff_rank(G2)

    def pct(x, k):
        return f"{float((x == k).float().mean()) * 100:.1f}%"

    print("\n== M observability (need span{delta} = R^3, rank 3) ==")
    for k in range(4):
        print(f"  rank(G1)={k}: {pct(r1, k)}")
    print(f"  mean effective rank: {float(r1.float().mean()):.3f} / 3")

    print("\n== P observability (need span{delta delta^T} = S^3, rank 6) ==")
    for k in range(7):
        print(f"  rank(G2)={k}: {pct(r2, k)}")
    print(f"  mean effective rank: {float(r2.float().mean()):.3f} / 6")

    # Conditioning: ratio of smallest to largest coverage eigenvalue (how
    # weakly the weakest observed direction is excited).
    cond1 = (ev1[:, 0].clamp_min(0) / ev1[:, -1].clamp_min(1e-30))
    cond2 = (ev2[:, 0].clamp_min(0) / ev2[:, -1].clamp_min(1e-30))
    q = lambda t, p: float(t.quantile(p))
    print(f"\n[G1 min/max eig ratio]  median={q(cond1,0.5):.2e}  p90={q(cond1,0.9):.2e}")
    print(f"[G2 min/max eig ratio]  median={q(cond2,0.5):.2e}  p90={q(cond2,0.9):.2e}")
    print(f"[views/primitive]       median={q(n_seen,0.5):.0f}  min={int(n_seen.min())}")

    frac_M = float((r1 == 3).float().mean())
    frac_P = float((r2 == 6).float().mean())
    print(f"\n[CONCLUSION] fully observable: M {frac_M*100:.1f}% of prims, "
          f"P {frac_P*100:.1f}% of prims.")
    print("The complement is unobservable parameter mass — free to drift "
          "without changing training renders, i.e. why the reparameterizations "
          "(whitened / cca) and the coupling penalty were quality-neutral.")


if __name__ == "__main__":
    main()
