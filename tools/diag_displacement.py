"""Displacement diagnostic (NeurIPS 2026 rebuttal, orhG Q3). No retraining.

For trained N-DGS and dGS checkpoints, evaluates test queries and reports the
distribution of size-normalized position displacement per primitive:

    iso  = ||delta_mu||_2 / s_bar            (isotropic normalization)
    ani  = ||S^{-1} R^T delta_mu||_2         (anisotropy-aware, primitive frame)

reported over (a) all primitives and (b) opacity-weighted primitives, with
median / p90 / p95 / p99 and the Pearson correlation between ||delta_mu|| and
s_bar. Internal diagnostic first: include in the discussion only if stable.

Usage (inside the ndsplat container):
  python tools/diag_displacement.py \
      --ndgs output/mcmc/ndgs/nerf_synthetic/lego \
      --dgs  output/mcmc/dgs/nerf_synthetic/lego \
      --source /code/dataset/nerf_synthetic/lego -w --views 8
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams
from scene import Scene, get_gaussian_model
from utils.general_utils import safe_state, build_scaling_rotation


def load(model_path, mode, source, white_bg, sh, dim, iteration):
    parser = argparse.ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    args.model_path, args.source_path, args.mode = model_path, source, mode
    args.sh_degree, args.input_dim, args.white_background = sh, dim, white_bg
    args.eval, args.data_device, args.resolution, args.images = True, "cuda", -1, "images"
    dataset = lp.extract(args)
    g = get_gaussian_model(mode)(sh, dim)
    scene = Scene(dataset, g, load_iteration=iteration, shuffle=False)
    return g, scene


def displacement(gaussians, cam, mode):
    """delta_mu for one camera, [N,3]."""
    dirs = gaussians.get_xyz - cam.camera_center.unsqueeze(0)
    q = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if getattr(gaussians, "input_dim", 6) == 7:
        t = float(getattr(cam, "timestamp", 0.0) or 0.0)
        q = torch.cat([q, torch.full((q.shape[0], 1), t, device=q.device)], dim=-1)
    with torch.no_grad():
        if "ndgs" in mode:
            lo = gaussians.get_lambda_opc.squeeze(-1)
            lot = (gaussians.get_lambda_opc_time.squeeze(-1)
                   if getattr(gaussians, "input_dim", 6) == 7 else None)
            m_cond, _, opa = gaussians.slice_gaussian(q, c_dim=3, lambda_opc=lo, lambda_opc_time=lot)
        else:
            m_cond, opa = gaussians.slice_gaussian_full_method(q)
    return m_cond - gaussians.get_xyz, opa.squeeze(-1)


def stats(x, w=None, label=""):
    x = x.flatten()
    if w is not None:
        idx = torch.argsort(x)
        xs, ws = x[idx], w.flatten()[idx]
        c = torch.cumsum(ws, 0) / ws.sum()
        qs = [xs[torch.searchsorted(c, torch.tensor(p, device=x.device))].item()
              for p in (0.5, 0.9, 0.95, 0.99)]
    else:
        qs = [torch.quantile(x, p).item() for p in (0.5, 0.9, 0.95, 0.99)]
    print(f"    {label:<20} median={qs[0]:8.4f}  p90={qs[1]:8.4f}  p95={qs[2]:8.4f}  p99={qs[3]:8.4f}")
    return qs


def run(tag, path, mode, args):
    g, scene = load(path, mode, args.source, args.white_background,
                    args.sh_degree, args.input_dim, args.iteration)
    cams = (scene.getTestCameras() or scene.getTrainCameras())[:args.views]
    # Model-agnostic principal frame: eigendecompose each model's spatial
    # covariance (dGS: R S S^T R^T from scaling/rotation; N-DGS: the [0:3,0:3]
    # block of its joint covariance). s_i = sqrt(eigenvalues), R = eigenvectors.
    with torch.no_grad():
        if "ndgs" in mode:
            cov = g.get_pc_v[:, :3, :3]
        else:
            L = build_scaling_rotation(g.get_scaling, g._rotation)
            cov = L @ L.transpose(1, 2)
        # cusolver's batched syev rejects some large 3x3 batches; CPU eigh on
        # [N,3,3] takes seconds and is robust.
        evals, evecs = torch.linalg.eigh(cov.double().cpu())
        evals, evecs = evals.float().cuda(), evecs.float().cuda()
        s = evals.clamp_min(1e-12).sqrt()              # [N,3] principal stds
    sbar = s.mean(dim=1)                               # [N]
    R = evecs                                          # [N,3,3]
    Sinv = 1.0 / s.clamp_min(1e-8)                     # [N,3]

    iso_all, ani_all, w_all, dn_all = [], [], [], []
    for cam in cams:
        dmu, opa = displacement(g, cam, mode)
        dn = dmu.norm(dim=1)
        iso_all.append(dn / sbar.clamp_min(1e-8))
        local = torch.bmm(R.transpose(1, 2), dmu.unsqueeze(-1)).squeeze(-1)  # R^T dmu
        ani_all.append((Sinv * local).norm(dim=1))
        w_all.append(opa)
        dn_all.append(dn)
    iso = torch.cat(iso_all); ani = torch.cat(ani_all)
    w = torch.cat(w_all); dn = torch.cat(dn_all)
    sb = sbar.repeat(len(cams))

    print(f"\n== {tag} ({mode})  N={g.get_xyz.shape[0]:,}  views={len(cams)} ==")
    print("  all primitives:")
    stats(iso, None, "||dmu||/s_bar")
    stats(ani, None, "||S^-1 R^T dmu||")
    print("  opacity-weighted:")
    stats(iso, w, "||dmu||/s_bar")
    stats(ani, w, "||S^-1 R^T dmu||")
    c = torch.corrcoef(torch.stack([dn, sb]))[0, 1].item()
    print(f"    corr(||dmu||, s_bar) = {c:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndgs", required=True)
    ap.add_argument("--dgs", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("-w", "--white_background", action="store_true")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--input_dim", type=int, default=6)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--views", type=int, default=8)
    args = ap.parse_args()
    safe_state(False, 0); sys.stdout = sys.__stdout__
    print(f"Device: {torch.cuda.get_device_name(0)}")
    run("N-DGS", args.ndgs, "ndgs", args)
    run("dGS", args.dgs, "dgs", args)
