#!/usr/bin/env python3
r"""Operator-swap render of the trained XClipGS (ours, dGS) checkpoint through the
RaRa Clipper operator, using the RaRa AUTHORS' RELEASED CUDA kernel (diff_gauss).

This is the RaRa row of the operator-swap comparison: the SAME trained interior that
Ours/MM/HC render (the ours _900 dGS checkpoint), the SAME cut-eval cameras and
planes, differing only in the render-time clip rule. Because RaRa is a render-time
operator on a fixed cloud, we drive their kernel directly with the per-view
conditioned primitives our dGS model produces (position conditioned on view dir,
opacity scaled by the view-dependent factor) so RaRa sees the identical primitives
every other operator sees; their kernel does SH->RGB and applies its chord-ratio
decay. Scale and rotation are not view-conditioned in dGS, matching what RaRa needs.

Plane sign map: our convention keeps n.x <= tau; RaRa keeps n.x + d > 0. So we pass
their clipper as (normal, d) = (-n, tau).

Their kernel is built (patched: decay_weight inits to 1.0 = keep, the method's
intended semantics; the released init-0 silently suppressed near-plane ray-misses)
into an isolated prefix; point PYTHONPATH there ahead of any stock diff_gauss.

Usage (ndgs container, RARA_PREFIX on PYTHONPATH):
  python scripts/benchmarks/render_rara.py <scene> <suffix> \
      --data-root /data --out-root /data/output/xclipgs
    <suffix> = cuteval (clipped) or cutevalfull (no clip)
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa
from scene import Scene, get_gaussian_model  # noqa


def build_clippers(cam, device, dtype):
    """RaRa clippers tensor [K,4] = (normal, d) for keep n.x + d > 0.
    Our camera.clip_plane is (nx, ny, nz, tau) keeping n.x <= tau. Map to
    RaRa's convention by (normal, d) = (-n, tau): (-n).x + tau > 0 <=> n.x < tau."""
    cp = getattr(cam, "clip_plane", None)
    if cp is None:
        return 0, torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
    n = np.asarray(cp[:3], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    tau = float(cp[3])
    row = [-n[0], -n[1], -n[2], tau]
    return 1, torch.tensor([row], device=device, dtype=dtype)


def dgs_conditioned(gaussians, cam):
    """Reproduce exactly what gaussian_model_dgs.render_tcgs feeds the rasterizer:
    view-conditioned position m_cond + view-scaled opacity, unconditioned scale/rot,
    SH features. Returns (means3D, opacity, scales, rotations, shs)."""
    xyz = gaussians.get_xyz
    dir_pp = xyz - cam.camera_center.repeat(xyz.shape[0], 1)
    mean_view = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    if getattr(gaussians, "input_dim", 6) == 7:
        ts = torch.full((mean_view.shape[0], 1),
                        getattr(cam, "timestamp", 0.0),
                        device=mean_view.device, dtype=mean_view.dtype)
        cond = torch.cat([mean_view, ts], dim=-1)
    else:
        cond = mean_view
    m_cond, opacity_scale = gaussians.slice_gaussian_full_method(cond)
    opacity = gaussians.get_opacity * opacity_scale
    scales = gaussians.get_scaling
    # Mip-Splatting 3D filter, matching the ours render path (no-op if filter_3D None)
    scales, opacity, _ = gaussians.mip_filtered(opacity, scales=scales)
    return m_cond, opacity, scales, gaussians.get_rotation, gaussians.get_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("suffix", choices=["cuteval", "cutevalfull"])
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--out-root", default="/data/output/xclipgs")
    ap.add_argument("--iteration", default="best")
    # Which trained interior to render RaRa on (operator-swap): "ours" (default) or
    # "hc". --out-tag names the output tree (e.g. "rara" or "raraHC") so a swap on a
    # different interior never clobbers the ours-interior results.
    ap.add_argument("--interior", default="ours")
    ap.add_argument("--out-tag", default="rara")
    args = ap.parse_args()

    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer  # patched RaRa build

    scene_name = args.scene
    nerf = os.path.join(args.data_root, "nerf_dataset", f"{scene_name}_{args.suffix}")
    trained = os.path.join(args.out_root, args.interior, f"{scene_name}_900")
    out_dir = os.path.join(args.out_root, f"{args.out_tag}_{args.suffix}", scene_name,
                           "test", "ours_best", "renders")
    os.makedirs(out_dir, exist_ok=True)

    # Load the ours dGS model + its trained cfg, but point source_path at the
    # cut-eval dataset (same trick as render_cutx.sh).
    from argparse import Namespace  # noqa: used by eval of the cfg_args repr
    cfg = eval(open(os.path.join(trained, "cfg_args")).read())
    cfg.source_path = nerf
    cfg.model_path = trained
    cfg.eval = True

    GaussianModel = get_gaussian_model(cfg.mode)
    gaussians = GaussianModel(
        cfg.sh_degree, input_dim=cfg.input_dim,
        use_view_dependent_pos=cfg.use_view_dependent_pos,
        use_opacity_pos_decouple=getattr(cfg, "use_opacity_pos_decouple", False),
        l_22_inv_init_scale=getattr(cfg, "l_22_inv_init_scale", 1.0),
        lambda_init=getattr(cfg, "lambda_init", -1.2),
        lambda_opc=getattr(cfg, "lambda_opc", 0.35),
    )
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False,
                  load_train_cameras=False, load_test_cameras=True)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    cams = scene.getTestCameras()
    import math
    with torch.no_grad():
        for idx, cam in enumerate(cams):
            m_cond, opacity, scales, rots, shs = dgs_conditioned(gaussians, cam)
            n_clips, clippers = build_clippers(cam, m_cond.device, m_cond.dtype)
            rs = GaussianRasterizationSettings(
                image_height=int(cam.image_height),
                image_width=int(cam.image_width),
                tanfovx=math.tan(cam.FoVx * 0.5),
                tanfovy=math.tan(cam.FoVy * 0.5),
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=cam.world_view_transform,
                projmatrix=cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=cam.camera_center,
                prefiltered=False,
                debug=False,
                rr_clipping=(n_clips > 0),      # RaRa cutoff classification
                rr_strategy=True,               # the RaRa chord-ratio strategy
                oenD_gs_strategy=False,         # default (not the 1D-GS variant)
                n_clips=n_clips,
                clipprs=clippers,
                vizPlane=False,
                n_inters=0,
                intersections_tensor=torch.zeros(3, device="cuda"),
            )
            rasterizer = GaussianRasterizer(raster_settings=rs)
            means2D = torch.zeros_like(m_cond, requires_grad=False)
            img, _, _, _ = rasterizer(
                means3D=m_cond, means2D=means2D, opacities=opacity, shs=shs,
                colors_precomp=None, scales=scales, rotations=rots, cov3D_precomp=None)
            arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(out_dir, f"{idx:05d}.png"))
    print(f"RaRa: {len(cams)} renders ({args.suffix}) -> {out_dir}")


if __name__ == "__main__":
    main()
