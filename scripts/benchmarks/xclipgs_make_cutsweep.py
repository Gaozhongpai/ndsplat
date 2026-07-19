#!/usr/bin/env python3
"""Build the <scene>_cutsweep nerf dataset for the SWEEP-FLICKER (popping) metric.

Per voxel axis we take ONE FIXED camera from the existing <scene>_cuteval set
(default: the dead-on PERP view, orbit_index 0) and emit K frames that differ
ONLY in clip_offset: the plane sweeps the middle half of the object's mass along
that axis (P25..P75 of the trained splat centres projected on the plane normal).
With the camera fixed there is ZERO parallax between consecutive frames -- the
true content change per step is identical across operators, so any EXCESS
frame-to-frame change is cull flicker (HC/ClipGS popping). No reference render
is needed. The perp (face-on) view is essential: the exposed cut face fills the
frame and its anatomy varies continuously with tau, so whole-splat pops stand
out; in a grazing view the face is edge-on and flicker drowns in the moving
silhouette sliver.

The frames reuse the grazing view's existing PNG via file_path (the reader opens
it only for width/height); renders are indexed by frame order, not file name.

Usage (ndgs container, /data = vengine_data):
  python scripts/benchmarks/xclipgs_make_cutsweep.py <scene> \
      --data-root /data --steps 41
"""
import argparse
import json
import os

import numpy as np
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--steps", type=int, default=41, help="tau steps per axis")
    ap.add_argument("--span", default="p25,p75",
                    help="sweep range as splat-projection percentiles lo,hi")
    ap.add_argument("--family", default="perp", choices=["perp", "graze"],
                    help="camera family for the fixed sweep view. PERP (dead-on, "
                         "the first perp frame per axis) is the right one for "
                         "popping: the exposed face fills the frame and anatomy "
                         "varies continuously with tau, so whole-splat pops stand "
                         "out; in a GRAZE view the face is edge-on and operator "
                         "flicker is buried in the moving silhouette sliver "
                         "(verified on heart: graze TV identical across methods).")
    args = ap.parse_args()

    nerf_src = os.path.join(args.data_root, "nerf_dataset", f"{args.scene}_cuteval")
    nerf_dst = os.path.join(args.data_root, "nerf_dataset", f"{args.scene}_cutsweep")
    ply_path = os.path.join(args.data_root, "output/xclipgs/ours",
                            f"{args.scene}_900/point_cloud/iteration_best/point_cloud.ply")

    tf = json.load(open(os.path.join(nerf_src, "transforms_test.json")))
    frames = sorted(tf["frames"], key=lambda f: f["file_path"])

    p = PlyData.read(ply_path)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float64)
    lo_q, hi_q = (float(s.strip("p")) for s in args.span.split(","))

    out_frames = []
    for axis in ("x", "y", "z"):
        fam = [f for f in frames
               if f.get("cut_axis") == axis and args.family in (f.get("mode") or "")]
        if not fam:
            print(f"  [warn] no {args.family} frame for axis {axis}; skip")
            continue
        base = fam[0]           # first per axis = orbit j=0 (dead-on for perp)
        n = np.asarray(base["plane_normal"], dtype=np.float64)
        n = n / np.linalg.norm(n)
        proj = xyz @ n
        t_lo, t_hi = np.percentile(proj, [lo_q, hi_q])
        taus = np.linspace(t_lo, t_hi, args.steps)
        print(f"  {axis}: n={np.round(n,3).tolist()} tau {t_lo:.2f}..{t_hi:.2f} "
              f"(cuteval tau {base['clip_offset']:.2f}) x{args.steps}")
        for k, tau in enumerate(taus):
            out_frames.append({
                "file_path": base["file_path"],             # size-only; renders are index-named
                "transform_matrix": base["transform_matrix"],
                "mode": "cut_sweep",
                "clip": True,
                "cut_axis": axis,
                "sweep_index": k,
                "plane_normal": [float(v) for v in n],
                "clip_offset": float(tau),
            })

    os.makedirs(nerf_dst, exist_ok=True)
    out = {"camera_angle_x": tf["camera_angle_x"], "frames": out_frames}
    json.dump(out, open(os.path.join(nerf_dst, "transforms_test.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(nerf_dst, "transforms_train.json"), "w"), indent=1)
    # reuse the cuteval images (sizes) + init points via relative symlinks
    for link, target in (("test", f"../{args.scene}_cuteval/test"),
                         ("points3d.ply", f"../{args.scene}_cuteval/points3d.ply")):
        lp = os.path.join(nerf_dst, link)
        if os.path.islink(lp) or os.path.exists(lp):
            if os.path.islink(lp):
                os.remove(lp)
        if not os.path.exists(lp):
            os.symlink(target, lp)
    print(f"wrote {len(out_frames)} sweep frames -> {nerf_dst}")


if __name__ == "__main__":
    main()
