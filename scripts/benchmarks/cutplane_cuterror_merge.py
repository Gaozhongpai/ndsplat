#!/usr/bin/env python3
"""Compute ONLY the GT-referenced cut-boundary mass error (hole / overshoot /
cut_error) and MERGE it into each scene's existing cutplane_results.json.

Why a separate script: the other metrics (band PSNR/SSIM/LPIPS, leak, edge width,
popping) do not depend on this and are expensive (LPIPS). This recomputes just the
three new keys from the already-rendered images (numpy+PIL only, no torch), and
patches results['methods'][m]['graze'] in place -- leaving every other value
untouched. Uses the SAME pooled-energy definition as cutplane_metrics.py:

  hole      = Σ_frames  Σ_{kept∩GTfg}  max(0, L_gt - L_method)   /  Σ L_gt(kept∩GTfg)
  overshoot = Σ_frames  Σ_{culled∩Mfg∩GTbg}  L_method            /  (same denominator)
  cut_error = hole + overshoot

Energies are POOLED across frames before the ratio (a per-frame ratio explodes on
thin grazing frames with tiny kept-side foreground -- the bug this script's design
avoids). Grazing views only; face-on frames skipped.

Usage:
  python scripts/benchmarks/cutplane_cuterror_merge.py \
      --gt-transforms <nerf>/<scene>_cuteval/transforms_test.json \
      --gt-dir <nerf>/<scene>_cuteval/test \
      --results <out>/cuteval/<scene>/cutplane_results.json \
      --methods ours=<...>/renders clipgs=<...> mm=<...> hc=<...> \
      --band-px 12
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

# reuse the exact geometry from the main metrics module
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cutplane_metrics as cm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-transforms", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--results", required=True, help="cutplane_results.json to patch")
    ap.add_argument("--methods", nargs="+", required=True, help="name=renders_dir")
    ap.add_argument("--band-px", type=float, default=12.0)
    args = ap.parse_args()

    tf = json.load(open(args.gt_transforms))
    fovx = float(tf["camera_angle_x"])
    frames = sorted(tf["frames"], key=lambda f: f["file_path"])
    methods = dict(m.split("=", 1) for m in args.methods)

    results = json.load(open(args.results))

    for mname, mdir in methods.items():
        hole_num = over_num = ref_den = 0.0
        leak_num = leak_den = 0.0
        edges = []
        n_used = 0
        for idx, fr in enumerate(frames):
            if not fr.get("clip") or "graze" not in (fr.get("mode") or ""):
                continue
            stem = os.path.basename(fr["file_path"])
            gtp = os.path.join(args.gt_dir, stem + ".png")
            rp = os.path.join(mdir, f"{idx:05d}.png")
            if not (os.path.isfile(gtp) and os.path.isfile(rp)):
                continue
            gt = cm.load_rgb(gtp); rimg = cm.load_rgb(rp)
            H, W = gt.shape[:2]
            fx = cm.fov2focal(fovx, W); fy = fx; cx, cy = W / 2.0, H / 2.0
            _t, _v, _s, geom = cm.plane_signed_distance_image(fr, W, H, fx, fy, cx, cy)
            signed_px, abc = cm.plane_line_distance(fr, W, H, fx, fy, cx, cy, geom)
            if math.hypot(abc[0], abc[1]) < 1e-6:      # face-on, no line
                continue
            gt_fg = cm.foreground_mask(gt); r_fg = cm.foreground_mask(rimg)
            Lg = gt.max(-1); Lr = rimg.max(-1)
            cband = np.abs(signed_px) <= args.band_px
            kept = (signed_px < 0.0) & cband
            culled = (signed_px > 0.0) & cband
            # cut error (holes + overshoot), pooled energies
            hole_num += float(np.clip(Lg - Lr, 0.0, None)[kept & gt_fg].sum())
            over_num += float(Lr[culled & r_fg & (~gt_fg)].sum())
            ref_den += float(Lg[kept & gt_fg].sum())
            # leak (same sign convention: culled = s>0), pooled energies
            near = np.abs(signed_px) <= 60.0
            leak_num += float(Lr[r_fg & (signed_px > 2.0) & (~gt_fg) & near].sum())
            leak_den += float(Lr[r_fg & near].sum())
            # edge spread (per-frame, averaged)
            ew = cm.edge_spread(rimg.mean(-1), signed_px, args.band_px)
            if ew is not None:
                edges.append(ew)
            n_used += 1

        g = results.setdefault("methods", {}).setdefault(mname, {}).setdefault("graze", {})
        if ref_den <= 0 or n_used == 0:
            g["hole"] = g["overshoot"] = g["cut_error"] = None
        else:
            g["hole"] = hole_num / ref_den
            g["overshoot"] = over_num / ref_den
            g["cut_error"] = g["hole"] + g["overshoot"]
        g["leak"] = (leak_num / leak_den) if leak_den > 0 else None
        g["edge_w"] = (float(np.mean(edges)) if edges else None)
        print(f"  {mname:8s} leak={g['leak']!r} edge={g['edge_w']!r} "
              f"cut_error={g['cut_error']!r}  ({n_used} graze frames)")

    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"patched {args.results}")


if __name__ == "__main__":
    main()
