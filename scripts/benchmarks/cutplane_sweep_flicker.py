#!/usr/bin/env python3
r"""SWEEP-FLICKER (popping) metric for the XClipGS operator ladder.

Setup: the <scene>_cutsweep set holds, per voxel axis, ONE FIXED grazing camera
and K frames whose ONLY difference is clip_offset (the plane sweeps the middle
half of the object's mass). With the camera fixed there is NO parallax: the true
content change between consecutive steps -- the thin slab of material between
plane t_k and t_{k+1} -- is IDENTICAL across operators. Any frame-to-frame change
beyond that slab is operator flicker: whole splats popping on/off as their
centres cross the threshold (HC, ClipGS's STE cull), which a smooth operator
(exact CDF truncation, moment fade) does not produce.

Per consecutive pair within an axis we take the luminance L1 difference,
normalized by the pair's mean foreground energy so it is scale-free:

    tv_k = sum |L_{k+1} - L_k|  /  mean(sum L_k, sum L_{k+1})

and report per method:
    flick_mean  -- mean tv over all pairs & axes  (the popping magnitude)
    flick_p95   -- 95th percentile               (pops are spiky; tail matters)
    flick_maxr  -- max_k tv_k / median_k tv_k    (spikiness: a smooth operator
                    has ~constant tv, a popping one has isolated jumps)

The renders are index-named (00000.png..) in TRANSFORMS FRAME ORDER (the nerf
reader does not sort), so frames are consumed in json order -- do NOT sort by
file_path here (sweep frames share one file_path).

Usage:
  python scripts/benchmarks/cutplane_sweep_flicker.py \
      --transforms /data/nerf_dataset/heart_cutsweep/transforms_test.json \
      --methods ours=<out>/ours_cutsweep/heart/test/ours_best/renders ... \
      --scene heart --out <out>/cuteval/heart
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image


def lum(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64).mean(-1) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="name=renders_dir")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tf = json.load(open(args.transforms))
    frames = tf["frames"]                      # JSON ORDER == render index order
    methods = dict(m.split("=", 1) for m in args.methods)

    # group consecutive render indices by axis (pairs only within an axis)
    axis_idx = defaultdict(list)
    for i, fr in enumerate(frames):
        axis_idx[fr["cut_axis"]].append(i)

    results = {"scene": args.scene, "methods": {}}
    for mname, mdir in methods.items():
        tvs = []
        per_axis = {}
        for axis, idxs in axis_idx.items():
            idxs = sorted(idxs)                # sweep_index order == frame order
            ax_tvs = []
            prev = None
            for i in idxs:
                p = os.path.join(mdir, f"{i:05d}.png")
                if not os.path.isfile(p):
                    print(f"  [warn] missing {p}")
                    prev = None
                    continue
                cur = lum(p)
                if prev is not None:
                    num = np.abs(cur - prev).sum()
                    den = 0.5 * (cur.sum() + prev.sum()) + 1e-9
                    ax_tvs.append(float(num / den))
                prev = cur
            per_axis[axis] = ax_tvs
            tvs.extend(ax_tvs)
        if not tvs:
            results["methods"][mname] = None
            continue
        tvs = np.array(tvs)
        med = float(np.median(tvs)) + 1e-12
        results["methods"][mname] = {
            "flick_mean": float(tvs.mean()),
            "flick_p95": float(np.percentile(tvs, 95)),
            "flick_max": float(tvs.max()),
            "flick_maxr": float(tvs.max() / med),
            "n_pairs": int(tvs.size),
            "per_axis_series": {a: [round(v, 6) for v in vs] for a, vs in per_axis.items()},
        }

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "sweep_flicker.json")
    json.dump(results, open(out_json, "w"), indent=1)

    print(f"\n=== sweep flicker: {args.scene} ===")
    print(f"{'method':10s} | {'mean↓':>8s} {'p95↓':>8s} {'max/med↓':>9s}")
    for mname in methods:
        r = results["methods"][mname]
        if r is None:
            print(f"{mname:10s} |     —")
            continue
        print(f"{mname:10s} | {r['flick_mean']:8.4f} {r['flick_p95']:8.4f} {r['flick_maxr']:9.2f}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
