#!/usr/bin/env python3
r"""CDE -- difference-referenced cut error (the repaired CErr).

The original cut error (hole+overshoot vs the clipped reference) is dominated by a
reconstruction floor shared by all operators: `hole` counts ANY render-vs-GT
deficit in the band, so near-exact operators are indistinguishable (Ours == HC on
average), and the halves reward opposite failure modes (leaked material FILLS
holes; kept-mass fading TRIMS overshoot).

CDE removes the floor by comparing CHANGE MAPS instead of images. Render every
method (and the volume reference) at the SAME cameras twice -- with and without
the clip -- and compare what the operator REMOVED against what SHOULD have been
removed:

    D_R = lum(R_full) - lum(R_clip)          (the method's removal map, signed)
    D_G = lum(G_full) - lum(G_clip)          (the reference removal map)

PRIMARY (mask CDE): compare the BINARY affected regions, not energies --
    A_R = |D_R| > thr ,  A_G = |D_G| > thr      (pixels the cut visibly affects)
    under = |A_G \ A_R| -- the cut should have affected here, method's didn't
                           (material left standing = leak-like)
    over  = |A_R \ A_G| -- method's cut affected where reference's didn't
                           (extra material removed = hole-like)
    CDE   = (under + over) / |A_G|
Binary masks are the crucial step: where BOTH cuts removed material, a brightness
mismatch of the removed content (reconstruction error) no longer counts -- only
disagreement about WHERE the cut acts does. The raw energy version
(sum|D_R - D_G| / sum|D_G|, also reported as cde_energy) is floor-dominated: on
the culled side both clip renders are empty, so D_R - D_G = R_full - G_full,
plain reconstruction error (verified on heart: energy-CDE 0.51 for every
plane-respecting operator; mask-CDE separates Ours .215 < MM .224 < HC .225 <<
ClipGS .325 at thr=0.04).

Reconstruction error common to a method's two renders cancels exactly (identical
checkpoint, identical camera); under-removal can no longer be paid for by
leaking, nor over-removal by fading.

Pooled numerators/denominators across frames, one ratio at the end (per-frame
ratios explode on thin-band frames -- same pooling as cutplane_metrics).

Families: graze (near-plane band |s|<=band_px, primary -- the cut edge) and
perp (foreground union, the whole exposed face).

Usage:
  python scripts/benchmarks/cutplane_cde.py \
      --gt-transforms /data/nerf_dataset/<s>_cuteval/transforms_test.json \
      --gt-clip-dir   /data/nerf_dataset/<s>_cuteval/test \
      --gt-full-dir   /data/nerf_dataset/<s>_cutevalfull/test \
      --methods ours=<out>/ours_cuteval/<s>/...:<out>/ours_cutevalfull/<s>/... \
      --scene <s> --out <out>/cuteval/<s> --band-px 12 [--debug]
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

# reuse the audited plane-projection geometry from cutplane_metrics.py
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cpm", os.path.join(_here, "cutplane_metrics.py"))
cpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpm)


def lum(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64).mean(-1) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-transforms", required=True)
    ap.add_argument("--gt-clip-dir", required=True)
    ap.add_argument("--gt-full-dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name=<clip_renders_dir>:<full_renders_dir>")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band-px", type=float, default=12.0)
    ap.add_argument("--fg-thr", type=float, default=0.04)
    ap.add_argument("--mask-thr", type=float, default=0.04,
                    help="|removal| threshold defining the affected-region masks")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tf = json.load(open(args.gt_transforms))
    fovx = float(tf["camera_angle_x"])
    frames = sorted(tf["frames"], key=lambda f: f["file_path"])  # == render index order

    methods = {}
    for m in args.methods:
        name, dirs = m.split("=", 1)
        cdir, fdir = dirs.split(":", 1)
        methods[name] = (cdir, fdir)

    # acc[method][family][key] pooled energies
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for idx, fr in enumerate(frames):
        if not fr.get("clip"):
            continue
        stem = os.path.basename(fr["file_path"])
        gclip_p = os.path.join(args.gt_clip_dir, stem + ".png")
        gfull_p = os.path.join(args.gt_full_dir, stem + ".png")
        if not (os.path.isfile(gclip_p) and os.path.isfile(gfull_p)):
            print(f"  [warn] missing GT pair for {stem}; skip")
            continue
        Gc, Gf = lum(gclip_p), lum(gfull_p)
        H, W = Gc.shape
        fx = cpm.fov2focal(fovx, W); fy = fx
        cx, cy = W / 2.0, H / 2.0
        family = "graze" if "graze" in (fr.get("mode") or "") else "perp"

        _t, _v, _s, geom = cpm.plane_signed_distance_image(fr, W, H, fx, fy, cx, cy)
        signed_px, line_abc = cpm.plane_line_distance(fr, W, H, fx, fy, cx, cy, geom)
        face_on = math.hypot(line_abc[0], line_abc[1]) < 1e-6

        DG = Gf - Gc
        AG = np.abs(DG) > args.mask_thr
        if family == "graze" and not face_on:
            band = np.abs(signed_px) <= args.band_px
        else:
            band = np.ones((H, W), dtype=bool)

        for mname, (cdir, fdir) in methods.items():
            rc_p = os.path.join(cdir, f"{idx:05d}.png")
            rf_p = os.path.join(fdir, f"{idx:05d}.png")
            if not (os.path.isfile(rc_p) and os.path.isfile(rf_p)):
                print(f"  [warn] missing render pair {mname} idx {idx}; skip")
                continue
            Rc, Rf = lum(rc_p), lum(rf_p)
            if Rc.shape != (H, W):
                continue
            DR = Rf - Rc
            AR = np.abs(DR) > args.mask_thr
            a = acc[mname][family]
            # primary: affected-region symmetric difference (appearance-robust)
            a["m_under"] += float((AG & ~AR & band).sum())
            a["m_over"] += float((AR & ~AG & band).sum())
            a["m_den"] += float((AG & band).sum())
            # secondary: raw energy difference (floor-dominated; kept for reference)
            diff = DR - DG
            a["under"] += float(np.clip(-diff, 0, None)[band].sum())
            a["over"] += float(np.clip(diff, 0, None)[band].sum())
            a["den"] += float(np.abs(DG)[band].sum())

            if args.debug and fr.get("cut_axis") and idx % 5 == 0:
                # removal-map triptych: reference | method | (method - reference)
                def viz(x):
                    v = np.clip(np.abs(x) * 3.0, 0, 1)
                    rgb = np.zeros((H, W, 3))
                    rgb[..., 0] = np.where(x < 0, v, 0)   # red = negative (reveal)
                    rgb[..., 1] = np.where(x >= 0, v, 0)  # green = positive (removed)
                    return rgb
                trip = np.concatenate([viz(DG), viz(DR), viz(diff)], axis=1)
                trip[np.tile(~band, (1, 3))] *= 0.25
                Image.fromarray((trip * 255).astype(np.uint8)).save(
                    os.path.join(args.out, f"cde_{mname}_{fr['cut_axis']}_{family}_{idx:05d}.png"))

    results = {"scene": args.scene, "band_px": args.band_px,
               "mask_thr": args.mask_thr, "methods": {}}
    for mname in methods:
        entry = {}
        for fam in ("graze", "perp"):
            a = acc[mname][fam]
            if a["m_den"] > 0:
                entry[fam] = {
                    "cde": (a["m_under"] + a["m_over"]) / a["m_den"],
                    "under": a["m_under"] / a["m_den"],
                    "over": a["m_over"] / a["m_den"],
                    "cde_energy": (a["under"] + a["over"]) / max(a["den"], 1e-9),
                }
            else:
                entry[fam] = None
        results["methods"][mname] = entry

    out_json = os.path.join(args.out, "cde_results.json")
    json.dump(results, open(out_json, "w"), indent=1)

    print(f"\n=== CDE (difference-referenced cut error): {args.scene} ===")
    print(f"{'method':10s} | {'graze CDE↓':>10s} {'under↓':>8s} {'over↓':>8s} | {'perp CDE↓':>10s}")
    for mname in methods:
        e = results["methods"][mname]
        g, p = e["graze"], e["perp"]
        def f(x, d=4): return f"{x:.{d}f}" if x is not None else "   —"
        print(f"{mname:10s} | {f(g['cde']) if g else '—':>10s} {f(g['under']) if g else '—':>8s} "
              f"{f(g['over']) if g else '—':>8s} | {f(p['cde']) if p else '—':>10s}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
