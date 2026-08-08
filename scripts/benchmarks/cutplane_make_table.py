#!/usr/bin/env python3
"""Aggregate the per-scene cut-plane-eval results into a comparison table.

Reads <out_root>/<eval_name>/<scene>/cutplane_results.json (written by
cutplane_metrics.py) for every scene and emits a Markdown + CSV table across the
four methods {ours, clipgs, mm, hc}, with a per-method average row.

Metrics reported (see cutplane_metrics.py for definitions):
  perp band-PSNR/SSIM/LPIPS   -- cut-face surface fidelity vs GT (head-on views)
  graze band-PSNR             -- cut-edge fidelity vs GT (edge-on views)
  leak                        -- spurious energy past the plane (lower better)
  edge_w                      -- 10-90% rise width of the cut edge, px (lower=sharper)
  pop                         -- popping over the grazing orbit (lower better)

Usage:
  python scripts/benchmarks/cutplane_make_table.py \
      --out-root /data/output/xclipgs --eval-name cuteval
"""
import argparse
import csv
import json
from pathlib import Path

METHODS = [("ours", "Ours (analytic)"),
           ("clipgs", "ClipGS (reimpl)"),
           ("mm", "MM (moment)"),
           ("hc", "HC (hard cull)")]

# (key, label, higher_is_better, getter path in results['methods'][m])
METRICS = [
    ("perp_psnr",  "perp band-PSNR",  True,  ("perp", "band_psnr")),
    ("perp_ssim",  "perp band-SSIM",  True,  ("perp", "band_ssim")),
    ("perp_lpips", "perp band-LPIPS", False, ("perp", "band_lpips")),
    ("graze_psnr", "graze band-PSNR", True,  ("graze", "band_psnr")),
    ("leak",       "leak",            False, ("graze", "leak")),
    ("edge_w",     "edge spread px",  False, ("graze", "edge_w")),
    ("hole",       "hole",            False, ("graze", "hole")),
    ("overshoot",  "overshoot",       False, ("graze", "overshoot")),
    ("cut_error",  "cut error (hole+overshoot)", False, ("graze", "cut_error")),
    # improved metrics (cutplane_cde.py / cutplane_sweep_flicker.py)
    ("cde",        "CDE (diff-referenced cut error)", False, ("cde", "graze", "cde")),
    ("cde_under",  "CDE under-removal (leak-like)",   False, ("cde", "graze", "under")),
    ("cde_over",   "CDE over-removal (hole-like)",    False, ("cde", "graze", "over")),
    ("cde_perp",   "CDE perp (whole face)",           False, ("cde", "perp", "cde")),
    ("flick_mean", "sweep flicker mean",              False, ("flicker", "flick_mean")),
    ("flick_p95",  "sweep flicker p95",               False, ("flicker", "flick_p95")),
    ("flick_maxr", "sweep flicker max/median",        False, ("flicker", "flick_maxr")),
]


def get_metric(mdict, path):
    cur = mdict
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def load_scene(out_root, eval_name, scene):
    p = Path(out_root) / eval_name / scene / "cutplane_results.json"
    if not p.is_file():
        return None
    try:
        r = json.load(open(p))
    except Exception:
        return None
    # graft the improved-metric result files (if present) under per-method keys
    for fname, key in (("cde_results.json", "cde"), ("sweep_flicker.json", "flicker")):
        fp = Path(out_root) / eval_name / scene / fname
        if fp.is_file():
            try:
                extra = json.load(open(fp))
                for m, entry in extra.get("methods", {}).items():
                    if m in r.get("methods", {}) and entry is not None:
                        r["methods"][m][key] = entry
            except Exception:
                pass
    return r


def discover_scenes(out_root, eval_name):
    d = Path(out_root) / eval_name
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "cutplane_results.json").is_file())


def fmt(v, key):
    if v is None:
        return "—"
    if key in ("leak", "pop", "hole", "overshoot", "cut_error",
               "cde", "cde_under", "cde_over", "cde_perp",
               "flick_mean", "flick_p95"):
        return f"{v:.4f}"
    if key == "flick_maxr":
        return f"{v:.1f}"
    if key == "edge_w":
        return f"{v:.1f}"
    if "psnr" in key:
        return f"{v:.2f}"
    return f"{v:.3f}"       # ssim, lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/data/output/xclipgs")
    ap.add_argument("--eval-name", default="cuteval",
                    help="evaluation directory under --out-root")
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    dest = Path(args.dest) if args.dest else out_root / args.eval_name
    dest.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(out_root, args.eval_name)
    # data[scene][method][metric_key] = value
    data = {}
    for s in scenes:
        r = load_scene(out_root, args.eval_name, s)
        if not r:
            continue
        data[s] = {}
        for mk, _ in METHODS:
            md = r.get("methods", {}).get(mk, {})
            data[s][mk] = {key: get_metric(md, path) for key, _, _, path in METRICS}

    # per-method averages across scenes (skip None)
    avg = {mk: {} for mk, _ in METHODS}
    for mk, _ in METHODS:
        for key, _, _, _ in METRICS:
            vals = [data[s][mk][key] for s in data if data[s][mk][key] is not None]
            avg[mk][key] = (sum(vals) / len(vals)) if vals else None

    # ---- Markdown: one section per metric, methods as columns ----
    lines = [f"# XClipGS cut-plane evaluation: {args.eval_name}", "",
             "Metrics restricted to a band around the clip plane (see "
             "`cutplane_metrics.py`). Perp = head-on cut-face views; graze = edge-on "
             "(plane→line) views. **Best per (scene, metric) in bold.**", ""]

    for key, label, higher, _ in METRICS:
        arrow = "↑" if higher else "↓"
        lines.append(f"## {label} {arrow}")
        lines.append("")
        lines.append("| Scene | " + " | ".join(name for _, name in METHODS) + " |")
        lines.append("|" + "---|" * (len(METHODS) + 1))
        for s in scenes:
            if s not in data:
                continue
            vals = [data[s][mk][key] for mk, _ in METHODS]
            present = [v for v in vals if v is not None]
            best = (max(present) if higher else min(present)) if present else None
            cells = []
            for v in vals:
                c = fmt(v, key)
                if v is not None and best is not None and abs(v - best) < 1e-9:
                    c = f"**{c}**"
                cells.append(c)
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        # avg row
        avals = [avg[mk][key] for mk, _ in METHODS]
        present = [v for v in avals if v is not None]
        best = (max(present) if higher else min(present)) if present else None
        cells = []
        for v in avals:
            c = fmt(v, key)
            if v is not None and best is not None and abs(v - best) < 1e-9:
                c = f"**{c}**"
            cells.append(c)
        lines.append(f"| **avg** | " + " | ".join(cells) + " |")
        lines.append("")

    (dest / "cutplane_table.md").write_text("\n".join(lines))

    # ---- CSV (long form) ----
    with open(dest / "cutplane_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "method"] + [key for key, *_ in METRICS])
        for s in scenes:
            if s not in data:
                continue
            for mk, _ in METHODS:
                w.writerow([s, mk] + [data[s][mk][key] for key, *_ in METRICS])
        for mk, _ in METHODS:
            w.writerow(["avg", mk] + [avg[mk][key] for key, *_ in METRICS])

    # console summary (avg)
    print(f"Scenes: {len(data)} ({', '.join(data)})")
    print(f"\n(averages over {len(data)} scenes)")
    print(f"{'method':16s} " + " ".join(f"{key:>10s}" for key, *_ in METRICS))
    for mk, name in METHODS:
        row = f"{name:16s} " + " ".join(
            f"{fmt(avg[mk][key], key):>10s}" for key, *_ in METRICS)
        print(row)
    print(f"\nWrote {dest/'cutplane_table.md'} and .csv")


if __name__ == "__main__":
    main()
