#!/usr/bin/env python3
"""Build an XClipGS clip-operator comparison table from the sweep outputs.

Reads every run's results.json under <out_root>/<operator>/<scene>/ (written by
metrics.py) and emits a comparison table (Markdown + CSV) of PSNR / SSIM / LPIPS
across the three clip operators {ours=analytic, mm=moment, hc=hardcull} for each
scene. Picks the best-iteration ("best") method if present, else the highest
numeric iteration.

Usage (run after the sweep's train+render+metrics finish):
    python scripts/benchmarks/xclipgs_make_table.py \
        --out-root /data/output/xclipgs \
        --dest    /data/output/xclipgs/ours/gel_900
"""
import argparse
import csv
import json
import os
from pathlib import Path

OPERATORS = [("ours", "Ours (analytic)"),
             ("mm", "MM (moment)"),
             ("hc", "HC (hard cull)"),
             ("ours_nomip", "Ours no-Mip")]
METRICS = ["PSNR", "SSIM", "LPIPS"]
# higher-is-better for arrow direction / best-bolding
HIGHER_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False}


def pick_method(scene_metrics):
    """From a results.json dict {method: {metrics}}, choose the reporting method.
    Prefer a 'best' method; else the highest-numbered iteration_<N> / <N> key."""
    if not scene_metrics:
        return None, None
    keys = list(scene_metrics.keys())
    for k in keys:
        if "best" in k.lower():
            return k, scene_metrics[k]

    def iter_num(k):
        digits = "".join(ch for ch in k if ch.isdigit())
        return int(digits) if digits else -1

    k = max(keys, key=iter_num)
    return k, scene_metrics[k]


def load_run(out_root, operator, scene):
    rj = Path(out_root) / operator / scene / "results.json"
    if not rj.is_file():
        return None
    try:
        with open(rj) as f:
            data = json.load(f)
    except Exception:
        return None
    # results.json is {scene_dir_str: {method: {...}}} OR {method: {...}}.
    # metrics.py keys the outer dict by the model dir path; unwrap one level if so.
    if data and all(isinstance(v, dict) for v in data.values()):
        first = next(iter(data.values()))
        if first and all(isinstance(vv, dict) for vv in first.values()):
            data = first  # unwrap scene_dir layer
    _, m = pick_method(data)
    return m


def discover_scenes(out_root):
    scenes = set()
    for op, _ in OPERATORS:
        d = Path(out_root) / op
        if d.is_dir():
            scenes.update(p.name for p in d.iterdir() if p.is_dir())
    return sorted(scenes)


def fmt(v, metric):
    if v is None:
        return "—"
    return f"{v:.4f}" if metric != "PSNR" else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/data/output/xclipgs",
                    help="Root holding <operator>/<scene>/results.json")
    ap.add_argument("--dest", default=None,
                    help="Directory to write the table (default: <out-root>)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    dest = Path(args.dest) if args.dest else out_root
    dest.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(out_root)
    # collect: results[scene][op] = {PSNR,SSIM,LPIPS,Number,...}
    results = {}
    for scene in scenes:
        results[scene] = {}
        for op, _ in OPERATORS:
            results[scene][op] = load_run(out_root, op, scene)

    # ---- Markdown ----
    lines = []
    lines.append("# XClipGS clip-operator comparison")
    lines.append("")
    lines.append("Opacity-only dGS + Mip-Splatting, warm-started from the RenderFM")
    lines.append("checkpoint, held-out test views. Operators: **Ours** = exact analytic")
    lines.append("half-space clip, **MM** = moment-matched truncation, **HC** = hard cull.")
    lines.append("Best per (scene, metric) in **bold**. ↑ higher-better, ↓ lower-better.")
    lines.append("")

    for metric in METRICS:
        arrow = "↑" if HIGHER_BETTER[metric] else "↓"
        header = f"| Scene | {' | '.join(name for _, name in OPERATORS)} |"
        sep = "|" + "---|" * (len(OPERATORS) + 1)
        lines.append(f"## {metric} {arrow}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for scene in scenes:
            vals = []
            for op, _ in OPERATORS:
                m = results[scene][op]
                vals.append(m.get(metric) if m else None)
            present = [v for v in vals if v is not None]
            best = (max(present) if HIGHER_BETTER[metric] else min(present)) if present else None
            cells = []
            for v in vals:
                s = fmt(v, metric)
                if v is not None and best is not None and abs(v - best) < 1e-9:
                    s = f"**{s}**"
                cells.append(s)
            lines.append(f"| {scene} | {' | '.join(cells)} |")
        lines.append("")

    # gaussian count + training time (context, from any operator that has it)
    lines.append("## Model size / training time")
    lines.append("")
    lines.append("| Scene | #Gaussians (ours) | Train time s (ours) |")
    lines.append("|---|---|---|")
    for scene in scenes:
        m = results[scene].get("ours")
        n = m.get("Number") if m else None
        t = m.get("Training_time") if m else None
        n_s = f"{int(n):,}" if isinstance(n, (int, float)) else "—"
        t_s = f"{t:.1f}" if isinstance(t, (int, float)) else "—"
        lines.append(f"| {scene} | {n_s} | {t_s} |")
    lines.append("")

    missing = [(s, op) for s in scenes for op, _ in OPERATORS if results[s][op] is None]
    if missing:
        lines.append("## Missing runs (no results.json yet)")
        lines.append("")
        for s, op in missing:
            lines.append(f"- {op}/{s}")
        lines.append("")

    md_path = dest / "comparison_table.md"
    md_path.write_text("\n".join(lines))

    # ---- CSV (long form) ----
    csv_path = dest / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "operator", "PSNR", "SSIM", "LPIPS", "Number", "Training_time"])
        for scene in scenes:
            for op, _ in OPERATORS:
                m = results[scene][op] or {}
                w.writerow([scene, op, m.get("PSNR"), m.get("SSIM"),
                            m.get("LPIPS"), m.get("Number"), m.get("Training_time")])

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Scenes: {len(scenes)}  Missing runs: {len(missing)}")


if __name__ == "__main__":
    main()
