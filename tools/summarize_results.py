"""Summarize best results from all scenes in a benchmark folder.

Usage:
    python tools/summarize_results.py output/mcmc/dbs/nerf_synthetic
    python tools/summarize_results.py output/mcmc/ubs/nerf_synthetic
    python tools/summarize_results.py output/standard/dgs/mipnerf360
"""

import json
import os
import sys
import numpy as np


def summarize(base_dir):
    scenes = sorted([s for s in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, s))])

    label = f"{os.path.basename(os.path.dirname(base_dir))}/{os.path.basename(base_dir)}"

    header = f"{'Scene':<12} {'PSNR':>8} {'PSNR_t':>8} {'SSIM':>8} {'LPIPS':>8} {'#GS':>8} {'FPS':>8} {'Time(s)':>8}"
    sep = '-' * 78

    results = []
    lines = []
    lines.append(f"{label} - ours_best")
    lines.append(header)
    lines.append(sep)

    for scene in scenes:
        rp = os.path.join(base_dir, scene, 'results.json')
        if not os.path.exists(rp):
            lines.append(f"{scene:<12} MISSING")
            continue
        with open(rp) as f:
            data = json.load(f)
        m = data.get('ours_best') or data.get('ours_30000')
        if m is None:
            lines.append(f"{scene:<12} no results")
            continue
        results.append((scene, m))
        lines.append(f"{scene:<12} {m.get('PSNR',0):>8.3f} {m.get('PSNR_train',0):>8.3f} {m.get('SSIM',0):>8.5f} {m.get('LPIPS',0):>8.5f} {m.get('Number',0):>8} {m.get('FPS',0):>8.2f} {m.get('Training_time',0):>8.1f}")

    lines.append(sep)
    if results:
        avg = lambda k: np.mean([m.get(k, 0) for _, m in results])
        lines.append(f"{'Average':<12} {avg('PSNR'):>8.3f} {avg('PSNR_train'):>8.3f} {avg('SSIM'):>8.5f} {avg('LPIPS'):>8.5f} {avg('Number'):>8.0f} {avg('FPS'):>8.2f} {avg('Training_time'):>8.1f}")
    lines.append("")

    # Print and save
    out = '\n'.join(lines)
    print(out)

    out_path = os.path.join(base_dir, 'summary.txt')
    with open(out_path, 'w') as f:
        f.write(out + '\n')
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/summarize_results.py <benchmark_dir>")
        sys.exit(1)

    for path in sys.argv[1:]:
        if os.path.isdir(path):
            summarize(path)
        else:
            print(f"Not a directory: {path}")
