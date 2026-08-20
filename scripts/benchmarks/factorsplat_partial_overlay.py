#!/usr/bin/env python3
"""Per-TF metrics from a PARTIAL test render (render.py --render_tf_indices).

The exact visibility gate changes only presets that hide labels, so a gate-ON
evaluation of an existing checkpoint only needs the hidden preset's frames plus
their paired base frames. This script reproduces, for the requested TFs, the
per-TF quantities of factorsplat_group_metrics.py (PSNR against GT) and
factorsplat_delta_metrics.py (paired delta vs the base preset, epsilon=0.04),
using identical formulas, and writes factorsplat_partial_overlay.json.

Aggregate splits are NOT written: callers merge these per-TF values with a
full no-gate run's metrics (bit-identical off the gated presets).
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--tf", action="append", required=True,
                        help="tf_id to evaluate (repeatable)")
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--out", type=Path, default=None,
                        help="output json path (default: <model>/factorsplat_partial_overlay.json)")
    args = parser.parse_args()

    frames = json.loads((args.dataset / "transforms_test.json").read_text())["frames"]
    method = args.model / "test" / f"ours_{args.iteration}"
    renders, gt = method / "renders", method / "gt"
    base_by_camera = {f["source_config"]: i for i, f in enumerate(frames)
                      if f["tf_id"] == "train_00_base"}
    acc = defaultdict(lambda: defaultdict(list))
    for index, frame in enumerate(frames):
        if frame["tf_id"] not in args.tf:
            continue
        pred_t = image(renders / f"{index:05d}.png")
        gt_t = image(gt / f"{index:05d}.png")
        mse = float(np.mean((pred_t - gt_t) ** 2))
        acc[frame["tf_id"]]["PSNR"].append(
            float("inf") if mse == 0 else -10.0 * np.log10(mse))
        base_index = base_by_camera[frame["source_config"]]
        pred_b = image(renders / f"{base_index:05d}.png")
        gt_b = image(gt / f"{base_index:05d}.png")
        reference_delta = gt_t - gt_b
        predicted_delta = pred_t - pred_b
        mask = np.max(np.abs(reference_delta), axis=-1) > args.epsilon
        unchanged = ~mask
        acc[frame["tf_id"]]["affected_fraction"].append(float(mask.mean()))
        acc[frame["tf_id"]]["delta_l1_full"].append(
            float(np.abs(predicted_delta - reference_delta).mean()))
        if mask.any():
            acc[frame["tf_id"]]["delta_l1_changed"].append(
                float(np.abs(predicted_delta - reference_delta)[mask].mean()))
        if unchanged.any():
            acc[frame["tf_id"]]["unchanged_delta_leak"].append(
                float(np.abs(predicted_delta)[unchanged].mean()))

    result = {"iteration": str(args.iteration), "epsilon": args.epsilon,
              "transfer_functions": {
                  tf: {m: float(np.mean(v)) for m, v in metrics.items()}
                  for tf, metrics in sorted(acc.items())}}
    destination = args.out or (args.model / "factorsplat_partial_overlay.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for tf, m in result["transfer_functions"].items():
        print(f"  {tf}: PSNR={m['PSNR']:.2f} delta_changed={m.get('delta_l1_changed', float('nan')):.4f}")
    print(f"  wrote {destination}")


if __name__ == "__main__":
    main()
