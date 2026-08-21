#!/usr/bin/env python3
"""Measure whether a model applies the requested TF image delta.

Each non-base TF frame is paired with the authored-base frame at the same source
camera. The metric compares predicted and reference image differences, focusing
on pixels whose reference appearance actually changes.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def image(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--epsilon", type=float, default=0.04,
                        help="reference RGB max-delta threshold defining changed pixels")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="parallel image-pair workers (default: min(8, CPU count))")
    args = parser.parse_args()

    frames = json.loads((args.dataset / "transforms_test.json").read_text())["frames"]
    method = args.model / "test" / f"ours_{args.iteration}"
    renders = method / "renders"
    gt = method / "gt"
    base_by_camera = {
        frame["source_config"]: index for index, frame in enumerate(frames)
        if frame["tf_id"] == "train_00_base"
    }
    if not base_by_camera:
        raise ValueError("no train_00_base test frames found")

    # Every base view is reused by all non-base TFs. Loading it once avoids
    # 40 redundant reads per camera (roughly 24 GB of PNG decode traffic at
    # 1600^2), while a bounded thread pool parallelizes the independent target
    # decodes and NumPy reductions. executor.map preserves manifest order, so
    # aggregation and JSON output remain deterministic.
    base_images = {
        index: (image(renders / f"{index:05d}.png"),
                image(gt / f"{index:05d}.png"))
        for index in sorted(set(base_by_camera.values()))
    }
    jobs = [(index, frame, base_by_camera.get(frame["source_config"]))
            for index, frame in enumerate(frames)
            if frame["tf_id"] != "train_00_base"]
    missing = [frame["source_config"] for _, frame, base_index in jobs
               if base_index is None]
    if missing:
        raise ValueError(f"no base pair for {missing[0]}")

    def measure(job):
        index, frame, base_index = job
        pred_t = image(renders / f"{index:05d}.png")
        gt_t = image(gt / f"{index:05d}.png")
        pred_b, gt_b = base_images[base_index]
        reference_delta = gt_t - gt_b
        predicted_delta = pred_t - pred_b
        mask = np.max(np.abs(reference_delta), axis=-1) > args.epsilon
        unchanged = ~mask
        values = {
            "affected_fraction": float(mask.mean()),
            "delta_l1_full": float(np.abs(predicted_delta - reference_delta).mean()),
            "delta_l1_changed": (float(np.abs(predicted_delta - reference_delta)[mask].mean())
                                 if mask.any() else None),
            "unchanged_delta_leak": (float(np.abs(predicted_delta)[unchanged].mean())
                                     if unchanged.any() else None),
        }
        return frame, values

    groups = defaultdict(lambda: defaultdict(list))
    per_tf = defaultdict(lambda: defaultdict(list))
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        measurements = executor.map(measure, jobs)
        for frame, values in measurements:
            keys = (f"split:{frame['tf_split']}", f"family:{frame['tf_family']}")
            for key in keys:
                for metric, value in values.items():
                    if value is not None:
                        groups[key][metric].append(value)
            for metric, value in values.items():
                if value is not None:
                    per_tf[frame["tf_id"]][metric].append(value)
    pair_count = len(jobs)

    metric_names = ("affected_fraction", "delta_l1_full", "delta_l1_changed",
                    "unchanged_delta_leak")
    def aggregate(source):
        return {key: {metric: mean_or_none(metrics.get(metric, []))
                      for metric in metric_names}
                for key, metrics in sorted(source.items())}

    result = {
        "iteration": str(args.iteration),
        "epsilon": args.epsilon,
        "pairs": pair_count,
        "groups": aggregate(groups),
        "transfer_functions": aggregate(per_tf),
    }
    destination = args.model / "factorsplat_delta_metrics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"FactorSplat TF-delta metrics: {pair_count} paired edits")
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    for group, values in result["groups"].items():
        if group.startswith("split:"):
            print(f"  {group:28s} affected={fmt(values['affected_fraction'])} "
                  f"delta-L1={fmt(values['delta_l1_changed'])} "
                  f"leak={fmt(values['unchanged_delta_leak'])}")
    print(f"  wrote {destination}")


if __name__ == "__main__":
    main()
