#!/usr/bin/env python3
"""Group standard per-view metrics by TF split, family, and identity."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iteration", default=None,
                        help="method suffix, e.g. 1000; default selects sole/latest method")
    args = parser.parse_args()

    payload = json.loads((args.model / "per_view.json").read_text())
    methods = list(payload)
    if args.iteration is not None:
        matches = [name for name in methods if name.endswith(f"_{args.iteration}")]
        if len(matches) != 1:
            raise ValueError(f"expected one method ending _{args.iteration}, got {matches}")
        method = matches[0]
    elif len(methods) == 1:
        method = methods[0]
    else:
        numeric = [(int(name.rsplit("_", 1)[-1]), name) for name in methods
                   if name.rsplit("_", 1)[-1].isdigit()]
        if not numeric:
            raise ValueError(f"cannot select method from {methods}")
        method = max(numeric)[1]

    frames = json.loads((args.dataset / "transforms_test.json").read_text())["frames"]
    metric_names = ("PSNR", "SSIM", "LPIPS")
    image_names = sorted(payload[method]["PSNR"])
    if len(image_names) != len(frames):
        raise ValueError(f"{len(image_names)} metric views but {len(frames)} dataset frames")

    groups = defaultdict(lambda: defaultdict(list))
    for frame, image_name in zip(frames, image_names):
        keys = {
            "all": "all",
            "split": frame["tf_split"],
            "family": frame["tf_family"],
            "tf": frame["tf_id"],
        }
        for kind, value in keys.items():
            group = f"{kind}:{value}"
            for metric in metric_names:
                groups[group][metric].append(float(payload[method][metric][image_name]))

    result = {"method": method, "groups": {}}
    for group in sorted(groups):
        result["groups"][group] = {
            "count": len(groups[group]["PSNR"]),
            **{metric: float(np.mean(groups[group][metric])) for metric in metric_names},
        }
    destination = args.model / "factorsplat_grouped_metrics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"FactorSplat grouped metrics: {method}")
    for group, values in result["groups"].items():
        if group.startswith("split:") or group == "all:all":
            print(f"  {group:28s} n={values['count']:4d} "
                  f"PSNR={values['PSNR']:.2f} SSIM={values['SSIM']:.4f} "
                  f"LPIPS={values['LPIPS']:.4f}")
    print(f"  wrote {destination}")


if __name__ == "__main__":
    main()
