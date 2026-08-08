#!/usr/bin/env python3
"""Sensitivity analysis for the XClipGS CDE and Leak metrics.

The sweep reuses existing clipped/unclipped cut-eval renders; it performs no
training and no rendering. CDE is evaluated on the full Cartesian product of
affected-region thresholds and grazing-band half-widths. Leak is evaluated on
the full Cartesian product of foreground thresholds, cut-edge exclusion
margins, and near-plane window half-widths.

Per scene, CDE and Leak pool their raw numerators and denominators across all
views of the relevant camera family before taking a ratio. The reported
eight-volume aggregate is the unweighted mean of the eight per-scene ratios,
matching the paper's table aggregation.
"""

import argparse
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "cutplane_metrics", HERE / "cutplane_metrics.py"
)
CPM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CPM)

SCENES = ("gel", "intestine", "kneejoint", "lower",
          "vascular", "heart", "nose", "hand")
METHODS = ("ours", "clipgs", "mm", "hc")


def parse_csv(text, cast=float):
    return tuple(cast(x) for x in text.split(","))


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def renders_dir(out_root, method, suffix, scene):
    test_dir = out_root / f"{method}_{suffix}" / scene / "test"
    for checkpoint in ("ours_30000", "ours_best"):
        candidate = test_dir / checkpoint / "renders"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no fixed-checkpoint renders under {test_dir}")


def nested_acc():
    return defaultdict(lambda: defaultdict(float))


def ratio(entry):
    den = entry["den"]
    return (entry["num"] / den) if den > 0 else None


def scene_metrics(data_root, out_root, scene, methods, mask_thresholds,
                  band_widths, fg_thresholds, leak_margins, leak_windows):
    dataset_clip = data_root / "nerf_dataset" / f"{scene}_cuteval"
    dataset_full = data_root / "nerf_dataset" / f"{scene}_cutevalfull"
    transforms = json.loads((dataset_clip / "transforms_test.json").read_text())
    frames = sorted(transforms["frames"], key=lambda f: f["file_path"])
    fovx = float(transforms["camera_angle_x"])

    render_dirs = {
        method: {
            "clip": renders_dir(out_root, method, "cuteval", scene),
            "full": renders_dir(out_root, method, "cutevalfull", scene),
        }
        for method in methods
    }

    # Keys:
    #   CDEg: (theta, band, method) -> pooled numerator/denominator
    #   CDEp: (theta, method)       -> pooled numerator/denominator
    #   Leak: (fg, margin, window, method) -> pooled energy ratio
    cdeg = defaultdict(nested_acc)
    cdep = defaultdict(nested_acc)
    leak = defaultdict(nested_acc)

    for idx, frame in enumerate(frames):
        if not frame.get("clip"):
            continue
        stem = os.path.basename(frame["file_path"])
        gt_clip_path = dataset_clip / "test" / f"{stem}.png"
        gt_full_path = dataset_full / "test" / f"{stem}.png"
        if not (gt_clip_path.is_file() and gt_full_path.is_file()):
            raise FileNotFoundError(f"missing GT pair for {scene}/{stem}")

        gt_clip = load_rgb(gt_clip_path)
        gt_full = load_rgb(gt_full_path)
        gt_clip_lum = gt_clip.mean(-1)
        gt_full_lum = gt_full.mean(-1)
        gt_max = gt_clip.max(-1)
        delta_gt = gt_full_lum - gt_clip_lum

        height, width = gt_clip_lum.shape
        fx = CPM.fov2focal(fovx, width)
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        _t, _valid, _side, geom = CPM.plane_signed_distance_image(
            frame, width, height, fx, fy, cx, cy
        )
        signed_px, line = CPM.plane_line_distance(
            frame, width, height, fx, fy, cx, cy, geom
        )
        face_on = math.hypot(line[0], line[1]) < 1e-6
        family = "graze" if "graze" in (frame.get("mode") or "") else "perp"
        gt_affected = {
            theta: np.abs(delta_gt) > theta for theta in mask_thresholds
        }
        bands = {
            band: np.abs(signed_px) <= band for band in band_widths
        }

        for method in methods:
            method_clip_path = render_dirs[method]["clip"] / f"{idx:05d}.png"
            method_full_path = render_dirs[method]["full"] / f"{idx:05d}.png"
            if not (method_clip_path.is_file() and method_full_path.is_file()):
                raise FileNotFoundError(
                    f"missing {method} render pair for {scene}/{idx:05d}"
                )
            method_clip = load_rgb(method_clip_path)
            method_full = load_rgb(method_full_path)
            method_max = method_clip.max(-1)
            delta_method = method_full.mean(-1) - method_clip.mean(-1)

            for theta in mask_thresholds:
                affected_gt = gt_affected[theta]
                affected_method = np.abs(delta_method) > theta
                mismatch = np.logical_xor(affected_gt, affected_method)
                if family == "graze" and not face_on:
                    for band in band_widths:
                        roi = bands[band]
                        key = (theta, band, method)
                        cdeg[key]["pooled"]["num"] += float((mismatch & roi).sum())
                        cdeg[key]["pooled"]["den"] += float(
                            (affected_gt & roi).sum()
                        )
                else:
                    key = (theta, method)
                    cdep[key]["pooled"]["num"] += float(mismatch.sum())
                    cdep[key]["pooled"]["den"] += float(affected_gt.sum())

            if family == "graze" and not face_on:
                abs_signed = np.abs(signed_px)
                for fg in fg_thresholds:
                    method_fg = method_max > fg
                    reference_bg = gt_max <= fg
                    for window in leak_windows:
                        near = abs_signed <= window
                        denominator = float(method_max[method_fg & near].sum())
                        for margin in leak_margins:
                            numerator = float(
                                method_max[
                                    method_fg
                                    & (signed_px > margin)
                                    & reference_bg
                                    & near
                                ].sum()
                            )
                            key = (fg, margin, window, method)
                            leak[key]["pooled"]["num"] += numerator
                            leak[key]["pooled"]["den"] += denominator

    return {
        "cdeg": {
            f"{theta:g}|{band:g}|{method}": ratio(cdeg[(theta, band, method)]["pooled"])
            for theta in mask_thresholds
            for band in band_widths
            for method in methods
        },
        "cdep": {
            f"{theta:g}|{method}": ratio(cdep[(theta, method)]["pooled"])
            for theta in mask_thresholds
            for method in methods
        },
        "leak": {
            f"{fg:g}|{margin:g}|{window:g}|{method}":
                ratio(leak[(fg, margin, window, method)]["pooled"])
            for fg in fg_thresholds
            for margin in leak_margins
            for window in leak_windows
            for method in methods
        },
    }


def aggregate(scene_results, scenes, methods, mask_thresholds, band_widths,
              fg_thresholds, leak_margins, leak_windows):
    def mean_values(metric, key):
        values = [scene_results[scene][metric][key] for scene in scenes]
        if any(value is None for value in values):
            return None
        return float(np.mean(values))

    cdeg = {
        f"{theta:g}|{band:g}": {
            method: mean_values("cdeg", f"{theta:g}|{band:g}|{method}")
            for method in methods
        }
        for theta in mask_thresholds
        for band in band_widths
    }
    cdep = {
        f"{theta:g}": {
            method: mean_values("cdep", f"{theta:g}|{method}")
            for method in methods
        }
        for theta in mask_thresholds
    }
    leak = {
        f"{fg:g}|{margin:g}|{window:g}": {
            method: mean_values(
                "leak", f"{fg:g}|{margin:g}|{window:g}|{method}"
            )
            for method in methods
        }
        for fg in fg_thresholds
        for margin in leak_margins
        for window in leak_windows
    }
    return {"cdeg": cdeg, "cdep": cdep, "leak": leak}


def summarize(aggregate_results, methods):
    summary = {}
    for metric, configurations in aggregate_results.items():
        per_method = {
            method: [values[method] for values in configurations.values()]
            for method in methods
        }
        winners = [
            min(values, key=values.get) for values in configurations.values()
        ]
        summary[metric] = {
            "configurations": len(configurations),
            "range": {
                method: {
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
                for method, values in per_method.items()
            },
            "wins": {
                method: sum(winner == method for winner in winners)
                for method in methods
            },
            "winner_all_configurations": (
                winners[0] if len(set(winners)) == 1 else None
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/mnt/uNeon/zhongpai/vengine_data")
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/mnt/uNeon/zhongpai/vengine_data/output/xclipgs")
    )
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--mask-thresholds", default="0.02,0.04,0.06,0.08")
    parser.add_argument("--band-widths", default="8,12,16,24")
    parser.add_argument("--fg-thresholds", default="0.02,0.04,0.08")
    parser.add_argument("--leak-margins", default="1,2,4")
    parser.add_argument("--leak-windows", default="40,60,80")
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/mnt/uNeon/zhongpai/vengine_data/output/xclipgs/"
            "cuteval/metric_sensitivity.json"
        )
    )
    args = parser.parse_args()

    mask_thresholds = parse_csv(args.mask_thresholds)
    band_widths = parse_csv(args.band_widths)
    fg_thresholds = parse_csv(args.fg_thresholds)
    leak_margins = parse_csv(args.leak_margins)
    leak_windows = parse_csv(args.leak_windows)

    scene_results = {}
    for scene in args.scenes:
        print(f"[{scene}] loading fixed cut-eval renders", flush=True)
        scene_results[scene] = scene_metrics(
            args.data_root, args.out_root, scene, tuple(args.methods),
            mask_thresholds, band_widths, fg_thresholds,
            leak_margins, leak_windows
        )

    aggregate_results = aggregate(
        scene_results, tuple(args.scenes), tuple(args.methods),
        mask_thresholds, band_widths, fg_thresholds,
        leak_margins, leak_windows
    )
    summary = summarize(aggregate_results, tuple(args.methods))
    payload = {
        "description": (
            "CDE and Leak parameter sensitivity on the eight-volume cut-eval set"
        ),
        "aggregation": (
            "pool numerator/denominator across views within each scene, then "
            "unweighted mean of per-scene ratios"
        ),
        "parameters": {
            "mask_thresholds": mask_thresholds,
            "band_widths_px": band_widths,
            "foreground_thresholds": fg_thresholds,
            "leak_margins_px": leak_margins,
            "leak_windows_px": leak_windows,
        },
        "scenes": scene_results,
        "aggregate": aggregate_results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for metric in ("cdeg", "cdep", "leak"):
        item = summary[metric]
        print(f"\n{metric}: {item['configurations']} configurations")
        for method in args.methods:
            bounds = item["range"][method]
            print(
                f"  {method:7s} {bounds['min']:.6f}--{bounds['max']:.6f}; "
                f"wins {item['wins'][method]}/{item['configurations']}"
            )
        print(f"  common winner: {item['winner_all_configurations']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
