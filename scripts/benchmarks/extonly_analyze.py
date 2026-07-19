#!/usr/bin/env python3
"""External-view-only ablation analysis for XClipGS.

Isolates the interior-supervision gain by comparing two identical analytic-operator
models that differ ONLY in training data:

  clip-aware : trained on <scene>_900 (405 intact + 405 clipped views)
  ext-only   : trained on <scene>_noclip (405 intact views, NO clipped supervision)

Both are evaluated on the SAME held-out <scene>_900 test split (90 views: 45 intact,
45 clipped). Global PSNR/SSIM are read from each model's per_view.json and split by
whether the test frame carries a clipping plane. The clipped-view PSNR gap is the
quantity the reviewer asked for: how much clip-aware supervision buys on the cut
views the external-only model never saw supervised.

Cut-face band/CDE/Leak for the ext-only model come from cuteval_extonly/<scene>
(written by _extonly_batch.sh); the clip-aware counterparts come from the existing
cuteval/<scene> tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SCENES = ("gel", "intestine", "kneejoint", "lower",
          "vascular", "heart", "nose", "hand")


def frame_labels(transforms_test):
    """Return {render_filename '{i:05d}.png' -> is_clipped} for the 900 test set.

    metrics.py renders in sorted(frames, key=file_path) order and names each
    render {index:05d}.png; per_view.json keys by that same name.
    """
    payload = json.loads(Path(transforms_test).read_text())
    frames = sorted(payload["frames"], key=lambda f: f["file_path"])
    labels = {}
    for i, frame in enumerate(frames):
        clipped = bool(frame.get("clip")) or ("plane_normal" in frame and
                                              frame.get("clip_offset") is not None)
        labels[f"{i:05d}.png"] = clipped
    return labels


def per_view_split(per_view_json, labels):
    """Mean PSNR/SSIM over intact vs clipped test frames from a per_view.json."""
    payload = json.loads(Path(per_view_json).read_text())
    # per_view.json = {method: {"PSNR": {name: v}, "SSIM": {...}, ...}}
    method = next(iter(payload))
    psnr = payload[method]["PSNR"]
    ssim = payload[method]["SSIM"]
    out = {}
    for split, want in (("intact", False), ("clipped", True), ("all", None)):
        names = [n for n in psnr if want is None or labels.get(n) == want]
        out[split] = {
            "psnr": float(np.mean([psnr[n] for n in names])),
            "ssim": float(np.mean([ssim[n] for n in names])),
            "n": len(names),
        }
    return out


def cutface_scalar(cut_json, key):
    """Read one averaged cut-face metric (band SSIM / CDE / Leak) if present."""
    if not Path(cut_json).exists():
        return None
    data = json.loads(Path(cut_json).read_text())
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("/data"))
    ap.add_argument("--out-root", type=Path, default=Path("/data/output/xclipgs"))
    ap.add_argument("--clipaware", default="ours",
                    help="output subdir of the clip-aware analytic model")
    ap.add_argument("--extonly", default="ours_extonly",
                    help="output subdir of the external-view-only model")
    ap.add_argument("--eval900-subdir", default="extonly_eval900",
                    help="where the ext-only 900-test per_view.json lives")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--output", type=Path,
                    default=Path("/data/output/xclipgs/cuteval/extonly_ablation.json"))
    args = ap.parse_args()

    rows = {}
    hdr = (f"{'scene':11s} "
           f"{'CA-clip':>8s} {'EX-clip':>8s} {'dPSNR':>7s} | "
           f"{'CA-all':>7s} {'EX-all':>7s} | {'CA-int':>7s} {'EX-int':>7s}")
    print(hdr)
    for scene in args.scenes:
        labels = frame_labels(args.data_root / "nerf_dataset" / f"{scene}_900" /
                              "transforms_test.json")
        ca = per_view_split(
            args.out_root / args.clipaware / f"{scene}_900" / "per_view.json", labels)
        ex = per_view_split(
            args.out_root / args.eval900_subdir / scene / "per_view.json", labels)
        rows[scene] = {"clip_aware": ca, "ext_only": ex}
        print(f"{scene:11s} "
              f"{ca['clipped']['psnr']:8.2f} {ex['clipped']['psnr']:8.2f} "
              f"{ca['clipped']['psnr']-ex['clipped']['psnr']:7.2f} | "
              f"{ca['all']['psnr']:7.2f} {ex['all']['psnr']:7.2f} | "
              f"{ca['intact']['psnr']:7.2f} {ex['intact']['psnr']:7.2f}")

    def avg(path):
        return float(np.mean([
            _dig(rows[s], path) for s in args.scenes]))

    def _dig(d, path):
        for k in path:
            d = d[k]
        return d

    summary = {
        split: {
            "clip_aware_psnr": avg(("clip_aware", split, "psnr")),
            "ext_only_psnr": avg(("ext_only", split, "psnr")),
            "delta_psnr": avg(("clip_aware", split, "psnr")) - avg(("ext_only", split, "psnr")),
        }
        for split in ("clipped", "intact", "all")
    }
    print(f"\n{'AVG':11s} "
          f"{summary['clipped']['clip_aware_psnr']:8.2f} "
          f"{summary['clipped']['ext_only_psnr']:8.2f} "
          f"{summary['clipped']['delta_psnr']:7.2f} | "
          f"{summary['all']['clip_aware_psnr']:7.2f} "
          f"{summary['all']['ext_only_psnr']:7.2f} | "
          f"{summary['intact']['clip_aware_psnr']:7.2f} "
          f"{summary['intact']['ext_only_psnr']:7.2f}")

    payload = {
        "description": "external-view-only vs clip-aware analytic model, same 900 test split",
        "summary": summary,
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
