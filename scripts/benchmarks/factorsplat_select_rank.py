#!/usr/bin/env python3
"""Machine-readable rank selection for the FactorSplat full study.

Scans <root>/<variant>/rank*/<scene>/<preset>/ and emits rank_selection.json.

Primary criterion: validation-preset changed-region TF-delta error
(split:val delta_l1_changed) -- minimize. Guardrails (reported per rank and
checked against the primary winner): unchanged-region leak, PSNR, SSIM, and
checkpoint size. Rank is NEVER selected on PSNR alone. Parsimony rule: if a
smaller rank is within --tie-tolerance (relative) of the best validation
delta error, the smaller rank is selected.
"""
import argparse
import glob
import json
import os


def load_rank(run_dir, iteration):
    delta_path = os.path.join(run_dir, "factorsplat_delta_metrics.json")
    group_path = os.path.join(run_dir, "factorsplat_grouped_metrics.json")
    if not (os.path.isfile(delta_path) and os.path.isfile(group_path)):
        return None
    delta = json.load(open(delta_path))
    group = json.load(open(group_path))

    def dsplit(name, key):
        entry = delta["groups"].get(f"split:{name}")
        return None if entry is None else float(entry[key])

    def gsplit(name, key):
        entry = group["groups"].get(f"split:{name}")
        return None if entry is None else float(entry[key])

    ply = os.path.join(run_dir, "point_cloud",
                       f"iteration_{delta.get('iteration', iteration)}",
                       "point_cloud.ply")
    size = 0
    for path in (ply, ply + ".factorsplat.pt"):
        if os.path.isfile(path):
            size += os.path.getsize(path)
    record = {
        "val_delta_l1_changed": dsplit("val", "delta_l1_changed"),
        "val_unchanged_delta_leak": dsplit("val", "unchanged_delta_leak"),
        "val_psnr": gsplit("val", "PSNR"),
        "val_ssim": gsplit("val", "SSIM"),
        "checkpoint_bytes": size or None,
        "splits": {
            name: {
                "delta_l1_changed": dsplit(name, "delta_l1_changed"),
                "unchanged_delta_leak": dsplit(name, "unchanged_delta_leak"),
                "psnr": gsplit(name, "PSNR"),
            }
            for name in ("train", "val", "test_interp", "test_ood")
        },
    }
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/data/output/factorsplat")
    ap.add_argument("--variant", default="residual")
    ap.add_argument("--scene", default="heart")
    ap.add_argument("--preset", default="full")
    ap.add_argument("--iteration", default="30000")
    ap.add_argument("--tie-tolerance", type=float, default=0.02,
                    help="relative val-delta margin within which the SMALLER "
                         "rank wins (parsimony)")
    ap.add_argument("--leak-guardrail", type=float, default=1.5,
                    help="flag ranks whose val leak exceeds this multiple of "
                         "the best rank's leak")
    ap.add_argument("--psnr-guardrail", type=float, default=0.3,
                    help="flag ranks whose val PSNR trails the best val PSNR "
                         "by more than this many dB")
    ap.add_argument("--out", default=None,
                    help="default <root>/<variant>/rank_selection_<scene>_<preset>.json")
    args = ap.parse_args()

    ranks = {}
    for run_dir in sorted(glob.glob(os.path.join(
            args.root, args.variant, "rank*", args.scene, args.preset))):
        rank = int(run_dir.split(os.sep + "rank", 1)[1].split(os.sep)[0])
        record = load_rank(run_dir, args.iteration)
        if record is None:
            print(f"[select-rank] rank {rank}: metrics incomplete, skipped")
            continue
        ranks[rank] = record
    if not ranks:
        raise SystemExit("no completed rank runs found")

    by_delta = sorted(ranks, key=lambda r: ranks[r]["val_delta_l1_changed"])
    best = by_delta[0]
    best_delta = ranks[best]["val_delta_l1_changed"]
    selected = best
    for rank in sorted(ranks):
        if rank >= selected:
            break
        rel = ranks[rank]["val_delta_l1_changed"] / best_delta - 1.0
        if rel <= args.tie_tolerance:
            selected = rank
            break

    best_leak = min(v["val_unchanged_delta_leak"] for v in ranks.values())
    best_psnr = max(v["val_psnr"] for v in ranks.values())
    warnings = []
    sel = ranks[selected]
    if best_leak > 0 and sel["val_unchanged_delta_leak"] > args.leak_guardrail * best_leak:
        warnings.append(f"selected rank leak {sel['val_unchanged_delta_leak']:.5f} "
                        f"exceeds {args.leak_guardrail}x best leak {best_leak:.5f}")
    if best_psnr - sel["val_psnr"] > args.psnr_guardrail:
        warnings.append(f"selected rank val PSNR {sel['val_psnr']:.2f} trails best "
                        f"{best_psnr:.2f} by more than {args.psnr_guardrail} dB")

    manifest = {
        "scene": args.scene,
        "preset": args.preset,
        "variant": args.variant,
        "rule": ("minimize split:val delta_l1_changed; smaller rank wins within "
                 f"{args.tie_tolerance:.0%} relative; guardrails: leak <= "
                 f"{args.leak_guardrail}x best, val PSNR within "
                 f"{args.psnr_guardrail} dB of best, size/latency reported"),
        "selected_rank": selected,
        "primary_ranking_by_val_delta": by_delta,
        "guardrail_warnings": warnings,
        "ranks": {str(k): ranks[k] for k in sorted(ranks)},
    }
    out = args.out or os.path.join(
        args.root, args.variant,
        f"rank_selection_{args.scene}_{args.preset}.json")
    json.dump(manifest, open(out, "w"), indent=2)
    print(f"[select-rank] ranks {sorted(ranks)}; by val delta: {by_delta}; "
          f"SELECTED r={selected}" + (f"; WARNINGS: {warnings}" if warnings else ""))
    print(f"[select-rank] wrote {out}")


if __name__ == "__main__":
    main()
