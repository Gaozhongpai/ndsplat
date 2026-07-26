#!/usr/bin/env python3
"""Aggregate multi-seed rebuttal runs into mean +/- std tables.

Produces the numbers for the [[FILL]] slots in pages/dGS/REBUTTAL.md
(Reviewer orhG W1/Q1: does the learned-Lambda ordering survive across seeds?).

Layout consumed:
    <root>/seed<N>/<config>/nerf_synthetic/<scene>/results.json

Usage:
    python3 scripts/benchmarks/dgs_rebuttal_summarize.py --root /code/output/rebuttal
    python3 scripts/benchmarks/dgs_rebuttal_summarize.py --root ... --markdown
"""
import argparse
import glob
import json
import os
import statistics as st

# Single-seed values published in Table 3 (NeRF Synthetic), for reference.
PAPER = {"lam0": 33.84, "lam1": 33.78, "lamlearn": 33.91}
LABEL = {
    "lam0": "Lambda=0 (dGS-O)",
    "lam1": "Lambda=1 (fixed)",
    "lamlearn": "Lambda learned (dGS)",
}
ITER = "ours_30000"


def collect(root):
    """-> {config: {seed: {scene: metrics}}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "seed*", "*", "*", "*", "results.json"))):
        parts = f.split(os.sep)
        seed = int(parts[-5].replace("seed", ""))
        config, scene = parts[-4], parts[-2]
        try:
            with open(f) as fh:
                d = json.load(fh)[ITER]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        out.setdefault(config, {}).setdefault(seed, {})[scene] = d
    return out


def mean_std(vals):
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--markdown", action="store_true", help="emit a REBUTTAL.md-ready table")
    args = ap.parse_args()

    data = collect(args.root)
    if not data:
        print(f"No results.json found under {args.root}")
        return

    configs = [c for c in ("lam0", "lam1", "lamlearn") if c in data] + \
              sorted(c for c in data if c not in LABEL)
    all_seeds = sorted({s for c in data.values() for s in c})

    # ---- per-seed scene means -------------------------------------------
    print("\n=== Per-seed scene-mean PSNR (NeRF Synthetic) ===")
    hdr = f"{'config':<24}" + "".join(f"{'seed'+str(s):>12}" for s in all_seeds) + f"{'paper':>10}"
    print(hdr)
    print("-" * len(hdr))
    per_seed = {}
    for c in configs:
        row = f"{LABEL.get(c, c):<24}"
        for s in all_seeds:
            scenes = data.get(c, {}).get(s, {})
            if scenes:
                m = st.mean(v["PSNR"] for v in scenes.values())
                per_seed.setdefault(c, {})[s] = m
                row += f"{m:>12.2f}" if len(scenes) == 8 else f"{str(round(m,2))+'*':>12}"
            else:
                row += f"{'--':>12}"
        row += f"{PAPER.get(c, float('nan')):>10.2f}"
        print(row)
    print("* = incomplete (fewer than 8 scenes)")

    # ---- across-seed mean +/- std ---------------------------------------
    print("\n=== Across-seed mean +/- std ===")
    print(f"{'config':<24}{'PSNR':>18}{'SSIM':>18}{'LPIPS':>18}")
    print("-" * 78)
    summary = {}
    for c in configs:
        cell = {}
        for key in ("PSNR", "SSIM", "LPIPS"):
            vals = [
                st.mean(v[key] for v in data[c][s].values())
                for s in sorted(data.get(c, {}))
                if data[c][s]
            ]
            cell[key] = mean_std(vals)
        summary[c] = cell
        n = len(data.get(c, {}))
        print(
            f"{LABEL.get(c, c):<24}"
            f"{cell['PSNR'][0]:>11.2f} +/-{cell['PSNR'][1]:>5.2f}"
            f"{cell['SSIM'][0]:>11.4f} +/-{cell['SSIM'][1]:>5.4f}"
            f"{cell['LPIPS'][0]:>11.4f} +/-{cell['LPIPS'][1]:>5.4f}"
            f"   (n={n})"
        )

    # ---- refuse to judge on unequal scene sets --------------------------
    # A partial config's mean is not comparable to a complete one: the runs
    # complete alphabetically, and `drums` (~27 dB) drags any early subset
    # down by ~1 dB. Comparing a 3-scene mean against an 8-scene mean once
    # produced a spurious "learned Lambda is 0.97 dB WORSE" verdict.
    counts = {c: {s: len(data[c][s]) for s in data.get(c, {})} for c in configs}
    incomplete = {c: n for c, per in counts.items() for n in per.values() if n != 8}
    if incomplete:
        print("\n=== Verdict withheld: unequal scene sets ===")
        for c in configs:
            per = counts.get(c, {})
            print(f"  {LABEL.get(c, c):<24} " + ", ".join(f"seed{s}: {n}/8" for s, n in sorted(per.items())))
        print("\n  Means above are NOT comparable across configs until every cell is 8/8.")
        print("  Runs complete alphabetically and `drums` (~27 dB) is the weakest scene,")
        print("  so a partial mean is biased low. Re-run this script when the sweep finishes.")
        common = set.intersection(*[set(data[c][s]) for c in configs for s in data.get(c, {})]) \
            if configs and all(data.get(c) for c in configs) else set()
        if len(common) >= 2:
            print(f"\n  Like-for-like on the {len(common)} scene(s) every config has finished:")
            for c in configs:
                vals = [st.mean(data[c][s][sc]["PSNR"] for sc in common) for s in sorted(data.get(c, {}))]
                if vals:
                    print(f"    {LABEL.get(c, c):<24} {st.mean(vals):.2f} dB")
            print("  (Subset means only — not the reportable numbers.)")

    # ---- the verdict orhG asked on complete data ------------------------
    if not incomplete and all(c in summary for c in ("lam0", "lam1", "lamlearn")):
        learned = summary["lamlearn"]["PSNR"]
        print("\n=== Verdict: does learned Lambda beat both fixed extremes? ===")
        worst_margin = float("inf")
        for other in ("lam0", "lam1"):
            o = summary[other]["PSNR"]
            delta = learned[0] - o[0]
            pooled = (learned[1] ** 2 + o[1] ** 2) ** 0.5
            worst_margin = min(worst_margin, delta - pooled)
            verdict = "SEPARATED" if delta > pooled and delta > 0 else "WITHIN NOISE"
            print(
                f"  learned - {LABEL[other]:<22} = {delta:+.3f} dB "
                f"(pooled std {pooled:.3f}) -> {verdict}"
            )
        print()
        if worst_margin > 0:
            print("  => Claim SURVIVES on NeRF Synthetic. Report mean +/- std and keep it.")
        else:
            print("  => Claim does NOT survive on NeRF Synthetic.")
            print("     Per the rebuttal commitment to orhG, soften to parity here and")
            print("     rest the learned-Lambda argument on 6DGS-PBR (+1.21 / +0.44 dB).")

    if args.markdown:
        print("\n=== REBUTTAL.md table ===\n")
        print("| Config | PSNR (mean ± std) |")
        print("|---|---|")
        for c in configs:
            p = summary[c]["PSNR"]
            print(f"| {LABEL.get(c, c)} | {p[0]:.2f} ± {p[1]:.2f} |")


if __name__ == "__main__":
    main()
