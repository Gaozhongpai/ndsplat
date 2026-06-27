#!/usr/bin/env python3
"""Finalize a DF3DV-41 method: local metrics + leaderboard packaging.

For a trained method whose renders live in
    <root>/DF3DV-41/<scene>/<scene>-All/MODELS/<method>/renders/extra_*.png
this script:

  1. (--score)   Computes PSNR / SSIM / LPIPS locally against the bundled
                 extra_* GT, using the OFFICIAL benchmark_df3dv functions
                 (works for arbitrary method names — no METHODS-list edit).
                 Writes <root>/DF3DV-41_<method>_metrics.csv and prints means.

  2. (--extract) Runs the official extract_leaderboard_images on DF3DV-41,
                 producing <root>/leaderboard/<method>/DF3DV-41/<scene>/extra_*.png.

  3. (--zip)     Zips that tree to <out_dir>/<method>_DF3DV-41.zip, ready to
                 upload to the leaderboard (Google Drive + email per the form).

Default (no flags) runs all three.

Usage:
    python finalize_df3dv41.py --root /code/dataset/DF3DV-1K --method dgs_mcmc_df3dv41
"""
import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

# Official modules live alongside this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_df3dv as B          # extract_rendering_from_concat, load_rgb_float01, find_existing_image
import extract_leaderboard_images as E


def list_df3dv41_scene_dirs(root: Path):
    df41 = root / "DF3DV-41"
    if not df41.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-41 under {root}")
    return sorted([p for p in df41.iterdir() if p.is_dir()])


def _build_metrics(device):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    # Match the official benchmark_df3dv settings exactly (incl. LPIPS normalize=True,
    # which accepts [0,1] input and rescales internally).
    return (
        PeakSignalNoiseRatio(data_range=1.0).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device),
    )


def score_one_scene(scene_dir: Path, method: str, metrics, device):
    """Return (psnr_list, ssim_list, lpips_list) for one scene, or empty lists."""
    import torch
    psnr_m, ssim_m, lpips_m = metrics
    scene_all = E.get_scene_all_dir(scene_dir)
    renders_dir = scene_all / "MODELS" / method / "renders"
    gt_dir = scene_all / "undistortion_images_8"
    if not renders_dir.is_dir():
        return [], [], []

    def to_t(arr):
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    render_paths = sorted(renders_dir.glob("extra_*.png")) + sorted(renders_dir.glob("extra_*.PNG"))
    ps, ss, ls = [], [], []
    for rp in render_paths:
        stem = rp.stem
        pred = B.extract_rendering_from_concat(rp)              # right half (float01 HWC)
        gt = B.load_rgb_float01(B.find_existing_image(gt_dir / stem))
        if pred.shape != gt.shape:
            print(f"  [warn] shape mismatch {stem}: pred {pred.shape} gt {gt.shape}; skipping")
            continue
        pt, gtt = to_t(pred), to_t(gt)
        with torch.no_grad():
            ps.append(float(psnr_m(pt, gtt)))
            ss.append(float(ssim_m(pt, gtt)))
            ls.append(float(lpips_m(pt.clamp(0, 1), gtt.clamp(0, 1))))  # normalize=True -> feed [0,1]
    return ps, ss, ls


def score_single_scene(root: Path, method: str, scene_name: str):
    """Score one scene and APPEND its mean row to a live progress CSV.

    Used for incremental, per-scene scoring during a sweep so official DF3DV
    metrics accumulate live instead of only at the end.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = _build_metrics(device)
    scene_dir = root / "DF3DV-41" / scene_name
    if not scene_dir.is_dir():
        print(f"[score-1] scene not found: {scene_dir}")
        return
    ps, ss, ls = score_one_scene(scene_dir, method, metrics, device)
    if not ps:
        print(f"[score-1] {scene_name}: no scorable pairs (renders missing?)")
        return
    m_psnr, m_ssim, m_lpips = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    print(f"[score-1] {scene_name:35s} PSNR {m_psnr:6.3f}  SSIM {m_ssim:.4f}  LPIPS {m_lpips:.4f}  ({len(ps)} views)")

    # Append to a running CSV (create header once; skip if scene already logged).
    prog_csv = root / f"DF3DV-41_{method}_progress.csv"
    existing = set()
    if prog_csv.exists():
        with open(prog_csv) as f:
            for row in csv.reader(f):
                if row:
                    existing.add(row[0])
    write_header = not prog_csv.exists()
    with open(prog_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["scene", "psnr", "ssim", "lpips", "n_views"])
        if scene_name not in existing:
            w.writerow([scene_name, m_psnr, m_ssim, m_lpips, len(ps)])

    # Print running mean across everything logged so far.
    rows = []
    with open(prog_csv) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if row and row[0] != "MEAN":
                rows.append((row[0], float(row[1]), float(row[2]), float(row[3]), int(row[4])))
    if rows:
        tot = sum(n for *_, n in rows)
        wp = sum(p * n for _, p, _, _, n in rows) / tot
        wsm = sum(s * n for _, _, s, _, n in rows) / tot
        wl = sum(l * n for _, _, _, l, n in rows) / tot
        print(f"[score-1] running mean over {len(rows)} scenes: "
              f"PSNR {wp:.3f}  SSIM {wsm:.4f}  LPIPS {wl:.4f}")


def score_method(root: Path, method: str):
    """Local PSNR/SSIM/LPIPS for `method` across all DF3DV-41 scenes."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = _build_metrics(device)

    rows = []
    all_psnr, all_ssim, all_lpips = [], [], []
    for scene_dir in list_df3dv41_scene_dirs(root):
        ps, ss, ls = score_one_scene(scene_dir, method, metrics, device)
        if not ps:
            print(f"[skip] {scene_dir.name}: no renders/scorable pairs for method {method}")
            continue
        m_psnr, m_ssim, m_lpips = np.mean(ps), np.mean(ss), np.mean(ls)
        rows.append((scene_dir.name, m_psnr, m_ssim, m_lpips, len(ps)))
        all_psnr += ps; all_ssim += ss; all_lpips += ls
        print(f"  {scene_dir.name:35s} PSNR {m_psnr:6.3f}  SSIM {m_ssim:.4f}  LPIPS {m_lpips:.4f}  ({len(ps)} views)")

    out_csv = root / f"DF3DV-41_{method}_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "psnr", "ssim", "lpips", "n_views"])
        for r in rows:
            w.writerow(r)
        if all_psnr:
            w.writerow(["MEAN", np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips), len(all_psnr)])
    if all_psnr:
        print("-" * 70)
        print(f"  MEAN over {len(rows)} scenes: "
              f"PSNR {np.mean(all_psnr):.3f}  SSIM {np.mean(all_ssim):.4f}  LPIPS {np.mean(all_lpips):.4f}")
    print(f"[score] wrote {out_csv}")


def extract_leaderboard(root: Path, method: str):
    # Official helper returns [(None, scene_dir), ...] for DF3DV-41.
    entries = E.collect_41_scenes(root)
    # Skip scenes without renders so a partial sweep still packages cleanly
    # (the upstream extract_dataset raises on the first missing renders dir).
    kept, skipped = [], []
    for chunk, scene_dir in entries:
        rd = E.get_scene_all_dir(scene_dir) / "MODELS" / method / "renders"
        (kept if rd.is_dir() and any(rd.iterdir()) else skipped).append((chunk, scene_dir))
    if skipped:
        print(f"[extract] WARNING: {len(skipped)}/{len(entries)} scenes have no renders and are EXCLUDED "
              f"from the submission: {', '.join(s.name for _, s in skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))
    if not kept:
        raise FileNotFoundError(f"No scenes have renders for method '{method}'.")
    E.extract_dataset(root, "DF3DV-41", kept, method)
    out_dir = root / "leaderboard" / method / "DF3DV-41"
    print(f"[extract] {len(kept)} scenes -> {out_dir}")
    return out_dir


def zip_leaderboard(root: Path, method: str, out_dir: Path):
    src = root / "leaderboard" / method        # zip the <method>/ folder so DF3DV-41/ is inside
    if not src.is_dir():
        raise FileNotFoundError(f"Nothing to zip: {src} (run --extract first)")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{method}_DF3DV-41.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent))   # arcname: <method>/DF3DV-41/<scene>/extra_*.png
                n += 1
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[zip] {zip_path}  ({n} files, {size_mb:.1f} MB)")
    print(f"      Upload this to Google Drive and submit via the DF3DV leaderboard form.")
    return zip_path


def main():
    p = argparse.ArgumentParser(description="Finalize DF3DV-41 method: score + package for leaderboard.")
    p.add_argument("--root", required=True, help="Dataset root containing DF3DV-41/")
    p.add_argument("--method", required=True, help="Method name (= MODELS/<method>/renders folder)")
    p.add_argument("--out_dir", default=None, help="Where to write the zip (default: <root>/submissions)")
    p.add_argument("--score", action="store_true", help="Only compute local metrics")
    p.add_argument("--extract", action="store_true", help="Only extract leaderboard images")
    p.add_argument("--zip", action="store_true", help="Only zip (requires prior --extract)")
    p.add_argument("--single_scene", default=None,
                   help="Score ONE scene and append to a live progress CSV (for incremental scoring during a sweep).")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "submissions")

    # Incremental per-scene scoring takes precedence and does nothing else.
    if args.single_scene:
        score_single_scene(root, args.method, args.single_scene)
        return

    do_all = not (args.score or args.extract or args.zip)

    if args.score or do_all:
        print(f"=== Scoring {args.method} ===")
        score_method(root, args.method)
    if args.extract or args.zip or do_all:
        print(f"=== Extracting leaderboard images for {args.method} ===")
        extract_leaderboard(root, args.method)
        print(f"=== Zipping submission for {args.method} ===")
        zip_leaderboard(root, args.method, out_dir)


if __name__ == "__main__":
    main()
