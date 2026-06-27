#!/usr/bin/env python3
"""Thin wrapper around the official downsample_images.py that processes ONLY
the DF3DV-41 subset (the upstream main() also requires DF3DV-1K-Star/, which
we don't download). Uses the exact official mediapy-based downscaling so the
generated undistortion_images_<factor>/ match the leaderboard's eval GT.

Usage:
    python downsample_df3dv41.py --root /code/dataset/DF3DV-1K --factor 8 \
        --num_workers 8 --overwrite
"""
import argparse
from pathlib import Path

# Import the official, unmodified module sitting next to this file.
import downsample_images as official


def main():
    p = argparse.ArgumentParser(description="Downsample DF3DV-41 only (official mediapy resize).")
    p.add_argument("--root", required=True, help="Dataset root containing DF3DV-41/")
    p.add_argument("--factor", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    if args.factor <= 1:
        raise ValueError("--factor should be larger than 1")
    if args.num_workers <= 0:
        raise ValueError("--num_workers should be larger than 0")

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    official.process_df3dv41(root, args.factor, args.overwrite, args.num_workers)
    print("Done (DF3DV-41 only).")


if __name__ == "__main__":
    main()
