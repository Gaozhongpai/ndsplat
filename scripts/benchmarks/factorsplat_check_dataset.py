#!/usr/bin/env python3
"""Validate FactorSplat TF metadata, splits, and matched test cameras."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    train = json.loads((root / "transforms_train.json").read_text())["frames"]
    test = json.loads((root / "transforms_test.json").read_text())["frames"]
    with np.load(root / "tf_bank.npz", allow_pickle=False) as data:
        tf_ids = data["tf_ids"].astype(str).tolist()
        rgba_shape = data["rgba"].shape
        label_count = len(data["label_ids"])

    errors = []
    tf_set = set(tf_ids)
    for split_name, frames in (("train", train), ("test", test)):
        for frame in frames:
            tf_id = frame.get("tf_id")
            tf_index = frame.get("tf_index")
            if tf_id not in tf_set:
                errors.append(f"{split_name}: unknown tf_id {tf_id}")
            elif tf_index is None or tf_ids[int(tf_index)] != tf_id:
                errors.append(f"{split_name}: inconsistent tf index/id {tf_index}/{tf_id}")
            image = root / f"{frame['file_path'][2:]}.png"
            if not image.exists():
                errors.append(f"{split_name}: missing {image}")
    bad_train = [f["tf_id"] for f in train if f.get("tf_split") != "train"]
    if bad_train:
        errors.append(f"combined train exposes held-out TFs: {sorted(set(bad_train))}")

    cameras_by_tf = defaultdict(set)
    for frame in test:
        cameras_by_tf[frame["tf_id"]].add(frame.get("source_config"))
    camera_sets = list(cameras_by_tf.values())
    if camera_sets and any(cameras != camera_sets[0] for cameras in camera_sets[1:]):
        errors.append("test TFs do not share the same source-camera set")

    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(f"FactorSplat dataset validation failed with {len(errors)} error(s)")

    train_counts = Counter(f["tf_id"] for f in train)
    test_counts = Counter(f["tf_id"] for f in test)
    print(f"FactorSplat dataset OK: {root}")
    print(f"  bank : {rgba_shape[0]} TFs x {label_count} labels x {rgba_shape[2]} samples")
    print(f"  train: {len(train)} frames / {len(train_counts)} TFs / counts={sorted(set(train_counts.values()))}")
    print(f"  test : {len(test)} frames / {len(test_counts)} TFs / counts={sorted(set(test_counts.values()))}")
    print(f"  paired test cameras: {len(camera_sets[0]) if camera_sets else 0}")


if __name__ == "__main__":
    main()
