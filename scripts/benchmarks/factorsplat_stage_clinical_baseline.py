#!/usr/bin/env python3
"""Stage a legacy 40-TF checkpoint for the 41-TF clinical evaluation.

The clinical bank retains 38 transfer functions byte-for-byte and replaces the
legacy inverse-gamma and region-hide stress tests with three clinical edits.
This helper links the already evaluated images for the common TFs into a fresh
model directory and leaves exactly the three new TF rows for rendering.  The
mapping is by ``(tf_id, camera ordinal)`` rather than by global frame index.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import os
import re
import shutil
from pathlib import Path


NEW_OOD = {
    "test_ood_02_target_isolation",
    "test_ood_03_occluder_suppression",
    "test_ood_04_target_context",
}
REMOVED_OOD = {"test_ood_02_invert_gamma", "test_ood_03_hide"}


def frames(path: Path) -> list[dict]:
    return json.loads(path.read_text())["frames"]


def source_path_from_cfg(path: Path) -> Path:
    text = path.read_text()
    match = re.search(r"(?:^|, )source_path=(['\"])(.*?)\1(?:,|\))", text)
    if not match:
        raise ValueError(f"cannot read source_path from {path}")
    return Path(match.group(2))


def tf_camera_keys(items: list[dict]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    keys = []
    for item in items:
        tf_id = item["tf_id"]
        ordinal = counts.get(tf_id, 0)
        counts[tf_id] = ordinal + 1
        keys.append((tf_id, ordinal))
    return keys


def link_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"refusing to replace {destination}")
        return
    destination.symlink_to(os.path.relpath(source, destination.parent))


def same_camera(left: list[list[float]], right: list[list[float]]) -> bool:
    """Compare serialized camera matrices up to JSON round-off.

    Regenerating a manifest can change the last binary digit of a matrix entry
    (the observed maximum is 3.4e-16) without changing the camera.  A strict
    list comparison incorrectly rejects those manifests, while the tight
    absolute tolerance still catches any meaningful pose or ordinal drift.
    """
    if len(left) != len(right):
        return False
    return all(
        len(left_row) == len(right_row)
        and all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def frame_image(dataset: Path, item: dict) -> Path:
    relative = item["file_path"].removeprefix("./")
    path = dataset / relative
    return path if path.suffix else path.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--clinical-dataset", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    args = parser.parse_args()

    source_model = args.source_model.resolve()
    clinical_dataset = args.clinical_dataset.resolve()
    target_model = args.target_model.resolve()
    source_cfg = source_model / "cfg_args"
    source_dataset = source_path_from_cfg(source_cfg)
    old_frames = frames(source_dataset / "transforms_test.json")
    new_frames = frames(clinical_dataset / "transforms_test.json")
    old_ids = {item["tf_id"] for item in old_frames}
    new_ids = {item["tf_id"] for item in new_frames}
    common = old_ids & new_ids

    if len(old_frames) != 40 * 40 or len(new_frames) != 41 * 40:
        raise ValueError(
            f"expected 1600 legacy and 1640 clinical frames, got "
            f"{len(old_frames)} and {len(new_frames)}")
    if len(common) != 38 or new_ids - old_ids != NEW_OOD:
        raise ValueError(
            f"unexpected bank difference: common={len(common)}, "
            f"new={sorted(new_ids-old_ids)}, old={sorted(old_ids-new_ids)}")
    if old_ids - new_ids != REMOVED_OOD:
        raise ValueError(f"unexpected removed TFs: {sorted(old_ids-new_ids)}")

    old_keys = tf_camera_keys(old_frames)
    new_keys = tf_camera_keys(new_frames)
    old_by_key = {key: index for index, key in enumerate(old_keys)}
    if len(old_by_key) != len(old_keys):
        raise ValueError("legacy manifest has duplicate (tf_id, camera) keys")

    point_cloud = source_model / "point_cloud"
    if not (point_cloud / "iteration_30000/point_cloud.ply").is_file():
        raise FileNotFoundError(point_cloud / "iteration_30000/point_cloud.ply")
    target_model.mkdir(parents=True, exist_ok=True)
    target_pc = target_model / "point_cloud"
    if target_pc.exists() or target_pc.is_symlink():
        if target_pc.resolve() != point_cloud.resolve():
            raise FileExistsError(f"refusing to replace {target_pc}")
    else:
        target_pc.symlink_to(os.path.relpath(point_cloud, target_model))
    target_cfg = target_model / "cfg_args"
    if not target_cfg.exists():
        shutil.copyfile(source_cfg, target_cfg)

    linked = 0
    pending_indices = []
    source_test = source_model / "test/ours_30000"
    target_test = target_model / "test/ours_30000"
    for new_index, key in enumerate(new_keys):
        if key[0] not in common:
            pending_indices.append(new_index)
            continue
        old_index = old_by_key[key]
        # The camera poses must also match; this catches accidental ordinal drift.
        if not same_camera(old_frames[old_index]["transform_matrix"],
                           new_frames[new_index]["transform_matrix"]):
            raise ValueError(f"camera mismatch for {key}")
        old_gt = frame_image(source_dataset, old_frames[old_index])
        new_gt = frame_image(clinical_dataset, new_frames[new_index])
        if not (os.path.samefile(old_gt, new_gt)
                or filecmp.cmp(old_gt, new_gt, shallow=False)):
            raise ValueError(f"reference-image mismatch for {key}")
        for leaf in ("renders", "gt"):
            link_checked(source_test / leaf / f"{old_index:05d}.png",
                         target_test / leaf / f"{new_index:05d}.png")
        linked += 1

    pending_tfs = sorted({new_frames[i]["tf_id"] for i in pending_indices})
    pending_rows = sorted({int(new_frames[i]["tf_index"]) for i in pending_indices})
    if len(pending_indices) != 120 or set(pending_tfs) != NEW_OOD:
        raise ValueError(
            f"expected 120 new clinical frames, got {len(pending_indices)} "
            f"from {pending_tfs}")
    print(json.dumps({
        "linked_frames": linked,
        "pending_frames": len(pending_indices),
        "pending_tf_ids": pending_tfs,
        "pending_tf_indices": pending_rows,
        "target_model": str(target_model),
    }, indent=2))


if __name__ == "__main__":
    main()
