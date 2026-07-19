#!/usr/bin/env python3
"""Camera-independent geometric error for the XClipGS operator ladder.

The diagnostic fixes the Ours-trained interior and evaluates the analytic,
moment-matched (MM), and hard-cull (HC) operators at the exact planes stored in
each ``<scene>_cuteval/transforms_test.json``.  Unlike the historical script,
this implementation uses each Gaussian's full rotated covariance and the
persisted Mip-Splatting 3D filter.

For a plane keeping ``n dot x <= tau``, the exact kept opacity mass of one
Gaussian is ``alpha * Phi(t)``, where
``t = (tau - n dot mu) / sqrt(n.T @ Sigma @ n)``.  CErr3D is wrong-side mass
normalized by total exact kept mass and then averaged equally across scenes.
The analytic operator is zero by construction.  MM contributes the tail of its
moment-matched surrogate beyond the plane.  HC contributes removed kept mass
(``hole``) plus retained culled mass (``overshoot``).

Example in the ndgs container::

    python scripts/benchmarks/cutplane_cuterror_3d.py \
      --data-root /data --out-root /data/output/xclipgs \
      --output /data/output/xclipgs/cuteval/cerr3d.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.special import log_ndtr, ndtr


SCENES = ("gel", "intestine", "kneejoint", "lower",
          "vascular", "heart", "nose", "hand")
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def load_planes(path):
    """Return unique normalized (normal, offset, label) evaluation planes."""
    payload = json.loads(Path(path).read_text())
    planes = []
    seen = set()
    for frame in payload["frames"]:
        if "plane_normal" not in frame or "clip_offset" not in frame:
            continue
        normal = np.asarray(frame["plane_normal"], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            raise ValueError(f"zero plane normal in {path}")
        normal /= norm
        offset = float(frame["clip_offset"]) / norm
        key = tuple(np.round(np.r_[normal, offset], 10))
        if key in seen:
            continue
        seen.add(key)
        planes.append((normal, offset, frame.get("cut_axis", str(len(planes)))))
    if not planes:
        raise ValueError(f"no clipping planes in {path}")
    return planes


def quaternion_matrices(quaternions):
    """Convert scalar-first quaternions [w,x,y,z] to rotation matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-15)
    w, x, y, z = q.T
    out = np.empty((len(q), 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[:, 0, 1] = 2.0 * (x * y - w * z)
    out[:, 0, 2] = 2.0 * (x * z + w * y)
    out[:, 1, 0] = 2.0 * (x * y + w * z)
    out[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[:, 1, 2] = 2.0 * (y * z - w * x)
    out[:, 2, 0] = 2.0 * (x * z - w * y)
    out[:, 2, 1] = 2.0 * (y * z + w * x)
    out[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def load_cloud(path):
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    raw_scales = np.exp(np.column_stack(
        [vertex[f"scale_{axis}"] for axis in range(3)]).astype(np.float64))
    rotations = quaternion_matrices(np.column_stack(
        [vertex[f"rot_{axis}"] for axis in range(4)]))
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float64)))

    names = {prop.name for prop in vertex.properties}
    if "filter_3D" in names:
        filter_var = np.maximum(np.asarray(vertex["filter_3D"], dtype=np.float64), 0.0)
        old_s2 = raw_scales * raw_scales
        new_s2 = old_s2 + filter_var[:, None]
        compensation = np.sqrt(
            np.prod(old_s2, axis=1) / np.maximum(np.prod(new_s2, axis=1), 1e-300))
        scales = np.sqrt(new_s2)
        opacity *= compensation
    else:
        scales = raw_scales
    return xyz, scales, rotations, opacity


def normal_sigma(scales, rotations, normal):
    """Compute sqrt(n.T Sigma n) for Sigma=R diag(scales^2) R.T."""
    local_normal = np.einsum("nji,j->ni", rotations, normal)
    variance = np.sum((local_normal * scales) ** 2, axis=1)
    return np.sqrt(np.maximum(variance, 1e-24))


def operator_components(t, opacity):
    log_z = log_ndtr(t)
    z = ndtr(t)
    exact = opacity * z

    hc_mass = np.where(t >= 0.0, opacity, 0.0)
    hc_hole = np.maximum(exact - hc_mass, 0.0)
    hc_overshoot = np.maximum(hc_mass - exact, 0.0)

    mills = np.exp(np.clip(-0.5 * t * t - LOG_SQRT_2PI - log_z, -745.0, 709.0))
    mm_mean = -mills
    mm_var = np.maximum(1.0 - t * mills - mills * mills, 1e-12)
    boundary = (t - mm_mean) / np.sqrt(mm_var)
    mm_overshoot = exact * ndtr(-boundary)
    return {
        "exact_kept": float(exact.sum()),
        "mm_hole": 0.0,
        "mm_overshoot": float(mm_overshoot.sum()),
        "hc_hole": float(hc_hole.sum()),
        "hc_overshoot": float(hc_overshoot.sum()),
    }


def analyze_scene(scene, data_root, out_root, geometry):
    transform_path = data_root / "nerf_dataset" / f"{scene}_cuteval" / "transforms_test.json"
    ply_path = out_root / geometry / f"{scene}_900" / "point_cloud" / \
        "iteration_best" / "point_cloud.ply"
    xyz, scales, rotations, opacity = load_cloud(ply_path)

    totals = {"exact_kept": 0.0, "mm_hole": 0.0, "mm_overshoot": 0.0,
              "hc_hole": 0.0, "hc_overshoot": 0.0}
    plane_results = []
    for normal, offset, label in load_planes(transform_path):
        sigma = normal_sigma(scales, rotations, normal)
        t = (offset - xyz @ normal) / sigma
        components = operator_components(t, opacity)
        for key, value in components.items():
            totals[key] += value
        plane_results.append({
            "axis": label,
            "normal": normal.tolist(),
            "offset": offset,
            "components": components,
        })

    denominator = max(totals["exact_kept"], 1e-300)
    metrics = {
        "ours": {"hole": 0.0, "overshoot": 0.0, "total": 0.0},
        "mm": {
            "hole": totals["mm_hole"] / denominator,
            "overshoot": totals["mm_overshoot"] / denominator,
            "total": (totals["mm_hole"] + totals["mm_overshoot"]) / denominator,
        },
        "hc": {
            "hole": totals["hc_hole"] / denominator,
            "overshoot": totals["hc_overshoot"] / denominator,
            "total": (totals["hc_hole"] + totals["hc_overshoot"]) / denominator,
        },
    }
    return {"metrics": metrics, "components": totals, "planes": plane_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/output/xclipgs"))
    parser.add_argument("--geometry", default="ours",
                        help="fixed checkpoint geometry used for all three operators")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/output/xclipgs/cuteval/cerr3d.json"))
    args = parser.parse_args()

    results = {}
    print(f"{'scene':10s} {'ours':>8s} {'MM':>8s} {'HC':>8s}  (CErr3D x10^-2)")
    for scene in args.scenes:
        result = analyze_scene(scene, args.data_root, args.out_root, args.geometry)
        results[scene] = result
        metrics = result["metrics"]
        print(f"{scene:10s} {0.0:8.3f} {100*metrics['mm']['total']:8.3f} "
              f"{100*metrics['hc']['total']:8.3f}")

    average = {
        operator: {
            component: float(np.mean([
                results[scene]["metrics"][operator][component] for scene in args.scenes]))
            for component in ("hole", "overshoot", "total")
        }
        for operator in ("ours", "mm", "hc")
    }
    print(f"{'AVG':10s} {0.0:8.3f} {100*average['mm']['total']:8.3f} "
          f"{100*average['hc']['total']:8.3f}")

    payload = {
        "metric": "CErr3D: wrong-side opacity mass / exact kept opacity mass",
        "geometry": args.geometry,
        "planes": "unique normals and offsets from each cuteval transforms_test.json",
        "covariance": "full quaternion covariance with persisted Mip-Splatting filter",
        "scene_average": average,
        "scenes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
