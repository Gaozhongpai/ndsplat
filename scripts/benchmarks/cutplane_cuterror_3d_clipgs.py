#!/usr/bin/env python3
"""Optional system-level CErr3D diagnostic for the ClipGS reimplementation.

This is deliberately separate from ``cutplane_cuterror_3d.py``.  It evaluates
ClipGS's learned deformation plus binary center cull on ClipGS's own vanilla-3DGS
cloud.  It therefore must not be presented as a fixed-interior operator comparison
with Ours/MM/HC.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from torch import nn

from cutplane_cuterror_3d import (SCENES, load_cloud, load_planes, normal_sigma)


class ClipDeformMLP(nn.Module):
    def __init__(self, width=64, deform_scale=False, out_scale=1e-3):
        super().__init__()
        self.deform_scale = deform_scale
        self.out_scale = out_scale
        output_dim = 6 if deform_scale else 3
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.ReLU(inplace=True),
            nn.Linear(width, width), nn.ReLU(inplace=True),
            nn.Linear(width, output_dim),
        )

    def forward(self, xyz, signed_distance):
        output = self.net(torch.cat([xyz, signed_distance], dim=1)) * self.out_scale
        delta_xyz = output[:, :3]
        delta_logscale = output[:, 3:6] if self.deform_scale else None
        return delta_xyz, delta_logscale


def analyze_scene(scene, data_root, out_root):
    model_dir = out_root / "clipgs" / f"{scene}_900" / "point_cloud" / "iteration_best"
    xyz, scales, rotations, opacity = load_cloud(model_dir / "point_cloud.ply")
    checkpoint = torch.load(model_dir / "deform_mlp.pt", map_location="cpu")
    deform_scale = bool(checkpoint.get("deform_scale", False))
    mlp = ClipDeformMLP(deform_scale=deform_scale)
    mlp.load_state_dict(checkpoint["state_dict"])
    mlp.eval()

    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32)
    totals = {"exact_kept": 0.0, "hole": 0.0, "overshoot": 0.0}
    plane_results = []
    transform_path = data_root / "nerf_dataset" / f"{scene}_cuteval" / "transforms_test.json"
    for normal, offset, label in load_planes(transform_path):
        normal_tensor = torch.as_tensor(normal, dtype=torch.float32)
        signed = (xyz_tensor @ normal_tensor - float(offset)).unsqueeze(1)
        with torch.no_grad():
            delta_xyz, delta_logscale = mlp(xyz_tensor, signed)
        deformed_xyz = xyz + delta_xyz.numpy().astype(np.float64)
        deformed_scales = scales
        if delta_logscale is not None:
            deformed_scales = scales * np.exp(delta_logscale.numpy().astype(np.float64))

        sigma = normal_sigma(deformed_scales, rotations, normal)
        t = (offset - deformed_xyz @ normal) / sigma
        exact = opacity * ndtr(t)
        hard_mass = np.where(deformed_xyz @ normal <= offset, opacity, 0.0)
        hole = np.maximum(exact - hard_mass, 0.0)
        overshoot = np.maximum(hard_mass - exact, 0.0)
        components = {
            "exact_kept": float(exact.sum()),
            "hole": float(hole.sum()),
            "overshoot": float(overshoot.sum()),
        }
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
        "hole": totals["hole"] / denominator,
        "overshoot": totals["overshoot"] / denominator,
        "total": (totals["hole"] + totals["overshoot"]) / denominator,
    }
    return {"metrics": metrics, "components": totals, "planes": plane_results,
            "deform_scale": deform_scale}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/output/xclipgs"))
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/output/xclipgs/cuteval/cerr3d_clipgs_system.json"))
    args = parser.parse_args()

    results = {}
    print(f"{'scene':10s} {'hole':>8s} {'over':>8s} {'total':>8s}  (x10^-2)")
    for scene in args.scenes:
        result = analyze_scene(scene, args.data_root, args.out_root)
        results[scene] = result
        metric = result["metrics"]
        print(f"{scene:10s} {100*metric['hole']:8.3f} {100*metric['overshoot']:8.3f} "
              f"{100*metric['total']:8.3f}")

    average = {
        key: float(np.mean([results[scene]["metrics"][key] for scene in args.scenes]))
        for key in ("hole", "overshoot", "total")
    }
    print(f"{'AVG':10s} {100*average['hole']:8.3f} {100*average['overshoot']:8.3f} "
          f"{100*average['total']:8.3f}")
    payload = {
        "scope": "system-level ClipGS diagnostic on its own vanilla-3DGS cloud",
        "comparable_to_fixed_interior_ladder": False,
        "scene_average": average,
        "scenes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
