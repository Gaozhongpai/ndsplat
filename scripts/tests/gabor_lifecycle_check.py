#!/usr/bin/env python3
"""Topology-growth lifecycle test for the gabor models (dgs-gabor, dbs-gabor).

External review (2026-07-20) found that every growth path crashed with
KeyError: 'gabor_omega' — densification_postfix clobbered the
_pending_new_gabor stash before cat_tensors_to_optimizer consumed it (dgs), and
the dbs base replace_tensors_to_optimizer indexes every optimizer group with no
membership guard. This test exercises, on a synthetic model:

    densify_and_clone -> densify_and_split -> prune_points ->
    relocate_gs -> add_new_gs -> optimizer step after growth

asserting after every op that all gabor tensors have exactly N rows (N = xyz
rows), the optimizer param groups alias the model attributes, and Adam state
shapes match — for BOTH model families.

Run inside the ndgs container:
    python scripts/tests/gabor_lifecycle_check.py
"""
import argparse
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import numpy as np
import torch


def make_training_args():
    from arguments import OptimizationParams
    parser = argparse.ArgumentParser()
    op = OptimizationParams(parser)
    return op.extract(parser.parse_args([]))


def make_model(mode, n=64):
    from scene import get_gaussian_model
    from utils.graphics_utils import BasicPointCloud
    rng = np.random.default_rng(0)
    pcd = BasicPointCloud(points=rng.normal(0, 1, (n, 3)),
                          colors=rng.uniform(0, 1, (n, 3)),
                          normals=np.zeros((n, 3)))
    if mode == "dgs-gabor":
        m = get_gaussian_model(mode)(3, input_dim=6, use_view_dependent_pos=False,
                                     use_opacity_pos_decouple=False,
                                     l_22_inv_init_scale=2.0, lambda_init=-1.2,
                                     lambda_opc=0.35)
    else:
        m = get_gaussian_model(mode)(3, input_dim=6, l_22_inv_init_scale=2.0)
    m.create_from_pcd(pcd, spatial_lr_scale=1.0)
    m.training_setup(make_training_args())
    return m


def check_invariants(m, label):
    n = m.get_xyz.shape[0]
    for attr, name, dim in m._GABOR_SPECS:
        t = getattr(m, attr)
        assert t.shape == (n, dim), f"{label}: {attr} {tuple(t.shape)} != ({n},{dim})"
        # optimizer group must alias the model attribute
        found = [g for g in m.optimizer.param_groups if g.get("name") == name]
        assert len(found) == 1, f"{label}: optimizer group {name} missing"
        assert found[0]["params"][0] is t, f"{label}: {name} group not aliasing model attr"
        state = m.optimizer.state.get(t, None)
        if state:
            assert state["exp_avg"].shape == t.shape, f"{label}: {name} Adam state shape"
    return n


def fake_step(m):
    """Give every param nonzero grad and step, so Adam state exists."""
    loss = sum(p.sum() for g in m.optimizer.param_groups for p in g["params"]
               if p.requires_grad and p.numel() > 0) * 1e-6
    loss.backward()
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)


def run(mode):
    print(f"--- {mode} ---")
    m = make_model(mode)
    n0 = check_invariants(m, "init")
    fake_step(m)  # populate Adam state before growth ops
    check_invariants(m, "post-step")

    # clone: force-select via zero threshold and tiny percent_dense
    m.percent_dense = 100.0  # every scale <= percent_dense * extent -> clone-eligible
    grads = torch.full((m.get_xyz.shape[0], 1), 1.0, device="cuda")
    m.densify_and_clone(grads, grad_threshold=0.5, scene_extent=1.0)
    n1 = check_invariants(m, "clone")
    print(f"[clone]    {n0} -> {n1}  OK")

    # split: make everything split-eligible
    m.percent_dense = 1e-12
    grads = torch.full((m.get_xyz.shape[0], 1), 1.0, device="cuda")
    m.densify_and_split(grads, grad_threshold=0.5, scene_extent=1.0)
    n2 = check_invariants(m, "split")
    print(f"[split]    {n1} -> {n2}  OK")

    # prune a third
    mask = torch.zeros(n2, dtype=torch.bool, device="cuda")
    mask[::3] = True
    m.prune_points(mask)
    n3 = check_invariants(m, "prune")
    print(f"[prune]    {n2} -> {n3}  OK")

    # MCMC: relocate a quarter, then add up to a cap. train.py invokes these
    # inside torch.no_grad() (the base does in-place index_copy_ on leaves).
    dead = torch.zeros(n3, dtype=torch.bool, device="cuda")
    dead[::4] = True
    with torch.no_grad():
        m.relocate_gs(dead_mask=dead)
    check_invariants(m, "relocate")
    amp_dead = m._gabor_amp[dead.nonzero(as_tuple=True)[0]].abs().max().item()
    assert amp_dead == 0.0, "relocated slots must restart with zero amp"
    print(f"[relocate] {int(dead.sum())} slots re-seeded  OK")

    with torch.no_grad():
        added = m.add_new_gs(cap_max=int(n3 * 1.5))
    n4 = check_invariants(m, "add_new_gs")
    print(f"[add]      {n3} -> {n4} (+{added})  OK")

    # a step after all growth ops must not blow up
    fake_step(m)
    check_invariants(m, "final-step")
    print(f"[step]     post-growth optimizer step  OK")


def main():
    torch.manual_seed(0)
    for mode in ("dgs-gabor", "dbs-gabor"):
        run(mode)
    print("ALL PASS")


if __name__ == "__main__":
    main()
