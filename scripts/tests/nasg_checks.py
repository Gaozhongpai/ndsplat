#!/usr/bin/env python3
"""Verification gates for the NASG-Gabor color model (`dgs-nasg`).

1. DC PARITY: with 0 active lobes the model's color is exactly the SH DC
   convention (SH_C0 * f_dc + 0.5, clamped), so rendering the heart SH
   checkpoint through dgs-nasg must match plain dgs with active_sh_degree=0
   on the same views (tolerance ~1e-6: torch vs CUDA fp32 FMA ordering).
2. EVAL REFERENCE: eval_nasg_gabor (fp32, vectorized) vs an independent
   float64 per-element port of upstream's spherical_nasg_gabor.cuh.
3. GRADFLOW / BOOTSTRAP: lobe weights are zero-init; dL/d(weight) must be
   alive at weight==0 while pos/shape grads are exactly zero there (they
   carry a factor weight), and all become alive at weight != 0.
4. LIFECYCLE: clone/split/prune/relocate/add_new_gs/post-growth step with
   row-count + optimizer invariants (the growth-path traps found in the
   Gabor band study).

Run inside the ndgs container (heart data mounted at /data):
    python scripts/tests/nasg_checks.py
"""
import argparse
import math
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import numpy as np
import torch

DATA = "/data/nerf_dataset/heart_900"
PLY = "/data/output/xclipgs/ours/heart_900/point_cloud/iteration_30000/point_cloud.ply"
MODEL_KW = dict(input_dim=6, use_view_dependent_pos=False,
                use_opacity_pos_decouple=False, l_22_inv_init_scale=2.0,
                lambda_init=-1.2, lambda_opc=0.35)
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def ref_nasg_gabor_f64(c0, pos, shape, weight, v, active):
    """Independent float64 per-element port of spherical_nasg_gabor.cuh."""
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    out = np.array(c0, dtype=np.float64).copy()
    for j in range(active):
        ct, cp, cu = [min(max(float(x), -0.999999), 0.999999) for x in pos[j]]
        st, sp, su = math.sqrt(1 - ct * ct), math.sqrt(1 - cp * cp), math.sqrt(1 - cu * cu)
        x = np.array([ct * cp * cu - st * su, st * cp * cu + ct * su, -sp * cu])
        z = np.array([ct * sp, st * sp, cp])
        lam = min(math.exp(float(shape[j][0])), 1e4)
        a = min(math.exp(float(shape[j][1])), 1e4)
        k = (math.tanh(float(shape[j][2])) + 1.0) * 20.0
        vz = float(v @ z)
        vx = float(v @ x)
        if vz >= 1.0 - 1e-7:
            pdf = 1.0
        elif vz <= -1.0 + 1e-7:
            pdf = 0.0
        else:
            K_base = (vz + 1.0) * 0.5
            K_exp = 5e-6 + a * vx * vx / (1.0 - vz * vz)
            E = K_base ** K_exp
            inv_norm = lam * math.sqrt(1.0 + a) / (2 * math.pi * (1.0 + 1e-8 - math.exp(-2.0 * lam)))
            gab = (1.0 + math.cos(k * vx)) * 0.5
            pdf = math.exp(2.0 * lam * (E * K_base - 1.0)) * E * gab * inv_norm
        out += pdf * np.asarray(weight[j], dtype=np.float64)
    return np.maximum(out, 0.0)


def make_training_args():
    from arguments import OptimizationParams
    parser = argparse.ArgumentParser()
    op = OptimizationParams(parser)
    return op.extract(parser.parse_args([]))


def load_cam(idx):
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    args = SimpleNamespace(resolution=-1, data_device="cuda",
                           white_background=False, use_jpeg_compression=False)
    infos = readCamerasFromTransforms(DATA, "transforms_test.json", False)
    return loadCam(args, idx, infos[idx], 1.0)


def main():
    from scene import get_gaussian_model
    from scene.gaussian_model_nasg import eval_nasg_gabor
    torch.manual_seed(0)

    # ---- 2. eval reference --------------------------------------------------
    print("== eval reference (fp32 vectorized vs float64 element port) ==")
    rng = np.random.default_rng(1)
    N, L = 64, 3
    c0 = rng.uniform(0, 1, (N, 3))
    pos = rng.uniform(-2, 2, (N, L, 3))
    shape = np.stack([rng.uniform(-2, 3, (N, L)), rng.uniform(-2, 3, (N, L)),
                      rng.uniform(-3, 3, (N, L))], axis=-1)
    weight = rng.uniform(-0.5, 1.0, (N, L, 3))
    dirs = rng.normal(0, 1, (N, 3))
    want = np.stack([ref_nasg_gabor_f64(c0[i], pos[i], shape[i], weight[i], dirs[i], L)
                     for i in range(N)])
    # float64 catches formula errors; float32 documents precision (the
    # exp(2*lam*(...)) term amplifies rounding by ~2*lam, lam up to ~20 in
    # this random-parameter sweep, so a few 1e-5 relative is expected).
    for dt, tol in ((torch.float64, 1e-10), (torch.float32, 2e-4)):
        got = eval_nasg_gabor(
            torch.tensor(c0, dtype=dt, device="cuda"),
            torch.tensor(pos, dtype=dt, device="cuda"),
            torch.tensor(shape, dtype=dt, device="cuda"),
            torch.tensor(weight, dtype=dt, device="cuda"),
            torch.tensor(dirs, dtype=dt, device="cuda"), L).cpu().numpy()
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
        check(f"eval matches reference ({dt})".replace("torch.", ""),
              rel < tol, f"max rel {rel:.2e} (tol {tol:.0e})")

    # ---- 1. DC parity on the heart checkpoint -------------------------------
    print("== DC parity (0 lobes vs plain dgs at SH degree 0) ==")
    cam = load_cam(0)
    dgs = get_gaussian_model("dgs")(3, **MODEL_KW)
    dgs.load_ply(PLY)
    dgs.active_sh_degree = 0
    dgs.background = torch.zeros(3, device="cuda")
    with torch.no_grad():
        img_dgs = dgs.render_tcgs(cam, use_tcgs=False)["render"]
    del dgs
    torch.cuda.empty_cache()

    m = get_gaussian_model("dgs-nasg")(3, lobe_number=1, **MODEL_KW)
    m.load_ply(PLY)
    m.background = torch.zeros(3, device="cuda")
    assert m.active_lobes == 0 and m._nasg_weight.abs().max().item() == 0.0
    with torch.no_grad():
        img_nasg = m.render_tcgs(cam, use_tcgs=False)["render"]
    d = (img_dgs - img_nasg).abs().max().item()
    check("dgs-nasg(0 lobes) == dgs(SH deg 0)", d < 1e-6,
          f"max|diff|={d:.2e} bitwise={torch.equal(img_dgs, img_nasg)}")
    check("render nonempty", img_dgs.max().item() > 0.01,
          f"max={img_dgs.max().item():.3f}")

    # ---- 3. gradflow / bootstrap --------------------------------------------
    print("== gradflow (weight bootstrap at 0) ==")
    m.active_lobes = 1
    img = m.render_tcgs(cam, use_tcgs=False)["render"]
    img.mean().backward()
    gw = m._nasg_weight.grad
    gp = m._nasg_pos.grad
    gs = m._nasg_shape.grad
    nzw = int((gw.abs().sum(1) > 0).sum())
    check("dL/d(weight) alive at weight=0", nzw > 0.5 * gw.shape[0],
          f"{nzw}/{gw.shape[0]} rows")
    mp = gp.abs().max().item() if gp is not None else 0.0
    ms = gs.abs().max().item() if gs is not None else 0.0
    check("pos/shape grads exactly 0 at weight=0", mp == 0.0 and ms == 0.0,
          f"max {mp:.1e}/{ms:.1e}")
    for p in (m._nasg_weight, m._nasg_pos, m._nasg_shape):
        p.grad = None
    with torch.no_grad():
        m._nasg_weight.uniform_(-0.05, 0.05)
    img = m.render_tcgs(cam, use_tcgs=False)["render"]
    img.mean().backward()
    nzp = int((m._nasg_pos.grad.abs().sum(1) > 0).sum())
    nzs = int((m._nasg_shape.grad.abs().sum(1) > 0).sum())
    check("pos/shape grads alive at weight!=0",
          nzp > 0.3 * gw.shape[0] and nzs > 0.3 * gw.shape[0],
          f"{nzp}/{nzs} rows")
    del m
    torch.cuda.empty_cache()

    # ---- 4. lifecycle ---------------------------------------------------------
    print("== lifecycle (clone/split/prune/relocate/add + step) ==")
    from utils.graphics_utils import BasicPointCloud
    rng = np.random.default_rng(0)
    n0 = 64
    pcd = BasicPointCloud(points=rng.normal(0, 1, (n0, 3)),
                          colors=rng.uniform(0, 1, (n0, 3)),
                          normals=np.zeros((n0, 3)))
    m = get_gaussian_model("dgs-nasg")(3, lobe_number=2, **MODEL_KW)
    m.create_from_pcd(pcd, spatial_lr_scale=1.0)
    m.training_setup(make_training_args())

    def invariants(label):
        n = m.get_xyz.shape[0]
        for attr, name in m._NASG_SPECS:
            t = getattr(m, attr)
            assert t.shape == (n, m.lobe_number * 3), f"{label}: {attr} {tuple(t.shape)}"
            found = [g for g in m.optimizer.param_groups if g.get("name") == name]
            assert len(found) == 1 and found[0]["params"][0] is t, f"{label}: {name} alias"
        return n

    def step():
        loss = sum(p.sum() for g in m.optimizer.param_groups for p in g["params"]
                   if p.requires_grad and p.numel() > 0) * 1e-6
        loss.backward()
        m.optimizer.step()
        m.optimizer.zero_grad(set_to_none=True)

    step(); invariants("post-step")
    m.percent_dense = 100.0
    m.densify_and_clone(torch.full((m.get_xyz.shape[0], 1), 1.0, device="cuda"), 0.5, 1.0)
    n1 = invariants("clone")
    m.percent_dense = 1e-12
    m.densify_and_split(torch.full((m.get_xyz.shape[0], 1), 1.0, device="cuda"), 0.5, 1.0)
    n2 = invariants("split")
    mask = torch.zeros(n2, dtype=torch.bool, device="cuda"); mask[::3] = True
    m.prune_points(mask)
    n3 = invariants("prune")
    dead = torch.zeros(n3, dtype=torch.bool, device="cuda"); dead[::4] = True
    with torch.no_grad():
        m.relocate_gs(dead_mask=dead)
    invariants("relocate")
    assert m._nasg_weight[dead.nonzero(as_tuple=True)[0]].abs().max().item() == 0.0
    with torch.no_grad():
        added = m.add_new_gs(cap_max=int(n3 * 1.5))
    n4 = invariants("add")
    step(); invariants("final-step")
    check("lifecycle", True, f"{n0}->{n1}->{n2}->{n3}->{n4}(+{added})")

    print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
