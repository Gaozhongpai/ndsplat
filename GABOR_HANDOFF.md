# Gabor Residual Integration — Handoff

**Goal:** Add a **residual Gabor band** on top of the existing **dGS** base in the
tcgs rasterizer + ndsplat, per-scan optimize it on the **heart** scene, and report
whether it improves render quality over plain dGS. Architecture constraint (from
the user, non-negotiable): the dGS base and its training stay **unchanged**; Gabor
is an **additive residual** (Gaussian base + residual Gabor kernels, per the Gabor
Fields paper). `amp` is zero-initialized so a fresh model renders **byte-identical
to dGS**, then trains up.

A Gabor atom modulates the footprint weight:
`alpha = opacity * kernel_weight * clip_mult * gabor_mult`, where
`gabor_mult = 1 + amp * cos(omega·d + phase)`, `d` = pixel offset from projected
center, `gabor = {omega_x, omega_y, phase, amp}` (a `float4` per Gaussian).

## Branch / commit state

- ndsplat repo: branch **`feat/gabor-residual`** (HEAD `4fb13ae` or later).
- tcgs submodule (`submodules/tcgs_speedy_rasterizer`): HEAD `7137219` or later.
- Base branch is `mip`. Do NOT touch `mip`/`master` or the XClipGS paper
  (`pages/XClipGS/**`).

Files implemented (all committed):
- `scene/gaussian_model_gabor.py` — `dgs-gabor` model: extends the dGS model with
  `_gabor_omega [N,3]`, `_gabor_phase [N,1]`, `_gabor_amp [N,1]`. omega random
  screen direction × 0.4 rad/px; amp/phase zero. save_ply/load_ply extended.
  Registered in `scene/__init__.py`. Also has a warm-start fix for loading a
  non-view-dependent dGS checkpoint into a view-dependent gabor fit.
- `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/forward.cu` (~L555-593):
  gabor_mult modulation in the blend loop. The half-space clip of the Gabor
  (complex-erf generalization of `clipPhi`) is NOT applied — approximate only
  when a clip plane crosses a high-freq atom; fine for the no-clip heart fit.
- `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/backward.cu` (~L780-945):
  gabor gradient block. Runs whenever a gabor buffer is bound (NOT gated on
  amp!=0) so dL/d(amp) is alive at amp==0 and the residual can bootstrap.
- Python wrapper `submodules/tcgs_speedy_rasterizer/tcgs_speedy_rasterizer/__init__.py`:
  threads `gabor` through forward and returns `grad_gabor` in the backward grads
  tuple. `gabor_active = gabor.numel() > 0` disables the bucket/tcgs fast paths
  (which lack the residual).
- `rasterize_points.cu`: allocates `dL_dgabor = zeros({P,4})` when
  `gabor.numel()>0`, passes to `BACKWARD::render`.
- `scripts/tests/gabor_bootstrap_test.py` — raw-rasterizer smoke test.
- `submodules/tcgs_speedy_rasterizer/test_cutting_plane.py` —
  `run_gabor_grad_checks`: full float64-reference gradient validation.

## STATUS: CUDA forward + backward VERIFIED CORRECT (2026-07-20)

The earlier "dL_dgabor path unwired" diagnosis was a **test artifact, not a
kernel bug**: the reproducer's synthetic camera used untransposed matrices, so
**zero splats were visible** — the render was black and ALL gradients (including
opacity) were zero. Parity "passed" vacuously (two black images) and the gabor
grads were zero for the same reason. After fixing the camera to the 3DGS
transposed convention (and using splats in the active, non-saturated regime):

`scripts/tests/gabor_bootstrap_test.py` (raw rasterizer, standard forward):
```
[parity] max|dGS - gabor(amp=0,omega=0)|   = 0.000e+00    PASS (byte-identical)
[parity] max|dGS - gabor(amp=0,omega=0.4)| = 0.000e+00    PASS
[grad@amp=0.0] max|dL/d(amp)| = 7.8e-05 (1645 nonzero entries)  BOOTSTRAP PASS
[grad@amp=0.1] omega/phase grads appear too (6388 nonzero)      PASS
```

`test_cutting_plane.py::run_gabor_grad_checks` (CUDA vs float64 autograd
reference; run from /opt, 4 configs: plane-inactive, amp=0 bootstrap, oblique
analytic clip, beta kernel):
- forward image rel L2 err ≤ 2.1e-4; all base grads (means3D/cov/opacity/shs/
  betas) still correct with a residual active (≤ 3.9e-4).
- d/d gabor omega/phase/amp rel L2 err ≤ 1.5e-4; **max elementwise rel err
  2.45e-4** (fast-math __cosf regime), ≤ 7e-6 in the other configs.
- amp=0 bootstrap: forward parity bit-exact (0.0), omega/phase grads exactly
  zero (they carry a factor amp), dL/d(amp) nonzero on 12/12 splats and matches
  the reference at 1.3e-6 rel L2.

## Rebuild after any `.cu` edit

The root-owned `build/` dir shadows incremental builds — remove it inside the
container:
```bash
docker run --rm --gpus '"device=6"' --cpuset-cpus 72-83 \
  -v /mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat:/workspace/ndsplat \
  -w /workspace/ndsplat/submodules/tcgs_speedy_rasterizer -e TORCH_CUDA_ARCH_LIST=8.0 \
  10.10.0.192:5555/zhongpai/ndgs:latest \
  bash -lc 'rm -rf build *.egg-info tcgs_speedy_rasterizer/_C*.so; pip install -e . 2>&1 | tail -2'
```
After a rebuild confirm the in-tree
`tcgs_speedy_rasterizer/tcgs_speedy_rasterizer/_C.*.so` mtime is fresh — a stale
in-tree `.so` silently shadows installed builds (fake OOMs, "no kernel image").

## Environment / infra

- Docker image: `10.10.0.192:5555/zhongpai/ndgs:latest`.
- **GPU 6 ONLY** (`--gpus '"device=6"'`), cpuset `72-83`. GPU 7 may be others'.
- Run tests from a cwd OUTSIDE the submodule root (e.g. copy to /opt) or via
  PYTHONPATH to the in-tree build; `test_cutting_plane.py` refuses to run from
  the submodule root by design.
- `use_tcgs=True` is a forward-only inference path; gradient work must use
  `use_tcgs=False` (the wrapper forces this automatically when gabor is active).
- Heart data (for the fit):
  - dGS checkpoint: `/mnt/uNeon/zhongpai/vengine_data/output/xclipgs/ours/heart_900/`
    (trained with `--mode dgs --input_dim 6 --use_view_dependent_pos False
    --l_22_inv_init_scale 2.0 --mip3dgs --eval`; warm-start the gabor fit with
    the SAME base flags plus `--mode dgs-gabor`).
  - dataset: `/mnt/uNeon/zhongpai/vengine_data/nerf_dataset/heart_900/`
  - Mount vengine_data at `/data` inside the container.

## Definition of done

1. `gabor_bootstrap_test.py` PARITY + BOOTSTRAP: **DONE (PASS)**.
2. Gradient check vs reference: **DONE** (max elementwise rel err 2.45e-4).
3. Remaining: end-to-end parity render of the heart dGS checkpoint through the
   model path (`dgs` vs `dgs-gabor`, same tcgs settings) — must be numerically
   identical; then warm-start the heart checkpoint into `dgs-gabor`, fit a few k
   iters (residual-focused; base frozen or tiny LRs), render, and report heart
   PSNR/SSIM of (dGS base) vs (dGS + gabor residual) at matched primitive count.
   **Report honestly** — a no-improvement result is valid and informative (it
   answers whether Gabor helps on smooth CT anatomy).
