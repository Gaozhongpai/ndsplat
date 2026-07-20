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

## Branch / commit state (start here)

- ndsplat repo: branch **`feat/gabor-residual`**, HEAD **`5b0a0e0`** (WIP).
- tcgs submodule (`submodules/tcgs_speedy_rasterizer`): HEAD **`dce5296`** (WIP).
- Base branch is `mip`. Do NOT touch `mip`/`master` or the XClipGS paper
  (`pages/XClipGS/**`).

Files already implemented (committed):
- `scene/gaussian_model_gabor.py` — `dgs-gabor` model: extends the dGS model with
  `_gabor_omega [N,3]`, `_gabor_phase [N,1]`, `_gabor_amp [N,1]`. omega random
  screen direction × 0.4 rad/px; amp/phase zero. save_ply/load_ply extended.
  Registered in `scene/__init__.py`. Also has a warm-start fix for loading a
  non-view-dependent dGS checkpoint into a view-dependent gabor fit.
- `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/forward.cu` (~L555-593):
  gabor_mult modulation in the blend loop. **Forward is CORRECT** (see parity
  below). Honestly notes the half-space clip of the Gabor (complex-erf
  generalization of `clipPhi`) is NOT applied — approximate only when a clip
  plane crosses a high-freq atom; fine for the no-clip heart fit.
- `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/backward.cu` (~L780-945):
  gabor gradient block. **This is where the bug is.**
- Python wrapper `submodules/tcgs_speedy_rasterizer/tcgs_speedy_rasterizer/__init__.py`:
  threads `gabor` through forward (L195-315) and returns `grad_gabor` last in the
  backward grads tuple (L384-400). `gabor_active = gabor.numel() > 0` correctly
  disables the bucket/tcgs fast paths (which lack the residual).
- `rasterize_points.cu`: allocates `dL_dgabor = zeros({P,4})` when
  `gabor.numel()>0` (L234), passes to `BACKWARD::render`.
- `scripts/tests/gabor_bootstrap_test.py` — the reproducer (see below).

## THE BUG (precisely diagnosed — do not re-derive)

Run the reproducer:
```bash
cd /mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat
docker run --rm --gpus '"device=6"' --cpuset-cpus 72-83 \
  -v $(pwd):/workspace/ndsplat -w /workspace/ndsplat \
  -e PYTHONPATH=/workspace/ndsplat/submodules/tcgs_speedy_rasterizer:/workspace/ndsplat/submodules/gsplat \
  10.10.0.192:5555/zhongpai/ndgs:latest \
  python scripts/tests/gabor_bootstrap_test.py
```
Current output:
```
[parity] max|dGS - gabor(amp=0,omega=0)|   = 0.000e+00   <- PASS: forward is dGS-identical at init
[parity] max|dGS - gabor(amp=0,omega=0.4)| = 0.000e+00   <- PASS
[grad@amp=0.0] whole gabor grad nonzero = 0/8000         <- FAIL
[grad@amp=0.1] whole gabor grad nonzero = 0/8000         <- FAIL  <-- KEY CLUE
```

**Interpretation:** The gabor gradient is **exactly zero even at amp=0.1**, where
the residual is fully active in the forward. So this is **NOT** a bootstrap-gate
problem (that theory is disproven — a gate blocking only amp==0 would still give
nonzero grad at amp=0.1). The **entire `dL_dgabor` path is unwired**: the backward
computes/returns zero, or the value never reaches the Python leaf. Forward value
path works; only the gradient path is dead.

## Ruled out (don't re-investigate)

- Stale `.so`: a fully clean rebuild was done (removed root-owned `build/` INSIDE
  the container, recompiled). Bug persists on fresh binary.
- Alpha-cull / saturation test artifact: reproduced with large splats (scale
  0.15, opacity 0.3) so many splat/pixel pairs are in the active,
  non-saturated regime. Still zero.
- Backward grads-tuple ordering: verified it matches the forward input signature
  (13 inputs → 13 grads, `grad_gabor` last aligns with `gabor` last). Correct.
- Bucket fast-path stealing the render: `BACKWARD::renderBuckets` is guarded by
  `gabor == nullptr` (rasterizer_impl.cu ~L599), so the full `BACKWARD::render`
  runs with `gabor` and `dL_dgabor` (L633/647/657). Correct.

## Prime suspects to check next (in order)

1. **`ctx.needs_input_grad` / autograd tracking of `gabor`.** In `__init__.py`
   backward, confirm `grad_gabor` is not being force-nulled. Line ~379:
   `if ctx.gabor_active and gabor is not None and gabor.numel()>0 and grad_gabor.numel()>0: reshape else None`.
   Check whether `grad_gabor` returned from `_C.rasterize_gaussians_backward` is
   actually shape `[P,4]` and whether `.numel()>0`. Add a debug print of
   `grad_gabor.abs().sum()` right after the `_C` call (before any guard) to see if
   the ZERO originates in CUDA or in the Python guard.

2. **The `_C.rasterize_gaussians_backward` arg list vs C++ signature.** Count and
   order the `args` tuple passed at `__init__.py` ~L330-360 against
   `RasterizeGaussiansBackwardCUDA` params in `rasterize_points.cu` (~L200-236). An
   off-by-one (e.g. `gabor` landing in the wrong positional slot) would leave the
   CUDA `gabor_ptr` null → the L934 gate `gabor_present=(gabor!=nullptr)` is false
   → zero grad, while the FORWARD (separate arg list) still works. **This is the
   most likely culprit given forward-works / backward-zero.**

3. **`dL_dgabor` return position** in the C++ `std::make_tuple` (rasterize_points.cu
   L284) vs the Python unpack (`__init__.py` L365/371). If the tuple has 11 elements
   and Python unpacks 11 names, confirm `grad_gabor` is the same slot on both sides.

4. If CUDA-side is confirmed receiving a valid `gabor_ptr` and `dL_dgabor_ptr`:
   put a `printf` (or write a sentinel) in backward.cu right before the L934 gate
   to confirm the block executes and `v_gab != 0`.

## Debug method that works here

The reproducer (`gabor_bootstrap_test.py`) tests the raw rasterizer directly with
random splats — model/training-independent, fast (~seconds). Use it as the
inner loop. Add prints at each suspected layer (Python grad_gabor sum → C++
pointer non-null → CUDA block reached) to localize where zero first appears.

**Rebuild after any `.cu` edit** (the root-owned build dir shadows incremental
builds — remove it inside the container):
```bash
docker run --rm --gpus '"device=6"' --cpuset-cpus 72-83 \
  -v /mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat:/workspace/ndsplat \
  -w /workspace/ndsplat/submodules/tcgs_speedy_rasterizer -e TORCH_CUDA_ARCH_LIST=8.0 \
  10.10.0.192:5555/zhongpai/ndgs:latest \
  bash -lc 'rm -rf build *.egg-info tcgs_speedy_rasterizer/_C*.so; pip install -e . 2>&1 | tail -2'
```

## Environment / infra

- Docker image: `10.10.0.192:5555/zhongpai/ndgs:latest`.
- **GPU 6 ONLY** (`--gpus '"device=6"'`), cpuset `72-83`. GPU 7 may be others'.
- Heart data (for the eventual fit, once gradients work):
  - dGS checkpoint / point cloud: `/mnt/uNeon/zhongpai/vengine_data/output/xclipgs/ours/heart_900/`
  - dataset: `/mnt/uNeon/zhongpai/vengine_data/nerf_dataset/heart_900/`
  - Mount vengine_data at `/data` inside the container.

## Definition of done

1. `gabor_bootstrap_test.py` prints **BOOTSTRAP: PASS** (nonzero dL/d(amp) at
   amp=0) and **PARITY: PASS** (already passing — keep it passing).
2. Optionally a finite-difference gradient check on amp/omega/phase for a couple
   splats (numerical vs analytic within ~1e-2 relative).
3. Then: warm-start the heart dGS checkpoint into `dgs-gabor`, fit ~a few k iters,
   render, and report heart PSNR/SSIM of (dGS base) vs (dGS + gabor residual) at
   matched primitive count. **Report honestly** — a no-improvement result is valid
   and informative (it answers whether Gabor helps on smooth CT anatomy).
