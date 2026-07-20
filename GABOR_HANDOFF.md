# Projected Gabor Residual — Handoff

> **OUTCOME (2026-07-20): COMPLETE. Decision — do NOT adopt Gabor for RenderFM.**
> A view-consistent **projected Gabor residual** on the dGS base gains
> **+0.14 dB PSNR** on heart (7k residual-only fit, matched primitives);
> making its half-space clipping **exact** (complex-erf operator) is
> quality-neutral; putting the same band on the **Beta kernel** base helps
> less. This matches representation theory: Gabor's advantage scales with
> high-frequency oscillatory content, and cinematic CT is spectrally smooth.
> Not worth the complexity for RenderFM (browser renderer, feed-forward
> prediction). RenderFM C5 stays on **LoD-Gaussians**. The branch is kept as a
> verified reference implementation + documented results. One experiment still
> running: 30k from-scratch co-training (base + band + densification), see
> Open items.

## What this is

A **projected Gabor residual** (Gabor-Fields-inspired; NOT a port — upstream
uses independent residual primitives and staged pyramid training, here one
co-located modulation rides each existing base primitive and shares its
envelope, opacity, color, center). Per primitive: world-space wave vector
`_gabor_omega` (k, [N,3]), `_gabor_phase` [N,1], `_gabor_amp` [N,1]
(tanh-activated). amp is zero-init, so a fresh model renders **byte-identical**
to the base; dL/d(amp) is alive at amp==0 (bootstrap).

Per view, the Python projection (`project_gabor_band`,
`scene/gaussian_model_gabor.py`) conditions the 3D modulated atom on the pixel
ray and packs a float4 + one scalar per splat:

    omega_2d = -Sigma2d^-1 T^T Sigma k          (screen frequency; T = W.J)
    amp_eff  = tanh(amp) * exp(-0.5 Var(k.delta | ray))   (along-ray fade)
    b        = Cov(clip coord, wave phase | pixel)/s      (clipped views only,
                clamped to its Cauchy-Schwarz bound |b| <= sigma_xi)

The CUDA blend loop multiplies the footprint weight by:

    unclipped / beta kernel:  1 + amp_eff * cos(omega_2d . d + phase)
    Gaussian + active plane:  Phi(l) + amp_eff * Re{e^{i ph} Phi_c(l, b)},
                              Phi_c(l, b) = 0.5 erfc(-(l - i b)/sqrt(2))

The second form is the **EXACT half-space clip of a Gabor-modulated Gaussian**
— the complex-erf (Faddeeva) generalisation of XClipGS's Phi(l); potential
supplement material. CUDA evaluates Phi_c via Humlicek w4 (4.5e-5 vs scipy
wofz); backward derivatives are the analytic complex Gaussian. Everything is
differentiable end-to-end (grad_gabor / grad_gabor_b chain back into k through
the Python projection).

Frequency lives in the whitened frame of the base primitive: init
`||S R^T k|| ~ U(0.7, 1.5)` rad/sigma (~one oscillation per footprint);
`clamp_gabor_frequency()` keeps it in `[0.5, 3.0]` after each optimizer step
(adapted from upstream's BoundedAdam bounds — post-step norm rescaling, not
the identical optimizer rule). The [0.5,3] bound also caps |b|, keeping the
CUDA exponentials finite by construction.

## Modes / files

- `dgs-gabor` — `scene/gaussian_model_gabor.py` (dGS base). Registered in
  `scene/__init__.py`; train.py/render.py dispatch via existing branches;
  train.py calls `clamp_gabor_frequency()` after each step for any gabor mode.
- `dbs-gabor` — `scene/gaussian_model_beta_gabor.py` (dBS-SH beta-kernel
  base). Whitening/init/clamp via a Cholesky factor of `get_covariance`
  (= the `cov3D_precomp` the renderer consumes); keeps the approximate clip
  (the exact form is Gaussian-envelope specific).
- tcgs rasterizer: `gabor` [N,4] + optional `gabor_b` [N] buffers through
  forward/backward (`forward.cu`, `backward.cu`, `auxiliary.h::clipPhiGabor`,
  wrapper `__init__.py`). A bound gabor buffer disables the TCGS/bucket fast
  paths (they lack the residual); the exact path additionally needs an active
  plane + gabor_b, else bit-identical approximate path.
- Both models support full topology growth (clone/split/prune/relocate/
  add_new_gs) — fixed after external review, see below.
- Legacy: pre-v2 gabor PLYs (screen-space omega, raw amp; e.g.
  `heart_900_resonly`) are incompatible — re-fit instead of loading.

## Results on heart_900

**The test set is 45 intact + 45 clipped views** (train is also half clipped);
headline PSNRs are the mixed average. Splits computed from saved PNG renders
by the dataset `clip` flag (base row reproduces supplement Table S2's heart
row). All fits: 7k residual-only warm-start (base frozen, densify off) from
the respective 30k base checkpoint; matched primitive counts within a base.

   | model             | N       | all     | intact  | clipped | SSIM    | LPIPS   |
   |-------------------|---------|---------|---------|---------|---------|---------|
   | dGS base          | 270,627 | 29.2025 | 31.0363 | 27.3687 | 0.94038 | 0.09508 |
   | + gabor (approx)  | 270,627 | 29.3423 | 31.2350 | 27.4496 | 0.94115 | 0.09444 |
   | + gabor (exact)   | 270,627 | 29.3418 | 31.2341 | 27.4496 | 0.94115 | 0.09444 |
   | dBS-SH base       | 195,279 | 28.6280 | 31.0121 | 26.2440 | 0.93059 | 0.10791 |
   | dBS-SH + gabor    | 195,279 | 28.7183 | 31.1615 | 26.2751 | 0.93107 | 0.10743 |

Findings:
1. **The band gives a modest, intact-skewed gain on both bases**: dGS
   +0.199/+0.081 dB (intact/clipped), dBS +0.150/+0.031. Capacity is NOT
   binding (97% atoms active, median |amp| ~0.12-0.15, <0.1% at tanh
   saturation, ~1% at the omega hi-clamp) — one band per atom suffices;
   multi-band in the paper is emergent across primitives, not per-primitive.
2. **Exact clipping is quality-neutral** (approx vs exact identical to
   0.001 dB on every slice, clipped half included). The intact-vs-clipped gain
   gap is content-driven — cut-face error is dominated by terms a footprint
   modulation cannot fix — not the clip approximation. Value of the exact
   operator: correctness + the closed form itself.
3. **The dBS deficit is entirely a clipping story**: it TIES dGS on intact
   views (31.01 vs 31.04) with 28% fewer primitives and no mip filter, and
   loses its whole 0.57 dB on the clipped half where the Gaussian-derived
   `clipPhi` is approximate on a beta envelope. Beta's unlock would be an
   exact beta half-space integral, not more primitives.
4. **Cost** (A100, clipped 1600^2 view, median of 20): gabor forward 12.9 ms
   (approx) / 14.9 ms (exact, +15.5%; zero overhead on intact views);
   fwd+bwd 49.7 / 53.7 ms. Plain dGS standard forward: 5.1 ms — the band
   itself is the main cost (disables the TCGS fast path, adds the per-view
   projection + per-sample modulation).

Outputs under `/data/output/xclipgs/gabor/`: `heart_900_resonly_v2` (approx),
`heart_900_resonly_v3` (exact), `heart_900_dbs_sh`, `heart_900_dbs_gabor_resonly`.
dGS base: `/data/output/xclipgs/ours/heart_900`.

## Verification (all PASS, 2026-07-20)

- `scripts/tests/gabor_projection_check.py` — CPU float64, non-circular:
  analytic screen map vs finite differences (~4e-10); projected band vs
  numerical line-integral conditional expectation (~2e-10); EXACT clipped band
  vs piecewise line integration with the half-space indicator (~3e-10,
  validates the Phi_c formula and the sign of b).
- `test_cutting_plane.py` (tcgs; run from /opt or via PYTHONPATH) — CUDA vs a
  float64 autograd reference (wofz-backed Phi_c with analytic backward):
  configs plane-inactive / amp=0 bootstrap / oblique clip approx / oblique
  clip EXACT (incl. d/d gabor_b, ~3e-6) / exact amp=0 bootstrap / beta kernel.
  Base grads stay correct with a residual active. Plus the pre-existing
  robustness/popping/leak/Adam suite.
- `scripts/tests/gabor_heart_parity.py` — amp=0 renders bitwise-equal to the
  base model on real 1600^2 heart views, INTACT and CLIPPED (`PARITY_VIEW`),
  for both model families (`GABOR_MODE`/`GABOR_PLY`).
- `scripts/tests/gabor_gradflow_check.py` — on the real checkpoint: amp
  bootstraps at 0 (k/phase grads exactly 0 there), k/phase grads alive at
  amp=0.1, packed buffer is view-dependent, whitened init/clamp in bounds.
- `scripts/tests/gabor_lifecycle_check.py` — clone/split/prune/relocate/
  add_new_gs/post-growth step invariants, both model families.
- FD check on a real clipped view: analytic dL/d(amp) vs finite differences,
  ratio ~1.0 (validates Python-b/CUDA-l consistency end-to-end).

## Numerics lessons (do not relearn these)

- **Clamp b to its Cauchy-Schwarz bound.** b is a ratio of two fp32
  Schur-complement DIFFERENCES; for splats whose plane nearly contains the
  ray, cancellation noise blew |b| to 7.6e3 (bound: sigma_xi <= 3) ->
  e^{b^2/2} overflow -> one NaN -> every splat killed via fmaxf(NaN,0)=0 ->
  black renders with finite flat loss (first exact-clip fit died at PSNR 12.6
  in ~80 iters). No synthetic test computed b from real degenerate geometry;
  an FD check + b-histogram on real data found it. Fixed in the projection
  (|b| <= sigma_xi, a mathematical no-op) + a CUDA-side guard in clipPhiGabor.
- fmaxf(NaN, 0) = 0 in CUDA: a NaN parameter silently BLANKS splats rather
  than crashing — flat loss at the GT mean means "black render", not "stuck".
- Trapezoid quadrature loses its spectral accuracy at a clipped endpoint
  (Euler-Maclaurin boundary term); integrate piecewise to the crossing or use
  adaptive quadrature with breakpoints when validating clipped integrals.

## How to run

Residual-only fit (current recipe; ~6 min train + render/metrics):
```bash
python train.py -s /data/nerf_dataset/heart_900 \
  -m /data/output/xclipgs/gabor/<out> \
  --mode dgs-gabor --input_dim 6 --use_view_dependent_pos False \
  --l_22_inv_init_scale 2.0 --mip3dgs --eval --disable_viewer \
  --iterations 7000 --gabor_residual_only --densify_until_iter 0 \
  --start_checkpoint /data/output/xclipgs/ours/heart_900/point_cloud/iteration_30000/point_cloud.ply
python render.py -m <out> --skip_train && python metrics.py -m <out>
```
dbs-gabor: same but `--mode dbs-gabor`, no `--mip3dgs`/`--use_view_dependent_pos`,
warm-start from the dbs-sh checkpoint. From scratch: drop the residual-only /
densify / start_checkpoint flags and use `--iterations 30000`.

Rebuild after any `.cu` edit (root-owned `build/` shadows incremental builds):
```bash
docker run --rm --gpus '"device=6"' --cpuset-cpus 72-83 \
  -v .../ndsplat:/workspace/ndsplat \
  -w /workspace/ndsplat/submodules/tcgs_speedy_rasterizer -e TORCH_CUDA_ARCH_LIST=8.0 \
  10.10.0.192:5555/zhongpai/ndgs:latest \
  bash -lc 'rm -rf build *.egg-info tcgs_speedy_rasterizer/_C*.so; pip install -e . 2>&1 | tail -2'
```
Confirm the in-tree `_C.*.so` mtime is fresh afterwards (a stale in-tree .so
silently shadows installs: fake OOMs, "no kernel image").

Environment:
- Image `10.10.0.192:5555/zhongpai/ndgs:latest`; mount vengine_data at /data.
- GPU 6 primary (cpuset 72-83). 2026-07-20: GPU 6 was taken by another user's
  job; user approved GPUs 2/3/4 as fallback (use the matching cpuset block,
  e.g. GPU 2 -> 24-35). Check nvidia-smi before launching.
- Run tcgs tests from OUTSIDE the submodule root (copy to /opt) AND with
  PYTHONPATH to the in-tree package — a stale installed wrapper otherwise
  shadows the editable install.
- `use_tcgs=True` is forward-only; gradient work uses `use_tcgs=False` (the
  wrapper forces this when gabor is active).
- Branch `feat/gabor-residual` (base `mip`); tcgs submodule branch ditto.
  Do NOT touch `mip`/`master` or `pages/XClipGS/**`.

## Open items

- **From-scratch co-training** (running 2026-07-20): 30k `dgs-gabor` with
  densification, identical flags to the dGS base training — tests whether
  co-adapting base + band beats the frozen-base residual (+0.14 dB). Output:
  `/data/output/xclipgs/gabor/heart_900_gabor_scratch/`. Record the result
  here when done.

## Independent implementation review (2026-07-20)

Reviewed `ndsplat` at `149631d`, `tcgs_speedy_rasterizer` at `7137219`, and
the upstream Gabor Fields repository at `009816f`. (Historical note: this
reviewed the pre-exact-clip state; points 1-2 are since resolved, see the
response below.)

**Verdict:** the fixed-topology, unclipped heart experiment is technically
sound, and its reported quality result is credible. The branch is not yet a
general densifying Gabor model, an exact clipped-Gabor implementation, or a
faithful port of the complete Gabor Fields system.

### Known correctness boundaries

1. **Densification and MCMC growth are broken.** `densify_and_clone`,
   `densify_and_split`, and `add_new_gs` place extension rows in
   `_pending_new_gabor`, but the dynamically dispatched
   `densification_postfix()` replaces that value with `None`. The base optimizer
   then requests a missing `gabor_omega` extension (`KeyError: 'gabor_omega'`),
   reproduced on the heart checkpoint. The reported fits are unaffected
   (`--densify_until_iter 0`).

2. **Half-space clipping is approximate for an active Gabor.** The true
   clipped-Gabor integral requires a complex error function (Faddeeva); the
   oblique CUDA test proves agreement with the implemented approximation, not
   with that exact integral.

3. **This is a constrained, Gabor-inspired residual rather than a Gabor Fields
   port.** Upstream uses independent base and residual primitives with separate
   centers, scales, orientations, signed opacities, budgets, and staged pyramid
   training; this branch attaches one co-located, nonnegative modulation to
   each existing primitive.

4. **The frequency bounds are adapted, not identical to upstream** (post-step
   norm rescaling of a 3D whitened vector vs upstream's scalar BoundedAdam;
   documentation should not imply an identical rule).

### Response (same day)

- Point 1 — **confirmed and fixed** (stash clobber in `densification_postfix`;
  sibling KeyError in the dbs base `replace_tensors_to_optimizer`; gabor Adam
  moments now reset at relocated slots). `gabor_lifecycle_check.py` covers all
  growth ops for both families: ALL PASS.
- Point 2 — **resolved**: the exact complex-erf operator is implemented and
  verified (see v3 above); measured quality-neutral on heart.
- Points 3-4 — **agreed**; terminology adjusted (projected Gabor residual),
  bounds documented as adapted. Point 4 is a documentation nit, not a defect:
  Adam moments are gradient EMAs, unaffected by parameter projection; the
  boundary population is ~1% of atoms and training was stable.
