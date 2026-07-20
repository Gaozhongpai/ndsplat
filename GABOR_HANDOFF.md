# Gabor Residual Integration — Handoff

> **OUTCOME (2026-07-20): COMPLETE. Decision — do NOT adopt Gabor for RenderFM.**
> The residual works and is verified correct. Two parameterizations were fit on
> heart: screen-space omega gave **+0.09 dB PSNR**; the improved view-consistent
> projected wave vector (v2, see below) gave **+0.14 dB PSNR**. Both match
> representation theory: Gabor's advantage scales with high-frequency oscillatory
> content, and cinematic CT is spectrally smooth, so the residual has little to
> bite on. Not worth the primitive swap (would complicate the exact clip /
> browser renderer / feed-forward prediction). RenderFM C5 stays on
> **LoD-Gaussians** (importance-ordered, anatomy-aware via the mask). This branch
> is kept as a working reference implementation + a documented result. kneejoint
> (high-freq bone) was NOT run — even a favorable gain there would not change the
> decision.


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
Since v2 (commit `411fcdf`) that float4 is BUILT PER VIEW in Python: the model
stores a world-space wave vector `k` [N,3] and `_gabor_tensors_for_raster`
projects it to `omega_2d` + a Gaussian along-ray amplitude attenuation
(differentiable, so the CUDA `grad_gabor` chains back into `k`; the CUDA kernels
are unchanged).

## Branch / commit state

- ndsplat repo: branch **`feat/gabor-residual`** (HEAD `4fb13ae` or later).
- tcgs submodule (`submodules/tcgs_speedy_rasterizer`): HEAD `7137219` or later.
- Base branch is `mip`. Do NOT touch `mip`/`master` or the XClipGS paper
  (`pages/XClipGS/**`).

Files implemented (all committed):
- `scene/gaussian_model_gabor.py` — `dgs-gabor` model: extends the dGS model with
  `_gabor_omega [N,3]` (world-space wave vector k since v2), `_gabor_phase
  [N,1]`, `_gabor_amp [N,1]` (tanh-activated since v2). Whitened-frame init
  ||S Rᵀ k|| ~ U(0.7, 1.5) rad/sigma; amp/phase zero. Per-view projection +
  attenuation in `_gabor_tensors_for_raster` (closed form documented there);
  `clamp_gabor_frequency()` keeps ||S Rᵀ k|| in [0.5, 3.0] (called from train.py
  after each optimizer step). save_ply/load_ply extended. Registered in
  `scene/__init__.py`. Also has a warm-start fix for loading a
  non-view-dependent dGS checkpoint into a view-dependent gabor fit.
- `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/forward.cu` (~L555-593):
  gabor_mult modulation in the blend loop. The half-space clip of the Gabor
  (complex-erf generalization of `clipPhi`) is NOT applied — approximate when
  a clip plane crosses an active atom. CORRECTION (2026-07-20, user-caught):
  heart_900 is HALF clipped views in BOTH train and test (see supplement
  Table S2: heart iP 31.04 / cP 27.37 / aP 29.20), so this approximation IS
  exercised by the fits and included in every reported metric — the earlier
  "fine for the no-clip heart fit" note here was wrong. Per-half split of the
  v2 result: +0.199 dB intact, +0.081 dB clipped (see v2 section).
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

## Definition of done — ALL DONE (2026-07-20)

1. `gabor_bootstrap_test.py` PARITY + BOOTSTRAP: **PASS**.
2. Gradient check vs float64 reference: **PASS** (max elementwise rel err 2.45e-4).
3. Heart end-to-end parity (`scripts/tests/gabor_heart_parity.py`): dgs vs
   dgs-gabor(amp=0) **bitwise-equal** on a real 1600x1600 test view (270,627
   gaussians), both with the buffer bound and with `use_gabor=False`.
4. Heart residual fit: warm-start `ours/heart_900` iteration_30000 into
   `dgs-gabor` with `--gabor_residual_only --densify_until_iter 0` (base
   FROZEN — verified bit-identical in the saved PLY), 7000 iters, 5.8 min:
   ```
   python train.py -s /data/nerf_dataset/heart_900 \
     -m /data/output/xclipgs/gabor/heart_900_resonly \
     --mode dgs-gabor --input_dim 6 --use_view_dependent_pos False \
     --l_22_inv_init_scale 2.0 --mip3dgs --eval --disable_viewer \
     --iterations 7000 --gabor_residual_only --densify_until_iter 0 \
     --start_checkpoint /data/output/xclipgs/ours/heart_900/point_cloud/iteration_30000/point_cloud.ply
   ```
   Test-set metrics (render.py + metrics.py, matched 270,627 primitives):

   | model                    | PSNR    | SSIM    | LPIPS   |
   |--------------------------|---------|---------|---------|
   | dGS base (30k)           | 29.2025 | 0.94038 | 0.09508 |
   | + Gabor residual (7k)    | 29.2900 | 0.94094 | 0.09387 |

   **+0.09 dB PSNR / +0.0006 SSIM / −0.0012 LPIPS** — a real but modest gain
   on smooth CT anatomy (94% of atoms activate, median |amp| 0.13, omega
   drifts from the 0.4 rad/px seed to mean 0.51, max 1.73). Output:
   `/data/output/xclipgs/gabor/heart_900_resonly/` (results.json, renders,
   training.log). The training-loop eval (float renders, no PNG quantization)
   read 29.203 -> 29.331 (+0.13 dB) over the same fit.

## v2: view-consistent parameterization (2026-07-20, commit `411fcdf`)

After porting design choices from the Gabor Fields reference implementation
(github.com/Arcanous98/gabor_fields — MIT; NOTE it is a tomographic
Mitsuba/DrJIT codebase: Gaussians + ONE Gabor level, scalar omega in the
primitive's whitened frame, no phase parameter, signed amplitudes, BoundedAdam
omega bounds [0.5, 3], two-stage fit against a blurred pyramid):

1. **Projected wave vector (the big one).** `_gabor_omega` is a world-space k.
   Per view: `omega_2d = -Sigma2d^-1 T^T Sigma k` (T = the same W·J affine map
   the covariance projection uses), `amp_eff = tanh(amp) * exp(-0.5·Var(k·delta
   | ray))` — Gaussian conditioning of the 3D modulated atom on the pixel ray.
   Stripes now foreshorten correctly with view and waves along the viewing ray
   wash out instead of painting arbitrary stripes. Python-side and
   differentiable; CUDA unchanged. Verified against a finite-difference
   Jacobian of the real projection pipeline and numerical line-integral
   conditional expectations at float64 (`scripts/tests/gabor_projection_check.py`,
   ALL PASS ~4e-10); heart amp=0 parity still bit-exact; grad flow to k/phase
   verified (`scripts/tests/gabor_gradflow_check.py`).
2. **Whitened init + bounds.** Init ||S Rᵀ k|| ~ U(0.7, 1.5) rad/sigma (about
   one oscillation per footprint regardless of splat size); clamped to
   [0.5, 3.0] after each step. This also FIXED A BUG: the warm-start load_ply
   path used to fill omega with a constant (v,v,v) — the 29.29 fit above
   started with every atom on the SAME 45-degree screen stripe.
3. **tanh amp** — the CUDA `gabor_mult < 0` clamp dead zone is unreachable.

**COMPAT:** gabor PLYs written before v2 store screen-space omega and raw amp;
re-fit rather than load them into v2 code (`heart_900_resonly` is pre-v2).

Same fit recipe (7000 iters, `--gabor_residual_only --densify_until_iter 0`,
same warm start; ~5.5 min train). Output:
`/data/output/xclipgs/gabor/heart_900_resonly_v2/`. Test metrics (PNG-quantized,
matched 270,627 primitives):

   | model                          | PSNR    | SSIM    | LPIPS   |
   |--------------------------------|---------|---------|---------|
   | dGS base (30k)                 | 29.2025 | 0.94038 | 0.09508 |
   | + Gabor, screen-space (7k)     | 29.2900 | 0.94094 | 0.09387 |
   | + Gabor, projected k v2 (7k)   | 29.3423 | 0.94115 | 0.09444 |

   v2 = **+0.14 dB over base** (vs +0.09 for screen-space), best SSIM; LPIPS
   better than base but a hair behind the screen-space fit. Float-render
   training-loop eval: 29.203 -> 29.381 (+0.18 dB); v2 passed the ENTIRE old
   fit's final quality by iteration 2000, so the parameterization both
   converges faster and lands higher.

   **Per-half split (user-caught correction: heart_900 test is 45 intact + 45
   clipped views; all "PSNR" above are the mixed average).** From the saved
   PNG renders, view-classified by the dataset `clip` flag (reproduces
   supplement Table S2's heart row for the base):

   | model                      | all     | intact  | clipped |
   |----------------------------|---------|---------|---------|
   | dGS base (30k)             | 29.2025 | 31.0363 | 27.3687 |
   | + Gabor projected v2 (7k)  | 29.3423 | 31.2350 | 27.4496 |
   | v2 gain                    | +0.140  | +0.199  | +0.081  |

   The residual helps the intact half ~2.5x more than the clipped half —
   consistent with the unclipped-cosine clip approximation blunting the band
   where planes cross active atoms (and with cut-face error being dominated by
   terms the footprint modulation cannot fix).

   Capacity stats at 7k (answers "do we need multiple bands per atom?" — no):
   97.3% atoms active, median |amp| 0.150, p90 0.514, only 0.05% near tanh
   saturation; whitened |omega| median 1.08, p90 1.49, only 0.04% at the 3.0
   clamp (1.3% at the 0.5 floor). Neither amplitude nor frequency capacity is
   binding — a second/third band per atom would have nothing to bite on. Multi-
   band in the paper is emergent (many primitives, each with ONE frequency,
   masked by per-primitive band for LOD), and that per-primitive LOD masking is
   still possible here on amp.

**Conclusion unchanged:** the improved parameterization is real (+56% more gain,
faster convergence) but the absolute ceiling on smooth CT is too low to justify
adopting Gabor for RenderFM.

## Independent implementation review (2026-07-20)

Reviewed `ndsplat` at `149631d`, `tcgs_speedy_rasterizer` at `7137219`, and
the upstream Gabor Fields repository at `009816f`.

**Verdict:** the fixed-topology, unclipped heart experiment is technically
sound, and its reported quality result is credible. The branch is not yet a
general densifying Gabor model, an exact clipped-Gabor implementation, or a
faithful port of the complete Gabor Fields system.

### Verified behavior

- `scripts/tests/gabor_projection_check.py` passed all trials. The affine-EWA
  world-to-screen Jacobian matched finite differences, and the projected Gabor
  matched numerical line integration with maximum error about `6.5e-10`.
- The four CUDA Gabor configurations in `test_cutting_plane.py` passed against
  the float64 reference, including amplitude bootstrap, oblique clipping under
  the implemented approximation, and the beta-kernel path.
- `scripts/tests/gabor_gradflow_check.py` passed on the 270,627-primitive heart
  checkpoint: amplitude gradients bootstrap at zero, world-frequency and phase
  gradients become active for nonzero amplitude, the packed band changes with
  view, and the whitened-frequency clamp behaves as implemented.
- `scripts/tests/gabor_heart_parity.py` confirmed bitwise-identical rendering
  between dGS and dGS-Gabor at zero amplitude on a real 1600x1600 view.

These checks support the `+0.14 dB` result for the exact protocol used here:
warm-started residual-only fitting, fixed topology, and no active clip plane.

### Known correctness boundaries

1. **Densification and MCMC growth are broken.** `densify_and_clone`,
   `densify_and_split`, and `add_new_gs` place extension rows in
   `_pending_new_gabor`, but the dynamically dispatched
   `densification_postfix()` replaces that value with `None`. The base optimizer
   then requests a missing `gabor_omega` extension. This was reproduced on the
   heart checkpoint by selecting one primitive for cloning:

   ```text
   KeyError: 'gabor_omega'
   ```

   The reported heart fits are unaffected because they use
   `--densify_until_iter 0`. Before using ordinary or MCMC densification, pass
   the new Gabor rows explicitly through `densification_postfix()` and add
   clone, split, add, relocate, prune, and optimizer-state lifecycle tests.

2. **Half-space clipping is approximate for an active Gabor.** The projected
   cosine is exact for the unclipped affine-EWA footprint. When a plane crosses
   an oscillatory atom, the code multiplies that footprint by the Gaussian
   `clipPhi`; the true clipped-Gabor integral requires a complex error function
   (Faddeeva). The oblique CUDA test proves agreement with the implemented
   approximation, not with that exact integral. This is harmless for the
   no-clip heart experiment but is insufficient for claiming exact XClipGS
   clipping of Gabor atoms.

3. **This is a constrained, Gabor-inspired residual rather than a Gabor Fields
   port.** Upstream uses independent Gaussian-base and Gabor-residual
   primitives with separate centers, scales, orientations, signed opacities,
   frequencies, primitive budgets, and staged pyramid training. This branch
   instead attaches one co-located, nonnegative modulation to each existing dGS
   primitive and shares its envelope, opacity, color, and center. It therefore
   answers the narrower question: "Does one projected oscillatory residual per
   dGS primitive improve this medical volume?"

4. **The frequency bounds are adapted, not identical to upstream.** Upstream
   bounds a scalar whitened frequency with `BoundedAdam`; this branch bounds the
   norm of an arbitrary 3D whitened wave vector by post-step rescaling without
   resetting Adam moments. The `[0.5, 3.0]` interval is reasonable for this
   experiment, but documentation should not imply an identical parameterization
   or optimizer rule.

### Review conclusion

Keep the experimental result and the decision not to adopt Gabor for RenderFM.
Describe the branch as a **projected Gabor residual** or **Gabor-inspired
residual modulation**, not as a complete implementation of Gabor Fields. Fix
the topology-growth path only if the branch will be reused beyond the verified
fixed-topology, unclipped ablation; derive the complex-CDF clip only if exact
Gabor clipping becomes a research requirement.

## Beta-kernel comparison: dBS-SH and dbs-gabor (2026-07-20)

Question: is the Beta kernel (envelope-shape control) a better capacity lever
than the Gabor band (interior oscillation) on this content? New mode
`dbs-gabor` (`scene/gaussian_model_beta_gabor.py`, commit `1916333`) puts the
same projected residual on the dBS-SH base — the CUDA already composed
beta+gabor; only the model layer was missing. Whitening/init/clamp go through
a Cholesky factor of `get_covariance` (dBS rotations are l-triangle, and that
IS the `cov3D_precomp` the renderer consumes); the projection is approximate
under a beta envelope (same status as the analytic clip on beta). Verified:
parity bit-exact vs plain dbs-sh, bootstrap/gradflow/clamp ALL PASS, lifecycle
ALL PASS.

Protocol: dBS-SH trained from scratch 30k (`--mode dbs-sh --input_dim 6
--l_22_inv_init_scale 2.0 --eval`; NO mip — unsupported by dBS), then the same
7k residual-only warm-start recipe. dBS densification settled at **195,279**
primitives (vs 270,627 for dGS — not count-matched; note both caveats when
comparing across bases). Outputs: `/data/output/xclipgs/gabor/heart_900_dbs_sh/`
and `.../heart_900_dbs_gabor_resonly/`.

   | model             | N       | all     | intact  | clipped | SSIM    | LPIPS   |
   |-------------------|---------|---------|---------|---------|---------|---------|
   | dGS base          | 270,627 | 29.2025 | 31.0363 | 27.3687 | 0.94038 | 0.09508 |
   | dGS + gabor v2    | 270,627 | 29.3423 | 31.2350 | 27.4496 | 0.94115 | 0.09444 |
   | dBS-SH base       | 195,279 | 28.6280 | 31.0121 | 26.2440 | 0.93059 | 0.10791 |
   | dBS-SH + gabor    | 195,279 | 28.7183 | 31.1615 | 26.2751 | 0.93107 | 0.10743 |

Findings:
1. **The dBS deficit is entirely a clipping story.** On intact views dBS-SH
   TIES dGS (31.01 vs 31.04) with 28% fewer primitives and no mip filter; it
   loses its whole 0.57 dB on the clipped half (-1.12 dB), where the
   Gaussian-derived `clipPhi` is approximate on a beta envelope. The beta
   kernel itself is competitive (even primitive-efficient) here; what it lacks
   is an exact clip operator. If beta ever matters strategically, the unlock
   is deriving the beta half-space integral, not more primitives.
2. **The gabor band gives the same modest, intact-skewed bump on both bases**:
   +0.199/+0.081 dB (intact/clipped) on dGS, +0.150/+0.031 dB on dBS-SH.
   Capacity again not binding (96.6% active, median |amp| 0.118, 0.01% at tanh
   saturation, 1.1% at the omega hi-clamp).
3. **Decision unchanged**: dGS + exact clip remains the best configuration
   overall; neither the beta swap nor the gabor band (nor both) changes the
   RenderFM picture.

### Response to review (2026-07-20, same day)

Point 1 (growth paths broken) — **confirmed and FIXED**:
- `dgs-gabor`: `densification_postfix` popped `new_gabor` with a `None`
  default, clobbering the `_pending_new_gabor` stash that the overridden
  clone/split/add paths set before the base dynamically dispatched back into
  the override → `KeyError: 'gabor_omega'` on every growth op. Now only
  touches the stash when the kwarg is explicitly present. Also resets the
  gabor Adam moments at relocated slots (the dgs base `replace` skips groups
  it does not know).
- `dbs-gabor` had a sibling bug: the dbs base `replace_tensors_to_optimizer`
  indexes every optimizer group with no membership guard → same KeyError on
  the MCMC paths. Reimplemented in the subclass with the gabor entries.
- New `scripts/tests/gabor_lifecycle_check.py` exercises
  clone → split → prune → relocate → add_new_gs → post-growth optimizer step
  with row-count/aliasing/Adam-state invariants after every op, for BOTH
  model families: **ALL PASS**. (The prior fits were unaffected —
  `--densify_until_iter 0` never exercised growth.)

Points 2-4 (approximate clip for active atoms; Gabor-inspired residual, not a
port; bounds adapted via post-step norm rescaling, not upstream's scalar
BoundedAdam) — **agreed**; terminology adjusted here and in the model header:
this branch is a **projected Gabor residual**. The clip statement stands as
documented: exact Gaussian-envelope clip, unclipped cosine, approximate when a
plane crosses an active atom.
