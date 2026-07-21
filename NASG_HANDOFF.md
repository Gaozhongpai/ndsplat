# NASG-Gabor Color (replacing SH) — Handoff

> **STATUS (2026-07-21): experiments complete on heart_900; decision open.**
> The NASG-Gabor appearance model (Miazga/Condor/Didyk, "Beyond Spherical
> Harmonics"; github.com/ewaMiazga/NASGabor) replaces SH view-dependent color
> on the dGS base. Under the STAGED protocol (frozen SH-checkpoint geometry,
> color-only 7k fit): **L=2 matches SH3 within 0.05 dB at 2.3x fewer color
> parameters; L=4 BEATS SH3 (+0.06 dB) at 0.8x.** The intact-view half is
> where NASG wins (up to +0.37 dB); the clipped half stays 0.26-0.46 dB below
> SH at all lobe counts. From-scratch training (the paper's protocol) does
> NOT transfer to our pipeline without NASG-specific recipe tuning: L=1
> scratch lands -0.55 dB under SH scratch.

## What this is

Mode `dgs-nasg` (`scene/gaussian_model_nasg.py`, branch `feat/nasg-color`,
commit `13227c2`): per Gaussian, color for view direction v is

    C(v) = c0 + sum_j pdf_j(v) * rgb_j        (j over L lobes; 3 + 9L scalars)
    pdf   = e^{2 lam (E*Kb - 1)} * E * (1 + cos(k*vx))/2 * inv_norm
    Kb = (vz+1)/2,  E = Kb^{eps + a*vx^2/(1-vz^2)},
    inv_norm = lam*sqrt(1+a) / (2pi(1+eps-e^{-2 lam}))

ported verbatim from upstream's `spherical_nasg_gabor.cuh`. Colors are
evaluated PER VIEW in torch (differentiable) and passed as `colors_precomp` —
**zero CUDA changes**; the model keeps the TCGS bucket-backward fast path (41+
it/s color-only fits, faster than the gabor-band study's fits).

- `_features_dc` doubles as c0 (SH DC convention), so 0 active lobes renders
  **bitwise-identical** to plain `dgs` at SH degree 0 (verified on heart) and
  SH checkpoints warm-start directly; `_features_rest` kept for compat,
  permanently frozen, never evaluated.
- Progressive lobe activation on the oneupSHdegree schedule (upstream recipe).
  DEVIATION: lobe weights zero-init (upstream: 0.5) so activation is seamless
  and bootstraps from the gradient (weight grads alive at 0; pos/shape grads
  exactly 0 there — verified).
- Flags: `--lobe_number L` (ModelParams), `--nasg_color_only` (freeze the
  entire dGS base; train only c0+lobes; pair with `--densify_until_iter 0`).
- Full growth plumbing (clone/split/prune/relocate/add) with the stash
  pattern + Adam-moment resets from the Gabor band study; lifecycle-tested.

## Results on heart_900 (45 intact + 45 clipped test views)

Base: `/data/output/xclipgs/ours/heart_900` (dGS + SH3, 30k from scratch,
270,627 prims). Staged fits: geometry frozen bit-identical to that checkpoint,
color-only 7k (`--nasg_color_only --densify_until_iter 0`), ~3.5 min each.
Splits from saved PNG renders via the dataset `clip` flag.

  | color model          | params | all     | intact  | clipped | SSIM    | LPIPS   |
  |----------------------|--------|---------|---------|---------|---------|---------|
  | SH degree 3 (base)   | 48     | 29.2025 | 31.0363 | 27.3687 | 0.94038 | 0.09508 |
  | NASG L=1, staged     | 12     | 28.9562 | 31.0048 | 26.9076 | 0.93793 | 0.09698 |
  | NASG L=2, staged     | 21     | 29.1561 | 31.3018 | 27.0105 | 0.93918 | 0.09594 |
  | NASG L=4, staged     | 39     | 29.2597 | 31.4095 | 27.1099 | 0.93987 | 0.09545 |
  | NASG L=1, scratch30k | 12     | 28.6532 | 30.5330 | 26.7734 | 0.93434 | 0.10529 |

  (scratch30k: 231,853 prims — its densification settled 14% below the SH
  run's; same formative-phase effect measured in the Gabor band study.)

Findings:
1. **Parameter efficiency replicates on intact views**: L=1 ties SH (-0.03)
   at 4x fewer color scalars; L=2 +0.27, L=4 +0.37 BETTER than SH. Monotone
   in L, mirroring the paper's Table 3 ranking.
2. **The clipped half is NASG's deficit** (-0.46/-0.36/-0.26 dB for L=1/2/4):
   cut-open views see splats from directions supervised only by the clipped
   half of the training set, and a few narrow lobes have less spare capacity
   for that secondary "interior" appearance mode than SH's 48-coefficient
   global basis. This regime does not exist in the paper's benchmarks
   (Mip-NeRF360 / DeepBlending / T&T). More lobes shrink the gap steadily.
3. **The paper's from-scratch protocol does not transfer as-is**: L=1 scratch
   is -0.55 dB under SH scratch (and under its own staged twin by -0.30).
   The staged results prove the BASIS is expressive enough, so the gap is
   optimization/recipe: our pipeline (LRs, densification, schedules) is tuned
   for SH-dGS; theirs is tuned for NASG (auto_lr, their Beta base, 0.5 weight
   init). Third confirmation this session that staged fitting is the right
   protocol on this pipeline.
4. Overall PSNR at L=4 beats SH3 while SSIM/LPIPS sit marginally behind
   (0.93987 vs 0.94038 / 0.09545 vs 0.09508) — call it parity at 0.8x params.
   The compression sweet spot is **L=2: -0.05 dB for 2.3x fewer color params**
   (48 -> 21 per splat; on 270k splats that is ~29 MB -> ~13 MB of color).

Outputs under `/data/output/xclipgs/nasg/`: `heart_900_nasg_L{1,2,4}`,
`heart_900_nasg_L1_scratch30k`.

## Verification (`scripts/tests/nasg_checks.py`, ALL PASS 2026-07-21)

- Eval vs an independent float64 port of upstream's kernel: 1.8e-16 (float64)
  / 4.1e-5 (fp32; the e^{2 lam(...)} term amplifies rounding by ~2 lam).
- DC parity: dgs-nasg with 0 lobes is BITWISE equal to plain dgs at SH deg 0
  on a real 1600^2 heart view.
- Bootstrap: dL/d(weight) alive at weight==0; pos/shape grads exactly 0 there
  and alive at weight != 0.
- Lifecycle: clone/split/prune/relocate/add_new_gs + post-growth step with
  row/aliasing invariants.

## Considerations for adoption (decision open)

- For RenderFM feed-forward prediction, NASG swaps 45 SH-rest channels for 9L
  lobe channels; predicting lobe ORIENTATION + frequency feed-forward is an
  open question (phaseless c0 + weights are easy; steerable direction less so).
- Browser renderer: eval is ~15 elementwise ops per splat per frame vs an SH
  polynomial — shader-friendly; upstream ships CUDA kernels
  (`compute_nasg_gabor_fwd/bwd.cu`) that could be ported to our gsplat fork /
  TCGS preprocess if fusion is ever needed (Python eval is NOT the bottleneck
  in training).
- The clipped-view deficit matters for the vengine use case specifically;
  L=4 halves it vs L=1 but does not close it. If pursued: NASG-tuned recipe
  (their auto_lr, 0.5 weight init, lobe schedule) or a clipped-supervision
  weighting are the obvious next levers.

## How to run

```bash
# staged color swap (recommended protocol)
python train.py -s /data/nerf_dataset/heart_900 -m <out> \
  --mode dgs-nasg --lobe_number 2 --input_dim 6 --use_view_dependent_pos False \
  --l_22_inv_init_scale 2.0 --mip3dgs --eval --disable_viewer \
  --iterations 7000 --nasg_color_only --densify_until_iter 0 \
  --start_checkpoint /data/output/xclipgs/ours/heart_900/point_cloud/iteration_30000/point_cloud.ply
python render.py -m <out> --skip_train && python metrics.py -m <out>
# from scratch: drop the last three flags, --iterations 30000
```

Environment as in the Gabor study (image `10.10.0.192:5555/zhongpai/ndgs:latest`,
vengine_data at /data; GPU 6 primary, 2/3/4 approved fallback with matching
cpuset blocks). No extension rebuild needed on this branch (CUDA untouched).

Branch `feat/nasg-color` (base `mip`). Related closed study: the projected
Gabor residual (footprint modulation, additive band) on `feat/gabor-residual`
— different mechanism (this replaces the COLOR basis; that modulated ALPHA).
