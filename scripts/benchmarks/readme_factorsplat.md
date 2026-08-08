# FactorSplat: per-scene transfer-function conditioning

## Claim

Train one Gaussian proxy for a volume from multiple label-aware transfer
functions, then render an unseen transfer function without optimization or
Gaussian regeneration. The first study is per scene; Render-FM is not required
by the conditional representation.

The vengine condition is a function

```text
T(label, intensity) -> RGBA
```

rather than a scalar style code. Every TF is exported as its exact sampled LUT,
and splits are made by TF operation rather than by random images.

## Fixed backbone

FactorSplat inherits the XClipGS dGS configuration:

```text
--mode dgs            (baselines)  /  --mode factorsplat  (conditioned)
--use_view_dependent_pos False
--l_22_inv_init_scale 2.0
--mip3dgs
```

Positions and covariances remain globally trainable, but neither is shifted by
view direction or transfer function. Existing dGS conditioning changes opacity
with view direction. The new TF branch changes color and opacity only:

```text
alpha_i(v,T) = alpha_i^coverage * s_i^view(v) * s_i^TF(T)
SH_i(T)      = SH_i^base + delta_SH_i^TF(T)
```

The proposed branch will use a physical local lookup followed by a low-rank
residual. A learned `tf_id` lookup is a seen-TF baseline, not the proposed input.

## Model (`--mode factorsplat`)

Implemented in `scene/gaussian_model_factorsplat.py` on top of the opacity-only
dGS class. Per TF `T`, the encoder consumes the bank's exact sampled RGBA
curves (all labels, subsampled to `--tf_samples`, default 32 of the 256 bank
samples) and produces a code `z_T` through a bias-free 2-layer MLP
(`input -> tf_hidden=64 -> tf_rank`). Per-Gaussian low-rank factors then map
the code to appearance offsets:

```text
delta_SH_dc_i = A_i^c z_T          A_i^c: [N, 3, r]   (color factors)
delta_logit_alpha_i = a_i^T z_T    a_i:   [N, r]      (opacity factors)
```

- `--tf_rank` r in {4, 8, 16, 32}; pilot default 8, smoke used 4.
- `--tf_condition_color / --tf_condition_opacity` gate the two channels
  (the hybrid/color/opacity ablation).
- Factors are per-Gaussian optimizer groups (`--tf_factor_lr 0.0025`); the
  encoder uses `--tf_encoder_lr 0.001`. Densify/clone/prune/reset carry the
  factors along; `capture()/restore()` checkpoints include them (11 optimizer
  param groups total).
- Saving writes the usual `point_cloud.ply` plus a `point_cloud.ply.factorsplat.pt`
  sidecar (encoder weights + factors); `load_ply` restores both.
- Geometry, covariance, and the dGS view-opacity path are untouched; the base
  TF renders bit-identically when the TF branch is zero.

## Transfer-function bank

`factorsplat_make_tf_bank.py` clones the scan's canonical bookmark and mutates
only `ColorPointList` / `AlphaPointList` / `Opacity` / label visibility; the
exact sampled LUT (per label x 256 HU samples x RGBA) is stored in
`tf_bank.npz` (`rgba [T,L,256,4]`, `tf_ids`, `label_ids`) and mirrored in
`tf_bank.json`. Explicit `AlphaPointList`s are preserved exactly (alpha is only
reconstructed from window/level when the bookmark has no alpha curve; both
[0,1] and [0,255] alpha conventions are normalized).

Pilot bank: 12 TFs in mutation families, split by operation (never randomly):

| tf_id | split | family | mutation |
|---|---|---|---|
| train_00_base | train | base | — |
| train_01_hue_p30 | train | color | hue +30° |
| train_02_hue_m30 | train | color | hue −30° |
| train_03_alpha_060 | train | opacity | alpha ×0.60 |
| train_04_alvl_p150 | train | window | alpha level +150 HU |
| train_05_coupled | train | coupled | hue +45° and alpha ×1.25 |
| val_00_hue_p15 | val | color | hue +15° |
| val_01_alpha_080 | val | opacity | alpha ×0.80 |
| test_interp_00_hue_m15 | test_interp | color | hue −15° |
| test_interp_01_alvl_p75 | test_interp | window | alpha level +75 HU |
| test_ood_00_gray | test_ood | color_ood | saturation 0 |
| test_ood_01_sharp | test_ood | opacity_ood | alpha gamma 2.2 |

Interpolation tests sit inside the span of training mutations; OOD tests are
operations never seen in any magnitude.

## Cameras and render budget

Per scene and preset (defaults in `factorsplat_prepare.sh`, overridable via
`FACTORSPLAT_TRAIN_VIEWS/ANCHOR_VIEWS/TEST_VIEWS`):

- 48 supervised training views per TF: 12 anchor cameras shared by every
  TF plus 36 TF-specific cameras. Anchors give the model exactly-paired
  appearance changes; the rest add view diversity.
- 40 held-out test cameras, identical across all 12 TFs (paired evaluation:
  the dataset checker enforces the shared source-camera set).
- Renders per scene: 12x48 train + 12x40 test = 1,056 vengine references
  (~2k for the two-scene pilot), versus 10,800 if all 900 XClipGS cameras were
  re-rendered per TF.

Camera geometry, materials, lighting, and sampler settings are byte-identical
across TFs; only the `<BookMark path=...>` differs between configs
(`factorsplat_prepare_configs.py` emits the camera x TF manifest).

## Dataset contract

```text
nerf_dataset/<scene>_factorsplat_<preset>/
  transforms_train.json      # ONLY the 6 training TFs (6x48 = 288 frames)
  transforms_test.json       # all 12 TFs x 40 shared cameras = 480 frames
  tf_bank.npz / tf_bank.json # canonical sampled LUTs (single copy)
  points3d.ply               # shared init (default <scan>_900/points3d.ply)
  train/ test/               # RELATIVE symlinks into render_dataset PNGs
nerf_dataset/factorsplat_oracle/<scene>/<preset>/<tf_id>/   # per-TF datasets
```

Every frame carries `tf_id`, `tf_index` (row into the bank), `tf_split`,
`tf_family`, and `source_config`. Validation/test TF supervision exists only in
the oracle directories, never in the combined set.

`scripts/benchmarks/factorsplat_check_dataset.py` (run automatically by every
trainer script) enforces: bank/frame index consistency, no held-out TF leakage
into combined train, shared test-camera pairing, and image existence.

**Symlinks must be relative** (vengine commit `50a13eb`): the converter used to
emit absolute links to `/home/vengine/app/external_data/...`, which dangle in
every non-vengine container — `ls` shows the names but `open()`/`exists()`
fail. If a dataset predates the fix, rewrite its links before training.

## Runbook

Generate and render inside the vengine container (GPU render, ~hours/scene):

```bash
bash /repo/factorsplat_prepare.sh heart pilot
bash /repo/factorsplat_prepare.sh vascular pilot
```

Training runs in the ndgs container. All trainer scripts share one env
contract — `FACTORSPLAT_DATA`, `FACTORSPLAT_OUT`, `FACTORSPLAT_SCENES`,
`FACTORSPLAT_PRESET`, `FACTORSPLAT_ITERS` (default 30000), and per-script
extras — and each runs check -> train -> render -> metrics -> group metrics ->
delta metrics, skipping outputs that already have `results.json`:

```bash
# supervised per-TF ceilings (subset with FACTORSPLAT_TF_IDS="...")
bash scripts/benchmarks/dgs_factorsplat_oracles.sh
# unconditioned mixed-TF floor (plain dgs on the combined set)
bash scripts/benchmarks/dgs_factorsplat_mixed.sh
# conditioned model; FACTORSPLAT_VARIANT=hybrid|color|opacity, FACTORSPLAT_RANK=4|8|16|32
bash scripts/benchmarks/factorsplat_train.sh
```

Reference container invocation (works on the A100 host and on 10.10.0.168,
whose A40s have GPUs 0/1/3 generally free; pin 12 CPUs per GPU):

```bash
docker run --rm --gpus '"device=0"' --cpuset-cpus 0-11 --entrypoint /bin/bash \
  -v <workspace>/ndsplat:/workspace \
  -v /mnt/uNeon/zhongpai/vengine_data:/mnt/uNeon/zhongpai/vengine_data \
  -w /workspace \
  -e PYTHONPATH=/workspace/submodules/gsplat:/workspace/submodules/tcgs_speedy_rasterizer \
  -e FACTORSPLAT_DATA=/mnt/uNeon/zhongpai/vengine_data/nerf_dataset \
  -e FACTORSPLAT_OUT=/mnt/uNeon/zhongpai/vengine_data/output/factorsplat \
  -e FACTORSPLAT_SCENES=heart -e FACTORSPLAT_PRESET=pilot \
  10.10.0.192:5555/zhongpai/ndgs:latest -lc "bash scripts/benchmarks/factorsplat_train.sh"
```

The in-tree gsplat/tcgs `.so` files are loaded via `PYTHONPATH` from shared NFS
and shadow any site-packages copy; they are built as fat binaries
(`TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"`, `setup.py build_ext --inplace`) so one
build serves both sm_80 (A100) and sm_86 (A40). Rebuild the same way after any
CUDA change; remove root-owned `build/` dirs first if a root container built
previously. Logs go to `vengine_data/output/factorsplat/logs/`.

Smoke preset (pipeline check, minutes): `FACTORSPLAT_TRAIN_VIEWS=2
FACTORSPLAT_ANCHOR_VIEWS=1 FACTORSPLAT_TEST_VIEWS=1` at prepare time, then
train with `FACTORSPLAT_PRESET=smoke FACTORSPLAT_ITERS=3000 FACTORSPLAT_RANK=4`.
Oracle datasets at smoke scale are 2-view floors, not ceilings.

## Main comparisons

1. Original-TF opacity-only dGS, frozen under edits.
2. Direct local TF lookup using each Gaussian's label/intensity statistics.
3. Seen-only learned TF embedding.
4. Dense TF-conditioned appearance MLP.
5. FactorSplat local lookup only.
6. FactorSplat low-rank residual only.
7. FactorSplat hybrid.
8. Separately trained dGS for each TF (supervised ceiling).
9. Optional Render-FM regeneration for each TF (amortized regeneration baseline).

The controlled FactorSplat ablation is TF-opacity only, TF-color only, and both.
No main variant enables TF- or view-dependent position.

## Evaluation

`metrics.py` reports PSNR/SSIM/LPIPS; `factorsplat_group_metrics.py` regroups
them by `tf_split` and `tf_family`. `factorsplat_delta_metrics.py` adds the
TF-specific scores, pairing each test render with the base-TF render from the
same camera:

```text
M_T   = |I_gt(T) - I_gt(T_base)| > epsilon        (epsilon = 0.04)
delta = |(I_pred(T)-I_pred(T_base)) - (I_gt(T)-I_gt(T_base))|
```

reported as `affected_fraction`, `delta_l1_full`, `delta_l1_changed` (inside
M_T), and `unchanged_delta_leak` (predicted change where GT says nothing
changed). The mixed baseline renders one image for all TFs, so its delta-L1 is
exactly the identity floor ("predict no change"); any conditioned model must
beat it. Also report disappearance leakage, continuous TF-sweep error, TF
update latency, FPS, and storage versus one dGS checkpoint per TF.

## Pipeline validation record (smoke, 2026-08-08)

heart, 6 train TFs x 2 views, 3k iters, rank 4, on 10.10.0.168 A40s; all four
pipelines exited 0 end-to-end. Delta-L1 (lower is better; floor = mixed):

| model | train | val | interp | ood |
|---|---|---|---|---|
| hybrid | 0.0572 | 0.0360 | 0.0432 | 0.1619 |
| color-only | 0.0557 | 0.0290 | 0.0363 | 0.1842 |
| opacity-only | 0.0586 | 0.0349 | 0.0418 | 0.1632 |
| mixed (floor) | 0.0690 | 0.0409 | 0.0436 | 0.1978 |

Directional only at this scale: all conditioned variants beat the floor on
every split; hybrid does not yet dominate; interp is ~flat — re-examine the
TF encoder if that persists at pilot scale.

## Go/no-go for the pilot

Proceed to the six-CT-scene, 40-TF study when the hybrid model:

- clearly beats original-TF dGS and direct lookup inside the changed region;
- approaches the supervised per-TF ceiling on interpolation TFs;
- responds to a new TF without optimization;
- preserves unchanged regions; and
- updates the TF condition interactively without rebuilding geometry.

After the per-scene result, evaluate the factorial composition of unseen TFs
with XClipGS analytic clipping. Render-FM integration is the subsequent
cross-scene/amortized stage, not a dependency of this experiment.
