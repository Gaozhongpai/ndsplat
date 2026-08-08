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
alpha_i(v,T) = sigmoid(o_i + delta_i^TF(T)) * s_i^view(v)   # logit OFFSET, can
SH_i(T)      = SH_i^base + delta_SH_i^TF(T)                 # reveal AND suppress
```

The TF branch is a physical local lookup (Eq. 6, joint p(l,h)) plus a low-rank
functional residual. A learned `tf_id` embedding is a seen-TF baseline, not the
proposed input.

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
  (the residual/color/opacity ablation).
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
# per-preset specialists (subset with FACTORSPLAT_TF_IDS="...")
bash scripts/benchmarks/dgs_factorsplat_oracles.sh
# unconditioned mixed-TF floor (plain dgs on the combined set)
bash scripts/benchmarks/dgs_factorsplat_mixed.sh
# conditioned model; FACTORSPLAT_VARIANT=residual|hybrid|lookup|color|opacity, FACTORSPLAT_RANK=4|8|16|32
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

Names correspond EXACTLY to enabled flags (FACTORSPLAT_VARIANT / output dir):

| name | low-rank residual | local lookup | encoder | notes |
|---|---|---|---|---|
| `mixed_unconditioned` | -- | -- | -- | identity floor (one image for all TFs) |
| `residual` | color+opacity | off | functional | the pilot model (was mislabeled "hybrid") |
| `lookup` | off | on | (unused) | Eq. 6 alone, learned global gain only |
| `hybrid` | color+opacity | on | functional | full model: lookup + residual |
| `residual_embedding` | color+opacity | off | embedding | seen-only baseline, nearest-train fallback |
| `color` / `opacity` | one channel | off | functional | channel ablation |
| `specialist` | -- | -- | -- | one dGS per preset ("per-preset specialist", NOT a strict ceiling: each sees 1/6 of the images) |

Plus: dense TF-conditioned appearance MLP (hypernetwork, TODO) and optional
Render-FM regeneration per preset. Render-FM stays a SEPARATE amortization
branch (it could generate the canonical FactorSplat representation for a new
volume; FactorSplat then handles instantaneous TF switching) -- it is not part
of the core per-scene study. No main variant enables TF- or view-dependent
position.

### Rank selection (full study)

Primary criterion: validation-preset changed-region TF-delta error
(`delta_l1_changed`). Guardrails: `unchanged_delta_leak`, PSNR/SSIM, model
size, and TF-switch latency. NEVER select rank on PSNR alone -- it is nearly
blind to localized edits.

### Matched ablation at the selected rank r*

Run all rows of the table above at r* on both scenes with identical data,
budget, and cameras: mixed floor, residual, lookup, hybrid,
residual_embedding, specialists. Table labels in the paper must match the
directory names. Budget wording: the comparison is at matched TOTAL DATA
COVERAGE, not matched compute -- the specialist suite trains one model per
preset and consumes several times the aggregate iterations.

### Deployment payload (packed lookup, sidecar v2)

`--tf_lookup_mode joint` (default) stores the empirical joint p(l,h) PACKED:
one uint16 id `label_col*S + hu_bin` per window voxel (valid-first) + one
uint8 count per Gaussian; uniform weights are reconstructed at lookup time.
Nothing dense is allocated in joint mode; `separable` keeps only p/q. Only
ACTIVE branches carry parameters (lookup-only saves no residual factors or
encoder; color/opacity ablations allocate one factor tensor). Sidecar format
v2; v1 checkpoints load (v1 joint arrays are converted, v1 separable-only
sidecars fall back to separable). Measured: heart pilot lookup checkpoint
(187,930 G) 69.6 MB -> 10.3 MB (85% smaller). Use packed for all matched
lookup/hybrid runs and final storage numbers.

### Frozen sequence (2026-08-08)

1. CODE FREEZE on the training path until the residual rank sweep completes.
   Exception staged, NOT merged: branch `feat/tf-aware-prune` (worktree
   ../ndsplat-tfprune) makes pruning TF-aware -- prune on
   max_{T in T_train} alpha_i(T) via a get_pruning_opacity() seam
   (--tf_aware_prune, default True) so no preset-revealed anatomy is deleted
   for a low shared logit. Unit-tested. MERGE THIS FIRST after the sweep, so
   the entire six-variant r* ablation trains under it.
2. `factorsplat_select_rank.py` emits the machine-readable manifest
   (`rank_selection_<scene>_<preset>.json`): val delta primary, guardrails
   recorded, parsimony tie-break to the smaller rank.
3. Six matched variants at r*: heart first, then vascular.
4. Label-selective presets extend the bank as a STRICT SUPERSET (append after
   index 39 only; never reorder) so all existing 40-TF renders stay reusable.
5. Validate the new presets on heart before any six-scene expansion.
6. `hybrid` becomes the headline FactorSplat model ONLY if it beats `residual`
   consistently across scenes/splits; otherwise `residual` stays primary and
   the lookup is analysis/ablation.

### v2 ladder (only if hybrid stays weak on label-selective/OOD)

Local functional residual: per-Gaussian code
`z_i(T) = sum_{l,h} p_i(l,h) phi(l, h, T(l,h)-T0(l,h))` so the LEARNED branch
becomes locally TF-aware too (currently only the analytic lookup is). Explore
before all-SH conditioning (entanglement risk) and never TF-dependent position
shifts (TFs edit appearance/support, not anatomy). Render-FM stays the
cross-scene amortization branch.

### Bank TODO before the six-scene study

Add label-SELECTIVE mutations (per-label hue/alpha edits, show/hide of
specific anatomy) to the bank families. The current bank is mostly global
(hue/opacity/window/gamma), so the label-aware contribution of Eq. 6 is
under-tested: on the pilot bank the joint p(l,h) and separable p(l)q(h)
lookups agree to 1.6% relative -- selective presets are what separates them.

### Baseline implementations (2026-08-08)

**Seen-only embedding (comparison 3).** `--tf_encoder_type embedding`
(`FACTORSPLAT_ENCODER=embedding`) swaps the functional encoder for an
`nn.Embedding(num_bank_TFs, tf_rank)` ID table, zero-initialized so untrained
rows give a zero code (= base appearance). Only training-preset rows ever
receive gradients. At test time unseen presets use the embedding of the
nearest *training* preset in descriptor space (`--tf_embedding_fallback
nearest`; `zero` renders base appearance instead). Output dirs get an
`_embedding` suffix. The sidecar records the encoder type and refuses a
mismatched load.

**Local lookup (comparisons 2/5 + hybrid).** `--tf_use_lookup True`
(`FACTORSPLAT_LOOKUP=1`, or `FACTORSPLAT_VARIANT=lookup` for lookup-alone with
the low-rank branch disabled). Per-Gaussian descriptors — a label distribution
over the bank's `label_ids` and an HU histogram over the bank's `hu` grid,
sampled in a 3^3 voxel window around each init Gaussian — are generated on the
HOST (SimpleITK reads the DICOM series; the mask raw is z-flipped when the
.ini `tm[8] < 0`) by
`trueview/vengine-runtime/factorsplat_lookup_descriptors.py <scan> <dataset>`,
writing `<dataset>/points3d_lookup.npz`. The model contracts them against the
raw (un-premultiplied) bank downsampled to `--tf_lookup_bins` (64):
`delta_i(T) = sum_l p_i(l) <q_i, R_T(l,:) - R_T0(l,:)>`, applied as a DC-color
offset (`/C0`) and an opacity-logit offset, each through a learned global RGBA
gain. Descriptors follow densification by nearest-pre-existing-Gaussian
inheritance and travel in the sidecar, so `render.py` needs no npz. The tool
prints in-grid / foreground-support fractions and refuses to write below
95% / 50% (frame-mapping guard). Verified: pilot heart + vascular 100%/100%;
unit tests (zero delta at base TF, NN inherit, prune, gain gradients) pass.

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
