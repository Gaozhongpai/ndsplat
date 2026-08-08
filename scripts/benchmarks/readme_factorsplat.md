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
--mode dgs
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

## Pilot protocol

Scenes: `heart` and `vascular`.

The pilot bank has 12 TFs:

- 6 training: base, two hue changes, opacity scale, intensity shift, coupled edit;
- 2 validation: unseen intermediate hue and opacity;
- 2 interpolation tests; and
- 2 OOD tests: grayscale and a sharp alpha curve.

Every TF receives 48 supervised training views for the oracle ceiling and the
same 40 held-out cameras. Twelve training cameras are anchors shared across all
TFs. The combined FactorSplat dataset exposes training images only for the six
training TFs; supervised images for validation/test TFs remain isolated in the
oracle directories.

Generate and render inside the vengine container:

```bash
bash /repo/factorsplat_prepare.sh heart pilot
bash /repo/factorsplat_prepare.sh vascular pilot
```

Outputs:

```text
/data/nerf_dataset/heart_factorsplat_pilot/
/data/nerf_dataset/vascular_factorsplat_pilot/
/data/nerf_dataset/factorsplat_oracle/<scene>/pilot/<tf_id>/
```

Train the supervised ceilings in the ndsplat container:

```bash
FACTORSPLAT_SCENES="heart vascular" FACTORSPLAT_PRESET=pilot \
  bash scripts/benchmarks/dgs_factorsplat_oracles.sh
```

Train one opacity-only dGS on the mixed training TFs without conditioning:

```bash
FACTORSPLAT_SCENES="heart vascular" FACTORSPLAT_PRESET=pilot \
  bash scripts/benchmarks/dgs_factorsplat_mixed.sh
```

For a fast pipeline check, select only the base and one held-out TF and reduce
the training budget:

```bash
FACTORSPLAT_TF_IDS="train_00_base test_interp_00_hue_m15" \
FACTORSPLAT_ITERS=1000 FACTORSPLAT_SCENES=heart \
  bash scripts/benchmarks/dgs_factorsplat_oracles.sh
```

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

Report PSNR, SSIM, and LPIPS by TF split and family. In addition, use the base
reference to define the changed region

```text
M_T = |I_gt(T) - I_gt(T_base)| > epsilon
```

and report masked image metrics plus transfer-function delta error:

```text
|(I_pred(T)-I_pred(T_base)) - (I_gt(T)-I_gt(T_base))|.
```

Also report disappearance leakage, continuous TF-sweep error, TF update latency,
FPS, total model size, and storage relative to one dGS checkpoint per TF.

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
