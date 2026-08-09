#!/bin/bash
# Train the functional low-rank FactorSplat model on combined multi-TF datasets.
set -euo pipefail

DATA_ROOT="${FACTORSPLAT_DATA:-/data/nerf_dataset}"
OUT_ROOT="${FACTORSPLAT_OUT:-/data/output/factorsplat}"
SCENES="${FACTORSPLAT_SCENES:-heart vascular}"
PRESET="${FACTORSPLAT_PRESET:-pilot}"
ITERS="${FACTORSPLAT_ITERS:-30000}"
# Extra train.py flags, e.g. --use_jpeg_compression for 40-preset datasets
# whose decoded frames exceed GPU memory.
EXTRA_NOTE="${FACTORSPLAT_EXTRA_FLAGS:-}"
RANK="${FACTORSPLAT_RANK:-8}"
# Variant names correspond EXACTLY to the enabled branches:
#   residual = functional low-rank residual only (color+opacity conditioning)
#   lookup   = local lookup only (low-rank branch off)
#   hybrid   = local lookup + functional residual
#   color / opacity = residual restricted to one channel (ablation)
VARIANT="${FACTORSPLAT_VARIANT:-residual}"
# functional (default) or embedding (seen-only per-preset codes baseline)
ENCODER="${FACTORSPLAT_ENCODER:-functional}"

ENCODER_FLAGS=""
if [ "$ENCODER" != "functional" ]; then
    ENCODER_FLAGS="--tf_encoder_type $ENCODER"
fi

LOOKUP=0
case "$VARIANT" in
    residual) TF_FLAGS="--tf_condition_color True --tf_condition_opacity True" ;;
    residual_dc) TF_FLAGS="--tf_condition_color True --tf_condition_opacity True --tf_color_sh_degree 0" ;;
    hybrid)   TF_FLAGS="--tf_condition_color True --tf_condition_opacity True"; LOOKUP=1 ;;
    hybrid_dc) TF_FLAGS="--tf_condition_color True --tf_condition_opacity True --tf_color_sh_degree 0"; LOOKUP=1 ;;
    color)    TF_FLAGS="--tf_condition_color True --tf_condition_opacity False" ;;
    opacity)  TF_FLAGS="--tf_condition_color False --tf_condition_opacity True" ;;
    lookup)   TF_FLAGS="--tf_condition_color False --tf_condition_opacity False"; LOOKUP=1 ;;
    veg)      TF_FLAGS="--tf_condition_color False --tf_condition_opacity False --tf_veg_packed True" ;;
    *) echo "unknown FACTORSPLAT_VARIANT=$VARIANT (residual|residual_dc|hybrid|hybrid_dc|color|opacity|lookup|veg)" >&2; exit 2 ;;
esac

VARIANT_DIR="$VARIANT"
if [ "$LOOKUP" = "1" ]; then
    TF_FLAGS="$TF_FLAGS --tf_use_lookup True"
fi
[ "$ENCODER" != "functional" ] && VARIANT_DIR="${VARIANT_DIR}_${ENCODER}"

for scene in $SCENES; do
    dataset="${scene}_factorsplat_${PRESET}"
    source="$DATA_ROOT/$dataset"
    output="$OUT_ROOT/$VARIANT_DIR/rank${RANK}/$scene/$PRESET"
    if [ ! -d "$source" ]; then
        echo "missing dataset: $source" >&2
        continue
    fi
    if [ -f "$output/results.json" ]; then
        echo "skip completed: $output"
        continue
    fi
    python scripts/benchmarks/factorsplat_check_dataset.py "$source"
    checkpoint=""
    [ -f "$source/points3d.ply" ] && checkpoint="--start_checkpoint $source/points3d.ply"
    python train.py -s "$source" --model_path "$output" --sh_degree 1 \
        --mode factorsplat --use_view_dependent_pos False \
        --l_22_inv_init_scale 2.0 --mip3dgs --tf_rank "$RANK" \
        $TF_FLAGS $ENCODER_FLAGS --iterations "$ITERS" --eval --disable_viewer $checkpoint ${FACTORSPLAT_EXTRA_FLAGS:-}
    python render.py -m "$output" --skip_train --iteration "$ITERS"
    python metrics.py -m "$output"
    python scripts/benchmarks/factorsplat_group_metrics.py \
        --model "$output" --dataset "$source" --iteration "$ITERS"
    python scripts/benchmarks/factorsplat_delta_metrics.py \
        --model "$output" --dataset "$source" --iteration "$ITERS"
done
