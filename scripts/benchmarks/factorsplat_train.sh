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
VARIANT="${FACTORSPLAT_VARIANT:-hybrid}"
# functional (default) or embedding (seen-only per-preset codes baseline)
ENCODER="${FACTORSPLAT_ENCODER:-functional}"

ENCODER_FLAGS=""
if [ "$ENCODER" != "functional" ]; then
    ENCODER_FLAGS="--tf_encoder_type $ENCODER"
fi

# FACTORSPLAT_LOOKUP=1 adds the local-lookup branch (needs points3d_lookup.npz
# in the dataset root). VARIANT=lookup runs the lookup ALONE (low-rank off).
LOOKUP="${FACTORSPLAT_LOOKUP:-0}"

case "$VARIANT" in
    hybrid)  TF_FLAGS="--tf_condition_color True --tf_condition_opacity True" ;;
    color)   TF_FLAGS="--tf_condition_color True --tf_condition_opacity False" ;;
    opacity) TF_FLAGS="--tf_condition_color False --tf_condition_opacity True" ;;
    lookup)  TF_FLAGS="--tf_condition_color False --tf_condition_opacity False"; LOOKUP=1 ;;
    *) echo "unknown FACTORSPLAT_VARIANT=$VARIANT (hybrid|color|opacity|lookup)" >&2; exit 2 ;;
esac

VARIANT_DIR="$VARIANT"
if [ "$LOOKUP" = "1" ]; then
    TF_FLAGS="$TF_FLAGS --tf_use_lookup True"
    [ "$VARIANT" != "lookup" ] && VARIANT_DIR="${VARIANT}_lookup"
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
    python train.py -s "$source" --model_path "$output" \
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
