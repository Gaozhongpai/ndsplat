#!/bin/bash
# Train the functional low-rank FactorSplat model on combined multi-TF datasets.
set -euo pipefail

DATA_ROOT="${FACTORSPLAT_DATA:-/data/nerf_dataset}"
OUT_ROOT="${FACTORSPLAT_OUT:-/data/output/factorsplat}"
SCENES="${FACTORSPLAT_SCENES:-heart vascular}"
PRESET="${FACTORSPLAT_PRESET:-pilot}"
ITERS="${FACTORSPLAT_ITERS:-30000}"
RANK="${FACTORSPLAT_RANK:-8}"
VARIANT="${FACTORSPLAT_VARIANT:-hybrid}"

case "$VARIANT" in
    hybrid)  TF_FLAGS="--tf_condition_color True --tf_condition_opacity True" ;;
    color)   TF_FLAGS="--tf_condition_color True --tf_condition_opacity False" ;;
    opacity) TF_FLAGS="--tf_condition_color False --tf_condition_opacity True" ;;
    *) echo "unknown FACTORSPLAT_VARIANT=$VARIANT (hybrid|color|opacity)" >&2; exit 2 ;;
esac

for scene in $SCENES; do
    dataset="${scene}_factorsplat_${PRESET}"
    source="$DATA_ROOT/$dataset"
    output="$OUT_ROOT/$VARIANT/rank${RANK}/$scene/$PRESET"
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
        $TF_FLAGS --iterations "$ITERS" --eval --disable_viewer $checkpoint
    python render.py -m "$output" --skip_train --iteration "$ITERS"
    python metrics.py -m "$output"
    python scripts/benchmarks/factorsplat_group_metrics.py \
        --model "$output" --dataset "$source" --iteration "$ITERS"
    python scripts/benchmarks/factorsplat_delta_metrics.py \
        --model "$output" --dataset "$source" --iteration "$ITERS"
done
