#!/bin/bash
# Non-conditional mixed-TF baseline: one opacity-only dGS sees all training TFs
# but receives no TF conditioning. This measures the appearance-averaging failure
# that FactorSplat must beat.
set -euo pipefail

DATA_ROOT="${FACTORSPLAT_DATA:-/data/nerf_dataset}"
OUT_ROOT="${FACTORSPLAT_OUT:-/data/output/factorsplat}"
SCENES="${FACTORSPLAT_SCENES:-heart vascular}"
PRESET="${FACTORSPLAT_PRESET:-pilot}"
ITERS="${FACTORSPLAT_ITERS:-30000}"

for scene in $SCENES; do
    dataset="${scene}_factorsplat_${PRESET}"
    source="$DATA_ROOT/$dataset"
    output="$OUT_ROOT/mixed_unconditioned/$scene/$PRESET"
    if [ ! -d "$source" ]; then
        echo "missing dataset: $source" >&2
        continue
    fi
    if [ -f "$output/results.json" ]; then
        echo "skip completed: $output"
        continue
    fi
    checkpoint=""
    [ -f "$source/points3d.ply" ] && checkpoint="--start_checkpoint $source/points3d.ply"
    python scripts/benchmarks/factorsplat_check_dataset.py "$source"
    python train.py -s "$source" --model_path "$output" \
        --mode dgs --use_view_dependent_pos False \
        --l_22_inv_init_scale 2.0 --mip3dgs \
        --iterations "$ITERS" --eval --disable_viewer $checkpoint
    python render.py -m "$output" --skip_train --iteration "$ITERS"
    python metrics.py -m "$output"
done
