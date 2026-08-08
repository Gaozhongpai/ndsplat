#!/bin/bash
# Per-transfer-function supervised dGS ceilings for FactorSplat.
#
# Each input directory contains one TF's train/test frames. This intentionally
# uses the XClipGS opacity-only dGS backbone with no view-dependent position:
#   --mode dgs --use_view_dependent_pos False --l_22_inv_init_scale 2.0
#
# Usage:
#   FACTORSPLAT_ORACLE_DATA=/data/nerf_dataset/factorsplat_oracle \
#   FACTORSPLAT_SCENES="heart vascular" FACTORSPLAT_PRESET=pilot \
#       bash scripts/benchmarks/dgs_factorsplat_oracles.sh
set -euo pipefail
shopt -s nullglob

DATA_ROOT="${FACTORSPLAT_ORACLE_DATA:-/data/nerf_dataset/factorsplat_oracle}"
OUT_ROOT="${FACTORSPLAT_OUT:-/data/output/factorsplat/oracle}"
SCENES="${FACTORSPLAT_SCENES:-heart vascular}"
PRESET="${FACTORSPLAT_PRESET:-pilot}"
ITERS="${FACTORSPLAT_ITERS:-30000}"
TF_FILTER="${FACTORSPLAT_TF_IDS:-}"

DGS_FLAGS="--mode dgs --use_view_dependent_pos False --l_22_inv_init_scale 2.0 --mip3dgs"
EVAL_FLAGS="--iterations ${ITERS} --eval --disable_viewer"

selected() {
    local tf_id="$1"
    [ -z "$TF_FILTER" ] && return 0
    local wanted
    for wanted in $TF_FILTER; do
        [ "$wanted" = "$tf_id" ] && return 0
    done
    return 1
}

for scene in $SCENES; do
    scene_root="$DATA_ROOT/$scene/$PRESET"
    if [ ! -d "$scene_root" ]; then
        echo "missing oracle root: $scene_root" >&2
        continue
    fi
    for source in "$scene_root"/*/; do
        [ -d "$source" ] || continue
        tf_id="${source%/}"
        tf_id="${tf_id##*/}"
        selected "$tf_id" || continue
        output="$OUT_ROOT/$scene/$PRESET/$tf_id"
        if [ -f "$output/results.json" ]; then
            echo "skip completed: $output"
            continue
        fi
        checkpoint=""
        [ -f "$source/points3d.ply" ] && checkpoint="--start_checkpoint $source/points3d.ply"
        echo "oracle: $scene / $tf_id -> $output"
        python train.py -s "$source" --model_path "$output" \
            $DGS_FLAGS $EVAL_FLAGS $checkpoint
        python render.py -m "$output" --skip_train --iteration "$ITERS"
        python metrics.py -m "$output"
    done
done
