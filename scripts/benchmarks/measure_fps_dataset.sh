#!/bin/bash
# Measure FPS across every scene in a dataset output directory, for both
# CUDA and PyTorch slicing implementations.
#
# Each scene gets a fps.txt with two lines (one per impl), e.g.:
#   cuda    594.77  iter=best   num_views=20  num_frames=200
#   torch   312.40  iter=best   num_views=20  num_frames=200
#
# Usage:
#   bash scripts/benchmarks/measure_fps_dataset.sh <output_dataset_dir> [num_views] [num_frames] [impls]
#
# impls: "both" (default), "cuda", or "torch"
#
# Examples:
#   bash scripts/benchmarks/measure_fps_dataset.sh output/mcmc/dbs/7dgs_pbr
#   bash scripts/benchmarks/measure_fps_dataset.sh output/mcmc/dbs/7dgs_pbr 20 200 torch
#   bash scripts/benchmarks/measure_fps_dataset.sh output/mcmc/ndgs/nerf_synthetic 20 500 cuda

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <output_dataset_dir> [num_views=20] [num_frames=200] [impls=both|cuda|torch]"
    exit 1
fi

ROOT="$1"
NUM_VIEWS="${2:-20}"
NUM_FRAMES="${3:-200}"
IMPLS="${4:-both}"

case "$IMPLS" in
    both)  IMPL_LIST="cuda torch" ;;
    cuda)  IMPL_LIST="cuda" ;;
    torch) IMPL_LIST="torch" ;;
    *)
        echo "Invalid impls: $IMPLS (expected: both|cuda|torch)"
        exit 1
        ;;
esac

if [ ! -d "$ROOT" ]; then
    echo "Directory not found: $ROOT"
    exit 1
fi

for scene_dir in "$ROOT"/*/; do
    scene=$(basename "${scene_dir%/}")
    if [ ! -d "${scene_dir}point_cloud" ]; then
        continue
    fi
    echo "=============================================="
    echo "Scene: $scene"
    echo "=============================================="

    for impl in $IMPL_LIST; do
        if [ "$impl" = "torch" ]; then
            export GSPLAT_TORCH_SLICE=1
        else
            unset GSPLAT_TORCH_SLICE
        fi
        echo "[${impl}] $scene_dir"
        python render_fps.py \
            -m "$scene_dir" \
            --iteration best \
            --skip_train \
            --num_views "$NUM_VIEWS" \
            --num_frames "$NUM_FRAMES" \
            --quiet
    done
    unset GSPLAT_TORCH_SLICE
done

echo ""
echo "Per-scene fps.txt files:"
find "$ROOT" -maxdepth 2 -name fps.txt -print
