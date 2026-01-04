#!/bin/bash

# Measure FPS for DGS MCMC models
#
# This script measures rendering FPS for trained MCMC models:
# - ndgs: N-DGS with full Cholesky precision
# - opacity_only: Opacity conditioning only (no position)
# - opacity_pos: Opacity + Position conditioning
# - opacity_pos_update: Opacity + Position conditioning (updated version)
#
# Only processes models that exist in output/mcmc/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_BASE="$PROJECT_ROOT/output/mcmc"

# FPS measurement parameters
NUM_FRAMES=500
NUM_VIEWS=20

echo "=============================================="
echo "Measuring FPS for DGS MCMC models"
echo "=============================================="
echo "Output base directory: $OUTPUT_BASE"
echo "FPS parameters: num_frames=$NUM_FRAMES, num_views=$NUM_VIEWS"
echo ""

# Function to measure FPS for a given mode and dataset
measure_fps() {
    local mode=$1
    local dataset=$2
    local model_path=$3
    local extra_args=$4

    if [ ! -d "$model_path" ]; then
        echo "  [SKIP] Model directory not found: $model_path"
        return
    fi

    # Check if best checkpoint exists (try both "best" and "iteration_best")
    if [ ! -d "$model_path/point_cloud/iteration_best" ] && [ ! -d "$model_path/point_cloud/best" ]; then
        echo "  [SKIP] Best checkpoint not found in: $model_path/point_cloud/"
        return
    fi

    # Skip if FPS results already exist
    if [ -f "$model_path/fps_results/fps_iteration_best.txt" ]; then
        echo "  [SKIP] FPS results already exist: $model_path/fps_results/fps_iteration_best.txt"
        return
    fi

    echo "  [FPS] Measuring FPS for $dataset ($mode)..."

    # Measure FPS for best iteration (CUDA optimizations enabled by default)
    python "$PROJECT_ROOT/render_fps.py" \
        -m "$model_path" \
        --iteration best \
        --skip_train \
        --num_frames $NUM_FRAMES \
        --num_views $NUM_VIEWS \
        $extra_args

    if [ $? -eq 0 ]; then
        echo "  [DONE] FPS measurement completed for $dataset ($mode)"
    else
        echo "  [ERROR] FPS measurement failed for $dataset ($mode)"
    fi
    echo ""
}

# Function to process all scenes for a given mode
process_mode() {
    local mode=$1
    local mode_dir=$2
    local extra_args=$3

    echo "=============================================="
    echo "Processing mode: $mode"
    echo "=============================================="

    if [ ! -d "$OUTPUT_BASE/$mode_dir" ]; then
        echo "[SKIP] Mode directory not found: $OUTPUT_BASE/$mode_dir"
        echo ""
        return
    fi

    # Process NeRF Synthetic scenes
    if [ -d "$OUTPUT_BASE/$mode_dir/nerf_synthetic" ]; then
        echo "Processing NeRF Synthetic scenes..."
        for scene_path in "$OUTPUT_BASE/$mode_dir/nerf_synthetic"/*; do
            if [ -d "$scene_path" ]; then
                scene_name=$(basename "$scene_path")
                measure_fps "$mode" "nerf_synthetic/$scene_name" "$scene_path" "$extra_args"
            fi
        done
    fi

    # Process Mip-NeRF 360 v2 scenes
    if [ -d "$OUTPUT_BASE/$mode_dir/360_v2" ]; then
        echo "Processing Mip-NeRF 360 v2 scenes..."
        for scene_path in "$OUTPUT_BASE/$mode_dir/360_v2"/*; do
            if [ -d "$scene_path" ]; then
                scene_name=$(basename "$scene_path")
                measure_fps "$mode" "360_v2/$scene_name" "$scene_path" "$extra_args"
            fi
        done
    fi

    # Process T&T PBR scenes (6DGS PBR dataset)
    if [ -d "$OUTPUT_BASE/$mode_dir/tandt_pbr" ]; then
        echo "Processing T&T PBR scenes..."
        for scene_path in "$OUTPUT_BASE/$mode_dir/tandt_pbr"/*; do
            if [ -d "$scene_path" ]; then
                scene_name=$(basename "$scene_path")
                measure_fps "$mode" "tandt_pbr/$scene_name" "$scene_path" "$extra_args"
            fi
        done
    fi

    echo ""
}

# ============================================
# Measure FPS for each mode
# ============================================

# 1. NDGS mode
process_mode "ndgs" "ndgs" "--mode ndgs"

# 2. Opacity only mode
process_mode "opacity_only" "opacity_only" "--mode dgs --use_view_dependent_pos False"

# 3. Opacity + Position mode
process_mode "opacity_pos" "opacity_pos" "--mode dgs --use_view_dependent_pos True"

# 4. Opacity + Position Update mode
process_mode "opacity_pos_update" "opacity_pos_update" "--mode dgs --use_view_dependent_pos True"

echo "=============================================="
echo "All FPS measurements completed!"
echo "=============================================="
echo ""
echo "Results are saved in: <model_path>/fps_results/fps_iteration_best.txt"
