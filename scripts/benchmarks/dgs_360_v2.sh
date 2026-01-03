#!/bin/bash

# Benchmark different modes on Mip-NeRF 360 v2 datasets
#
# Modes:
# | Mode                 | Output Dir                        | Description                            |
# |----------------------|-----------------------------------|----------------------------------------|
# | opacity_only         | output/standard/opacity_only/...           | Opacity conditioning only (no position)|
# | opacity_pos          | output/standard/opacity_pos/...            | Opacity + Position conditioning        |
# | opacity_pos_decouple | output/standard/opacity_pos_decouple/...   | Decoupled position + opacity (λ=0)     |
# | ndgs                 | output/standard/ndgs/...                   | N-DGS with full Cholesky precision     |
# | ndgs_v2_no_pos       | output/standard/ndgs_v2_no_pos/...         | N-DGS V2: v_11 only, no position shift |
# | ndgs_v2_with_pos     | output/standard/ndgs_v2_with_pos/...       | N-DGS V2: v_11 only, with position shift|
#
# Note: Rotation conditioning is only available for dynamic scenes (C=4 with time)
# Note: Scale is NOT view-dependent (use get_scaling directly)

shopt -s dotglob

base_dir="/code/dataset/360_v2/"

# List of all scenes in 360_v2 dataset
SCENES=(
    "bicycle"
    "bonsai"
    "counter"
    "flowers"
    "garden"
    "kitchen"
    "room"
    "stump"
    "treehill"
)

# Function to run experiment for a given mode and output directory
run_experiment() {
    local mode=$1
    local output_dir=$2
    local dir=$3
    local extra_args=$4

    # Skip if results already exist
    if [ -f "$output_dir/results.json" ]; then
        echo "Skipping (results.json exists)"
        return
    fi

    # Train (training time is saved internally by train.py)
    python train.py -s "$dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        $extra_args \
        --eval \
        --disable_viewer \
        -w

    # Render at multiple iterations (including best)
    for iter in 7000 30000 best; do
        python render.py -m "$output_dir" \
            --skip_train \
            --iteration ${iter} \
            $extra_args
    done

    # Compute metrics
    python metrics.py -m "$output_dir"
}

# ============================================
# 1. opacity_only mode (no position shift)
# ============================================
echo "=============================================="
echo "Running opacity_only mode benchmarks"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    dir="${base_dir}${scene_name}"
    if [ -d "$dir" ]; then
        output_dir="output/standard/opacity_only/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_only..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False"
    fi
done

# ============================================
# 2. opacity_pos mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos mode benchmarks"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    dir="${base_dir}${scene_name}"
    if [ -d "$dir" ]; then
        output_dir="output/standard/opacity_pos/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
    fi
done

# ============================================
# 2. opacity_pos_update mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos_update mode benchmarks"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    dir="${base_dir}${scene_name}"
    if [ -d "$dir" ]; then
        output_dir="output/standard/opacity_pos_update/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_update..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
    fi
done

# ============================================
# 3. opacity_pos_decouple mode (decoupled λ=0)
# ============================================
echo "=============================================="
echo "Running opacity_pos_decouple mode benchmarks"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    dir="${base_dir}${scene_name}"
    if [ -d "$dir" ]; then
        output_dir="output/standard/opacity_pos_decouple/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_decouple..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --use_opacity_pos_decouple True"
    fi
done

# ============================================
# 4. NDGS mode (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running NDGS mode benchmarks"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    dir="${base_dir}${scene_name}"
    if [ -d "$dir" ]; then
        output_dir="output/standard/ndgs/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs..."
        run_experiment "ndgs" "$output_dir" "$dir" ""
    fi
done

echo "Benchmark completed!"
