#!/bin/bash

# Benchmark dGS and N-DGS (7DGS) on NeRF-DS dataset (dynamic specular scenes)
#
# NeRF-DS: real-world dynamic scenes with specular/view-dependent effects
# Converted from HyperNeRF format to Blender format using convert_nerfds_to_blender.py
#
# Modes:
# | Mode        | Output Dir                              | Description                      |
# |-------------|-----------------------------------------|----------------------------------|
# | ndgs (7DGS) | output/standard/ndgs/nerfds/...         | N-DGS baseline (7D, with time)   |
# | dgs-O       | output/standard/opacity_only/nerfds/... | dGS opacity-only (no position)   |
# | dgs         | output/standard/opacity_pos/nerfds/...  | dGS full (opacity + position)    |

shopt -s dotglob

base_dir="/code/dataset/NeRF-DS-blender/"

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

    # Train
    python train.py -s "$dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        --input_dim 7 \
        $extra_args \
        --eval \
        --disable_viewer

    # Render at multiple iterations
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
# 1. N-DGS (7DGS) baseline
# ============================================
echo "=============================================="
echo "Running N-DGS (7DGS) mode benchmarks on NeRF-DS"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        output_dir="output/standard/ndgs/nerfds/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs..."
        run_experiment "ndgs" "$output_dir" "$dir" ""
    fi
done

# ============================================
# 2. dGS opacity-only mode
# ============================================
echo "=============================================="
echo "Running dGS opacity-only mode benchmarks on NeRF-DS"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        output_dir="output/standard/opacity_only/nerfds/${scene_name}"
        echo "Processing ${scene_name} with mode dGS-O..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False"
    fi
done

# ============================================
# 3. dGS full mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running dGS full mode benchmarks on NeRF-DS"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        output_dir="output/standard/opacity_pos/nerfds/${scene_name}"
        echo "Processing ${scene_name} with mode dGS..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
    fi
done

echo "NeRF-DS benchmark completed!"
