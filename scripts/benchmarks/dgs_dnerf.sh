#!/bin/bash

# Benchmark different modes on D-NeRF dynamic datasets
#
# Modes:
# | Mode         | Output Dir                | Description                            |
# |--------------|---------------------------|----------------------------------------|
# | opacity_only | output/opacity_only/...   | Opacity conditioning only (no position)|
# | opacity_pos  | output/opacity_pos/...    | Opacity + Position conditioning        |
# | opacity_pos_rot | output/opacity_pos_rot/... | Opacity + Position + Rotation cond. |
# | ndgs         | output/ndgs/...           | N-DGS with full Cholesky precision     |
#
# Note: Rotation conditioning is only available for dynamic scenes (C=4 with time)
# Note: Scale is NOT view-dependent (use get_scaling directly)

shopt -s dotglob

base_dir="/code/dataset/dnerf/"

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
        --mv 4 \
        --input_dim 7 \
        --resolution 2 \
        $extra_args \
        --eval

    # Render at multiple iterations (including best)
    for iter in 7000 30000 best; do
        python render.py -m "$output_dir" \
            --skip_train \
            --iteration ${iter} \
            --input_dim 7 \
            --resolution 2 \
            $extra_args
    done

    # Compute metrics
    python metrics.py -m "$output_dir"
}

# # ============================================
# # 4. NDGS mode (full Cholesky precision)
# # ============================================
# echo "=============================================="
# echo "Running NDGS mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         scene_name=$(basename "${dir%/}")
#         if [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/ndgs/dnerf/${scene_name}"
#         echo "Processing ${scene_name} with mode ndgs..."
#         run_experiment "ndgs" "$output_dir" "$dir" ""
#     fi
# done

# # ============================================
# # 1. opacity_only mode (no position shift)
# # ============================================
# echo "=============================================="
# echo "Running opacity_only mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         scene_name=$(basename "${dir%/}")
#         if [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/opacity_only/dnerf/${scene_name}"
#         echo "Processing ${scene_name} with mode opacity_only..."
#         run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False --use_view_dependent_rot False"
#     fi
# done

# ============================================
# 2. opacity_pos mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos mode benchmarks"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/opacity_pos/dnerf/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --use_view_dependent_rot False"
    fi
done

# # ============================================
# # 3. opacity_pos_rot mode (opacity + position + rotation)
# # ============================================
# echo "=============================================="
# echo "Running opacity_pos_rot mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         scene_name=$(basename "${dir%/}")
#         if [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/opacity_pos_rot/dnerf/${scene_name}"
#         echo "Processing ${scene_name} with mode opacity_pos_rot..."
#         run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --use_view_dependent_rot True"
#     fi
# done


echo "Benchmark completed!"
