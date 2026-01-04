#!/bin/bash

# Benchmark different modes on Tanks & Temples PBR datasets
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

base_dir="/code/dataset/tandt_db/6dgs-pbr/"

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
        --disable_viewer

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

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_only/tandt_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_only..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False --l_22_inv_init_scale 2.0"
    fi
done


# # ============================================
# # 2. opacity_pos_woscale mode (opacity + position)
# # ============================================
# echo "=============================================="
# echo "Running opacity_pos_woscale mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         scene_name=$(basename "${dir%/}")
#         if [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/standard/opacity_pos_woscale/tandt_pbr/${scene_name}"
#         echo "Processing ${scene_name} with mode opacity_pos_woscale..."
#         run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --l_22_inv_init_scale 2.0"
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

        output_dir="output/standard/opacity_pos/tandt_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --l_22_inv_init_scale 2.0"
    fi
done


# ============================================
# 2. opacity_pos_update mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos_update mode benchmarks"
echo "=============================================="
## bunny and cloud lambda_init=-1.2
for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_pos_update/tandt_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_update..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --l_22_inv_init_scale 2.0 --lambda_init 0.0" 
    fi
done

# ============================================
# 3. opacity_pos_decouple mode (decoupled λ=0)
# ============================================
echo "=============================================="
echo "Running opacity_pos_decouple mode benchmarks"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_pos_decouple/tandt_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_decouple..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --use_opacity_pos_decouple True --l_22_inv_init_scale 2.0"
    fi
done

# ============================================
# 4. NDGS mode (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running NDGS mode benchmarks"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/ndgs/tandt_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs..."
        run_experiment "ndgs" "$output_dir" "$dir" ""
    fi
done

echo "Benchmark completed!"
