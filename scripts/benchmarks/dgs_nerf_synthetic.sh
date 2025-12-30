#!/bin/bash

# Benchmark different modes on NeRF synthetic datasets
#
# Modes:
# | Mode                 | Output Dir                        | Description                            |
# |----------------------|-----------------------------------|----------------------------------------|
# | opacity_only         | output/standard/opacity_only/...           | Opacity conditioning only (no position)|
# | opacity_pos          | output/standard/opacity_pos/...            | Opacity + Position conditioning        |
# | opacity_pos_decouple | output/standard/opacity_pos_decouple/...   | Decoupled position + opacity (λ=0)     |
# | ndgs                 | output/standard/ndgs/...                   | N-DGS with full Cholesky precision     |
# | ubs                  | output/standard/ubs/...                    | Unbounded Splatting baseline           |
# | ndgs_v2_no_pos       | output/standard/ndgs_v2_no_pos/...         | N-DGS V2: v_11 only, no position shift |
# | ndgs_v2_with_pos     | output/standard/ndgs_v2_with_pos/...       | N-DGS V2: v_11 only, with position shift|
#
# Note: Rotation conditioning is only available for dynamic scenes (C=4 with time)
# Note: Scale is NOT view-dependent (use get_scaling directly)

shopt -s dotglob

base_dir="/code/dataset/nerf_synthetic/"

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

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_only/nerf_synthetic/${scene_name}"
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

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_pos/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
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
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_pos_decouple/nerf_synthetic/${scene_name}"
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

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/ndgs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs..."
        run_experiment "ndgs" "$output_dir" "$dir" ""
    fi
done

# ============================================
# 5. UBS mode (unbounded splatting baseline)
# ============================================
echo "=============================================="
echo "Running UBS mode benchmarks"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/ubs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode ubs..."
        run_experiment "ubs" "$output_dir" "$dir" ""
    fi
done

# # ============================================
# # 6. NDGS V2 no position shift mode
# # ============================================
# echo "=============================================="
# echo "Running NDGS V2 (no position shift) mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         clean_dir="${dir%/}"
#         scene_name=$(basename "$clean_dir")
#         if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/standard/ndgs_v2_no_pos/nerf_synthetic/${scene_name}"
#         echo "Processing ${scene_name} with mode ndgs-v2 (no pos)..."
#         run_experiment "ndgs-v2" "$output_dir" "$dir" "--use_rot_scale_l_triangle True --use_view_dependent_pos False"
#     fi
# done

# # ============================================
# # 6. NDGS V2 with position shift mode
# # ============================================
# echo "=============================================="
# echo "Running NDGS V2 (with position shift) mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         clean_dir="${dir%/}"
#         scene_name=$(basename "$clean_dir")
#         if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/standard/ndgs_v2_with_pos/nerf_synthetic/${scene_name}"
#         echo "Processing ${scene_name} with mode ndgs-v2 (with pos)..."
#         run_experiment "ndgs-v2" "$output_dir" "$dir" "--use_rot_scale_l_triangle True --use_view_dependent_pos True"
#     fi
# done

echo "Benchmark completed!"
