#!/bin/bash

# Benchmark different modes on 7DGS PBR dynamic datasets
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

base_dir="/code/dataset/dyct/7dgs_pbr/"

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
        --eval \
        --disable_viewer

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

        # Set l_22_inv_init_scale: 2 for cloud, 0.2 for others
        if [[ "$scene_name" == "cloud" ]]; then
            l_22_scale=2.5
        else
            l_22_scale=0.4
        fi

        output_dir="output/standard/opacity_only/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_only (l_22_inv_init_scale=${l_22_scale})..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False --l_22_inv_init_scale ${l_22_scale}"
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
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        # Set l_22_inv_init_scale: 2 for cloud, 0.2 for others
        if [[ "$scene_name" == "cloud" ]]; then
            l_22_scale=2.5
        else
            l_22_scale=0.4
        fi

        output_dir="output/standard/opacity_pos/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos (l_22_inv_init_scale=${l_22_scale})..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --l_22_inv_init_scale ${l_22_scale}"
    fi
done


# ============================================
# 2. opacity_pos_update mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos_update mode benchmarks"
echo "=============================================="
## claude and dust lambda_init=-1.2
for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        # Set l_22_inv_init_scale: 2 for cloud, 0.2 for others
        if [[ "$scene_name" == "cloud" ]]; then
            l_22_scale=2.5
        else
            l_22_scale=0.4
        fi

        output_dir="output/standard/opacity_pos_update/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_update (l_22_inv_init_scale=${l_22_scale})..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --l_22_inv_init_scale ${l_22_scale} --lambda_init -2.5"
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

        # Set l_22_inv_init_scale: 2 for cloud, 0.2 for others
        if [[ "$scene_name" == "cloud" ]]; then
            l_22_scale=2.5
        else
            l_22_scale=0.4
        fi

        output_dir="output/standard/opacity_pos_decouple/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_decouple (l_22_inv_init_scale=${l_22_scale})..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --use_opacity_pos_decouple True --l_22_inv_init_scale ${l_22_scale}"
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

        output_dir="output/standard/ndgs/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs..."
        run_experiment "ndgs" "$output_dir" "$dir" "--lambda_opc 0.2"
    fi
done

echo "Benchmark completed!"
