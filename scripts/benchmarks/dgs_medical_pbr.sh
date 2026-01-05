#!/bin/bash

# Benchmark different modes on Medical PBR datasets
#
# Modes:
# | Mode                 | Output Dir                        | Description                            |
# |----------------------|-----------------------------------|----------------------------------------|
# | opacity_only         | output/opacity_only/...           | Opacity conditioning only (no position)|
# | opacity_pos          | output/opacity_pos/...            | Opacity + Position conditioning        |
# | opacity_pos_decouple | output/opacity_pos_decouple/...   | Decoupled position + opacity (λ=0)     |
# | ndgs                 | output/ndgs/...                   | N-DGS with full Cholesky precision     |
# | ubs                  | output/ubs/...                    | Unbounded Splatting baseline           |
# | ndgs_v2_no_pos       | output/ndgs_v2_no_pos/...         | N-DGS V2: v_11 only, no position shift |
# | ndgs_v2_with_pos     | output/ndgs_v2_with_pos/...       | N-DGS V2: v_11 only, with position shift|
# | 3dgs                 | output/3dgs/...                   | Standard 3DGS baseline                 |
#
# Note: Rotation conditioning is only available for dynamic scenes (C=4 with time)
# Note: Scale is NOT view-dependent (use get_scaling directly)

shopt -s dotglob

base_dir="/code/dataset/tandt_db/medical_pbr/"

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
# 4. 3DGS mode (standard 3DGS baseline)
# ============================================
echo "=============================================="
echo "Running 3DGS mode benchmarks"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/3dgs/medical_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode 3dgs..."
        run_experiment "3dgs" "$output_dir" "$dir" ""
    fi
done

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

        output_dir="output/standard/opacity_only/medical_pbr/${scene_name}"
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

        output_dir="output/standard/opacity_pos/medical_pbr/${scene_name}"
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

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/standard/opacity_pos_update/medical_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_update..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
    fi
done

# # ============================================
# # 2. opacity_pos_sh1 mode (opacity + position)
# # ============================================
# echo "=============================================="
# echo "Running opacity_pos_sh1 mode benchmarks"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         clean_dir="${dir%/}"
#         scene_name=$(basename "$clean_dir")
#         if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/standard/opacity_pos_sh1/medical_pbr/${scene_name}"
#         echo "Processing ${scene_name} with mode opacity_pos_sh1..."
#         run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False --sh_degree 1"
#     fi
# done

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

        output_dir="output/standard/opacity_pos_decouple/medical_pbr/${scene_name}"
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

        output_dir="output/standard/ndgs/medical_pbr/${scene_name}"
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

        output_dir="output/standard/ubs/medical_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode ubs..."
        run_experiment "ubs" "$output_dir" "$dir" ""
    fi
done

echo "Benchmark completed!"
