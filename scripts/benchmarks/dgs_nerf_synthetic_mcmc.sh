#!/bin/bash

# Benchmark different DGS modes with MCMC densification on NeRF synthetic datasets
#
# Modes:
# | Mode         | Output Dir                     | Description                            |
# |--------------|--------------------------------|----------------------------------------|
# | 3dgs         | output/mcmc/3dgs/...           | Baseline 3D Gaussian Splatting         |
# | opacity_only | output/mcmc/opacity_only/...   | Opacity conditioning only (no position)|
# | opacity_pos  | output/mcmc/opacity_pos/...    | Opacity + Position conditioning        |
# | ndgs         | output/mcmc/ndgs/...           | N-DGS with full Cholesky precision     |
#
# MCMC Parameters:
# - densification_strategy: mcmc
# - mcmc_cap_max: Maximum Gaussians (default 300k)
# - noise_lr: Noise learning rate for spatial perturbation (default 1.0)
# - opacity_reg: Opacity regularization weight (default 0.01)
# - scale_reg: Scale regularization weight (default 0.01)
#
# Note: Rotation conditioning is only available for dynamic scenes (C=4 with time)
# Note: Scale is NOT view-dependent (use get_scaling directly)

shopt -s dotglob

base_dir="/code/dataset/nerf_synthetic/"

# MCMC parameters
MCMC_CAP_MAX=300000
NOISE_LR=1.0
OPACITY_REG=0.01
SCALE_REG=0

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

    # Train with MCMC densification
    python train.py -s "$dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        --densification_strategy mcmc \
        --mcmc_cap_max $MCMC_CAP_MAX \
        --opacity_reg $OPACITY_REG \
        --scale_reg $SCALE_REG \
        $extra_args \
        --eval \
        --disable_viewer \
        -w

    # Render best iteration
    python render.py -m "$output_dir" \
        --skip_train \
        --iteration best \
        $extra_args

    # Compute metrics
    python metrics.py -m "$output_dir"
}

# ============================================
# 1. 3DGS baseline mode with MCMC
# ============================================
echo "=============================================="
echo "Running 3DGS baseline benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/3dgs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode 3dgs (MCMC)..."
        run_experiment "3dgs" "$output_dir" "$dir" ""
    fi
done
python tools/summarize_results.py output/mcmc/3dgs/nerf_synthetic

# ============================================
# 2. opacity_only mode with MCMC (no position shift)
# ============================================
echo "=============================================="
echo "Running opacity_only mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/opacity_only/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_only (MCMC)..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos False"
    fi
done
python tools/summarize_results.py output/mcmc/opacity_only/nerf_synthetic

# ============================================
# 3. opacity_pos mode with MCMC (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/opacity_pos/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos (MCMC)..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
    fi
done
python tools/summarize_results.py output/mcmc/opacity_pos/nerf_synthetic

# ============================================
# 3. opacity_pos_update mode with MCMC (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos_update mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/opacity_pos_update/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_update (MCMC)..."
        run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True"
    fi
done
python tools/summarize_results.py output/mcmc/opacity_pos_update/nerf_synthetic


# # ============================================
# # 3. opacity_pos_beta mode with MCMC (opacity + position)
# # ============================================
# echo "=============================================="
# echo "Running opacity_pos_beta mode benchmarks (MCMC)"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         clean_dir="${dir%/}"
#         scene_name=$(basename "$clean_dir")
#         if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         output_dir="output/mcmc/opacity_pos_beta/nerf_synthetic/${scene_name}"
#         echo "Processing ${scene_name} with mode opacity_pos_beta (MCMC)..."
#         run_experiment "dgs" "$output_dir" "$dir" "--use_view_dependent_pos True --use_beta True --noise_lr 1000"
#     fi
# done

# ============================================
# 4. NDGS mode with MCMC (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running NDGS mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/ndgs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs (MCMC)..."
        run_experiment "ndgs" "$output_dir" "$dir"
    fi
done
python tools/summarize_results.py output/mcmc/ndgs/nerf_synthetic


# ============================================
# 4. UBS mode with MCMC (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running UBS mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/ubs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode ubs (MCMC)..."
        run_experiment "ubs" "$output_dir" "$dir" "--use_gsplat --noise_lr 1000000"
    fi
done
python tools/summarize_results.py output/mcmc/ubs/nerf_synthetic


# ============================================
# 5. dBS mode with MCMC (gsplat rasterizer)
# ============================================
echo "=============================================="
echo "Running dBS mode benchmarks (MCMC, gsplat)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/dbs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode dbs (MCMC, gsplat)..."
        run_experiment "dbs" "$output_dir" "$dir" "--use_gsplat --noise_lr 1000000"
    fi
done
python tools/summarize_results.py output/mcmc/dbs/nerf_synthetic

echo "MCMC Benchmark completed!"
