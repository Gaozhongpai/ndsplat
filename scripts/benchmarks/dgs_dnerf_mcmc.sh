#!/bin/bash

# Benchmark different DGS modes with MCMC densification on D-NeRF dynamic datasets
#
# Modes:
# | Mode             | Output Dir                              | Description                            |
# |------------------|-----------------------------------------|----------------------------------------|
# | opacity_pos      | output/mcmc/opacity_pos/dnerf/...       | dGS with position shift (MCMC)         |
# | ndgs             | output/mcmc/ndgs/dnerf/...              | N-DGS full Cholesky precision (MCMC)   |
# | ubs              | output/mcmc/ubs/dnerf/...               | UBS Beta kernel (MCMC)                 |
# | dbs              | output/mcmc/dbs/dnerf/...               | dBS Direct Beta Splatting (MCMC)       |

shopt -s dotglob

base_dir="/code/dataset/dnerf/"

SCENES=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "mutant" "standup" "trex")

# MCMC parameters (matching ubs_fresh benchmark.py settings)
NOISE_LR=1.0
OPACITY_REG=0.01
SCALE_REG=0.01
CAP_MAX=150000

# Function to run experiment for a given mode and output directory
run_experiment() {
    local mode=$1
    local output_dir=$2
    local scene_dir=$3
    local scene_name=$4
    local extra_args=$5

    # Skip if results already exist
    if [ -f "$output_dir/results.json" ]; then
        echo "Skipping (results.json exists)"
        return
    fi

    echo "  Using cap_max: $CAP_MAX"

    # # Train with MCMC densification (skip if point cloud already exists)
    # if [ -d "$output_dir/point_cloud" ]; then
    #     echo "  Skipping training (point_cloud exists)"
    # else
    python train.py -s "$scene_dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        --input_dim 7 \
        -r 2 \
        --densification_strategy mcmc \
        --mcmc_cap_max $CAP_MAX \
        --noise_lr $NOISE_LR \
        --opacity_reg $OPACITY_REG \
        --scale_reg $SCALE_REG \
        $extra_args \
        --eval \
        --disable_viewer
    # fi

    # Render best iteration
    python render.py -m "$output_dir" \
        --skip_train \
        --iteration best \
        --input_dim 7 \
        -r 2 \
        $extra_args

    # Compute metrics
    python metrics.py -m "$output_dir"
}

# ============================================
# 1. dgs mode with MCMC (dGS with position shift)
# ============================================
echo "=============================================="
echo "Running dgs mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/dgs/dnerf/${scene_name}"
        echo "Processing ${scene_name} with mode dgs (MCMC)..."
        run_experiment "dgs" "$output_dir" "$scene_dir" "$scene_name" "--use_view_dependent_pos True --l_22_inv_init_scale 0.02"
    fi
done
python tools/summarize_results.py output/mcmc/dgs/dnerf

# ============================================
# 2. NDGS mode with MCMC
# ============================================
echo "=============================================="
echo "Running NDGS mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/ndgs/dnerf/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs (MCMC)..."
        run_experiment "ndgs" "$output_dir" "$scene_dir" "$scene_name" "--lambda_opc 0.1"
    fi
done
python tools/summarize_results.py output/mcmc/ndgs/dnerf

# # ============================================
# # 3. UBS mode with MCMC (gsplat rasterizer)
# # ============================================
# echo "=============================================="
# echo "Running UBS mode benchmarks (MCMC, gsplat)"
# echo "=============================================="

# for scene_name in "${SCENES[@]}"; do
#     scene_dir="${base_dir}${scene_name}"
#     if [ -d "$scene_dir" ]; then
#         output_dir="output/mcmc/ubs/dnerf/${scene_name}"
#         echo "Processing ${scene_name} with mode ubs (MCMC, gsplat)..."
#         run_experiment "ubs" "$output_dir" "$scene_dir" "$scene_name" "--use_gsplat"
#     fi
# done
# python tools/summarize_results.py output/mcmc/ubs/dnerf

# # ============================================
# # 4. dBS mode with MCMC (gsplat rasterizer)
# # ============================================
# echo "=============================================="
# echo "Running dBS mode benchmarks (MCMC, gsplat)"
# echo "=============================================="

# for scene_name in "${SCENES[@]}"; do
#     scene_dir="${base_dir}${scene_name}"
#     if [ -d "$scene_dir" ]; then
#         output_dir="output/mcmc/dbs/dnerf/${scene_name}"
#         echo "Processing ${scene_name} with mode dbs (MCMC, gsplat)..."
#         run_experiment "dbs" "$output_dir" "$scene_dir" "$scene_name" "--use_gsplat"
#     fi
# done
# python tools/summarize_results.py output/mcmc/dbs/dnerf

echo "D-NeRF MCMC Benchmark completed!"
