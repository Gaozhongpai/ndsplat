#!/bin/bash

# Benchmark different DGS modes with MCMC densification on 7DGS-PBR dynamic datasets
#
# Modes:
# | Mode         | Output Dir                     | Description                            |
# |--------------|--------------------------------|----------------------------------------|
# | 3dgs         | output/mcmc/3dgs/...           | Baseline 3D Gaussian Splatting         |
# | opacity_only | output/mcmc/opacity_only/...   | Opacity conditioning only (no position)|
# | opacity_pos  | output/mcmc/opacity_pos/...    | Opacity + Position conditioning        |
# | ndgs         | output/mcmc/ndgs/...           | N-DGS with full Cholesky precision     |
# | ubs          | output/mcmc/ubs/...            | UBS (Beta kernel, full covariance)     |
# | dbs          | output/mcmc/dbs/...            | dBS (Beta kernel, direct Cholesky)     |
#
# MCMC Parameters:
# - densification_strategy: mcmc
# - mcmc_cap_max: Maximum Gaussians (scene-specific)
# - noise_lr: Noise learning rate for spatial perturbation (default 1.0)
# - opacity_reg: Opacity regularization weight (default 0.01)
# - scale_reg: Scale regularization weight (default 0.01)

shopt -s dotglob

base_dir="/code/dataset/dyct/7dgs_pbr/"

# MCMC parameters
NOISE_LR=1.0
SCALE_REG=0

# Scene-specific cap_max values
declare -A MCMC_CAP_MAX
MCMC_CAP_MAX["cloud"]=150000
MCMC_CAP_MAX["dust"]=150000
MCMC_CAP_MAX["flame"]=150000
MCMC_CAP_MAX["heart"]=150000
MCMC_CAP_MAX["heart_1600"]=150000
MCMC_CAP_MAX["suzanne"]=300000

# Scene-specific opacity_reg values
declare -A OPACITY_REG_MAP
OPACITY_REG_MAP["cloud"]=0
OPACITY_REG_MAP["dust"]=0
OPACITY_REG_MAP["flame"]=0
OPACITY_REG_MAP["heart"]=0
OPACITY_REG_MAP["heart_1600"]=0
OPACITY_REG_MAP["suzanne"]=0.01

# Function to run experiment for a given mode and output directory
run_experiment() {
    local mode=$1
    local output_dir=$2
    local dir=$3
    local scene_name=$4
    local extra_args=$5

    # Skip if results already exist
    if [ -f "$output_dir/results.json" ]; then
        echo "Skipping (results.json exists)"
        return
    fi

    # Get scene-specific cap_max and opacity_reg, or use defaults
    local cap_max=${MCMC_CAP_MAX[$scene_name]:-150000}
    local opacity_reg=${OPACITY_REG_MAP[$scene_name]:-0}
    echo "Using mcmc_cap_max=$cap_max, opacity_reg=$opacity_reg for scene $scene_name"

    # Train with MCMC densification
    python train.py -s "$dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        --input_dim 7 \
        --resolution 2 \
        --densification_strategy mcmc \
        --mcmc_cap_max $cap_max \
        --noise_lr $NOISE_LR \
        --opacity_reg $opacity_reg \
        --scale_reg $SCALE_REG \
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

# # ============================================
# # 2. opacity_only mode with MCMC (no position shift)
# # ============================================
# echo "=============================================="
# echo "Running opacity_only mode benchmarks (MCMC)"
# echo "=============================================="

# for dir in "$base_dir"*/; do
#     if [ -d "$dir" ]; then
#         scene_name=$(basename "${dir%/}")
#         if [[ "$scene_name" == *.zip ]]; then
#             continue
#         fi

#         # Set l_22_inv_init_scale: 2.5 for cloud, 0.4 for others
#         if [[ "$scene_name" == "cloud" ]]; then
#             l_22_scale=2.5
#         else
#             l_22_scale=0.4
#         fi

#         output_dir="output/mcmc/opacity_only/7dgs_pbr/${scene_name}"
#         echo "Processing ${scene_name} with mode opacity_only (MCMC, l_22_inv_init_scale=${l_22_scale})..."
#         run_experiment "dgs" "$output_dir" "$dir" "$scene_name" "--use_view_dependent_pos False --l_22_inv_init_scale ${l_22_scale}"
#     fi
# done

# ============================================
# 3. opacity_pos mode with MCMC (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        if [[ "$scene_name" == "cloud" ]]; then
            l_22_scale=2.5
        else
            l_22_scale=0.4
        fi

        output_dir="output/mcmc/opacity_pos/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos (MCMC, l_22_inv_init_scale=${l_22_scale})..."
        run_experiment "dgs" "$output_dir" "$dir" "$scene_name" "--use_view_dependent_pos True --l_22_inv_init_scale ${l_22_scale}"
    fi
done

# ============================================
# 4. NDGS mode with MCMC (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running NDGS mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/ndgs/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs (MCMC)..."
        run_experiment "ndgs" "$output_dir" "$dir" "$scene_name" "--lambda_opc 0.2 --use_rot_scale_l_triangle True"
    fi
done


# ============================================
# 5. UBS mode with MCMC (full covariance, Beta kernel)
# ============================================
echo "=============================================="
echo "Running UBS mode benchmarks (MCMC)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/mcmc/ubs/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode ubs (MCMC)..."
        run_experiment "ubs" "$output_dir" "$dir" "$scene_name" "--use_gsplat"
    fi
done
python tools/summarize_results.py output/mcmc/ubs/7dgs_pbr


# ============================================
# 6. dBS mode with MCMC (direct Cholesky, Beta kernel)
# ============================================
echo "=============================================="
echo "Running dBS mode benchmarks (MCMC, gsplat)"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "${dir%/}")
        if [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        if [[ "$scene_name" == "cloud" ]]; then
            l_22_scale=2.5
        else
            l_22_scale=0.4
        fi

        output_dir="output/mcmc/dbs/7dgs_pbr/${scene_name}"
        echo "Processing ${scene_name} with mode dbs (MCMC, gsplat, l_22_inv_init_scale=${l_22_scale})..."
        run_experiment "dbs" "$output_dir" "$dir" "$scene_name" "--use_gsplat --l_22_inv_init_scale ${l_22_scale}"
    fi
done
python tools/summarize_results.py output/mcmc/dbs/7dgs_pbr


echo "MCMC Benchmark completed!"
