#!/bin/bash

# Benchmark different DGS modes with MCMC densification on 360_v2 datasets
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

# MCMC parameters (per-scene cap_max)
NOISE_LR=1.0
OPACITY_REG=0.01
SCALE_REG=0.01

# Scene-specific cap_max values
declare -A MCMC_CAP_MAX
MCMC_CAP_MAX["bicycle"]=6000000
MCMC_CAP_MAX["flowers"]=3000000
MCMC_CAP_MAX["garden"]=5000000
MCMC_CAP_MAX["stump"]=4500000
MCMC_CAP_MAX["treehill"]=3500000
MCMC_CAP_MAX["room"]=1500000
MCMC_CAP_MAX["counter"]=1500000
MCMC_CAP_MAX["kitchen"]=1500000
MCMC_CAP_MAX["bonsai"]=1500000

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

    # Get scene-specific cap_max
    local cap_max=${MCMC_CAP_MAX[$scene_name]}
    if [ -z "$cap_max" ]; then
        cap_max=300000  # default fallback
    fi

    echo "  Using cap_max: $cap_max"

    # Train with MCMC densification
    python train.py -s "$scene_dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        --densification_strategy mcmc \
        --mcmc_cap_max $cap_max \
        --noise_lr $NOISE_LR \
        --opacity_reg $OPACITY_REG \
        --scale_reg $SCALE_REG \
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
# 1. 3DGS baseline mode with MCMC
# ============================================
echo "=============================================="
echo "Running 3DGS baseline benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/3dgs/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode 3dgs (MCMC)..."
        run_experiment "3dgs" "$output_dir" "$scene_dir" "$scene_name" ""
    fi
done

# ============================================
# 2. opacity_only mode with MCMC (no position shift)
# ============================================
echo "=============================================="
echo "Running opacity_only mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/opacity_only/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_only (MCMC)..."
        run_experiment "dgs" "$output_dir" "$scene_dir" "$scene_name" "--use_view_dependent_pos False"
    fi
done

# ============================================
# 3. opacity_pos mode with MCMC (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/opacity_pos/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos (MCMC)..."
        run_experiment "dgs" "$output_dir" "$scene_dir" "$scene_name" "--use_view_dependent_pos True"
    fi
done

# ============================================
# 3. opacity_pos_update mode with MCMC (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos_update mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/opacity_pos_update/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode opacity_pos_update (MCMC)..."
        run_experiment "dgs" "$output_dir" "$scene_dir" "$scene_name" "--use_view_dependent_pos True"
    fi
done

# ============================================
# 4. NDGS mode with MCMC (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running NDGS mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/ndgs/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs (MCMC)..."
        run_experiment "ndgs" "$output_dir" "$scene_dir" "$scene_name" ""
    fi
done

echo "MCMC Benchmark for 360_v2 completed!"
