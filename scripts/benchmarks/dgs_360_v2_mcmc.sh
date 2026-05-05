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

# Outdoor scenes: -r 4 (high-res source images)
# Indoor scenes: -r 2 (lower-res source images)
# Following 3DGS (Kerbl et al., SIGGRAPH 2023) standard protocol
OUTDOOR_SCENES=("bicycle" "flowers" "garden" "stump" "treehill")
INDOOR_SCENES=("bonsai" "counter" "kitchen" "room")
SCENES=("${OUTDOOR_SCENES[@]}" "${INDOOR_SCENES[@]}")

# MCMC parameters
NOISE_LR=1.0
OPACITY_REG=0.01
SCALE_REG=0

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

# Scene-specific resolution factors (-r 4 outdoor, -r 2 indoor)
declare -A RESOLUTION
for s in "${OUTDOOR_SCENES[@]}"; do RESOLUTION[$s]=4; done
for s in "${INDOOR_SCENES[@]}"; do RESOLUTION[$s]=2; done

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

    # Get scene-specific cap_max and resolution
    local cap_max=${MCMC_CAP_MAX[$scene_name]}
    if [ -z "$cap_max" ]; then
        cap_max=300000  # default fallback
    fi
    local res=${RESOLUTION[$scene_name]}
    if [ -z "$res" ]; then
        res=2  # default fallback
    fi

    echo "  Using cap_max: $cap_max, resolution: -r $res"

    # Train with MCMC densification (skip if point cloud already exists)
    if [ -d "$output_dir/point_cloud" ]; then
        echo "  Skipping training (point_cloud exists)"
    else
        python train.py -s "$scene_dir" \
            --model_path "$output_dir" \
            --mode "$mode" \
            -r $res \
            --densification_strategy mcmc \
            --mcmc_cap_max $cap_max \
            --noise_lr $NOISE_LR \
            --opacity_reg $OPACITY_REG \
            --scale_reg $SCALE_REG \
            $extra_args \
            --eval \
            --disable_viewer
    fi
        
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

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/3dgs/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode 3dgs (MCMC)..."
        run_experiment "3dgs" "$output_dir" "$scene_dir" "$scene_name" ""
    fi
done
python tools/summarize_results.py output/mcmc/3dgs/360_v2

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
python tools/summarize_results.py output/mcmc/opacity_only/360_v2

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
python tools/summarize_results.py output/mcmc/opacity_pos/360_v2

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
python tools/summarize_results.py output/mcmc/opacity_pos_update/360_v2

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
python tools/summarize_results.py output/mcmc/ndgs/360_v2


# ============================================
# 6. dBS mode with MCMC (gsplat rasterizer)
# ============================================
echo "=============================================="
echo "Running dBS mode benchmarks (MCMC, gsplat)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/dbs/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode dbs (MCMC, gsplat)..."
        run_experiment "dbs" "$output_dir" "$scene_dir" "$scene_name" "--use_gsplat --noise_lr 1000000"
    fi
done
python tools/summarize_results.py output/mcmc/dbs/360_v2


# ============================================
# 7. dBS-SH mode with MCMC (gsplat rasterizer, SH colors)
# ============================================
echo "=============================================="
echo "Running dBS-SH mode benchmarks (MCMC, gsplat)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/dbs-sh/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode dbs-sh (MCMC, gsplat)..."
        run_experiment "dbs-sh" "$output_dir" "$scene_dir" "$scene_name" "--use_gsplat --noise_lr 1000000"
    fi
done
python tools/summarize_results.py output/mcmc/dbs-sh/360_v2

echo "MCMC Benchmark for 360_v2 completed!"


# ============================================
# 5. UBS mode with MCMC (full covariance, Beta kernel)
# ============================================
echo "=============================================="
echo "Running UBS mode benchmarks (MCMC)"
echo "=============================================="

for scene_name in "${SCENES[@]}"; do
    scene_dir="${base_dir}${scene_name}"
    if [ -d "$scene_dir" ]; then
        output_dir="output/mcmc/ubs/360_v2/${scene_name}"
        echo "Processing ${scene_name} with mode ubs (MCMC)..."
        run_experiment "ubs" "$output_dir" "$scene_dir" "$scene_name" "--use_gsplat --noise_lr 1000000"
    fi
done
python tools/summarize_results.py output/mcmc/ubs/360_v2