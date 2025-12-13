#!/bin/bash

# Benchmark different modes and view-dependent configurations on NeRF synthetic datasets
#
# Modes:
# | Mode     | Output Dir              | Description                                    |
# |----------|-------------------------|------------------------------------------------|
# | dgs      | output/dgs/...          | DGS with bounded v_12 parameterization         |
# | ndgs     | output/ndgs/...         | N-DGS with full Cholesky precision             |
# | dgs-full | output/dgs-full_*/...   | DGS-full with configurable view-dependent flags|
#
# DGS-full Configurations:
# | Config       | Output Dir                        | pos | scale | rot |
# |--------------|-----------------------------------|-----|-------|-----|
# | no_view_dep  | output/dgs-full_no_view_dep/...   |  -  |   -   |  -  |
# | pos_only     | output/dgs-full_pos_only/...      |  +  |   -   |  -  |
# | pos_rot      | output/dgs-full_pos_rot/...       |  +  |   -   |  +  |
# | rot_only     | output/dgs-full_rot_only/...      |  -  |   -   |  +  |

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

    # Render at multiple iterations
    for iter in 7000 30000; do
        python render.py -m "$output_dir" \
            --skip_train \
            --iteration ${iter}
    done

    # Compute metrics
    python metrics.py -m "$output_dir"
}

# ============================================
# 1. DGS mode (bounded v_12 parameterization)
# ============================================
echo "=============================================="
echo "Running DGS mode benchmarks"
echo "=============================================="

for dir in "$base_dir"*/; do
    if [ -d "$dir" ]; then
        clean_dir="${dir%/}"
        scene_name=$(basename "$clean_dir")
        if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
            continue
        fi

        output_dir="output/dgs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode dgs..."
        run_experiment "dgs" "$output_dir" "$dir" ""
    fi
done

# ============================================
# 2. NDGS mode (full Cholesky precision)
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

        output_dir="output/ndgs/nerf_synthetic/${scene_name}"
        echo "Processing ${scene_name} with mode ndgs..."
        run_experiment "ndgs" "$output_dir" "$dir" ""
    fi
done

# ============================================
# 3. DGS-full mode with view-dependent configs
# ============================================
# Define configurations as "name:pos:scale:rot"
configs=(
    "pos_rot:True:False:True"
    "no_view_dep:False:False:False"
    "pos_only:True:False:False"
    "rot_only:False:False:True"
)

for config in "${configs[@]}"; do
    # Parse config
    IFS=':' read -r config_name use_pos use_scale use_rot <<< "$config"

    echo "=============================================="
    echo "Running DGS-full benchmarks with config: $config_name"
    echo "  use_view_dependent_pos=$use_pos"
    echo "  use_view_dependent_scale=$use_scale"
    echo "  use_view_dependent_rotation=$use_rot"
    echo "=============================================="

    for dir in "$base_dir"*/; do
        if [ -d "$dir" ]; then
            clean_dir="${dir%/}"
            scene_name=$(basename "$clean_dir")
            if [[ "$scene_name" == "README.txt" ]] || [[ "$scene_name" == *.zip ]]; then
                continue
            fi

            output_dir="output/dgs-full_${config_name}/nerf_synthetic/${scene_name}"
            echo "Processing ${scene_name} with config ${config_name}..."

            extra_args="--use_view_dependent_pos $use_pos --use_view_dependent_scale $use_scale --use_view_dependent_rotation $use_rot"
            run_experiment "dgs-full" "$output_dir" "$dir" "$extra_args"
        fi
    done
done

echo "Benchmark completed!"
