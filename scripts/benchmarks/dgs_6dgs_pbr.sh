#!/bin/bash

# Benchmark dgs with different view-dependent configurations on Tanks & Temples PBR datasets
#
# Configurations:
# | Config       | Output Dir                   | pos | scale | rot |
# |--------------|------------------------------|-----|-------|-----|
# | no_view_dep  | output/dgs_no_view_dep/...   |  -  |   -   |  -  |
# | pos_only     | output/dgs_pos_only/...      |  +  |   -   |  -  |
# | pos_rot      | output/dgs_pos_rot/...       |  +  |   -   |  +  |
# | rot_only     | output/dgs_rot_only/...      |  -  |   -   |  +  |

shopt -s dotglob

base_dir="/code/dataset/tandt_db/6dgs-pbr/"
mode="dgs"

# Define configurations as "name:pos:scale:rot"
configs=(
    "no_view_dep:False:False:False"
    "pos_only:True:False:False"
    "pos_rot:True:False:True"
    "rot_only:False:False:True"
)

for config in "${configs[@]}"; do
    # Parse config
    IFS=':' read -r config_name use_pos use_scale use_rot <<< "$config"

    echo "=============================================="
    echo "Running benchmarks with config: $config_name"
    echo "  use_view_dependent_pos=$use_pos"
    echo "  use_view_dependent_scale=$use_scale"
    echo "  use_view_dependent_rotation=$use_rot"
    echo "=============================================="

    for dir in "$base_dir"*/; do
        if [ -d "$dir" ]; then
            # Extract scene name
            scene_name=$(basename "${dir%/}")

            # Skip zip files
            if [[ "$scene_name" == *.zip ]]; then
                continue
            fi

            output_dir="output/${mode}_${config_name}/tandt_pbr/${scene_name}"

            # Skip if results already exist
            if [ -f "$output_dir/results.json" ]; then
                echo "Skipping ${scene_name} with config ${config_name} (results.json exists)"
                continue
            fi

            echo "Processing ${scene_name} with config ${config_name}..."

            # Train (training time is saved internally by train.py)
            python train.py -s "$dir" \
                --model_path "$output_dir" \
                --mode "$mode" \
                --use_view_dependent_pos "$use_pos" \
                --use_view_dependent_scale "$use_scale" \
                --use_view_dependent_rotation "$use_rot" \
                --eval

            # Render at multiple iterations
            for iter in 7000 30000; do
                python render.py -m "$output_dir" \
                    --skip_train \
                    --iteration ${iter}
            done

            # Compute metrics
            python metrics.py -m "$output_dir"
        fi
    done
done

echo "Benchmark completed!"
