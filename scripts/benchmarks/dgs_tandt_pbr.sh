#!/bin/bash

# Benchmark dgs and dgs-color on Tanks & Temples PBR datasets

shopt -s dotglob

base_dir="/code/dataset/tandt_db/6dgs-pbr/"
modes=("dgs")

for mode in "${modes[@]}"; do
    echo "=============================================="
    echo "Running benchmarks with mode: $mode"
    echo "=============================================="

    for dir in "$base_dir"*/; do
        if [ -d "$dir" ]; then
            # Extract scene name
            scene_name=$(basename "${dir%/}")

            # Skip zip files
            if [[ "$scene_name" == *.zip ]]; then
                continue
            fi

            output_dir="output/${mode}/tandt_pbr/${scene_name}"
            echo "Processing ${scene_name} with mode ${mode}..."

            # Train (training time is saved internally by train.py)
            python train.py -s "$dir" \
                --model_path "$output_dir" \
                --mode "$mode" \
                --eval

            # Render at multiple iterations
            for iter in 7000 30000; do
                python render.py -m "$output_dir" \
                    --mode "$mode" \
                    --skip_train \
                    --iteration ${iter}
            done

            # Compute metrics
            python metrics.py -m "$output_dir"
        fi
    done
done

echo "Benchmark completed!"
