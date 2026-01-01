#!/bin/bash

# Train and render on NeRF synthetic datasets

shopt -s dotglob

base_dirs=("/code/dataset/nerf_synthetic/")

for base_dir in "${base_dirs[@]}"; do
    echo "Listing subdirectories in: $base_dir"

    for dir in "$base_dir"*/; do
        if [ -d "$dir" ]; then
            # Extract clean directory name
            clean_dir="${dir%/}"
            clean_dir="${clean_dir#/code/dataset/}"

            echo "Processing ${clean_dir}..."

            # Train
            python train.py -s "../dataset/${clean_dir}" \
                --model_path "../output/6dgs/${clean_dir}" \
                --eval \
        --disable_viewer

            # Render at multiple iterations
            for iter in 500 2000 7000 15000 30000; do
                python render.py -m "../output/6dgs/${clean_dir}" \
                    --skip_train \
                    --iteration ${iter}
            done

            # Compute metrics
            python metrics.py -m "../output/6dgs/${clean_dir}"
        fi
    done

    echo
done
