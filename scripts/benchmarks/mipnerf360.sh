#!/bin/bash

# Train and render on Mip-NeRF 360 datasets

shopt -s dotglob

datasets=(
    "/code/dataset/360_v2/room"
    "/code/dataset/360_v2/stump"
    "/code/dataset/360_v2/treehill"
)

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Train
        python train.py -s "../dataset/${clean_dir}" \
            --model_path "../output/6dgs/${clean_dir}" \
            --eval

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
