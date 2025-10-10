#!/bin/bash

# Train and render on NeRF synthetic dataset for ablation studies

shopt -s dotglob

datasets=(
    "/code/dataset/nerf_synthetic/lego"
    # Add more datasets as needed:
    # "/code/dataset/nerf_synthetic/chair"
    # "/code/dataset/nerf_synthetic/drums"
    # "/code/dataset/nerf_synthetic/ficus"
    # "/code/dataset/nerf_synthetic/hotdog"
    # "/code/dataset/nerf_synthetic/materials"
    # "/code/dataset/nerf_synthetic/mic"
    # "/code/dataset/nerf_synthetic/ship"
)

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Train
        python train.py -s "../dataset/${clean_dir}" \
            --model_path "../output/6dgs_ablation/${clean_dir}_v2" \
            --eval

        # Render at multiple iterations
        for iter in 500 2000 7000 15000 30000; do
            python render.py -m "../output/6dgs_ablation/${clean_dir}_v2" \
                --skip_train \
                --iteration ${iter}
        done

        # Compute metrics
        python metrics.py -m "../output/6dgs_ablation/${clean_dir}_v2"
    fi
done
