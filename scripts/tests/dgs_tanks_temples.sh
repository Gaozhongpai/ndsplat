#!/bin/bash

# Train and render on T&T volumetric datasets with DGS

shopt -s dotglob

datasets=(
    "/code/dataset/tandt_db/bunny_cloud_cut"
    "/code/dataset/tandt_db/cloud_cut"
    "/code/dataset/tandt_db/explosion"
    "/code/dataset/tandt_db/smoke"
    "/code/dataset/tandt_db/translucent_suzanne_cut"
    "/code/dataset/tandt_db/subsurface_dragon2"
)

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Train
        python train.py -s "dataset/${clean_dir}" \
            --model_path "output/dgs/${clean_dir}" \
            --eval

        # Render at multiple iterations
        for iter in 500 2000 7000 15000 30000; do
            python render.py -m "output/dgs/${clean_dir}" \
                --skip_train \
                --iteration ${iter}
        done

        # Compute metrics
        python metrics.py -m "output/dgs/${clean_dir}"
    fi
done
