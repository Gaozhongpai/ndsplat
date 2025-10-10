#!/bin/bash

# Train and render on Tanks & Temples datasets (rebuttal)

shopt -s dotglob

datasets=(
    "/code/dataset/tandt_db/subsurface_dragon2"
    # Add more as needed:
    # "/code/dataset/tandt_db/drjohnson"
    # "/code/dataset/tandt_db/playroom"
    # "/code/dataset/tandt_db/train"
)

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Train
        python train.py -s "../dataset/${clean_dir}" \
            --model_path "../output/6dgs_rebuttal/${clean_dir}" \
            --eval

        # Render at multiple iterations
        for iter in 500 2000 7000 15000 30000; do
            python render.py -m "../output/6dgs_rebuttal/${clean_dir}" \
                --skip_train \
                --iteration ${iter}
        done

        # Compute metrics
        python metrics.py -m "../output/6dgs_rebuttal/${clean_dir}"
    fi
done
