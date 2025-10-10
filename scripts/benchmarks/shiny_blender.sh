#!/bin/bash

# Train and render on Shiny Blender datasets (rebuttal)

shopt -s dotglob

datasets=(
    "/code/dataset/shiny/shiny/cd"
    "/code/dataset/shiny/shiny/crest"
    "/code/dataset/shiny/shiny/food"
    "/code/dataset/shiny/shiny/giants"
    "/code/dataset/shiny/shiny/lab"
    "/code/dataset/shiny/shiny/pasta"
    "/code/dataset/shiny/shiny/seasoning"
    "/code/dataset/shiny/shiny/tools"
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
