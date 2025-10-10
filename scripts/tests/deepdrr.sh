#!/bin/bash

# Train and render on DeepDRR datasets

shopt -s dotglob

base_dirs=("/code/dataset/deepdrr/")

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
                --model_path "../output/ablation/init-even/${clean_dir}" \
                --eval

            # Render
            python render.py -m "../output/ablation/init-even/${clean_dir}" --skip_train

            # Compute metrics
            python metrics.py -m "../output/ablation/init-even/${clean_dir}"
        fi
    done

    echo
done
