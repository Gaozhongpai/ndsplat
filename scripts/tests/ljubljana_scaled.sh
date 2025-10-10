#!/bin/bash

# Train and render on Ljubljana datasets (scaled versions only)

shopt -s dotglob

base_dirs=("/code/dataset/ljubljana/")

for base_dir in "${base_dirs[@]}"; do
    echo "Listing subdirectories in: $base_dir"

    for dir in "$base_dir"*/; do
        # Only process directories ending with "-scaled"
        if [[ "$dir" == *"-scaled/" ]]; then
            # Extract clean directory name
            clean_dir="${dir%/}"
            clean_dir="${clean_dir#/code/dataset/}"

            echo "Processing ${clean_dir}..."

            # Train
            python train.py -s "../dataset/${clean_dir}" \
                --model_path "output/${clean_dir}" \
                --eval

            # Render
            python render.py -m "output/${clean_dir}" --skip_train

            # Note: Metrics computation is commented out
            # python metrics.py -m "output_fast/${clean_dir}"
        fi
    done

    echo
done
