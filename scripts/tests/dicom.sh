#!/bin/bash

# Train and render on CT DICOM datasets

shopt -s dotglob

datasets=("/code/dataset/ct_data/22022107v2")

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Train
        python train.py -s "../dataset/${clean_dir}" \
            --model_path "../output/test/3dgs/${clean_dir}" \
            --eval

        # Render at final iteration
        python render.py -m "../output/test/3dgs/${clean_dir}" \
            --skip_train \
            --iteration 30000

        # Compute metrics
        python metrics.py -m "../output/test/3dgs/${clean_dir}"
    fi
done
