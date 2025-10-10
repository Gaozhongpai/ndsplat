#!/bin/bash

# Train and render on DeepDRR datasets for ablation study

shopt -s dotglob

datasets=(
    "/code/dataset/deepdrr/dataset6_CLINIC_0003_data"
    "/code/dataset/deepdrr/dataset6_CLINIC_0002_data"
)

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Train
        python train.py -s "../dataset/${clean_dir}" \
            --model_path "../output/ablation/init-even/${clean_dir}" \
            --eval

        # Render at final iteration
        python render.py -m "../output/ablation/init-even/${clean_dir}" \
            --skip_train \
            --iteration 30000

        # Compute metrics
        python metrics.py -m "../output/ablation/init-even/${clean_dir}"
    fi
done
