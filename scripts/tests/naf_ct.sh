#!/bin/bash

# Train and render on NAF CT datasets (test/debug)

shopt -s dotglob

datasets=(
    "/code/dataset/naf_ct_data/abdomen_50"
    "/code/dataset/naf_ct_data/chest_50"
    "/code/dataset/naf_ct_data/foot_50"
    "/code/dataset/naf_ct_data/jaw_50"
)

for dir in "${datasets[@]}"; do
    if [ -d "$dir" ]; then
        # Extract clean directory name
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"

        echo "Processing ${clean_dir}..."

        # Note: Training is commented out - only rendering/metrics
        # python train.py -s "../dataset/${clean_dir}" \
        #     --model_path "../output/test/reduceopacity/${clean_dir}" \
        #     --eval

        # Render at final iteration
        python render.py -m "../output/orignal-separate/${clean_dir}" \
            --skip_train \
            --iteration 30000

        # Compute metrics
        python metrics.py -m "../output/orignal-separate/${clean_dir}"
    fi
done
