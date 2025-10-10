#!/bin/bash

# Train and render on CT datasets
datasets=("chest" "foot" "abdomen" "jaw")

for dataset in "${datasets[@]}"; do
    echo "Processing ${dataset}..."

    # Train the model
    python train.py -s "../dataset/ct_data/${dataset}_50" --model_path "output/${dataset}_50"

    # Render results
    python render.py -m "output/${dataset}_50" --skip_train
done
