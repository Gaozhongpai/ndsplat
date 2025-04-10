#!/bin/bash

# Array of strings
str=("chest" "foot" "abdomen" "jaw")

# Loop through the array and run the command for each item
for item in "${str[@]}"
do
    # Train the model for the current item
    python train.py -s "../dataset/ct_data/${item}_50" --model_path "output/${item}_50"
    
    # Render results for the current item's model
    python render.py -m "output/${item}_50" --skip_train
done