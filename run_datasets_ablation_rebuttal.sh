# Enable globbing to include hidden directories
shopt -s dotglob

# Define an array of base directories
# base_dirs=("/code/dataset/deepfluoro/" "/code/dataset/ljubljana/" "/code/dataset/deepdrr/" "/code/dataset/naf_ct_data/")
base_dirs=("/code/dataset/deepdrr/") # "/code/dataset/naf_ct_data/"
# base_dirs=("/code/dataset/deepfluoro/" "/code/dataset/ljubljana/")

# Loop through each base directory
for base_dir in "${base_dirs[@]}"; do
    # Print the base directory for clarity
    echo "Listing subdirectories in: $base_dir"
    
    # Loop through all items in the specified directory and check if they are directories
    for dir in "$base_dir"*/ ; do
        # Check if it is actually a directory to handle no match cases
        if [ -d "$dir" ]; then
            # Removes the trailing slash for cleaner output and removes prefix "/code/dataset"
            clean_dir="${dir%/}"
            clean_dir="${clean_dir#/code/dataset/}"  # Remove the prefix
            echo "$clean_dir"
            # echo "../dataset/${clean_dir}"
            # echo "output/${clean_dir}"
            # Train the model for the current item
            python train.py -s "../dataset/${clean_dir}" --model_path "../output/ablation2/ddgs_entangled/${clean_dir}" --eval
            python render.py -m "../output/ablation2/ddgs_entangled/${clean_dir}" --iteration 500 --skip_train
            python render.py -m "../output/ablation2/ddgs_entangled/${clean_dir}" --iteration 2000 --skip_train
            python render.py -m "../output/ablation2/ddgs_entangled/${clean_dir}" --iteration 7000 --skip_train
            python render.py -m "../output/ablation2/ddgs_entangled/${clean_dir}" --iteration 15000 --skip_train
            python render.py -m "../output/ablation2/ddgs_entangled/${clean_dir}" --iteration 30000 --skip_train
            python metrics.py -m "../output/ablation2/ddgs_entangled/${clean_dir}" # Compute error metrics on renderings 
        fi
    done
    
    # Add a newline for better readability between different base directories
    echo
done