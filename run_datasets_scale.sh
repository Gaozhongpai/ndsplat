# Enable globbing to include hidden directories
shopt -s dotglob

# Define an array of base directories
base_dirs=("/code/dataset/ljubljana/")

# Loop through each base directory
for base_dir in "${base_dirs[@]}"; do
    # Print the base directory for clarity
    echo "Listing subdirectories in: $base_dir"
    
    # Loop through all items in the specified directory and check if they are directories
    for dir in "$base_dir"*/ ; do
        # Check if it is actually a directory to handle no match cases
        if [[ "$dir" == *"-scaled/" ]]; then
            # Removes the trailing slash for cleaner output and removes prefix "/code/dataset"
            clean_dir="${dir%/}"
            clean_dir="${clean_dir#/code/dataset/}"  # Remove the prefix
            echo "$clean_dir"
            # echo "../dataset/${clean_dir}"
            # echo "output/${clean_dir}"
            # Train the model for the current item
            python train.py -s "../dataset/${clean_dir}" --model_path "output/${clean_dir}" --eval
            python render.py -m "output/${clean_dir}" --skip_train
            # python metrics.py -m "output_fast/${clean_dir}"
        fi
    done
    
    # Add a newline for better readability between different base directories
    echo
done