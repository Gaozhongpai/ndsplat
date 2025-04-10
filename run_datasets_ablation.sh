# Enable globbing to include hidden directories
shopt -s dotglob

# Define an array of base directories
# base_dirs=("/code/dataset/deepfluoro/" "/code/dataset/ljubljana/" "/code/dataset/deepdrr/" "/code/dataset/naf_ct_data/")
# base_dirs=("/code/dataset/deepdrr/" "/code/dataset/naf_ct_data/")

# Define the base directories as an array
base_dirs=("/code/dataset/deepdrr/dataset6_CLINIC_0003_data" "/code/dataset/deepdrr/dataset6_CLINIC_0002_data")

# Loop through all items in the specified directory and check if they are directories
for dir in "${base_dirs[@]}" ; do
    # Check if it is actually a directory to handle no match cases
    if [ -d "$dir" ]; then
        # Removes the trailing slash for cleaner output and removes prefix "/code/dataset"
        clean_dir="${dir%/}"
        clean_dir="${clean_dir#/code/dataset/}"  # Remove the prefix
        echo "$clean_dir"
        # echo "../dataset/${clean_dir}"
        # echo "output/${clean_dir}"
        # Train the model for the current item
        python train.py -s "../dataset/${clean_dir}" --model_path "../output/ablation/init-even/${clean_dir}" --eval
        # python render.py -m "../output/ablation/feature8/${clean_dir}" --skip_train --iteration 500
        # python render.py -m "../output/ablation/feature8/${clean_dir}" --skip_train --iteration 2000
        # python render.py -m "../output/ablation/init-mc/${clean_dir}" --skip_train --iteration 7000
        # python render.py -m "../output/ablation/feature8/${clean_dir}" --skip_train --iteration 15000
        python render.py -m "../output/ablation/init-even/${clean_dir}" --skip_train --iteration 30000
        python metrics.py -m "../output/ablation/init-even/${clean_dir}" # Compute error metrics on renderings 
    fi
done

    