#!/bin/bash

# Benchmark different modes on the scan_12_200views NeRF dataset
#
# Dataset: /code/workspace/ndsplat/output/12/finetune/nerf_dataset/scan_12_200views
#   - 190 train views, 10 test views, 10 val views (test == val)
#
# Modes mirror dgs_medical_pbr.sh:
# | Mode                 | Output Dir                                        | Description                            |
# |----------------------|---------------------------------------------------|----------------------------------------|
# | opacity_only         | output/standard/opacity_only/scan_12_200views     | Opacity conditioning only (no position)|
# | opacity_pos          | output/standard/opacity_pos/scan_12_200views      | Opacity + Position conditioning        |
# | opacity_pos_decouple | output/standard/opacity_pos_decouple/...          | Decoupled position + opacity (lambda=0)|
# | dgs                  | output/standard/dgs/scan_12_200views              | Full DGS                               |
# | ndgs                 | output/standard/ndgs/scan_12_200views             | N-DGS with full Cholesky precision     |
# | ubs                  | output/standard/ubs/scan_12_200views              | Unbounded Splatting baseline           |
# | 3dgs                 | output/standard/3dgs/scan_12_200views             | Standard 3DGS baseline                 |

shopt -s dotglob

scene_dir="/code/workspace/ndsplat/output/12/finetune/nerf_dataset/scan_12_200views"
scene_name="scan_12_200views"

run_experiment() {
    local mode=$1
    local output_dir=$2
    local extra_args=$3

    # Skip if results already exist
    if [ -f "$output_dir/results.json" ]; then
        echo "Skipping ${output_dir} (results.json exists)"
        return
    fi

    # Train
    python train.py -s "$scene_dir" \
        --model_path "$output_dir" \
        --mode "$mode" \
        $extra_args \
        --eval \
        --disable_viewer

    # Render at multiple iterations (including best)
    for iter in 7000 30000 best; do
        python render.py -m "$output_dir" \
            --skip_train \
            --iteration ${iter} \
            $extra_args
    done

    # Compute metrics
    python metrics.py -m "$output_dir"
}

# ============================================
# 1. 3DGS mode (standard 3DGS baseline)
# ============================================
echo "=============================================="
echo "Running 3DGS mode on ${scene_name}"
echo "=============================================="
run_experiment "3dgs" "output/standard/3dgs/${scene_name}" ""

# ============================================
# 2. opacity_only mode (no position shift)
# ============================================
echo "=============================================="
echo "Running opacity_only mode on ${scene_name}"
echo "=============================================="
run_experiment "dgs" "output/standard/opacity_only/${scene_name}" "--use_view_dependent_pos False"

# ============================================
# 3. opacity_pos mode (opacity + position)
# ============================================
echo "=============================================="
echo "Running opacity_pos mode on ${scene_name}"
echo "=============================================="
run_experiment "dgs" "output/standard/opacity_pos/${scene_name}" "--use_view_dependent_pos True"

# ============================================
# 4. dgs mode (opacity + position, alias of opacity_pos)
# ============================================
echo "=============================================="
echo "Running dgs mode on ${scene_name}"
echo "=============================================="
run_experiment "dgs" "output/standard/dgs/${scene_name}" "--use_view_dependent_pos True"

# ============================================
# 5. opacity_pos_decouple mode (decoupled lambda=0)
# ============================================
echo "=============================================="
echo "Running opacity_pos_decouple mode on ${scene_name}"
echo "=============================================="
run_experiment "dgs" "output/standard/opacity_pos_decouple/${scene_name}" "--use_view_dependent_pos True --use_opacity_pos_decouple True"

# ============================================
# 6. NDGS mode (full Cholesky precision)
# ============================================
echo "=============================================="
echo "Running NDGS mode on ${scene_name}"
echo "=============================================="
run_experiment "ndgs" "output/standard/ndgs/${scene_name}" ""

# ============================================
# 7. UBS mode (unbounded splatting baseline)
# ============================================
echo "=============================================="
echo "Running UBS mode on ${scene_name}"
echo "=============================================="
run_experiment "ubs" "output/standard/ubs/${scene_name}" ""

echo "Benchmark completed!"
