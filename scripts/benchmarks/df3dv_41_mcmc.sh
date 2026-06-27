#!/bin/bash

# Benchmark dGS / dBS with MCMC densification on the DF3DV-41 benchmark.
#
# Benchmark: https://github.com/johnnylu305/DF3DV/tree/main/DF3DV_Benchmark
#
# Modes:
# | Mode | Output Dir                       | Leaderboard method     | Description              |
# |------|----------------------------------|------------------------|--------------------------|
# | dgs  | output/df3dv41_mcmc/dgs/<scene>  | dgs_mcmc_df3dv41       | Direct Gaussian Splatting|
# | dbs  | output/df3dv41_mcmc/dbs/<scene>  | dbs_mcmc_df3dv41       | Direct Beta Splatting    |
#
# MCMC Parameters:
# - densification_strategy: mcmc
# - mcmc_cap_max: Maximum Gaussians (per-scene table below, default 1M)
# - noise_lr / opacity_reg / scale_reg
#
# Pipeline per scene:
#   - Train on clutter_* (undistortion_images_8); eval on extra_* (clean)
#   - Render side-by-side |GT|Rendering| extra_*.png into <scene>-All/MODELS/<method>/renders/
#
# Data prep (download/unzip/downsample) handled by scripts/df3dv_benchmark/ helpers.
#
# Note: DF3DV scenes are COLMAP (PINHOLE), view-dependent -> --input_dim 6, no -w.

shopt -s dotglob nullglob

DF3DV_ROOT="${DF3DV_ROOT:-/code/dataset/DF3DV-1K}"
base_dir="${DF3DV_ROOT}/DF3DV-41/"
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../df3dv_benchmark" && pwd)"

ITERATIONS="${ITERATIONS:-30000}"

# Test every 500 iters so the BEST checkpoint is chosen from a fine grid
# (overrides the default static-scene schedule of only [500,2k,7k,15k,30k]).
TEST_ITERATIONS="$(seq 500 500 "$ITERATIONS")"

# MCMC parameters
NOISE_LR=5e5
OPACITY_REG=0.01
SCALE_REG=0.01
DEFAULT_CAP_MAX="${MCMC_CAP_MAX:-1000000}"

# Per-scene cap_max overrides (optional). Scenes not listed use $DEFAULT_CAP_MAX.
declare -A SCENE_CAP_MAX
# e.g. SCENE_CAP_MAX["301025-Temple"]=2000000

collect_scenes() {
    SCENES=()
    for scene_all in "${base_dir}"*/*-All; do
        [ -d "$scene_all" ] || continue
        [ -d "$scene_all/undistortion_sparse" ] || continue
        [ -d "$scene_all/undistortion_images_8" ] || continue
        SCENES+=("$(basename "$(dirname "$scene_all")")")
    done
}

# ============================================
# Data preparation (download + unzip + downsample). No-ops if already present.
# ============================================
prepare_data() {
    if [ ! -d "${base_dir}" ] || [ -z "$(ls -A "${base_dir}" 2>/dev/null)" ]; then
        echo "[prep] downloading DF3DV-41 from HuggingFace..."
        command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
        hf download ChengYou305/DF3DV-1K --repo-type dataset \
            --local-dir "$DF3DV_ROOT" --include "DF3DV-41/*"
    fi
    for zip in "${base_dir}"*.zip; do
        [ -e "$zip" ] || continue
        name="$(basename "$zip" .zip)"
        [ -d "${base_dir}${name}" ] && continue
        echo "[prep] unzip $name"
        if command -v unzip >/dev/null 2>&1; then
            unzip -q -o "$zip" -d "${base_dir}"
        else
            python - "$zip" "${base_dir}" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z: z.extractall(sys.argv[2])
PY
        fi
    done
    if ! ls -d "${base_dir}"*/*-All/undistortion_images_8 >/dev/null 2>&1; then
        echo "[prep] downsampling (official mediapy, factor 8)"
        ( cd "$BENCH_DIR" && python downsample_df3dv41.py \
            --root "$DF3DV_ROOT" --factor 8 --num_workers 8 --overwrite )
    fi
}

# ============================================
# Run one (scene, mode) experiment with MCMC densification.
# ============================================
run_experiment() {
    local mode=$1
    local output_dir=$2
    local scene_all=$3
    local method=$4
    local scene_name=$5
    local extra_args=$6

    local renders_dir="${scene_all}/MODELS/${method}/renders"
    if [ -d "$renders_dir" ] && [ -n "$(ls -A "$renders_dir" 2>/dev/null)" ]; then
        echo "  Skipping (renders exist): $renders_dir"
        return
    fi

    # Scene-specific cap_max
    local cap_max=${SCENE_CAP_MAX[$scene_name]}
    if [ -z "$cap_max" ]; then
        cap_max=$DEFAULT_CAP_MAX
    fi
    echo "  Using cap_max: $cap_max"

    if [ -d "$output_dir/point_cloud" ]; then
        echo "  Skipping training (point_cloud exists)"
    else
        python train.py -s "$scene_all" \
            --model_path "$output_dir" \
            --mode "$mode" \
            --input_dim 6 \
            --iterations "$ITERATIONS" \
            --test_iterations $TEST_ITERATIONS \
            --densification_strategy mcmc \
            --mcmc_cap_max $cap_max \
            --noise_lr $NOISE_LR \
            --opacity_reg $OPACITY_REG \
            --scale_reg $SCALE_REG \
            $extra_args \
            --eval \
            --disable_viewer
    fi

    # Render the BEST checkpoint (highest test PSNR during training).
    python render_df3dv.py -m "$output_dir" -s "$scene_all" \
        --mode "$mode" \
        --method "$method" \
        --iteration best \
        $extra_args

    # Incremental scoring: official DF3DV PSNR/SSIM/LPIPS for this scene,
    # appended to <root>/DF3DV-41_<method>_progress.csv (live running mean).
    python "${BENCH_DIR}/finalize_df3dv41.py" --root "$DF3DV_ROOT" \
        --method "$method" --single_scene "$scene_name"
}

# ============================================
# Main
# ============================================
prepare_data
collect_scenes
echo "=============================================="
echo "DF3DV-41 (MCMC): ${#SCENES[@]} scenes found"
echo "=============================================="

# ============================================
# 1. dgs mode with MCMC
# ============================================
echo "=============================================="
echo "Running dgs mode benchmarks (MCMC)"
echo "=============================================="
for scene_name in "${SCENES[@]}"; do
    scene_all="${base_dir}${scene_name}/${scene_name}-All"
    if [ -d "$scene_all" ]; then
        output_dir="output/df3dv41_mcmc/dgs/${scene_name}"
        echo "Processing ${scene_name} with mode dgs (MCMC)..."
        run_experiment "dgs" "$output_dir" "$scene_all" "dgs_mcmc_df3dv41" "$scene_name" ""
    fi
done

# ============================================
# 2. dbs mode with MCMC
# ============================================
echo "=============================================="
echo "Running dbs mode benchmarks (MCMC)"
echo "=============================================="
for scene_name in "${SCENES[@]}"; do
    scene_all="${base_dir}${scene_name}/${scene_name}-All"
    if [ -d "$scene_all" ]; then
        output_dir="output/df3dv41_mcmc/dbs/${scene_name}"
        echo "Processing ${scene_name} with mode dbs (MCMC)..."
        run_experiment "dbs" "$output_dir" "$scene_all" "dbs_mcmc_df3dv41" "$scene_name" ""
    fi
done

echo ""
echo "=============================================="
echo "Scoring + packaging submissions"
echo "=============================================="
# Local PSNR/SSIM/LPIPS + leaderboard zip (writes <root>/submissions/<method>_DF3DV-41.zip).
python "${BENCH_DIR}/finalize_df3dv41.py" --root "$DF3DV_ROOT" --method dgs_mcmc_df3dv41
python "${BENCH_DIR}/finalize_df3dv41.py" --root "$DF3DV_ROOT" --method dbs_mcmc_df3dv41

echo ""
echo "Benchmark completed!"
echo "  Metrics CSVs : ${DF3DV_ROOT}/DF3DV-41_<method>_metrics.csv"
echo "  Upload zips  : ${DF3DV_ROOT}/submissions/<method>_DF3DV-41.zip"
echo "  Submit via the DF3DV leaderboard form (Google Drive link + email)."
