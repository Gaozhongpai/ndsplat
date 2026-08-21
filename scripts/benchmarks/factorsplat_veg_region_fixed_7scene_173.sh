#!/usr/bin/env bash
# Retrain the seven-scene adapted-VEG reference with a fixed categorical
# region and a learnable bounded within-region intensity coordinate.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
WORKTREE="${FACTORSPLAT_WORKTREE:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun}"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data/nerf_dataset}"
OUT_ROOT="${FACTORSPLAT_OUT_ROOT:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/clinical_ood_baselines/veg_region_fixed}"
PYDEPS="${FACTORSPLAT_PYDEPS:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100}"
TORCH_HOME_DIR="${FACTORSPLAT_TORCH_HOME:-/mnt/uNeon/zhongpai/vengine_data/cache/torch}"
ITERS="${FACTORSPLAT_ITERS:-30000}"

mkdir -p "$OUT_ROOT/logs"

run_scene() {
    local scene="$1"
    local gpu="$2"
    local source="$DATA_ROOT/${scene}_factorsplat_clinical"
    local output="$OUT_ROOT/$scene/clinical"
    local name="fs_veg_fixed_${scene}"
    local cap_flags=""
    if [[ "$scene" == "intestine" ]]; then
        cap_flags="--tf_veg_max_gaussians 450000"
    fi

    if [[ ! -f "$source/transforms_train.json" || ! -f "$source/tf_bank.npz" || \
          ! -f "$source/points3d_lookup.npz" || ! -f "$source/points3d.ply" ]]; then
        echo "incomplete dataset: $source" >&2
        return 2
    fi
    if [[ -f "$output/factorsplat_delta_metrics.json" ]]; then
        echo "skip completed: $scene"
        return 0
    fi

    mkdir -p "$output"
    docker run --rm --gpus "device=$gpu" --network none --name "$name" \
        -v "$WORKTREE:/workspace:ro" \
        -v /mnt/uNeon:/mnt/uNeon \
        -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="$PYDEPS:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            set -euo pipefail
            python scripts/benchmarks/factorsplat_check_dataset.py '$source'
            python train.py -s '$source' --model_path '$output' -r 1 \
                --seed 0 --sh_degree 1 --mode factorsplat \
                --use_view_dependent_pos False --l_22_inv_init_scale 2.0 \
                --mip3dgs --tf_rank 8 \
                --tf_condition_color False --tf_condition_opacity False \
                --tf_veg_packed True --tf_veg_v_lr 0.01 \
                $cap_flags \
                --tf_exact_visibility_gate True --tf_aware_prune True \
                --iterations '$ITERS' --eval --disable_viewer \
                --start_checkpoint '$source/points3d.ply' \
                --use_jpeg_compression False --skip_test_camera_loading
            python render.py -m '$output' --skip_train --iteration '$ITERS'
            python metrics.py -m '$output'
            python scripts/benchmarks/factorsplat_group_metrics.py \
                --model '$output' --dataset '$source' --iteration '$ITERS'
            python scripts/benchmarks/factorsplat_delta_metrics.py \
                --model '$output' --dataset '$source' --iteration '$ITERS'
        "
}

# GPU 2 is intentionally excluded on host 173.
DEFAULT_JOBS="heart:0 vascular:1 intestine:3 lower:4 kneejoint:5 nose:6 hand:7"
read -r -a jobs <<< "${FACTORSPLAT_SCENE_JOBS:-$DEFAULT_JOBS}"

pids=()
for job in "${jobs[@]}"; do
    scene="${job%%:*}"
    gpu="${job##*:}"
    run_scene "$scene" "$gpu" >"$OUT_ROOT/logs/${scene}.log" 2>&1 &
    pids+=("$!")
    echo "launched $scene on GPU $gpu (pid ${pids[-1]})"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
