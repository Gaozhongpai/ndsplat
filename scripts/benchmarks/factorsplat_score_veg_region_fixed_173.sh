#!/usr/bin/env bash
# Score completed region-constrained VEG renders without rerendering them.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
WORKTREE="${FACTORSPLAT_WORKTREE:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun}"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data/nerf_dataset}"
BASE_ROOT="${FACTORSPLAT_BASE_ROOT:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/clinical_ood_baselines}"
PYDEPS="${FACTORSPLAT_PYDEPS:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100}"
TORCH_HOME_DIR="${FACTORSPLAT_TORCH_HOME:-/mnt/uNeon/zhongpai/vengine_data/cache/torch}"

score_scene() {
    local scene="$1" gpu="$2" family="veg_region_fixed"
    [[ "$scene" == "intestine" ]] && family="veg_region_fixed_capped"
    local model="$BASE_ROOT/$family/$scene/clinical"
    local dataset="$DATA_ROOT/${scene}_factorsplat_clinical"
    local renders="$model/test/ours_30000/renders"
    local train_name="fs_veg_fixed_${scene}"
    local score_name="fs_veg_fixed_${scene}_score"

    while [[ ! -d "$renders" ]] || \
          [[ "$(find "$renders" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" -ne 1640 ]] || \
          docker ps --format '{{.Names}}' | grep -qx "$train_name"; do
        sleep 30
    done
    [[ -f "$model/factorsplat_delta_metrics.json" ]] && return 0

    docker run --rm --gpus "device=$gpu" --network none --name "$score_name" \
        -v "$WORKTREE:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="$PYDEPS:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            set -euo pipefail
            python metrics.py -m '$model'
            python scripts/benchmarks/factorsplat_group_metrics.py \
                --model '$model' --dataset '$dataset' --iteration 30000
            python scripts/benchmarks/factorsplat_delta_metrics.py \
                --model '$model' --dataset '$dataset' --iteration 30000
        "
}

mkdir -p "$BASE_ROOT/veg_region_fixed/score_logs"
jobs=(heart:0 vascular:1 intestine:3 lower:4 kneejoint:5 nose:6 hand:7)
pids=()
for job in "${jobs[@]}"; do
    scene="${job%%:*}"
    gpu="${job##*:}"
    score_scene "$scene" "$gpu" \
        >"$BASE_ROOT/veg_region_fixed/score_logs/${scene}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
