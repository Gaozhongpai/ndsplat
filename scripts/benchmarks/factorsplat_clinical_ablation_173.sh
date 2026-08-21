#!/usr/bin/env bash
# Matched heart/vascular clinical ablations for the fixed-descriptor model. Runs only
# after the seven-scene baseline/specialist comparison has completed.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
SNAPSHOT="${FACTORSPLAT_SNAPSHOT:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/_run_snapshots/factorsplat_clinical_20260820_2110}"
REPO="$SNAPSHOT/ndsplat"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data}"
FS_ROOT="$DATA_ROOT/output/factorsplat"
CANONICAL="$FS_ROOT/clinical_ood/hybrid_dc_local/rank8"
REFRESH="$FS_ROOT/clinical_ood/hybrid_dc_local_refresh/rank8"
OUT_ROOT="$FS_ROOT/clinical_ood_ablation"
BASE_ROOT="$FS_ROOT/clinical_ood_baselines"
SPEC_ROOT="$FS_ROOT/clinical_ood_specialist"
LOG_ROOT="$OUT_ROOT/logs"
TORCH_HOME_DIR="$DATA_ROOT/cache/torch"
OOD_CLINICAL=(test_ood_02_target_isolation test_ood_03_occluder_suppression test_ood_04_target_context)

mkdir -p "$LOG_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

clinical_metrics_complete() {
    local output="$1"
    python3 - "$output" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
need = {"test_ood_00_gray", "test_ood_01_sharp",
        "test_ood_02_target_isolation", "test_ood_03_occluder_suppression",
        "test_ood_04_target_context"}
try:
    grouped = json.loads((root / "factorsplat_grouped_metrics.json").read_text())
    delta = json.loads((root / "factorsplat_delta_metrics.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if not need <= set(delta.get("transfer_functions", {})):
    raise SystemExit(1)
if not {f"tf:{x}" for x in need} <= set(grouped.get("groups", {})):
    raise SystemExit(1)
PY
}

wait_for_comparisons() {
    local scene tf ready
    while true; do
        ready=1
        for scene in heart vascular lower kneejoint nose intestine hand; do
            clinical_metrics_complete "$CANONICAL/$scene/clinical" || ready=0
            clinical_metrics_complete "$BASE_ROOT/mixed/$scene/clinical" || ready=0
            clinical_metrics_complete "$BASE_ROOT/veg/$scene/clinical" || ready=0
            for tf in "${OOD_CLINICAL[@]}"; do
                [ -f "$SPEC_ROOT/$scene/$tf/results.json" ] || ready=0
            done
        done
        if [ "$ready" = 1 ] && ! docker ps --format '{{.Names}}' | grep -q '^factorsplat_clinical_'; then
            return
        fi
        note "waiting for seven-scene clinical comparisons"
        sleep 30
    done
}

arm_flags() {
    case "$1" in
        lookup)
            echo "--tf_condition_color False --tf_condition_opacity False --tf_use_lookup True --tf_encoder_local False"
            ;;
        residual)
            echo "--tf_condition_color True --tf_condition_opacity True --tf_use_lookup False --tf_encoder_local True"
            ;;
        embedding)
            echo "--tf_condition_color True --tf_condition_opacity True --tf_use_lookup True --tf_encoder_local False --tf_encoder_type embedding"
            ;;
        *) return 2 ;;
    esac
}

train_eval_arm() {
    local gpu="$1" scene="$2" arm="$3" source output final name flags
    source="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    output="$OUT_ROOT/$arm/$scene/clinical"
    final="$output/point_cloud/iteration_30000/point_cloud.ply"
    name="factorsplat_clinical_ablation_${arm}_${scene}_g${gpu}"
    flags="$(arm_flags "$arm")"
    if clinical_metrics_complete "$output"; then
        note "$scene $arm already complete"
        return
    fi
    if [ ! -f "$final" ]; then
        if [ -d "$output" ] && [ -n "$(find "$output" -mindepth 1 -print -quit)" ]; then
            note "refusing to overwrite incomplete output: $output"
            return 1
        fi
        note "GPU $gpu training $scene $arm ablation"
        # flags contains only fixed, in-repository argument names and values.
        docker run --rm --gpus "device=$gpu" --network none --name "$name" \
            -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
            -e TORCH_HOME="$TORCH_HOME_DIR" \
            -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
            "$IMAGE" bash -lc "
                python train.py -s '$source' --model_path '$output' --seed 0 --sh_degree 1 \
                    --mode factorsplat --use_view_dependent_pos False \
                    --l_22_inv_init_scale 2.0 --mip3dgs --tf_rank 8 \
                    --tf_color_sh_degree 0 $flags \
                    --tf_refresh_descriptors False \
                    --tf_opacity_alpha_identity_gate True \
                    --tf_exact_visibility_gate True --tf_gate_removed_mass 0.5 \
                    --iterations 30000 --eval --disable_viewer \
                    --start_checkpoint '$source/points3d.ply' \
                    --use_jpeg_compression False --skip_test_camera_loading
            "
    fi
    note "GPU $gpu evaluating $scene $arm ablation"
    docker run --rm --gpus "device=$gpu" --network none --name "${name}_eval" \
        -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            python render.py -m '$output' -s '$source' --skip_train --skip_fps \
                --iteration 30000 --use_jpeg_compression False &&
            python metrics.py -m '$output' &&
            python scripts/benchmarks/factorsplat_group_metrics.py \
                --model '$output' --dataset '$source' --iteration 30000 &&
            python scripts/benchmarks/factorsplat_delta_metrics.py \
                --model '$output' --dataset '$source' --iteration 30000
        "
    clinical_metrics_complete "$output"
    note "GPU $gpu completed $scene $arm ablation"
}

stage_checkpoint() {
    local source="$1" target="$2"
    mkdir -p "$target"
    if [ ! -e "$target/point_cloud" ]; then
        ln -s "$(realpath --relative-to="$target" "$source/point_cloud")" "$target/point_cloud"
    fi
    [ -f "$target/cfg_args" ] || cp "$source/cfg_args" "$target/cfg_args"
}

eval_without_gate() {
    local gpu="$1" scene="$2" source_model target dataset name
    source_model="$CANONICAL/$scene/clinical"
    target="$OUT_ROOT/no_exact_gate/$scene/clinical"
    dataset="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    name="factorsplat_clinical_ablation_nogate_${scene}_g${gpu}"
    if clinical_metrics_complete "$target"; then
        note "$scene no-exact-gate already complete"
        return
    fi
    stage_checkpoint "$source_model" "$target"
    note "GPU $gpu evaluating $scene without exact visibility gate"
    docker run --rm --gpus "device=$gpu" --network none --name "$name" \
        -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            python render.py -m '$target' -s '$dataset' --skip_train --skip_fps \
                --iteration 30000 --use_jpeg_compression False \
                --tf_exact_visibility_gate False &&
            python metrics.py -m '$target' &&
            python scripts/benchmarks/factorsplat_group_metrics.py \
                --model '$target' --dataset '$dataset' --iteration 30000 &&
            python scripts/benchmarks/factorsplat_delta_metrics.py \
                --model '$target' --dataset '$dataset' --iteration 30000
        "
    clinical_metrics_complete "$target"
    note "GPU $gpu completed $scene no-exact-gate evaluation"
}

gate_queue() {
    eval_without_gate 7 heart
    eval_without_gate 7 vascular
}

launch_all() {
    local p0 p1 p3 p4 p5 p6 p7
    wait_for_comparisons
    note "launching matched heart/vascular clinical ablations"
    train_eval_arm 0 heart lookup >"$LOG_ROOT/gpu0_heart_lookup.log" 2>&1 & p0=$!
    train_eval_arm 1 vascular lookup >"$LOG_ROOT/gpu1_vascular_lookup.log" 2>&1 & p1=$!
    train_eval_arm 3 heart residual >"$LOG_ROOT/gpu3_heart_residual.log" 2>&1 & p3=$!
    train_eval_arm 4 vascular residual >"$LOG_ROOT/gpu4_vascular_residual.log" 2>&1 & p4=$!
    train_eval_arm 5 heart embedding >"$LOG_ROOT/gpu5_heart_embedding.log" 2>&1 & p5=$!
    train_eval_arm 6 vascular embedding >"$LOG_ROOT/gpu6_vascular_embedding.log" 2>&1 & p6=$!
    gate_queue >"$LOG_ROOT/gpu7_no_exact_gate.log" 2>&1 & p7=$!
    note "ablation pids: $p0 $p1 $p3 $p4 $p5 $p6 $p7"
    wait "$p0" "$p1" "$p3" "$p4" "$p5" "$p6" "$p7"
    # Refresh-enabled runs are retained as the matched final diagnostic arm.
    clinical_metrics_complete "$REFRESH/heart/clinical"
    clinical_metrics_complete "$REFRESH/vascular/clinical"
    note "matched clinical ablation complete"
}

case "${1:-wait-and-run}" in
    wait-and-run) launch_all ;;
    status)
        for log in "$LOG_ROOT"/*.log; do
            [ -f "$log" ] || continue
            echo "=== $(basename "$log") ==="
            tail -n 8 "$log"
        done
        ;;
    *) echo "usage: $0 [wait-and-run|status]" >&2; exit 2 ;;
esac
