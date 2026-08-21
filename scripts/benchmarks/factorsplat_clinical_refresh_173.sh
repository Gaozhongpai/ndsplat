#!/usr/bin/env bash
# Train and evaluate the position-refresh diagnostic. The fixed inherited
# descriptors are the canonical material-association protocol.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
SNAPSHOT="${FACTORSPLAT_SNAPSHOT:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/_run_snapshots/factorsplat_clinical_20260820_2110}"
REPO="$SNAPSHOT/ndsplat"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data}"
OUTPUT_ROOT="$DATA_ROOT/output/factorsplat/clinical_ood"
OLD_ROOT="$OUTPUT_ROOT/hybrid_dc_local/rank8"
NEW_ROOT="$OUTPUT_ROOT/hybrid_dc_local_refresh/rank8"
LOG_ROOT="$OUTPUT_ROOT/refresh_logs"
TORCH_HOME_DIR="$DATA_ROOT/cache/torch"
SCENES=(heart vascular lower kneejoint nose intestine hand)

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

wait_for_no_refresh_study() {
    local scene ready output
    while true; do
        ready=1
        for scene in "${SCENES[@]}"; do
            output="$OLD_ROOT/$scene/clinical"
            clinical_metrics_complete "$output" || ready=0
        done
        if [ "$ready" = 1 ] && ! docker ps --format '{{.Names}}' | grep -q '^factorsplat_clinical_'; then
            return
        fi
        note "waiting for no-refresh clinical evaluations to finish"
        sleep 30
    done
}

run_scene() {
    local gpu="$1" scene="$2"
    local source="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    local output="$NEW_ROOT/$scene/clinical"
    local final="$output/point_cloud/iteration_30000/point_cloud.ply"
    local name="factorsplat_clinical_refresh_${scene}_g${gpu}"

    if clinical_metrics_complete "$output"; then
        note "$scene canonical refresh run already complete"
        return
    fi
    [ -f "$source/points3d_refresh_grid.npz" ] || {
        note "missing refresh grid: $source/points3d_refresh_grid.npz"
        return 1
    }
    if [ ! -f "$final" ]; then
        if [ -d "$output" ] && [ -n "$(find "$output" -mindepth 1 -print -quit)" ]; then
            note "refusing to overwrite incomplete output: $output"
            return 1
        fi
        note "GPU $gpu training refresh-enabled canonical model for $scene"
        docker run --rm --gpus "device=$gpu" --network none --name "$name" \
            -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
            -e TORCH_HOME="$TORCH_HOME_DIR" \
            -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
            "$IMAGE" python train.py \
                -s "$source" --model_path "$output" --seed 0 --sh_degree 1 \
                --mode factorsplat --use_view_dependent_pos False \
                --l_22_inv_init_scale 2.0 --mip3dgs --tf_rank 8 \
                --tf_condition_color True --tf_condition_opacity True \
                --tf_color_sh_degree 0 --tf_use_lookup True --tf_encoder_local True \
                --tf_refresh_descriptors True \
                --tf_opacity_alpha_identity_gate True \
                --tf_exact_visibility_gate True --tf_gate_removed_mass 0.5 \
                --iterations 30000 --eval --disable_viewer \
                --start_checkpoint "$source/points3d.ply" \
                --use_jpeg_compression False --skip_test_camera_loading
    fi
    note "GPU $gpu evaluating refresh-enabled canonical model for $scene"
    docker run --rm --gpus "device=$gpu" --network none --name "${name}_eval" \
        -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            python scripts/benchmarks/factorsplat_check_dataset.py '$source' &&
            python render.py -m '$output' -s '$source' --skip_train --skip_fps \
                --iteration 30000 --use_jpeg_compression False &&
            python metrics.py -m '$output' &&
            python scripts/benchmarks/factorsplat_group_metrics.py \
                --model '$output' --dataset '$source' --iteration 30000 &&
            python scripts/benchmarks/factorsplat_delta_metrics.py \
                --model '$output' --dataset '$source' --iteration 30000
        "
    clinical_metrics_complete "$output"
    note "GPU $gpu completed canonical refresh run for $scene"
}

launch_all() {
    local p0 p1 p3 p4 p5 p6 p7
    wait_for_no_refresh_study
    note "launching seven canonical refresh runs"
    run_scene 0 heart >"$LOG_ROOT/gpu0_heart.log" 2>&1 & p0=$!
    run_scene 1 vascular >"$LOG_ROOT/gpu1_vascular.log" 2>&1 & p1=$!
    run_scene 3 lower >"$LOG_ROOT/gpu3_lower.log" 2>&1 & p3=$!
    run_scene 4 kneejoint >"$LOG_ROOT/gpu4_kneejoint.log" 2>&1 & p4=$!
    run_scene 5 nose >"$LOG_ROOT/gpu5_nose.log" 2>&1 & p5=$!
    run_scene 6 intestine >"$LOG_ROOT/gpu6_intestine.log" 2>&1 & p6=$!
    run_scene 7 hand >"$LOG_ROOT/gpu7_hand.log" 2>&1 & p7=$!
    note "refresh pids: $p0 $p1 $p3 $p4 $p5 $p6 $p7"
    wait "$p0" "$p1" "$p3" "$p4" "$p5" "$p6" "$p7"
    note "seven canonical refresh runs complete"
}

case "${1:-wait-and-run}" in
    wait-and-run) launch_all ;;
    scene) run_scene "${2:?GPU required}" "${3:?scene required}" ;;
    status)
        for log in "$LOG_ROOT"/*.log; do
            [ -f "$log" ] || continue
            echo "=== $(basename "$log") ==="
            tail -n 8 "$log"
        done
        ;;
    *) echo "usage: $0 [wait-and-run|scene GPU SCENE|status]" >&2; exit 2 ;;
esac
