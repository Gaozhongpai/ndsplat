#!/usr/bin/env bash
# Train the seven clinical-bank models immediately on 173 GPUs 4--7, then
# evaluate after the asynchronously rendered 41-TF test manifests are ready.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
VENGINE_IMAGE="${VENGINE_IMAGE:-10.10.0.192:5555/zhongpai/vengine-runtime:latest}"
SNAPSHOT="${FACTORSPLAT_SNAPSHOT:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/_run_snapshots/factorsplat_clinical_20260820_2110}"
REPO="$SNAPSHOT/ndsplat"
VENGINE_REPO="$SNAPSHOT/vengine-runtime"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data}"
OUTPUT_ROOT="$DATA_ROOT/output/factorsplat/clinical_ood"
LOG_ROOT="$OUTPUT_ROOT/launcher_logs_now"
TORCH_HOME_DIR="$DATA_ROOT/cache/torch"

mkdir -p "$LOG_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

train_scene() {
    local gpu="$1" scene="$2"
    local source="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    local output="$OUTPUT_ROOT/hybrid_dc_local/rank8/$scene/clinical"
    local final="$output/point_cloud/iteration_30000/point_cloud.ply"
    local name="factorsplat_clinical_train_${scene}_g${gpu}"

    if [ -f "$final" ]; then
        note "$scene training already complete; keeping checkpoint"
        return
    fi
    if [ -d "$output" ] && [ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        note "refusing to overwrite incomplete output: $output"
        return 1
    fi
    note "GPU $gpu starting 30k training for $scene"
    docker run --rm --gpus "device=$gpu" --network none \
        --name "$name" \
        -v "$REPO:/workspace:ro" \
        -v /mnt/uNeon:/mnt/uNeon \
        -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" python train.py \
        -s "$source" --model_path "$output" --seed 0 --sh_degree 1 \
        --mode factorsplat --use_view_dependent_pos False \
        --l_22_inv_init_scale 2.0 --mip3dgs --tf_rank 8 \
        --tf_condition_color True --tf_condition_opacity True \
        --tf_color_sh_degree 0 --tf_use_lookup True --tf_encoder_local True \
        --tf_opacity_alpha_identity_gate True \
        --tf_exact_visibility_gate True --tf_gate_removed_mass 0.5 \
        --iterations 30000 --eval --disable_viewer \
        --start_checkpoint "$source/points3d.ply" \
        --use_jpeg_compression False --skip_test_camera_loading
    note "GPU $gpu completed training for $scene"
}

test_ready() {
    local scene="$1"
    local source="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    python3 - "$source" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "transforms_test.json"
if not path.is_file():
    raise SystemExit(1)
frames = json.loads(path.read_text()).get("frames", [])
if len(frames) != 41 * 40:
    raise SystemExit(1)
if any(not (root / (f["file_path"][2:] + ".png")).is_file() for f in frames):
    raise SystemExit(1)
PY
}

evaluate_scene() {
    local gpu="$1" scene="$2"
    local source="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    local output="$OUTPUT_ROOT/hybrid_dc_local/rank8/$scene/clinical"
    local name="factorsplat_clinical_eval_${scene}_g${gpu}"

    while ! test_ready "$scene"; do
        note "$scene training done; waiting for 41-TF clinical test set"
        sleep 30
    done
    note "GPU $gpu starting clinical evaluation for $scene"
    docker run --rm --gpus "device=$gpu" --network none \
        --name "$name" \
        -v "$REPO:/workspace:ro" \
        -v /mnt/uNeon:/mnt/uNeon \
        -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            python scripts/benchmarks/factorsplat_check_dataset.py '$source' &&
            python render.py -m '$output' -s '$source' --skip_train --iteration 30000 &&
            python metrics.py -m '$output' &&
            python scripts/benchmarks/factorsplat_group_metrics.py \\
                --model '$output' --dataset '$source' --iteration 30000 &&
            python scripts/benchmarks/factorsplat_delta_metrics.py \\
                --model '$output' --dataset '$source' --iteration 30000
        "
    note "GPU $gpu completed clinical evaluation for $scene"
}

scene_queue() {
    local gpu="$1"
    shift
    local scene
    for scene in "$@"; do
        train_scene "$gpu" "$scene"
    done
    for scene in "$@"; do
        evaluate_scene "$gpu" "$scene"
    done
}

evaluation_queue() {
    local gpu="$1"
    shift
    local scene
    for scene in "$@"; do
        evaluate_scene "$gpu" "$scene"
    done
}

evaluation_queue_after_render() {
    local gpu="$1" prerequisite="$2" status
    shift 2
    note "GPU $gpu waiting for $prerequisite before evaluation"
    status="$(docker wait "$prerequisite")"
    [ "$status" = 0 ] || {
        note "$prerequisite failed with exit code $status"
        return 1
    }
    evaluation_queue "$gpu" "$@"
}

render_tail() {
    local status
    if docker inspect factorsplat_clinical_gpu3_intestine_hand >/dev/null 2>&1; then
        note "remaining intestine/hand render container already exists"
        return
    fi
    status="$(docker inspect factorsplat_clinical_gpu3 --format '{{.State.Status}}')"
    if [ "$status" = running ]; then
        note "waiting for lower render before intestine and hand"
        status="$(docker wait factorsplat_clinical_gpu3)"
    else
        status="$(docker inspect factorsplat_clinical_gpu3 --format '{{.State.ExitCode}}')"
    fi
    [ "$status" = 0 ] || { note "lower render failed: $status"; return 1; }
    docker run --rm --gpus 'device=3' --network none \
        --name factorsplat_clinical_gpu3_intestine_hand \
        -v "$VENGINE_REPO:/repo:ro" \
        -v "$DATA_ROOT:/home/vengine/app/external_data" \
        --entrypoint bash "$VENGINE_IMAGE" -lc \
        'for scene in intestine hand; do bash /repo/factorsplat_render_clinical_ood.sh "$scene"; done'
}

case "${1:-start}" in
    start)
        note "starting immediate training on GPUs 4, 5, 6, and 7"
        render_tail >"$LOG_ROOT/render_tail.log" 2>&1 & pr=$!
        scene_queue 4 heart hand >"$LOG_ROOT/gpu4.log" 2>&1 & p4=$!
        scene_queue 5 vascular intestine >"$LOG_ROOT/gpu5.log" 2>&1 & p5=$!
        scene_queue 6 lower kneejoint >"$LOG_ROOT/gpu6.log" 2>&1 & p6=$!
        scene_queue 7 nose >"$LOG_ROOT/gpu7.log" 2>&1 & p7=$!
        note "pids: render=$pr GPU4=$p4 GPU5=$p5 GPU6=$p6 GPU7=$p7"
        wait "$pr" "$p4" "$p5" "$p6" "$p7"
        note "seven-scene clinical study complete"
        ;;
    status)
        for name in render_tail gpu4 gpu5 gpu6 gpu7; do
            echo "=== $name ==="
            tail -n 12 "$LOG_ROOT/$name.log" 2>/dev/null || echo "no log yet"
        done
        ;;
    evaluate-only)
        note "queuing evaluation on GPUs 4, 5, 6, and 7"
        evaluation_queue_after_render 4 factorsplat_clinical_kneejoint_g4 \
            heart hand >"$LOG_ROOT/eval_gpu4.log" 2>&1 & p4=$!
        evaluation_queue_after_render 5 factorsplat_clinical_nose_g5 \
            vascular intestine >"$LOG_ROOT/eval_gpu5.log" 2>&1 & p5=$!
        evaluation_queue_after_render 6 factorsplat_clinical_intestine_g6 \
            lower kneejoint >"$LOG_ROOT/eval_gpu6.log" 2>&1 & p6=$!
        evaluation_queue_after_render 7 factorsplat_clinical_hand_g7 \
            nose >"$LOG_ROOT/eval_gpu7.log" 2>&1 & p7=$!
        note "evaluation pids: GPU4=$p4 GPU5=$p5 GPU6=$p6 GPU7=$p7"
        wait "$p4" "$p5" "$p6" "$p7"
        note "seven-scene clinical evaluation complete"
        ;;
    evaluate-scene)
        gpu="${2:?GPU index required}"
        scene="${3:?scene required}"
        prerequisite="${4:-}"
        if [ -n "$prerequisite" ]; then
            evaluation_queue_after_render "$gpu" "$prerequisite" "$scene"
        else
            evaluate_scene "$gpu" "$scene"
        fi
        ;;
    *)
        echo "usage: $0 [start|status|evaluate-only|evaluate-scene GPU SCENE [RENDER_CONTAINER]]" >&2
        exit 2
        ;;
esac
