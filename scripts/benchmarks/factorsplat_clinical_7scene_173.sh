#!/usr/bin/env bash
# Host-side 173 launcher for the seven-scene clinical-OOD FactorSplat study.
#
# GPU 2 is intentionally never referenced.  The three jobs wait for their
# corresponding vengine render/conversion queues, attach the existing packed
# lookup descriptors to the new dataset, and then train scenes sequentially.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
VENGINE_IMAGE="${VENGINE_IMAGE:-10.10.0.192:5555/zhongpai/vengine-runtime:latest}"
WORKSPACE="${FACTORSPLAT_WORKSPACE:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace}"
REPO="${FACTORSPLAT_REPO:-$WORKSPACE/ndsplat-factorsplat-rerun}"
VENGINE_REPO="${VENGINE_REPO_PATH:-$WORKSPACE/trueview/vengine-runtime}"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data}"
OUTPUT_ROOT="$DATA_ROOT/output/factorsplat/clinical_ood"
LOG_ROOT="$OUTPUT_ROOT/launcher_logs"
TORCH_HOME_DIR="$DATA_ROOT/cache/torch"

mkdir -p "$LOG_ROOT" "$TORCH_HOME_DIR/hub/checkpoints"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

wait_container_success() {
    local name="$1" status
    if ! docker inspect "$name" >/dev/null 2>&1; then
        note "missing prerequisite container: $name"
        return 1
    fi
    status="$(docker inspect "$name" --format '{{.State.Status}}')"
    if [ "$status" = running ]; then
        note "waiting for $name"
        status="$(docker wait "$name")"
    else
        status="$(docker inspect "$name" --format '{{.State.ExitCode}}')"
    fi
    if [ "$status" != 0 ]; then
        note "$name failed with exit code $status"
        return 1
    fi
    note "$name completed"
}

prepare_scene() {
    local scene="$1"
    local dataset="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    local descriptor="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_full/points3d_lookup.npz"

    python3 - "$dataset" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"train": 24 * 48, "test": 41 * 40}
for split, count in expected.items():
    path = root / f"transforms_{split}.json"
    if not path.is_file():
        raise SystemExit(f"missing completed dataset marker: {path}")
    actual = len(json.loads(path.read_text())["frames"])
    if actual != count:
        raise SystemExit(f"{path}: expected {count} frames, found {actual}")
if not (root / "points3d.ply").exists():
    raise SystemExit(f"missing initialization: {root / 'points3d.ply'}")
PY

    if [ ! -f "$descriptor" ]; then
        note "missing packed descriptor for $scene: $descriptor"
        return 1
    fi
    if [ ! -e "$dataset/points3d_lookup.npz" ]; then
        ln -s "../${scene}_factorsplat_full/points3d_lookup.npz" \
            "$dataset/points3d_lookup.npz"
    fi
    note "$scene dataset verified"
}

train_scene() {
    local gpu="$1" scene="$2"
    local name="factorsplat_clinical_train_${scene}_g${gpu}"
    note "GPU $gpu starting $scene"
    docker run --rm --gpus "device=$gpu" --network none \
        --name "$name" \
        -v "$REPO:/workspace" \
        -v /mnt/uNeon:/mnt/uNeon \
        -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e FACTORSPLAT_DATA="$DATA_ROOT/nerf_dataset" \
        -e FACTORSPLAT_OUT="$OUTPUT_ROOT" \
        -e FACTORSPLAT_SCENES="$scene" \
        -e FACTORSPLAT_PRESET=clinical \
        -e FACTORSPLAT_VARIANT=hybrid_dc \
        -e FACTORSPLAT_ENCODER=local \
        -e FACTORSPLAT_RANK=8 \
        -e FACTORSPLAT_ITERS=30000 \
        -e 'FACTORSPLAT_EXTRA_FLAGS=--use_jpeg_compression False --skip_test_camera_loading' \
        "$IMAGE" bash scripts/benchmarks/factorsplat_train.sh
    note "GPU $gpu completed $scene"
}

train_queue() {
    local gpu="$1"
    shift
    local scene
    for scene in "$@"; do
        prepare_scene "$scene"
        train_scene "$gpu" "$scene"
    done
}

gpu0_job() {
    wait_container_success factorsplat_clinical_gpu0
    train_queue 0 heart kneejoint
}

gpu1_job() {
    wait_container_success factorsplat_clinical_gpu1
    train_queue 1 vascular nose
}

gpu3_job() {
    wait_container_success factorsplat_clinical_gpu3
    note "GPU 3 rendering remaining intestine and hand clinical references"
    docker run --rm --gpus 'device=3' --network none \
        --name factorsplat_clinical_gpu3_intestine_hand \
        -v "$VENGINE_REPO:/repo:ro" \
        -v "$DATA_ROOT:/home/vengine/app/external_data" \
        --entrypoint bash "$VENGINE_IMAGE" -lc \
        'for scene in intestine hand; do bash /repo/factorsplat_render_clinical_ood.sh "$scene"; done'
    note "GPU 3 completed remaining clinical reference renders"
    train_queue 3 lower intestine hand
}

case "${1:-start}" in
    start)
        note "launching seven-scene queues on allowed GPUs 0, 1, and 3"
        gpu0_job >"$LOG_ROOT/gpu0.log" 2>&1 & p0=$!
        gpu1_job >"$LOG_ROOT/gpu1.log" 2>&1 & p1=$!
        gpu3_job >"$LOG_ROOT/gpu3.log" 2>&1 & p3=$!
        note "queue pids: GPU0=$p0 GPU1=$p1 GPU3=$p3"
        wait "$p0" "$p1" "$p3"
        note "all seven scene queues completed"
        ;;
    status)
        for gpu in 0 1 3; do
            echo "=== GPU $gpu ==="
            tail -n 12 "$LOG_ROOT/gpu${gpu}.log" 2>/dev/null || echo "no log yet"
        done
        ;;
    *)
        echo "usage: $0 [start|status]" >&2
        exit 2
        ;;
esac
