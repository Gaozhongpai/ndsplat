#!/usr/bin/env bash
# Evaluate matched identity/VEG baselines and train the three new clinical-OOD
# per-preset dGS specialists after the seven FactorSplat evaluations finish.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
SNAPSHOT="${FACTORSPLAT_SNAPSHOT:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/_run_snapshots/factorsplat_clinical_20260820_2110}"
REPO="$SNAPSHOT/ndsplat"
TOOLS_REPO="${FACTORSPLAT_TOOLS_REPO:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun}"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data}"
OUTPUT_ROOT="$DATA_ROOT/output/factorsplat"
CLINICAL_ROOT="$OUTPUT_ROOT/clinical_ood"
CANONICAL_ROOT="$CLINICAL_ROOT/hybrid_dc_local/rank8"
BASELINE_ROOT="$OUTPUT_ROOT/clinical_ood_baselines"
SPECIALIST_ROOT="$OUTPUT_ROOT/clinical_ood_specialist"
LOG_ROOT="$CLINICAL_ROOT/comparison_logs"
TORCH_HOME_DIR="$DATA_ROOT/cache/torch"
SCENES=(heart vascular lower kneejoint nose intestine hand)
TF_IDS=(test_ood_02_target_isolation test_ood_03_occluder_suppression test_ood_04_target_context)

mkdir -p "$LOG_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

source_model() {
    local scene="$1" method="$2" root="$OUTPUT_ROOT/table1_5scan"
    case "$scene:$method" in
        heart:mixed) echo "$OUTPUT_ROOT/mixed_unconditioned/heart/full" ;;
        heart:veg) echo "$OUTPUT_ROOT/veg_native/rank8/heart/full" ;;
        vascular:mixed) echo "$OUTPUT_ROOT/mixed_unconditioned/vascular/full" ;;
        vascular:veg) echo "$OUTPUT_ROOT/veg_native/rank8/vascular/full" ;;
        kneejoint:mixed) echo "$root/mixed_unconditioned_visible/kneejoint/full" ;;
        kneejoint:veg) echo "$root/veg_visible/rank8/kneejoint/full" ;;
        intestine:veg) echo "$root/veg_capped/rank8/intestine/full" ;;
        *:mixed) echo "$root/mixed_unconditioned/$scene/full" ;;
        *:veg) echo "$root/veg/rank8/$scene/full" ;;
        *) return 2 ;;
    esac
}

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

evaluate_baseline() {
    local gpu="$1" scene="$2" method="$3"
    local source target dataset name
    source="$(source_model "$scene" "$method")"
    target="$BASELINE_ROOT/$method/$scene/clinical"
    dataset="$DATA_ROOT/nerf_dataset/${scene}_factorsplat_clinical"
    name="factorsplat_clinical_${method}_${scene}_g${gpu}"
    if clinical_metrics_complete "$target"; then
        note "$scene $method clinical metrics already complete"
        return
    fi
    note "GPU $gpu staging $scene $method (38 reused TFs, 3 new TFs)"
    python3 "$TOOLS_REPO/scripts/benchmarks/factorsplat_stage_clinical_baseline.py" \
        --source-model "$source" --clinical-dataset "$dataset" --target-model "$target"
    docker run --rm --gpus "device=$gpu" --network none --name "$name" \
        -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            python render.py -m '$target' -s '$dataset' --skip_train --skip_fps \
                --iteration 30000 --render_tf_indices 38,39,40 \
                --use_jpeg_compression False &&
            python metrics.py -m '$target' &&
            python scripts/benchmarks/factorsplat_group_metrics.py \
                --model '$target' --dataset '$dataset' --iteration 30000 &&
            python scripts/benchmarks/factorsplat_delta_metrics.py \
                --model '$target' --dataset '$dataset' --iteration 30000
        "
    clinical_metrics_complete "$target"
    note "GPU $gpu completed $scene $method clinical evaluation"
}

train_specialist() {
    local gpu="$1" scene="$2" tf_id="$3"
    local source output name
    source="$DATA_ROOT/nerf_dataset/factorsplat_oracle/$scene/clinical/$tf_id"
    output="$SPECIALIST_ROOT/$scene/$tf_id"
    name="factorsplat_clinical_specialist_${scene}_${tf_id}_g${gpu}"
    if [ -f "$output/results.json" ] && [ -f "$output/per_view.json" ]; then
        note "$scene $tf_id specialist already complete"
        return
    fi
    if [ -d "$output" ] && [ -n "$(find "$output" -mindepth 1 -print -quit)" ]; then
        note "refusing to overwrite incomplete specialist: $output"
        return 1
    fi
    [ -f "$source/points3d.ply" ] || { note "missing specialist dataset: $source"; return 1; }
    note "GPU $gpu training $scene $tf_id specialist"
    docker run --rm --gpus "device=$gpu" --network none --name "$name" \
        -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon -w /workspace \
        -e TORCH_HOME="$TORCH_HOME_DIR" \
        -e PYTHONPATH="/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100:/workspace/submodules/gsplat" \
        "$IMAGE" bash -lc "
            python train.py -s '$source' --model_path '$output' --seed 0 --sh_degree 1 \
                --mode dgs --use_view_dependent_pos False --l_22_inv_init_scale 2.0 \
                --mip3dgs --iterations 30000 --eval --disable_viewer \
                --start_checkpoint '$source/points3d.ply' --use_jpeg_compression False &&
            python render.py -m '$output' -s '$source' --skip_train --skip_fps \
                --iteration 30000 --use_jpeg_compression False &&
            python metrics.py -m '$output'
        "
    note "GPU $gpu completed $scene $tf_id specialist"
}

scene_comparisons() {
    local gpu="$1" scene="$2" tf_id
    evaluate_baseline "$gpu" "$scene" mixed
    evaluate_baseline "$gpu" "$scene" veg
    for tf_id in "${TF_IDS[@]}"; do
        train_specialist "$gpu" "$scene" "$tf_id"
    done
    note "GPU $gpu completed all clinical comparisons for $scene"
}

wait_for_factor_study() {
    local scene output ready
    while true; do
        ready=1
        for scene in "${SCENES[@]}"; do
            output="$CANONICAL_ROOT/$scene/clinical"
            clinical_metrics_complete "$output" || ready=0
        done
        if [ "$ready" = 1 ] && ! docker ps --format '{{.Names}}' | grep -q '^factorsplat_clinical_'; then
            return
        fi
        note "waiting for seven FactorSplat clinical evaluations"
        sleep 30
    done
}

launch_all() {
    local p0 p1 p3 p4 p5 p6 p7
    wait_for_factor_study
    note "FactorSplat study complete; launching seven matched comparison queues"
    scene_comparisons 0 heart >"$LOG_ROOT/gpu0_heart.log" 2>&1 & p0=$!
    scene_comparisons 1 vascular >"$LOG_ROOT/gpu1_vascular.log" 2>&1 & p1=$!
    scene_comparisons 3 lower >"$LOG_ROOT/gpu3_lower.log" 2>&1 & p3=$!
    scene_comparisons 4 kneejoint >"$LOG_ROOT/gpu4_kneejoint.log" 2>&1 & p4=$!
    scene_comparisons 5 nose >"$LOG_ROOT/gpu5_nose.log" 2>&1 & p5=$!
    scene_comparisons 6 intestine >"$LOG_ROOT/gpu6_intestine.log" 2>&1 & p6=$!
    scene_comparisons 7 hand >"$LOG_ROOT/gpu7_hand.log" 2>&1 & p7=$!
    note "comparison pids: $p0 $p1 $p3 $p4 $p5 $p6 $p7"
    wait "$p0" "$p1" "$p3" "$p4" "$p5" "$p6" "$p7"
    note "all seven clinical comparison queues complete"
}

case "${1:-wait-and-run}" in
    wait-and-run) launch_all ;;
    scene) scene_comparisons "${2:?GPU required}" "${3:?scene required}" ;;
    specialist) train_specialist "${2:?GPU required}" "${3:?scene required}" "${4:?TF id required}" ;;
    baseline) evaluate_baseline "${2:?GPU required}" "${3:?scene required}" "${4:?method required}" ;;
    status)
        for log in "$LOG_ROOT"/*.log; do
            [ -f "$log" ] || continue
            echo "=== $(basename "$log") ==="
            tail -n 8 "$log"
        done
        ;;
    *) echo "usage: $0 [wait-and-run|scene GPU SCENE|specialist GPU SCENE TF|baseline GPU SCENE METHOD|status]" >&2; exit 2 ;;
esac
