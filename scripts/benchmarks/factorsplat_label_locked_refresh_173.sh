#!/usr/bin/env bash
# Matched 30k label-locked descriptor-refresh test on heart, nose, and hand.
# Host 173 GPU 2 is intentionally excluded.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
REPO="${FACTORSPLAT_REPO:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun}"
DATA="${FACTORSPLAT_DATA:-/mnt/uNeon/zhongpai/vengine_data}"
OUT_ROOT="${FACTORSPLAT_OUT_ROOT:-$DATA/output/factorsplat/clinical_ood/hybrid_dc_local_label_locked_refresh/rank8}"
PYDEPS="${FACTORSPLAT_PYDEPS:-$DATA/output/factorsplat/ood_fix_tests/pydeps_h100}"
CACHE="${FACTORSPLAT_TORCH_HOME:-$DATA/cache/torch}"

common=(--rm --network none -v "$REPO:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon
        -w /workspace -e "TORCH_HOME=$CACHE"
        -e "PYTHONPATH=$PYDEPS:/workspace/submodules/gsplat" "$IMAGE")

train_scene() {
    local scene="$1" gpu="$2"
    local source="$DATA/nerf_dataset/${scene}_factorsplat_clinical"
    local output="$OUT_ROOT/$scene/clinical"
    if [[ -f "$output/point_cloud/iteration_30000/point_cloud.ply" ]]; then
        echo "[$scene] training already complete"
        return
    fi
    docker run --gpus "device=$gpu" --name "fs_locked_train_$scene" \
        "${common[@]}" python train.py -s "$source" --model_path "$output" \
        --seed 0 --sh_degree 1 --mode factorsplat \
        --use_view_dependent_pos False --l_22_inv_init_scale 2.0 --mip3dgs \
        --tf_rank 8 --tf_condition_color True --tf_condition_opacity True \
        --tf_color_sh_degree 0 --tf_use_lookup True --tf_encoder_local True \
        --tf_opacity_alpha_identity_gate True --tf_exact_visibility_gate True \
        --tf_gate_removed_mass 0.5 --tf_aware_prune True \
        --tf_refresh_descriptors True --tf_refresh_label_locked True \
        --iterations 30000 --eval --disable_viewer \
        --start_checkpoint "$source/points3d.ply" \
        --use_jpeg_compression False --skip_test_camera_loading
}

train_pids=()
for job in heart:0 nose:1 hand:3; do
    train_scene "${job%%:*}" "${job##*:}" &
    train_pids+=("$!")
done
for pid in "${train_pids[@]}"; do wait "$pid"; done

# Render all three scenes concurrently. Each shard preserves global frame ids.
render_jobs=(
    "heart:0:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    "heart:4:21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
    "nose:1:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    "nose:5:21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
    "hand:3:0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    "hand:6:14,15,16,17,18,19,20,21,22,23,24,25,26,27"
    "hand:7:28,29,30,31,32,33,34,35,36,37,38,39,40"
)
render_pids=()
for job in "${render_jobs[@]}"; do
    IFS=: read -r scene gpu ids <<< "$job"
    output="$OUT_ROOT/$scene/clinical"
    docker run --gpus "device=$gpu" --name "fs_locked_render_${scene}_$gpu" \
        "${common[@]}" python render.py -m "$output" --iteration 30000 \
        --skip_train --skip_fps --render_tf_indices "$ids" &
    render_pids+=("$!")
done
for pid in "${render_pids[@]}"; do wait "$pid"; done

score_pids=()
for job in heart:0 nose:1 hand:3; do
    scene="${job%%:*}"; gpu="${job##*:}"
    source="$DATA/nerf_dataset/${scene}_factorsplat_clinical"
    output="$OUT_ROOT/$scene/clinical"
    docker run --gpus "device=$gpu" --name "fs_locked_score_$scene" \
        "${common[@]}" bash -lc "python metrics.py -m '$output' && \
        python scripts/benchmarks/factorsplat_group_metrics.py \
          --model '$output' --dataset '$source' --iteration 30000 && \
        python scripts/benchmarks/factorsplat_delta_metrics.py \
          --model '$output' --dataset '$source' --iteration 30000" &
    score_pids+=("$!")
done
for pid in "${score_pids[@]}"; do wait "$pid"; done

echo "Label-locked refresh experiment complete: $OUT_ROOT"
