#!/usr/bin/env bash
# Matched-budget heart ablation: local code + global TF context at total rank 8.
# GPU 2 on host 173 is intentionally excluded.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
WORKTREE="${FACTORSPLAT_WORKTREE:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun}"
SOURCE="${FACTORSPLAT_SOURCE:-/mnt/uNeon/zhongpai/vengine_data/nerf_dataset/heart_factorsplat_clinical}"
GLOBAL_CONTEXT_RANK="${FACTORSPLAT_GLOBAL_CONTEXT_RANK:-4}"
LOCAL_CONTEXT_RANK=$((8 - GLOBAL_CONTEXT_RANK))
OUTPUT="${FACTORSPLAT_OUTPUT:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/context_ablation/local${LOCAL_CONTEXT_RANK}_global${GLOBAL_CONTEXT_RANK}/heart/clinical}"
PYDEPS="${FACTORSPLAT_PYDEPS:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100}"
CACHE="${FACTORSPLAT_TORCH_HOME:-/mnt/uNeon/zhongpai/vengine_data/cache/torch}"

common=(--rm --network none -v "$WORKTREE:/workspace:ro" -v /mnt/uNeon:/mnt/uNeon
        -w /workspace -e "TORCH_HOME=$CACHE"
        -e "PYTHONPATH=$PYDEPS:/workspace/submodules/gsplat" "$IMAGE")

if [[ ! -f "$OUTPUT/point_cloud/iteration_30000/point_cloud.ply" ]]; then
  docker run --gpus device=1 --name fs_context_heart_full "${common[@]}" \
    python train.py -s "$SOURCE" --model_path "$OUTPUT" -r 1 --seed 0 \
      --sh_degree 1 --mode factorsplat --use_view_dependent_pos False \
      --l_22_inv_init_scale 2.0 --mip3dgs --tf_rank 8 \
      --tf_global_context_rank "$GLOBAL_CONTEXT_RANK" --tf_condition_color True \
      --tf_condition_opacity True --tf_color_sh_degree 0 \
      --tf_use_lookup True --tf_encoder_local True --tf_lookup_mode joint \
      --tf_lookup_bins 64 --tf_exact_visibility_gate True \
      --tf_gate_removed_mass 0.5 --tf_aware_prune True --iterations 30000 \
      --eval --disable_viewer --start_checkpoint "$SOURCE/points3d.ply" \
      --use_jpeg_compression False --skip_test_camera_loading
fi

# Seven disjoint TF shards; global frame numbering is preserved by render.py.
jobs=("0:0,1,2,3,4,5" "1:6,7,8,9,10,11" "3:12,13,14,15,16,17"
      "4:18,19,20,21,22,23" "5:24,25,26,27,28,29"
      "6:30,31,32,33,34,35" "7:36,37,38,39,40")
pids=()
for job in "${jobs[@]}"; do
  gpu="${job%%:*}"
  ids="${job#*:}"
  docker run --gpus "device=$gpu" --name "fs_context_heart_render_$gpu" \
    "${common[@]}" python render.py -m "$OUTPUT" --skip_train \
    --iteration 30000 --render_tf_indices "$ids" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

docker run --gpus device=0 --name fs_context_heart_score "${common[@]}" \
  bash -lc "python metrics.py -m '$OUTPUT' && \
    python scripts/benchmarks/factorsplat_group_metrics.py \
      --model '$OUTPUT' --dataset '$SOURCE' --iteration 30000 && \
    python scripts/benchmarks/factorsplat_delta_metrics.py \
      --model '$OUTPUT' --dataset '$SOURCE' --iteration 30000"
