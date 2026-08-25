#!/usr/bin/env bash
# Matched FactorSplat opacity-code ablation on heart and lower.
# GPU 2 on host 173 is intentionally excluded. Docker networking is disabled.
set -euo pipefail

IMAGE="${FACTORSPLAT_IMAGE:-10.10.0.192:5555/zhongpai/ndgs:latest}"
WORKTREE="${FACTORSPLAT_WORKTREE:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun}"
GSPLAT_SOURCE="${FACTORSPLAT_GSPLAT_SOURCE:-/mnt/UIIUSA/zhongpai/code/gaussian/workspace/ndsplat-factorsplat-rerun/submodules/gsplat}"
DATA_ROOT="${FACTORSPLAT_DATA_ROOT:-/mnt/uNeon/zhongpai/vengine_data/nerf_dataset}"
OUTPUT_ROOT="${FACTORSPLAT_OUTPUT_ROOT:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/opacity_code_ablation}"
PYDEPS="${FACTORSPLAT_PYDEPS:-/mnt/uNeon/zhongpai/vengine_data/output/factorsplat/ood_fix_tests/pydeps_h100}"
CACHE="${FACTORSPLAT_TORCH_HOME:-/mnt/uNeon/zhongpai/vengine_data/cache/torch}"

common=(--rm --network none -v "$WORKTREE:/workspace:ro"
        -v "$GSPLAT_SOURCE:/workspace/submodules/gsplat:ro"
        -v /mnt/uNeon:/mnt/uNeon
        -w /workspace -e "TORCH_HOME=$CACHE"
        -e "PYTHONPATH=$PYDEPS:/workspace/submodules/gsplat" "$IMAGE")

run_case() {
  local scene="$1"
  local variant="$2"
  local gpu="$3"
  local source="$DATA_ROOT/${scene}_factorsplat_clinical"
  local output="$OUTPUT_ROOT/$variant/$scene/clinical"
  local container="fs_opacity_${variant}_${scene}"
  local extra=(--tf_opacity_alpha_only True)
  if [[ "$variant" == "alphaonly_logratio" ]]; then
    extra+=(--tf_opacity_log_ratio True --tf_log_ratio_encoder True)
  fi

  mkdir -p "$output"
  if [[ ! -f "$output/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    docker run --gpus "device=$gpu" --name "${container}_train" "${common[@]}" \
      python train.py -s "$source" --model_path "$output" -r 1 --seed 0 \
        --sh_degree 1 --mode factorsplat --use_view_dependent_pos False \
        --l_22_inv_init_scale 2.0 --mip3dgs --tf_rank 8 \
        --tf_condition_color True --tf_condition_opacity True \
        --tf_color_sh_degree 0 --tf_use_lookup True --tf_encoder_local True \
        --tf_lookup_mode joint --tf_lookup_bins 64 \
        --tf_exact_visibility_gate True --tf_gate_removed_mass 0.5 \
        --tf_aware_prune True --iterations 30000 --eval --disable_viewer \
        --start_checkpoint "$source/points3d.ply" \
        --use_jpeg_compression False --skip_test_camera_loading "${extra[@]}"
  fi

  docker run --gpus "device=$gpu" --name "${container}_render" "${common[@]}" \
    python render.py -m "$output" --skip_train --iteration 30000

  docker run --gpus "device=$gpu" --name "${container}_score" "${common[@]}" \
    bash -lc "python metrics.py -m '$output' && \
      python scripts/benchmarks/factorsplat_group_metrics.py \
        --model '$output' --dataset '$source' --iteration 30000 && \
      python scripts/benchmarks/factorsplat_delta_metrics.py \
        --model '$output' --dataset '$source' --iteration 30000"
}

run_case heart alphaonly 0 & p0=$!
run_case heart alphaonly_logratio 1 & p1=$!
run_case lower alphaonly 3 & p3=$!
run_case lower alphaonly_logratio 4 & p4=$!

status=0
for pid in "$p0" "$p1" "$p3" "$p4"; do
  wait "$pid" || status=1
done
exit "$status"
