#!/bin/bash
#
# NeurIPS 2026 rebuttal (Submission 8371): matched-pair seeds, Gaussian pair.
#
# Reviewer orhG Q1 asks for multi-seed results "for at least the matched pairs
# on NeRF Synthetic and D-NeRF, and for the Lambda ablation on NeRF Synthetic",
# with an explicit score criterion attached. The Lambda ablation is done; this
# script covers the dGS/N-DGS half of the matched pairs, re-running BOTH arms
# under the exact Table 2 MCMC protocol with new seeds.
#
# Flags are copied verbatim from dgs_nerf_synthetic_mcmc.sh and
# dgs_dnerf_mcmc.sh (the scripts that produced Table 2; verified: mcmc/dgs
# nerf_synthetic mean 34.34, mcmc/ndgs 34.04, matching the paper).
#
# Usage (inside the ndsplat container, cwd /code):
#   DATASET=nerf_synthetic bash scripts/benchmarks/dgs_rebuttal_pair_seeds.sh 1 2
#   DATASET=dnerf          bash scripts/benchmarks/dgs_rebuttal_pair_seeds.sh 1 2
#
# Output: /code/output/rebuttal_pairs/seed<N>/<mode>/<dataset>/<scene>/

set -uo pipefail
shopt -s dotglob

DATASET="${DATASET:-nerf_synthetic}"
SEEDS=("${@:-1}")
[ $# -eq 0 ] && SEEDS=(1 2)
out_root="/code/output/rebuttal_pairs"

run_one() {
    local seed=$1 mode=$2 scene=$3 src=$4 common=$5 extra=$6
    local out="${out_root}/seed${seed}/${mode}/${DATASET}/${scene}"
    if [ -f "$out/results.json" ]; then echo "  [skip] seed=$seed $mode/$scene"; return 0; fi
    mkdir -p "$out"
    echo "  [run ] seed=$seed $mode/$scene"
    # shellcheck disable=SC2086
    if ! python train.py -s "$src" --model_path "$out" --mode "$mode" --seed "$seed" \
            $common $extra > "$out/train_stdout.log" 2>&1; then
        echo "  [FAIL] train seed=$seed $mode/$scene" >&2; return 1
    fi
    # shellcheck disable=SC2086
    python render.py -m "$out" --skip_train --iteration 30000 $extra >> "$out/train_stdout.log" 2>&1 \
        && python metrics.py -m "$out" >> "$out/train_stdout.log" 2>&1 \
        || { echo "  [FAIL] render/metrics seed=$seed $mode/$scene" >&2; return 1; }
}

failed=0; n=0
if [ "$DATASET" = "nerf_synthetic" ]; then
    # Verbatim from dgs_nerf_synthetic_mcmc.sh: cap 300000, opacity_reg 0.01,
    # scale_reg 0, default noise_lr, white background.
    COMMON="--densification_strategy mcmc --mcmc_cap_max 300000 --opacity_reg 0.01 --scale_reg 0 --eval --disable_viewer -w"
    base="/code/dataset/nerf_synthetic/"
    scenes=(chair drums ficus hotdog lego materials mic ship)
    declare -A EXTRA=( [dgs]="--use_view_dependent_pos True" [ndgs]="" )
elif [ "$DATASET" = "dnerf" ]; then
    # Verbatim from dgs_dnerf_mcmc.sh: 7 scenes, cap 150000, -r 2, input_dim 7,
    # noise_lr 1.0, opacity_reg 0.01, scale_reg 0.01, batch 1 (default).
    COMMON="--input_dim 7 -r 2 --densification_strategy mcmc --mcmc_cap_max 150000 --noise_lr 1.0 --opacity_reg 0.01 --scale_reg 0.01 --eval --disable_viewer"
    base="/code/dataset/dnerf/"
    scenes=(bouncingballs hellwarrior hook jumpingjacks mutant standup trex)
    declare -A EXTRA=( [dgs]="--use_view_dependent_pos True --l_22_inv_init_scale 0.02" [ndgs]="--lambda_opc 0.1" )
else
    echo "unknown DATASET: $DATASET" >&2; exit 2
fi

total=$(( ${#SEEDS[@]} * 2 * ${#scenes[@]} ))
echo "=============================================="
echo "Matched-pair seeds (Gaussian pair, Table 2 MCMC protocol)"
echo "  dataset : $DATASET   seeds: ${SEEDS[*]}   runs: $total"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

for seed in "${SEEDS[@]}"; do
    for mode in ndgs dgs; do
        for scene in "${scenes[@]}"; do
            n=$((n+1)); echo "[$n/$total] ------------------------------"
            run_one "$seed" "$mode" "$scene" "${base}${scene}" "$COMMON" "${EXTRA[$mode]}" || failed=$((failed+1))
        done
    done
done
echo "Done: $((n-failed))/$n succeeded, $failed failed"
