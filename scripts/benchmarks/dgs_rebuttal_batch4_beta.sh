#!/bin/bash
#
# NeurIPS 2026 rebuttal (Submission 8371): Beta pair on 7DGS-PBR at BATCH 4.
#
# Reviewer orhG Q4 carries the only explicit downside criterion in the reviews:
# "evidence of under-optimized or under-tuned baselines would move me down."
# The exposed number is the paper's largest headline gain, +1.26 dB dBS/UBS on
# 7DGS-PBR, measured at batch 1 while our UBS baseline sits at 30.61 dB against
# a published 33.00 dB at batch 4. This script re-runs BOTH arms at batch 4
# (--mv 4, gradient accumulation over 4 views) -- UBS's published operating
# point -- under the otherwise-identical Table 2 MCMC protocol, to test whether
# the pairwise ordering survives.
#
# All flags besides --mv 4 are verbatim from dgs_7dgs_pbr_mcmc.sh (the script
# that produced Table 2's Beta rows). Seed 0 (the published seed).
#
# Usage (inside the ndsplat container, cwd /code):
#   bash scripts/benchmarks/dgs_rebuttal_batch4_beta.sh
#
# Output: /code/output/rebuttal_batch4/{ubs,dbs}/7dgs_pbr/<scene>/

set -uo pipefail
shopt -s dotglob

base_dir="/code/dataset/dyct/7dgs_pbr/"
out_root="/code/output/rebuttal_batch4"
NOISE_LR=1.0
SCALE_REG=0

declare -A MCMC_CAP_MAX=( [cloud]=150000 [dust]=150000 [flame]=150000 [heart]=150000 [heart_1600]=150000 [suzanne]=300000 )
declare -A OPACITY_REG_MAP=( [cloud]=0 [dust]=0 [flame]=0 [heart]=0 [heart_1600]=0 [suzanne]=0.01 )

run_experiment() {
    local mode=$1 scene_name=$2 dir=$3 extra_args=$4
    local output_dir="${out_root}/${mode}/7dgs_pbr/${scene_name}"
    if [ -f "$output_dir/results.json" ]; then echo "  [skip] $mode/$scene_name"; return 0; fi
    mkdir -p "$output_dir"
    local cap_max=${MCMC_CAP_MAX[$scene_name]:-150000}
    local opacity_reg=${OPACITY_REG_MAP[$scene_name]:-0}
    echo "  [run ] $mode/$scene_name (cap=$cap_max, opacity_reg=$opacity_reg, mv=4)"

    # shellcheck disable=SC2086
    if ! python train.py -s "$dir" \
            --model_path "$output_dir" \
            --mode "$mode" \
            --input_dim 7 \
            --resolution 2 \
            --mv 4 \
            --densification_strategy mcmc \
            --mcmc_cap_max $cap_max \
            --noise_lr $NOISE_LR \
            --opacity_reg $opacity_reg \
            --scale_reg $SCALE_REG \
            $extra_args \
            --eval \
            --disable_viewer > "$output_dir/train_stdout.log" 2>&1; then
        echo "  [FAIL] train $mode/$scene_name" >&2; return 1
    fi
    # shellcheck disable=SC2086
    python render.py -m "$output_dir" --skip_train --iteration 30000 \
        --input_dim 7 --resolution 2 $extra_args >> "$output_dir/train_stdout.log" 2>&1 \
        && python metrics.py -m "$output_dir" >> "$output_dir/train_stdout.log" 2>&1 \
        || { echo "  [FAIL] render/metrics $mode/$scene_name" >&2; return 1; }
}

echo "=============================================="
echo "Beta pair, 7DGS-PBR, BATCH 4 (mv=4), Table 2 MCMC protocol"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

failed=0; n=0
# ONLY_SCENES="cloud dust flame" restricts this instance to a scene subset so
# two GPUs can partition the sweep without racing on the same output dirs.
for dir in "$base_dir"*/; do
    [ -d "$dir" ] || continue
    scene_name=$(basename "${dir%/}")
    [[ "$scene_name" == *.zip ]] && continue
    if [ -n "${ONLY_SCENES:-}" ] && ! grep -qw "$scene_name" <<< "$ONLY_SCENES"; then continue; fi

    n=$((n+1)); echo "[$n] ------------------------------"
    run_experiment "ubs" "$scene_name" "$dir" "--use_gsplat" || failed=$((failed+1))

    if [[ "$scene_name" == "cloud" ]]; then l22=2.5; else l22=0.4; fi
    n=$((n+1)); echo "[$n] ------------------------------"
    run_experiment "dbs" "$scene_name" "$dir" "--use_gsplat --l_22_inv_init_scale ${l22}" || failed=$((failed+1))
done
echo "Done: $((n-failed))/$n succeeded, $failed failed"
