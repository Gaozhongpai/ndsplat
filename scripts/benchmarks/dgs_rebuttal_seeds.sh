#!/bin/bash
#
# NeurIPS 2026 rebuttal (Submission 8371): multi-seed Lambda ablation.
#
# Answers Reviewer orhG W1/Q1: the Lambda ordering on NeRF Synthetic spans only
# 0.13 dB (33.91 learned / 33.84 Lambda=0 / 33.78 Lambda=1) on a single seed, so
# "learned Lambda beats both fixed extremes" is not currently supported. This
# script re-runs the three ablation configs across seeds to get mean +/- std.
#
# NOTE: --seed did not exist before this rebuttal. safe_state() hardcoded
# seed 0, so every published number is seed 0 and re-running reproduced it
# bit-identically. --seed now defaults to 0, so seed 0 here MUST reproduce the
# existing output/standard/{opacity_only,opacity_pos_decouple_lambda1,dgS}
# numbers. That reproduction is the patch's correctness check (see --check).
#
# Configs (matching Table 3 of the paper):
#   lam0      Lambda=0, opacity-only (dGS-O)     -> paper 33.84
#   lam1      Lambda=1, fixed full coupling      -> paper 33.78
#   lamlearn  learned Lambda (full dGS)          -> paper 33.91
#
# Usage (inside container, cwd /code):
#   bash scripts/benchmarks/dgs_rebuttal_seeds.sh            # seeds 1 2
#   bash scripts/benchmarks/dgs_rebuttal_seeds.sh 1 2 3      # explicit seeds
#   bash scripts/benchmarks/dgs_rebuttal_seeds.sh --check    # seed 0 repro only
#
# Output: /code/output/rebuttal/seed<N>/<config>/nerf_synthetic/<scene>/
#         (host: trueview/vengine-runtime/vengine_data/output/rebuttal/)
#
# Host launch, pinned to GPU6 on uiiusls169 (8 GPUs / 96 CPUs = 12 CPUs per GPU):
#   docker run -d --name dgs_seed_gpu6 \
#     --gpus='"device=6"' --cpuset-cpus="72-83" --shm-size=32g \
#     -v <ndsplat>:/code -v /mnt/public_data_02/surgical:/code/dataset:ro \
#     -v <vengine_data/output>:/code/output -w /code \
#     10.10.0.192:5555/zhongpai/ndgs:latest sleep infinity

set -uo pipefail
shopt -s dotglob

# DATASET=nerf_synthetic (default) | 6dgs_pbr
#
# 6DGS-PBR is the scientifically more important target: NeRF Synthetic is
# diffuse and weakly view-dependent, so the coupling knob Lambda has little to
# act on there and configs are EXPECTED to land within noise. 6DGS-PBR is built
# for view-dependent specularity, and that is where Lambda separates in the
# paper (+1.21 dB over Lambda=0, +0.44 dB over Lambda=1). Seeds here are what
# license the learned-Lambda claim.
DATASET="${DATASET:-nerf_synthetic}"
case "$DATASET" in
    nerf_synthetic) base_dir="/code/dataset/nerf_synthetic/"; WHITE_BG="-w" ;;
    6dgs_pbr)       base_dir="/code/dataset/tandt_db/6dgs-pbr/"; WHITE_BG="" ;;
    *) echo "unknown DATASET: $DATASET (expected nerf_synthetic|6dgs_pbr)" >&2; exit 2 ;;
esac
out_root="/code/output/rebuttal"

# Config name -> extra train.py args. Mirrors dgs_nerf_synthetic.sh exactly so
# these are comparable to the published standard-densification numbers.
#   lam0:     --use_view_dependent_pos False              (= opacity_only)
#   lam1:     decouple + lambda_init 1.0                  (= opacity_pos_decouple_lambda1)
#   lamlearn: learned lambda, lambda_init -2.5            (= dgs)
# Flags mirror the published per-dataset benchmark scripts exactly
# (dgs_nerf_synthetic.sh / dgs_6dgs_pbr.sh), so these runs are directly
# comparable to the Table 3 numbers. PBR scenes use l_22_inv_init_scale 2.0.
CONFIGS=("lam0" "lam1" "lamlearn")
config_args() {
    local pbr=""
    [ "$DATASET" = "6dgs_pbr" ] && pbr="--l_22_inv_init_scale 2.0"
    case "$1" in
        lam0)
            echo "--use_view_dependent_pos False $pbr" ;;
        lam1)
            echo "--use_view_dependent_pos True --use_opacity_pos_decouple True --lambda_init 1.0 $pbr" ;;
        lamlearn)
            if [ "$DATASET" = "6dgs_pbr" ]; then
                echo "--use_view_dependent_pos True --lambda_init 0.0 $pbr"
            else
                echo "--use_view_dependent_pos True --lambda_init -2.5"
            fi ;;
        *) echo "unknown config: $1" >&2; return 1 ;;
    esac
}

CHECK_MODE=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_MODE=1
    SEEDS=(0)
    shift
elif [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2)
fi

scenes=()
for dir in "$base_dir"*/; do
    [ -d "$dir" ] || continue
    s=$(basename "${dir%/}")
    [[ "$s" == "README.txt" || "$s" == *.zip ]] && continue
    scenes+=("$s")
done

total=$(( ${#SEEDS[@]} * ${#CONFIGS[@]} * ${#scenes[@]} ))
echo "=============================================="
echo "Rebuttal Lambda ablation"
echo "  dataset : $DATASET"
echo "  seeds   : ${SEEDS[*]}"
echo "  configs : ${CONFIGS[*]}"
echo "  scenes  : ${#scenes[@]} (${scenes[*]})"
echo "  runs    : $total"
[ "$CHECK_MODE" = 1 ] && echo "  MODE    : seed-0 reproduction check"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

run_one() {
    local seed=$1 config=$2 scene=$3
    local extra; extra=$(config_args "$config")
    local out="${out_root}/seed${seed}/${config}/${DATASET}/${scene}"

    if [ -f "$out/results.json" ]; then
        echo "  [skip] seed=$seed $config/$scene (results.json exists)"
        return 0
    fi
    mkdir -p "$out"

    echo "  [run ] seed=$seed $config/$scene"
    # shellcheck disable=SC2086
    if ! python train.py -s "${base_dir}${scene}" \
            --model_path "$out" \
            --mode dgs \
            --seed "$seed" \
            $extra \
            --eval --disable_viewer $WHITE_BG \
            > "$out/train_stdout.log" 2>&1; then
        echo "  [FAIL] train seed=$seed $config/$scene -- see $out/train_stdout.log" >&2
        return 1
    fi

    # Only iteration 30000 is needed for the ablation table (paper reports 30k).
    # shellcheck disable=SC2086
    if ! python render.py -m "$out" --skip_train --iteration 30000 $extra \
            >> "$out/train_stdout.log" 2>&1; then
        echo "  [FAIL] render seed=$seed $config/$scene" >&2
        return 1
    fi

    if ! python metrics.py -m "$out" >> "$out/train_stdout.log" 2>&1; then
        echo "  [FAIL] metrics seed=$seed $config/$scene" >&2
        return 1
    fi
    return 0
}

failed=0; done_n=0
start_all=$(date +%s)
for seed in "${SEEDS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        for scene in "${scenes[@]}"; do
            done_n=$((done_n + 1))
            echo "[$done_n/$total] ------------------------------"
            run_one "$seed" "$config" "$scene" || failed=$((failed + 1))
        done
    done
done
elapsed=$(( $(date +%s) - start_all ))

echo "=============================================="
echo "Completed $((done_n - failed))/$total runs in $((elapsed / 60)) min ($failed failed)"
echo "=============================================="

python3 scripts/benchmarks/dgs_rebuttal_summarize.py --root "$out_root" || true

if [ "$CHECK_MODE" = 1 ]; then
    cat <<'EOF'

--------------------------------------------------------------
SEED-0 REPRODUCTION CHECK
Compare the seed0 means above against the published Table 3
NeRF Synthetic column:
    lam0      expected 33.84
    lam1      expected 33.78
    lamlearn  expected 33.91
Agreement to ~0.01 dB confirms the --seed patch is inert at its
default and that seeds 1,2 are a genuine second/third sample.
A mismatch means the patch changed behavior -- STOP and
investigate before using any multi-seed number in the rebuttal.
--------------------------------------------------------------
EOF
fi

exit $(( failed > 0 ? 1 : 0 ))
