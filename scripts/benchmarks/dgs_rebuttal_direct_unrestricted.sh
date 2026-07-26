#!/bin/bash
#
# NeurIPS 2026 rebuttal (Submission 8371): Direct-Unrestricted ablation.
#
# Isolates the COORDINATE CHANGE from the ADDED PRIORS, per Appendix D:
#
#   Model                Parameters                        Model class
#   -----------------    ------------------------------    -----------------------------
#   N-DGS                joint covariance                  unrestricted joint Gaussian
#   Direct-unrestricted  (Sigma_cond, M, V_qq), M free      exactly the same class
#   dGS                  structured factorization of M      constrained subfamily
#
#   N-DGS vs Direct-unrestricted -> effect of the coordinate change alone
#   Direct-unrestricted vs dGS   -> effect of the spatial normalization, bounded Lambda,
#                                   and structured regression prior
#
# Direct-unrestricted learns M in R^{3xC} directly and renders
# mu_{p|q} = mu_p + M (q - mu_q), keeping Sigma_cond (scale/rotation) and
# V_qq = L L^T unchanged. Verified before training: the opacity path is
# bit-identical to dGS, and setting M = V_pq diag(Lambda) V_qq reproduces dGS
# exactly (max err 2.4e-07), so DU strictly contains dGS.
#
# NOTE ON SPEED: the DU position shift runs in PyTorch (extra kernel launch), so
# FPS from these runs is NOT comparable. This ablation is about QUALITY only.
#
# Usage (inside container, cwd /code):
#   DATASET=nerf_synthetic bash scripts/benchmarks/dgs_rebuttal_direct_unrestricted.sh
#   DATASET=6dgs_pbr       bash scripts/benchmarks/dgs_rebuttal_direct_unrestricted.sh
#
# Output: /code/output/rebuttal_du/<dataset>/<scene>/

set -uo pipefail
shopt -s dotglob

DATASET="${DATASET:-nerf_synthetic}"
case "$DATASET" in
    nerf_synthetic) base_dir="/code/dataset/nerf_synthetic/"; WHITE_BG="-w"; EXTRA="--lambda_init -2.5" ;;
    6dgs_pbr)       base_dir="/code/dataset/tandt_db/6dgs-pbr/"; WHITE_BG=""; EXTRA="--l_22_inv_init_scale 2.0 --lambda_init 0.0" ;;
    *) echo "unknown DATASET: $DATASET" >&2; exit 2 ;;
esac
out_root="/code/output/rebuttal_du"

scenes=()
for d in "$base_dir"*/; do
    [ -d "$d" ] || continue
    s=$(basename "${d%/}")
    [[ "$s" == "README.txt" || "$s" == *.zip ]] && continue
    scenes+=("$s")
done

echo "=============================================="
echo "Direct-Unrestricted ablation"
echo "  dataset : $DATASET"
echo "  scenes  : ${#scenes[@]} (${scenes[*]})"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

failed=0; n=0
for scene in "${scenes[@]}"; do
    n=$((n+1))
    out="${out_root}/${DATASET}/${scene}"
    if [ -f "$out/results.json" ]; then
        echo "[$n/${#scenes[@]}] [skip] $scene"; continue
    fi
    mkdir -p "$out"
    echo "[$n/${#scenes[@]}] [run ] $scene"

    # shellcheck disable=SC2086
    if ! python train.py -s "${base_dir}${scene}" --model_path "$out" \
            --mode dgs --direct_unrestricted \
            --use_view_dependent_pos True $EXTRA \
            --eval --disable_viewer $WHITE_BG > "$out/train_stdout.log" 2>&1; then
        echo "  [FAIL] train $scene -- see $out/train_stdout.log" >&2
        failed=$((failed+1)); continue
    fi
    # shellcheck disable=SC2086
    python render.py -m "$out" --skip_train --iteration 30000 \
        --direct_unrestricted $EXTRA >> "$out/train_stdout.log" 2>&1 \
        && python metrics.py -m "$out" >> "$out/train_stdout.log" 2>&1 \
        || { echo "  [FAIL] render/metrics $scene" >&2; failed=$((failed+1)); }
done

echo "=============================================="
echo "Done: $((n-failed))/$n succeeded, $failed failed"
echo "=============================================="
