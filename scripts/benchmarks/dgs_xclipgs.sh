#!/bin/bash
#
# XClipGS training: opacity-conditioned dGS + Mip-Splatting, under the THREE clip
# operators from the paper (pages/XClipGS/paper/main.tex, §Method), on the
# vengine_data XClipGS nerf datasets. Trains every nerf_dataset/<scene>_900 with
# each operator and saves to vengine_data/output/xclipgs/<operator>/<scene>/.
#
# Variants (label -> mode + clip operator + Mip):
#   ours       = analytic half-space (exact; our contribution) [dgs/analytic], Mip ON
#   mm         = moment-matched truncation (Gaussian surrogate) [dgs/moment],  Mip ON
#   hc         = hard cull (per-primitive keep/drop; prior-art) [dgs/hardcull],Mip ON
#   ours_nomip = analytic (Ours) with Mip-Splatting OFF -- ablation isolating
#                whether Ours' gain depends on Mip
#   clipgs     = ClipGS BASELINE, our reimpl: 3DGS backbone + STE trainable
#                hard-cull + deformation MLP [--mode clipgs]. Its own model, so it
#                uses --mode clipgs (not dgs) and no Mip. Labeled a baseline, not
#                their released code. See scene/gaussian_model_clipgs.py.
#
# Fixed for the dGS variants (ours/mm/hc/ours_nomip):
#   1. dGS OPACITY-ONLY : --mode dgs --use_view_dependent_pos False --l_22_inv_init_scale 2.0
#   2. clip modes       : the loop's --clip_operator
#   3. MIP              : ON for ours/mm/hc, OFF for ours_nomip (per-label, see MIP[])
# clipgs uses its own --mode; MODE[] selects per-label.
#
# Init: each dataset ships a full dGS RenderFM checkpoint as points3d.ply
# (xyz + SH + opacity + scale + rot + L_22_inv). We warm-start from it via
# --start_checkpoint (ndsplat's dgs load_ply reads that exact schema; the
# nx/ny/nz view-direction fallback handles RenderFM plys), matching the
# model-viewer finetune recipe (renderfm/scan_renders/upload_root_finetune.py).
# Unlike that FINETUNE, we keep densification ON here (training from the init,
# not refining a frozen prediction). The points3d.ply/cameras are already
# centered on the CT volume center, so the checkpoint is used as-is.
#
# Per-frame clip planes (n, tau) come from each dataset's transforms_*.json.
# Run from the ndsplat root (train.py lives here), e.g. in ndgs:latest with the
# ndsplat repo and vengine_data mounted. The tcgs rasterizer AND the gsplat fork
# must be rebuilt in-tree (their shipped _C.so predate the clip API / are stale);
# import them via PYTHONPATH pointing at submodules/{gsplat,tcgs_speedy_rasterizer}.
#
# Usage:
#   bash scripts/benchmarks/dgs_xclipgs.sh
#   XCLIPGS_DATA=/data/nerf_dataset XCLIPGS_OUT=/data/output/xclipgs \
#       XCLIPGS_ITERS=30000 bash scripts/benchmarks/dgs_xclipgs.sh
#
set -e
shopt -s nullglob

# vengine_data mounts (override for your container layout)
BASE_DIR="${XCLIPGS_DATA:-/data/nerf_dataset}"        # vengine_data/nerf_dataset
OUT_ROOT="${XCLIPGS_OUT:-/data/output/xclipgs}"        # vengine_data/output/xclipgs
ITERS="${XCLIPGS_ITERS:-30000}"

# Shared eval flags. The MODEL flags (mode + dGS opacity-only knobs) are per-label
# via COMMON[] so the clipgs baseline (its own --mode) coexists with the dGS ops.
EVAL_COMMON="--iterations ${ITERS} --eval --disable_viewer"
DGS_MODEL="--mode dgs --use_view_dependent_pos False --l_22_inv_init_scale 2.0"

# per-label MODEL flags (mode + model-specific knobs)
declare -A COMMON=(
  ["ours"]="${DGS_MODEL}" ["mm"]="${DGS_MODEL}" ["hc"]="${DGS_MODEL}" ["ours_nomip"]="${DGS_MODEL}"
  ["clipgs"]="--mode clipgs" )
# operator label -> --clip_operator value (ignored by clipgs, which uses its own model)
declare -A OPS=( ["ours"]="analytic" ["mm"]="moment" ["hc"]="hardcull" ["ours_nomip"]="analytic" ["clipgs"]="clipgs" )
# per-label Mip-Splatting flag: ON for the main dGS 3, OFF for ours_nomip and clipgs.
declare -A MIP=( ["ours"]="--mip3dgs" ["mm"]="--mip3dgs" ["hc"]="--mip3dgs" ["ours_nomip"]="" ["clipgs"]="" )
# labels to sweep (override with XCLIPGS_LABELS="clipgs" to run just the baseline)
LABELS="${XCLIPGS_LABELS:-ours mm hc ours_nomip clipgs}"

# Preflight: the installed tcgs rasterizer MUST expose the clip API, otherwise a
# stale in-tree _C.so silently ignores the clip planes and all three operators
# render identically (see the "stale _C.so" gotcha). Fail fast before 18 runs.
echo "Preflight: checking tcgs_speedy_rasterizer clip API..."
python - <<'PY' || { echo "ABORT: tcgs rasterizer lacks the clip API (stale _C.so?). Rebuild submodules/tcgs_speedy_rasterizer."; exit 1; }
from tcgs_speedy_rasterizer import GaussianRasterizationSettings, hard_clip_mask  # noqa
assert "analytic_clip" in GaussianRasterizationSettings._fields, "no analytic_clip field"
assert "clip_plane" in GaussianRasterizationSettings._fields, "no clip_plane field"
print("  OK: analytic_clip + clip_plane + hard_clip_mask present")
PY

run_one() {
    local scene_dir=$1 scene=$2 label=$3 op=$4
    local out="${OUT_ROOT}/${label}/${scene}"
    if [ -f "${out}/results.json" ]; then
        echo "  skip (results.json exists): ${out}"
        return
    fi
    # Warm-start from the RenderFM dGS checkpoint if present (else train.py
    # falls back to the reader's random point cloud).
    local ckpt_arg=""
    if [ -f "${scene_dir}/points3d.ply" ]; then
        ckpt_arg="--start_checkpoint ${scene_dir}/points3d.ply"
    fi
    # dGS ops take --clip_operator; clipgs is its own model and ignores it.
    local clipop_arg="--clip_operator ${op}"
    [ "${label}" = "clipgs" ] && clipop_arg=""
    echo "  train: ${scene} [${label} / mode=${COMMON[${label}]} / op=${op} / mip=${MIP[${label}]:-off}] -> ${out}"
    python train.py -s "${scene_dir}" --model_path "${out}" ${COMMON[${label}]} ${EVAL_COMMON} ${MIP[${label}]} ${ckpt_arg} ${clipop_arg}
    # Render held-out test views at a few iterations (incl. best) + metrics
    for it in 7000 "${ITERS}" best; do
        python render.py -m "${out}" --skip_train --iteration "${it}" || true
    done
    python metrics.py -m "${out}" || true
}

echo "XClipGS training: data=${BASE_DIR}  out=${OUT_ROOT}  iters=${ITERS}"
for scene_dir in "${BASE_DIR}"/*_900/; do
    [ -d "${scene_dir}" ] || continue
    scene=$(basename "${scene_dir%/}")
    for label in ${LABELS}; do
        echo "==================== ${scene} : ${label} ===================="
        run_one "${scene_dir}" "${scene}" "${label}" "${OPS[${label}]}"
    done
done
echo "XClipGS training complete. Outputs under ${OUT_ROOT}/{ours,mm,hc,ours_nomip}/<scene>_900/."

# Build the clip-operator comparison table (Markdown + CSV) across all runs.
# Requested location: under the ours/gel_900 run dir; also drop a copy at the
# sweep root for convenience.
TABLE_DEST="${XCLIPGS_TABLE_DEST:-${OUT_ROOT}/ours/gel_900}"
echo "Building comparison table -> ${TABLE_DEST}/comparison_table.md"
python scripts/benchmarks/xclipgs_make_table.py --out-root "${OUT_ROOT}" --dest "${TABLE_DEST}" || true
python scripts/benchmarks/xclipgs_make_table.py --out-root "${OUT_ROOT}" --dest "${OUT_ROOT}" || true
