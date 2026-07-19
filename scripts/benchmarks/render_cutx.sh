#!/bin/bash
# Render trained XClipGS models on an auxiliary cut-eval dataset variant
# (generalizes render_cuteval.sh to any <scene>_<suffix> nerf dataset):
#   suffix=cutevalfull  -> UNCLIPPED renders at the cut-eval cameras (metric B/CDE)
#   suffix=cutsweep     -> fixed-camera plane-sweep renders (metric A/flicker)
#
# Runs INSIDE ndgs:latest with PYTHONPATH to the submodules (same as training).
# Usage (in container): bash scripts/benchmarks/render_cutx.sh <scene> <suffix> [methods...]
set -uo pipefail

SCENE="${1:?scene, e.g. heart}"
SUFFIX="${2:?dataset suffix, e.g. cutevalfull or cutsweep}"
shift 2 || true
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(ours clipgs mm hc)

DATA="${XCLIPGS_DATA_ROOT:-/data}"
OUT="${XCLIPGS_OUT:-$DATA/output/xclipgs}"
NERF="$DATA/nerf_dataset/${SCENE}_${SUFFIX}"

if [ ! -f "$NERF/transforms_test.json" ]; then
  echo "!! missing dataset $NERF/transforms_test.json"; exit 1
fi

# Fail fast if the tcgs rasterizer lacks the clip API (stale in-tree _C.so) — else
# every operator renders identically (the clip planes are silently ignored).
python - <<'PY' || { echo "ABORT: tcgs rasterizer lacks the clip API (stale _C.so?). Rebuild submodules/tcgs_speedy_rasterizer."; exit 1; }
from tcgs_speedy_rasterizer import GaussianRasterizationSettings, hard_clip_mask  # noqa
assert "analytic_clip" in GaussianRasterizationSettings._fields, "no analytic_clip field"
assert "clip_plane" in GaussianRasterizationSettings._fields, "no clip_plane field"
print("  OK: analytic_clip + clip_plane + hard_clip_mask present")
PY

for m in "${METHODS[@]}"; do
  TRAINED="$OUT/$m/${SCENE}_900"
  CE="$OUT/${m}_${SUFFIX}/${SCENE}"
  if [ ! -d "$TRAINED/point_cloud/iteration_best" ]; then
    echo "!! $m: no iteration_best in $TRAINED; skip"; continue
  fi
  echo "=== $m/${SUFFIX}: building model dir $CE ==="
  rm -rf "$CE"; mkdir -p "$CE"
  # symlink the trained point cloud (keeps deform_mlp.pt sibling for clipgs)
  ln -s "$TRAINED/point_cloud" "$CE/point_cloud"
  python3 - "$TRAINED/cfg_args" "$CE/cfg_args" "$NERF" "$CE" <<'PY'
import sys
src_cfg, dst_cfg, nerf, model_path = sys.argv[1:5]
from argparse import Namespace  # noqa: needed for eval of Namespace(...)
ns = eval(open(src_cfg).read())
ns.source_path = nerf
ns.model_path = model_path
open(dst_cfg, "w").write(repr(ns))
print(f"   cfg_args: source_path -> {nerf}")
PY
  echo "=== $m/${SUFFIX}: render.py (iteration best) ==="
  python render.py -m "$CE" -s "$NERF" --skip_train --iteration best \
    --eval 2>&1 | grep -Ei "Rendering|FPS|error|traceback|Loading trained" | head -20
  n=$(ls "$CE"/test/ours_best/renders/*.png 2>/dev/null | wc -l)
  echo "=== $m/${SUFFIX}: $n renders in $CE/test/ours_best/renders ==="
done

chown -R 1000073:1000001 "$OUT"/*_${SUFFIX} 2>/dev/null || true
echo "=== ${SCENE} ${SUFFIX} render DONE ==="
