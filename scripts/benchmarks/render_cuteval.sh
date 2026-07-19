#!/bin/bash
# Render trained XClipGS models at the CUT-EVAL cameras (perp + grazing on x/y/z
# planes) without re-training. For each method we build a throwaway model dir that
# reuses the trained point_cloud (symlink, so clipgs's sibling deform_mlp.pt comes
# along) but points source_path at the *_cuteval nerf dataset, then render.py loads
# iteration_best and renders the cut-eval test split.
#
# Runs INSIDE ndgs:latest (same as training) with PYTHONPATH to the submodules.
# Mounts: <ndsplat>:/workspace/ndsplat, <DATA>/vengine_data:/data, -w ndsplat.
#
# Usage (in container):  bash scripts/benchmarks/render_cuteval.sh <scene> [methods...]
#   scene   e.g. heart   (uses <scene>_900 trained model + <scene>_cuteval dataset)
#   methods default: ours clipgs mm hc
set -uo pipefail

SCENE="${1:?scene, e.g. heart}"; shift || true
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(ours clipgs mm hc)

DATA="${XCLIPGS_DATA_ROOT:-/data}"
OUT="${XCLIPGS_OUT:-$DATA/output/xclipgs}"
NERF="$DATA/nerf_dataset/${SCENE}_cuteval"

if [ ! -f "$NERF/transforms_test.json" ]; then
  echo "!! missing cut-eval dataset $NERF/transforms_test.json"; exit 1
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
  CE="$OUT/${m}_cuteval/${SCENE}"
  if [ ! -d "$TRAINED/point_cloud/iteration_best" ]; then
    echo "!! $m: no iteration_best in $TRAINED; skip"; continue
  fi
  echo "=== $m: building cut-eval model dir $CE ==="
  rm -rf "$CE"; mkdir -p "$CE"
  # symlink the trained point cloud (keeps deform_mlp.pt sibling for clipgs)
  ln -s "$TRAINED/point_cloud" "$CE/point_cloud"
  # cfg_args with source_path -> cut-eval dataset (all other flags identical)
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
  echo "=== $m: render.py cut-eval (iteration best) ==="
  python render.py -m "$CE" -s "$NERF" --skip_train --iteration best \
    --eval 2>&1 | grep -Ei "Rendering|FPS|error|traceback|Loading trained" | head -20
  n=$(ls "$CE"/test/ours_best/renders/*.png 2>/dev/null | wc -l)
  echo "=== $m: $n cut-eval renders in $CE/test/ours_best/renders ==="
done

chown -R 1000073:1000001 "$OUT"/*_cuteval 2>/dev/null || true
echo "=== ${SCENE} cut-eval render DONE ==="
