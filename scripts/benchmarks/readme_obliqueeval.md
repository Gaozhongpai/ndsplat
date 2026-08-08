# XClipGS arbitrary-normal evaluation (`obliqueeval`) — data, inference, metrics

End-to-end guide for the arbitrary-normal results in the XClipGS paper. This is
an **evaluation-only** orientation test: it renders existing checkpoints on
planes whose physical-world normals were not used for training. It does not
generate training views or retrain any model.

The pipeline has three stages:

1. **Data preparation** — sample deterministic arbitrary normals and render
   matched clipped/unclipped reference images.
2. **Inference** — render the existing Ours, ClipGS, MM, and HC checkpoints on
   both paired datasets.
3. **Metrics** — compute band fidelity, CDE, leakage, CErr3D, and the aggregate
   arbitrary-normal table.

Two containers are used:

- **vengine** `10.10.0.192:5555/zhongpai/vengine-runtime:latest` — reference
  volume rendering. Mount `trueview/vengine-runtime` at `/repo` and
  `vengine_data` at `/home/vengine/app/external_data` (`$ED`).
- **ndgs** `10.10.0.192:5555/zhongpai/ndgs:latest` — trained-model rendering and
  metrics. Mount `ndsplat` at `/workspace/ndsplat` and `vengine_data` at
  `/data`. Set
  `PYTHONPATH=submodules/gsplat:submodules/tcgs_speedy_rasterizer`.

Data root: `vengine_data` (symlink to
`/mnt/uNeon/zhongpai/vengine_data`).

Scene keys:

```text
gel intestine kneejoint lower vascular heart nose hand
```

The paper displays `gel` as `abdomen`, `kneejoint` as `knee-joint`, and `lower`
as `lower-limb`; the scripts and output directories retain the original keys.

---

## 0. Protocol and prerequisites

For each scene, the default protocol samples:

- 5 deterministic, sphere-uniform physical-world plane normals;
- a minimum unoriented angle of 20 degrees from every voxel axis;
- a minimum unoriented pairwise angle of 20 degrees;
- one plane through the renderer volume center for each normal;
- one near-perpendicular camera (10 degrees from the normal) and two
  near-grazing cameras (1.5 degrees from edge-on) per plane.

This gives 15 clipped views and 15 camera-matched unclipped views per scene.
Sampling uses base seed 27 plus a stable SHA-256-derived scene component. The
generator records both values, all physical geometry, and every plane in
`cut_manifest.json`.

Required trained models:

```text
output/xclipgs/<method>/<scene>_900/
```

for `method in {ours, clipgs, mm, hc}`, each with
`point_cloud/iteration_best/` and `cfg_args`.

The in-tree tcgs/gsplat extension must expose `analytic_clip`, `clip_plane`, and
`hard_clip_mask`. `render_cutx.sh` checks this before rendering. A stale `_C.so`
can silently ignore arbitrary planes and produce an apparently intact volume.

---

## 1. Paired reference-data preparation

### 1a. Generate, render, and convert one scene

Run inside the vengine container:

```bash
export ED=/home/vengine/app/external_data
bash /repo/xclipgs_prepare_obliqueeval.sh <scene> 1.25
```

For example:

```bash
bash /repo/xclipgs_prepare_obliqueeval.sh heart 1.25
```

The wrapper performs all of the following:

1. `xclipgs_gen_obliqueeval_configs.py` samples the normals and writes matched
   clipped and unclipped camera XMLs.
2. `render_chunked.sh` renders both configuration trees.
3. The wrapper converts BMP output to PNG.
4. `xclipgs_dataset_to_nerf.py` converts the paired sets to NeRF transforms.
5. The test cameras are mirrored into a train split because the ndsplat reader
   obtains scene normalization from `transforms_train.json`.
6. The original `<scene>_900/points3d.ply` initialization is copied when
   available.

The primary outputs are:

```text
configs/<scene>_obliqueeval/                 # clipped camera XML + manifest
configs/<scene>_obliqueevalfull/             # matched unclipped XML + manifest
render_dataset/<scene>_obliqueeval/          # clipped reference rendering
render_dataset/<scene>_obliqueevalfull/      # unclipped reference rendering
nerf_dataset/<scene>_obliqueeval/            # clipped NeRF dataset
nerf_dataset/<scene>_obliqueevalfull/        # matched unclipped NeRF dataset
```

The clipped transforms contain `clip=true`, `plane_normal`, and `clip_offset`.
The full transforms retain the same camera order and view-family labels but set
`clip=false` and omit the plane.

Generator options can be appended to the wrapper command:

```bash
bash /repo/xclipgs_prepare_obliqueeval.sh heart 1.25 \
  --num-planes 5 \
  --seed 27 \
  --min-axis-angle-deg 20 \
  --min-pair-angle-deg 20 \
  --perp-tilt-deg 10 \
  --graze-tilt-deg 1.5
```

Keep these defaults when reproducing the paper.

### 1b. Host-side container example

```bash
SCENE=heart
GPU=2
CPUSET=24-35
VRUNTIME=/path/to/trueview/vengine-runtime
DATA=/mnt/uNeon/zhongpai/vengine_data
VENGINE=10.10.0.192:5555/zhongpai/vengine-runtime:latest

docker run --rm --gpus "\"device=${GPU}\"" --cpuset-cpus "$CPUSET" \
  --entrypoint /bin/bash \
  -e ED=/home/vengine/app/external_data \
  -v "$VRUNTIME:/repo" \
  -v "$DATA:/home/vengine/app/external_data" \
  "$VENGINE" -lc \
  "bash /repo/xclipgs_prepare_obliqueeval.sh ${SCENE} 1.25"
```

Run different scenes on different GPUs/CPU blocks. Do not launch two reference
renders on the same GPU.

### 1c. Validate clipping before model inference

First verify the protocol metadata:

```bash
python3 - <<'PY'
import json
from pathlib import Path

scene = "heart"
root = Path("/home/vengine/app/external_data")
manifest = json.loads(
    (root / "configs" / f"{scene}_obliqueeval" / "cut_manifest.json").read_text()
)
transforms = json.loads(
    (root / "nerf_dataset" / f"{scene}_obliqueeval" /
     "transforms_test.json").read_text()
)

assert manifest["num_planes"] == 5
assert manifest["views_per_plane"] == 3
assert len(manifest["planes"]) == 5
assert len(transforms["frames"]) == 15
assert all(frame["clip"] for frame in transforms["frames"])
assert all("plane_normal" in frame and "clip_offset" in frame
           for frame in transforms["frames"])
print("OK: 5 planes, 15 clipped frames, plane metadata present")
PY
```

Then verify that every clipped reference differs from its camera-matched full
reference. This catches the earlier failure mode in which all images showed the
intact CT:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

scene = "heart"
root = Path("/home/vengine/app/external_data/nerf_dataset")
clip = sorted((root / f"{scene}_obliqueeval" / "test").glob("*.png"))
full = sorted((root / f"{scene}_obliqueevalfull" / "test").glob("*.png"))
assert len(clip) == len(full) == 15

diffs = []
for clipped_path, full_path in zip(clip, full):
    assert clipped_path.name == full_path.name
    clipped = np.asarray(Image.open(clipped_path).convert("RGB"), dtype=np.float32)
    intact = np.asarray(Image.open(full_path).convert("RGB"), dtype=np.float32)
    diffs.append(float(np.mean(np.abs(clipped - intact)) / 255.0))

print("paired mean-absolute differences:", [round(value, 4) for value in diffs])
assert all(value > 1e-4 for value in diffs), \
    "a clipped reference is indistinguishable from the full volume"
print("OK: all 15 reference pairs show a nontrivial clip")
PY
```

Also inspect several clipped/full pairs visually. The clipped image must expose
a planar interior face; merely seeing a plane-shaped dark overlay is not enough.

---

## 2. Render existing checkpoints without retraining

Run inside the ndgs container from `/workspace/ndsplat`:

```bash
export PYTHONPATH=submodules/gsplat:submodules/tcgs_speedy_rasterizer
export XCLIPGS_DATA_ROOT=/data
export XCLIPGS_OUT=/data/output/xclipgs

bash scripts/benchmarks/render_cutx.sh <scene> obliqueeval \
  ours clipgs mm hc
bash scripts/benchmarks/render_cutx.sh <scene> obliqueevalfull \
  ours clipgs mm hc
```

For example:

```bash
bash scripts/benchmarks/render_cutx.sh heart obliqueeval \
  ours clipgs mm hc
bash scripts/benchmarks/render_cutx.sh heart obliqueevalfull \
  ours clipgs mm hc
```

`render_cutx.sh` creates a temporary evaluation model directory whose
`point_cloud` symlinks the trained checkpoint and whose `source_path` points to
the auxiliary dataset. It invokes `render.py --iteration best`; no optimizer or
training step runs.

The paired outputs are:

```text
output/xclipgs/<method>_obliqueeval/<scene>/test/ours_best/renders/
output/xclipgs/<method>_obliqueevalfull/<scene>/test/ours_best/renders/
```

Each directory must contain `00000.png` through `00014.png`. Render index `i`
corresponds to frame `i` after sorting `transforms_test.json` by `file_path`.

To render all scenes sequentially inside one container:

```bash
for scene in gel intestine kneejoint lower vascular heart nose hand; do
  bash scripts/benchmarks/render_cutx.sh "$scene" obliqueeval \
    ours clipgs mm hc
  bash scripts/benchmarks/render_cutx.sh "$scene" obliqueevalfull \
    ours clipgs mm hc
done
```

---

## 3. Metrics

The arbitrary-normal evaluation reuses the audited cut-face metrics. Keep its
results under `output/xclipgs/obliqueeval/`; do not mix them with the voxel-axis
`cuteval` directory.

Set common paths inside the ndgs container:

```bash
SCENE=heart
DATA=/data
OUT="$DATA/output/xclipgs"
EVAL=obliqueeval

O_CLIP="$OUT/ours_${EVAL}/$SCENE/test/ours_best/renders"
C_CLIP="$OUT/clipgs_${EVAL}/$SCENE/test/ours_best/renders"
M_CLIP="$OUT/mm_${EVAL}/$SCENE/test/ours_best/renders"
H_CLIP="$OUT/hc_${EVAL}/$SCENE/test/ours_best/renders"

O_FULL="$OUT/ours_${EVAL}full/$SCENE/test/ours_best/renders"
C_FULL="$OUT/clipgs_${EVAL}full/$SCENE/test/ours_best/renders"
M_FULL="$OUT/mm_${EVAL}full/$SCENE/test/ours_best/renders"
H_FULL="$OUT/hc_${EVAL}full/$SCENE/test/ours_best/renders"

DEST="$OUT/$EVAL/$SCENE"
mkdir -p "$DEST"
```

### 3a. Band fidelity and leakage

`cutplane_metrics.py` reports cut-face PSNR/SSIM/LPIPS on the near-perpendicular
views and near-edge fidelity/leakage on the grazing views. For grazing cameras,
the band is localized using the same least-squares screen-line proxy as
`cuteval`.

```bash
python scripts/benchmarks/cutplane_metrics.py \
  --gt-transforms "$DATA/nerf_dataset/${SCENE}_${EVAL}/transforms_test.json" \
  --gt-dir "$DATA/nerf_dataset/${SCENE}_${EVAL}/test" \
  --scene "$SCENE" \
  --out "$DEST" \
  --band-px 12 \
  --fg-thr 0.04 \
  --leak-margin-px 2 \
  --leak-window-px 60 \
  --debug \
  --methods \
    "ours=$O_CLIP" \
    "clipgs=$C_CLIP" \
    "mm=$M_CLIP" \
    "hc=$H_CLIP"
```

Output:

```text
output/xclipgs/obliqueeval/<scene>/cutplane_results.json
```

Use `--debug` for the first scene and inspect the band/line overlays. Remove it
for the final batch if the extra PNGs are not needed.

### 3b. Difference-referenced cut error (CDE)

CDE compares the clipped/full *change map* for each method against the
camera-matched reference change map:

```bash
python scripts/benchmarks/cutplane_cde.py \
  --gt-transforms "$DATA/nerf_dataset/${SCENE}_${EVAL}/transforms_test.json" \
  --gt-clip-dir "$DATA/nerf_dataset/${SCENE}_${EVAL}/test" \
  --gt-full-dir "$DATA/nerf_dataset/${SCENE}_${EVAL}full/test" \
  --scene "$SCENE" \
  --out "$DEST" \
  --band-px 12 \
  --fg-thr 0.04 \
  --mask-thr 0.04 \
  --methods \
    "ours=$O_CLIP:$O_FULL" \
    "clipgs=$C_CLIP:$C_FULL" \
    "mm=$M_CLIP:$M_FULL" \
    "hc=$H_CLIP:$H_FULL"
```

Output:

```text
output/xclipgs/obliqueeval/<scene>/cde_results.json
```

### 3c. Camera-independent CErr3D

Evaluate the five arbitrary-normal planes directly on the fixed Ours geometry:

```bash
python scripts/benchmarks/cutplane_cuterror_3d.py \
  --data-root /data \
  --out-root /data/output/xclipgs \
  --geometry ours \
  --eval-name obliqueeval \
  --output /data/output/xclipgs/obliqueeval/cerr3d.json
```

The analytic operator must report exactly zero. MM and HC report their
wrong-side opacity mass relative to exact kept mass.

### 3d. Aggregate the eight scenes

After `cutplane_results.json` and `cde_results.json` exist for all scenes:

```bash
python scripts/benchmarks/cutplane_make_table.py \
  --out-root /data/output/xclipgs \
  --eval-name obliqueeval \
  --dest /data/output/xclipgs/obliqueeval
```

Outputs:

```text
output/xclipgs/obliqueeval/cutplane_table.md
output/xclipgs/obliqueeval/cutplane_table.csv
output/xclipgs/obliqueeval/cerr3d.json
```

The paper's arbitrary-normal row uses the eight-scene averages of:

- `perp band_ssim` as band SSIM;
- grazing `cde` as edge CDE;
- perpendicular `cde` as face CDE;
- grazing `leak` as culled-side leakage.

CDE and Leak are displayed as `x10^-2` in the paper, so multiply the JSON/CSV
fractions by 100 when transcribing them.

---

## 4. Compact all-scene metric loop

Run after all clipped and full model renders are complete:

```bash
export PYTHONPATH=submodules/gsplat:submodules/tcgs_speedy_rasterizer
DATA=/data
OUT="$DATA/output/xclipgs"
EVAL=obliqueeval

for SCENE in gel intestine kneejoint lower vascular heart nose hand; do
  DEST="$OUT/$EVAL/$SCENE"
  mkdir -p "$DEST"

  O_CLIP="$OUT/ours_${EVAL}/$SCENE/test/ours_best/renders"
  C_CLIP="$OUT/clipgs_${EVAL}/$SCENE/test/ours_best/renders"
  M_CLIP="$OUT/mm_${EVAL}/$SCENE/test/ours_best/renders"
  H_CLIP="$OUT/hc_${EVAL}/$SCENE/test/ours_best/renders"
  O_FULL="$OUT/ours_${EVAL}full/$SCENE/test/ours_best/renders"
  C_FULL="$OUT/clipgs_${EVAL}full/$SCENE/test/ours_best/renders"
  M_FULL="$OUT/mm_${EVAL}full/$SCENE/test/ours_best/renders"
  H_FULL="$OUT/hc_${EVAL}full/$SCENE/test/ours_best/renders"

  python scripts/benchmarks/cutplane_metrics.py \
    --gt-transforms "$DATA/nerf_dataset/${SCENE}_${EVAL}/transforms_test.json" \
    --gt-dir "$DATA/nerf_dataset/${SCENE}_${EVAL}/test" \
    --scene "$SCENE" --out "$DEST" \
    --band-px 12 --fg-thr 0.04 \
    --leak-margin-px 2 --leak-window-px 60 \
    --methods "ours=$O_CLIP" "clipgs=$C_CLIP" "mm=$M_CLIP" "hc=$H_CLIP"

  python scripts/benchmarks/cutplane_cde.py \
    --gt-transforms "$DATA/nerf_dataset/${SCENE}_${EVAL}/transforms_test.json" \
    --gt-clip-dir "$DATA/nerf_dataset/${SCENE}_${EVAL}/test" \
    --gt-full-dir "$DATA/nerf_dataset/${SCENE}_${EVAL}full/test" \
    --scene "$SCENE" --out "$DEST" \
    --band-px 12 --fg-thr 0.04 --mask-thr 0.04 \
    --methods \
      "ours=$O_CLIP:$O_FULL" \
      "clipgs=$C_CLIP:$C_FULL" \
      "mm=$M_CLIP:$M_FULL" \
      "hc=$H_CLIP:$H_FULL"
done

python scripts/benchmarks/cutplane_cuterror_3d.py \
  --data-root "$DATA" --out-root "$OUT" \
  --geometry ours --eval-name "$EVAL" \
  --output "$OUT/$EVAL/cerr3d.json"

python scripts/benchmarks/cutplane_make_table.py \
  --out-root "$OUT" --eval-name "$EVAL" --dest "$OUT/$EVAL"
```

---

## 5. Reproduction checks

Before accepting a run, verify:

1. Every clipped and full reference dataset contains exactly 15 test frames.
2. Every method/suffix render directory contains exactly 15 PNGs.
3. Every clipped reference visibly differs from its paired full reference.
4. `transforms_test.json` contains five unique plane normals and three views per
   `plane_id`.
5. Each normal is at least 20 degrees from every physical voxel axis and each
   plane pair is separated by at least 20 degrees, treating `n` and `-n` as the
   same orientation.
6. Ours' `CErr3D` is exactly zero.
7. The aggregate table discovers all eight scenes.
8. No training command was launched; all results use the existing fixed final
   checkpoints.

A quick render-count audit:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/data/output/xclipgs")
scenes = "gel intestine kneejoint lower vascular heart nose hand".split()
methods = "ours clipgs mm hc".split()
suffixes = "obliqueeval obliqueevalfull".split()

bad = []
for scene in scenes:
    for method in methods:
        for suffix in suffixes:
            path = root / f"{method}_{suffix}" / scene / \
                "test" / "ours_best" / "renders"
            count = len(list(path.glob("*.png")))
            print(f"{scene:10s} {method:7s} {suffix:19s} {count:2d}")
            if count != 15:
                bad.append((scene, method, suffix, count))
assert not bad, f"unexpected render counts: {bad}"
PY
```

---

## 6. Gotchas

1. **Do not pass the physical-world normal directly to VEngine.** VEngine keeps
   `r dot index >= d` in voxel-index coordinates; ndsplat keeps
   `n dot x_centered <= tau` in physical-world coordinates.
   `xclipgs_gen_obliqueeval_configs.py` performs the sign, orientation, spacing,
   and center conversion and stores both representations in the manifest.
2. **An intact CT means clipping failed.** Check the paired reference images
   first. If reference clipping is correct but ndsplat renders remain intact,
   rebuild the in-tree tcgs/gsplat extension and rerun the clip-API preflight.
3. **Clipped and full images must be camera-identical.** CDE is invalid if the
   two datasets were generated independently or reordered. Always generate both
   with `xclipgs_prepare_obliqueeval.sh`.
4. **`perp` is near-perpendicular, not mathematically face-on.** The default
   camera is tilted 10 degrees from the plane normal. `graze` is 1.5 degrees
   from edge-on.
5. **The grazing screen line is a metric proxy.** Under perspective, a general
   plane does not project to one exact image line. The least-squares line is
   used only to localize the narrow band for the near-grazing cameras.
6. **Do not mix `obliqueeval` with `cuteval`.** They test different plane
   distributions and must have separate per-scene JSON and aggregate tables.
7. **`render_cutx.sh` replaces its evaluation model directory.** Do not run the
   same scene/method/suffix concurrently.
8. **Changing `--num-planes` can leave stale renderer PNGs.** Remove the old
   `configs`, `render_dataset`, and `nerf_dataset` trees for that scene before
   regenerating a nondefault protocol.
9. **Frame order is contractual.** Metric scripts sort transforms by
   `file_path` and pair frame `i` with render `i:05d.png`. Do not rename or
   independently sort one side of the pair.
10. **This experiment tests orientation transfer only.** All planes pass through
    the volume center, and existing checkpoints are evaluated without
    arbitrary-normal training.
