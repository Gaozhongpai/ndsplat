# XClipGS cut-face evaluation (`cuteval`) — data, inference, metrics

End-to-end guide for the cut-face evaluation behind the paper's cut-face tables
(band-SSIM, CDE, leak, CErr3D) and the operator-swap tables. Covers three things:

1. **Data prep** — generate the fixed cut-eval cameras + vengine ground truth, and
   the clipped / unclipped / sweep dataset variants.
2. **Inference** — render each trained model (and each render-time operator) at the
   cut-eval cameras.
3. **Metrics** — compute band fidelity, CDE (difference-referenced cut error), leak,
   CErr3D (geometric mass error), and assemble the aggregate table.

Two containers are used throughout:
- **vengine** `10.10.0.192:5555/zhongpai/vengine-runtime:latest` — reference volume
  renderer (ground truth). Mounts `vengine-runtime` at `/repo` and `vengine_data`
  at `/home/vengine/app/external_data` (`$ED`).
- **ndgs** `10.10.0.192:5555/zhongpai/ndgs:latest` — trained-model renderer +
  metrics. Mounts `ndsplat` at `/workspace/ndsplat`, `vengine_data` at `/data`.
  Needs `PYTHONPATH=submodules/gsplat:submodules/tcgs_speedy_rasterizer` and the
  **in-tree tcgs/gsplat `.so` must expose the clip API** (a stale `_C.so` silently
  ignores clip planes — every operator then renders identically; see Gotchas).

Data root: `vengine_data` (symlink → `/mnt/uNeon/zhongpai/vengine_data`).
Scenes: `gel intestine kneejoint lower vascular heart` (CT) + `nose hand` (MRI).
GPUs 2–5 only; pin `--cpuset-cpus` to the GPU's CPU block (e.g. gpu2→24-35).

---

## 0. Prerequisites

- Trained per-scene models under `output/xclipgs/<method>/<scene>_900/` for
  `method ∈ {ours, mm, hc, clipgs}` (from the training sweep `dgs_xclipgs.sh`;
  `ours/mm/hc` are dGS with `--clip_operator {analytic,moment,hardcull}`, `clipgs`
  is its own `--mode clipgs` model). Each has `point_cloud/iteration_best/` + `cfg_args`.
- RaRa kernel built once into `raraclipper/_rara_install` (see §2c).

---

## 1. Data preparation (vengine container)

### 1a. Cut-eval cameras + clipped GT → `<scene>_cuteval`

Per scene, 3 fixed clip planes (one per voxel axis, at the volume-core midpoint) ×
2 camera families (5 **perp** = head-on cut face, 5 **graze** = edge-on so the plane
projects to a line), = 30 views. Ground truth is rendered by vengine with the plane
applied, then converted to a NeRF `transforms_test.json` carrying `plane_normal` +
`clip_offset` per frame.

```bash
# host, per scene (issues the vengine docker run itself):
ED=/mnt/uNeon/zhongpai/vengine_data bash vengine_data/_cuteval_batch.sh <scene> 1.25
#   hand needs the rotated-mask flag:
ED=... bash vengine_data/_cuteval_batch.sh hand 1.25 --allow-rotated-mask
```

`_cuteval_batch.sh` runs `xclipgs_gen_cuteval_configs.py` (cameras, `--graze-tilt-deg
1.5` — keep small so graze stays edge-on), `render_chunked.sh` (vengine GT),
bmp→png, `xclipgs_dataset_to_nerf.py` (→ transforms), and `_cuteval_fix_train.py`
(mirror test frames into an empty train split so the ndsplat reader normalizes).

Full fan-out (host, one scene per GPU): `_cuteval_allscenes.sh <scene> <gpu> <cpuset>
<fit> [gen-extra]`.

Output: `nerf_dataset/<scene>_cuteval/{transforms_test.json,test/*.png}`.

### 1b. Unclipped GT at the same cameras → `<scene>_cutevalfull` (needed for CDE)

CDE compares each cut against the *unclipped* render. Re-render the identical
cut-eval cameras with the volume crop removed (patch each config's `maxPoint` to the
full shape — `maxPoint` is the renderer's crop; the `x_threshold*` attrs are
converter metadata only), and strip the clip fields from the transforms.

```bash
ED=/mnt/uNeon/zhongpai/vengine_data bash vengine_data/_cutx_batch.sh <scene>
# fan-out (does 1b for all + downstream): _cutx_allscenes.sh <scene> <gpu> <cpuset>
```

Output: `nerf_dataset/<scene>_cutevalfull/{transforms_test.json,test/*.png}` (uncut).

### 1c. Plane-sweep dataset → `<scene>_cutsweep` (optional; sweep-flicker metric)

One fixed **perp** camera per axis, K=41 frames differing only in `clip_offset`
(plane sweeps P25→P75 of the splat mass). Zero parallax between frames, so any
per-step change beyond the moving slab is cull flicker.

```bash
# ndgs container:
python scripts/benchmarks/xclipgs_make_cutsweep.py <scene> --data-root /data --steps 41 --family perp
```

---

## 2. Inference — render trained models at the cut-eval cameras

All render outputs land at
`output/xclipgs/<tag>/<scene>/test/ours_best/renders/{00000..}.png`, index-ordered
in `transforms_test.json` frame order (the reader does NOT sort; metric scripts rely
on this index↔frame mapping).

### 2a. Table-1/2 methods, each on its OWN interior (ndgs container)

`render_cuteval.sh` builds a throwaway model dir that symlinks the trained
`point_cloud` and points `source_path` at the `_cuteval` dataset, then runs
`render.py`. **`render.py` reads `clip_operator` from the model's `cfg_args`** (see
the critical fix in Gotchas) so each dGS model renders with its trained operator;
`clipgs` uses its own render path.

```bash
# ndgs container, PYTHONPATH set, XCLIPGS_DATA_ROOT=/data XCLIPGS_OUT=/data/output/xclipgs
bash scripts/benchmarks/render_cuteval.sh <scene> ours clipgs mm hc
```
Also render the unclipped companions (for CDE) with `render_cutx.sh`:
```bash
bash scripts/benchmarks/render_cutx.sh <scene> cutevalfull ours clipgs mm hc
```
Tags produced: `{ours,clipgs,mm,hc}_cuteval` and `..._cutevalfull`.

### 2b. Operator swap — ONE fixed interior through every render-time operator

Isolates the clip rule from the interior. Renders one checkpoint through
`analytic/moment/hardcull` via a per-op `cfg_args` override, and through RaRa (§2c).

```bash
# host: _swap_interior.sh <scene> <gpu> <cpuset> <interior> <tag>
bash vengine_data/_swap_interior.sh <scene> 2 24-35 ours ""    # ours interior  -> Table 3
bash vengine_data/_swap_interior.sh <scene> 2 24-35 hc   HC    # HC interior     -> Table 4 (invariance)
```
Tags: `swap<TAG>_{analytic,moment,hardcull}_{cuteval,cutevalfull}` and
`rara<TAG>_{cuteval,cutevalfull}`; metrics → `cuteval3d<TAG>/`.

### 2c. RaRa (authors' released kernel)

RaRa is render-time only (no training path), driven directly with the dGS
conditioned primitives so it sees the same interior every operator does.

```bash
# build once (isolated prefix; do NOT let it shadow the stock diff_gaussian_rasterization):
docker run --rm --gpus '"device=2"' -v <raraclipper>:/rara \
  -w /rara/gaussian-splatting/submodules/diff-gaussian-rasterization \
  -e TORCH_CUDA_ARCH_LIST=8.0 <ndgs> bash -lc \
  'pip install --target /rara/_rara_install --no-deps --no-build-isolation .'
# render (PYTHONPATH=/rara/_rara_install:...):
python scripts/benchmarks/render_rara.py <scene> cuteval     --data-root /data --interior ours --out-tag rara
python scripts/benchmarks/render_rara.py <scene> cutevalfull --data-root /data --interior ours --out-tag rara
```
Plane sign map: our convention keeps `n·x ≤ τ`; RaRa keeps `n·x + d > 0`, so pass
their clipper `(normal, d) = (−n, τ)` (handled inside `render_rara.py`). The kernel
is patched so `decay_weight` inits to 1.0 (the method's intended "keep" semantics;
the released init-0 silently suppresses near-plane ray-misses).

---

## 3. Metrics (ndgs container; CPU-only for the metric scripts)

### 3a. Band fidelity + leak + edge → `cutplane_results.json`

`cutplane_metrics.py` projects the clip plane into each view
(`l = pinv(P)ᵀ·[n;−τ]`, oriented off the camera so kept = s<0), restricts to a
±`band-px` strip (graze) or the GT foreground (perp), and reports per family:
band **PSNR/SSIM/LPIPS** (perp, vs GT), **leak** (culled-side method energy where GT
is background / near-plane fg), **edge spread** (gradient-weighted std of the
cross-plane luminance change), and **hole/overshoot/cut_error** (pooled energies).

```bash
python scripts/benchmarks/cutplane_metrics.py \
  --gt-transforms /data/nerf_dataset/<scene>_cuteval/transforms_test.json \
  --gt-dir        /data/nerf_dataset/<scene>_cuteval/test \
  --scene <scene> --out /data/output/xclipgs/cuteval/<scene> --band-px 12 --debug \
  --methods ours=<...>/ours_cuteval/<scene>/test/ours_best/renders \
            clipgs=<...>/clipgs_cuteval/<scene>/test/ours_best/renders \
            mm=<...>/mm_cuteval/<scene>/test/ours_best/renders \
            hc=<...>/hc_cuteval/<scene>/test/ours_best/renders
```
`--debug` writes green-band / red-cut-line overlay PNGs per frame (verify placement).

### 3b. CDE — difference-referenced cut error → `cde_results.json`

Compares each cut's binarized removal map (`|unclipped − clipped| > mask-thr`)
against the reference's, as a region symmetric difference; cancels the
reconstruction floor a direct render-vs-reference comparison carries. Reports
`cde` (graze = edge, perp = face), and `under`/`over` splits.

```bash
python scripts/benchmarks/cutplane_cde.py \
  --gt-transforms /data/nerf_dataset/<scene>_cuteval/transforms_test.json \
  --gt-clip-dir   /data/nerf_dataset/<scene>_cuteval/test \
  --gt-full-dir   /data/nerf_dataset/<scene>_cutevalfull/test \
  --scene <scene> --out /data/output/xclipgs/cuteval/<scene> --band-px 12 \
  --methods ours=<clip_renders>:<full_renders>  clipgs=...:...  mm=...:...  hc=...:...
#   (each method value is "<cuteval_renders>:<cutevalfull_renders>")
```

### 3c. CErr3D — geometric mass error (no rendering) → `cerr3d.json`

Closed-form from the trained `.ply`: per Gaussian, kept mass `a·Φ(t)`; HC = whole
keep/drop by center, MM = moment-matched tail, Ours = 0 by construction. Camera- and
interior-independent (property of the operator).

```bash
python scripts/benchmarks/cutplane_cuterror_3d.py        # ours/mm/hc ladder
python scripts/benchmarks/cutplane_cuterror_3d_clipgs.py # ClipGS MLP+cull on the ours cloud
```

### 3d. Sweep-flicker (optional) → `sweep_flicker.json`

```bash
python scripts/benchmarks/cutplane_sweep_flicker.py \
  --transforms /data/nerf_dataset/<scene>_cutsweep/transforms_test.json \
  --scene <scene> --out /data/output/xclipgs/cuteval/<scene> --methods ours=... clipgs=... mm=... hc=...
```

### 3e. Aggregate table

`cutplane_make_table.py` discovers all scenes under `<out-root>/cuteval/`, grafts
`cde_results.json` + `sweep_flicker.json`, and emits `cutplane_table.{md,csv}` with a
per-method average row.

```bash
python scripts/benchmarks/cutplane_make_table.py --out-root /data/output/xclipgs --dest /data/output/xclipgs/cuteval
```

Whole-pipeline fan-outs (host, GPUs 2–5): `_cutx_allscenes.sh` (unclipped GT + method
renders + metrics), `_swap_interior.sh` (operator swap). `_fix_t2.sh` /
`_remetric_t2.sh` re-render + re-metric only `mm`/`hc` (used for the operator-bug fix).

---

## 4. Gotchas (each of these bit us)

1. **`render.py` did not apply `clip_operator`** (fixed 2026-07-19). Before the fix,
   only `train.py` set `gaussians.clip_operator`; `render.py` left it at the default
   `"analytic"`, so **every cut-eval render came out as the exact operator regardless
   of the model's trained operator** — mm/hc cut-face metrics were silently wrong
   (HC/MM leak looked as low as Ours). Fix: `render.py` now sets
   `gaussians.clip_operator = getattr(dataset, "clip_operator", "analytic")` after
   scene load. **Always confirm the render used the intended operator** (e.g. HC leak
   should be large, not ~0). ClipGS is unaffected (own render path); `cutevalfull`
   (uncut) and CErr3D (primitive-based) are operator-independent.
2. **Stale in-tree `_C.so`** for tcgs/gsplat silently ignores clip planes → all
   operators render identically. `render_cuteval.sh`/`render_cutx.sh` preflight-check
   `analytic_clip` + `clip_plane` in `GaussianRasterizationSettings`; rebuild the
   submodule if it fails.
3. **plane→line** must use `l = pinv(P)ᵀ·[n;−τ]` (the projected plane, not the
   vanishing line); orient the sign off the camera (grazing cam sits on the kept side).
4. **perp is face-on** → no meaningful cut line; band = GT foreground, and never draw
   the red debug line on perp overlays.
5. **overlays must always overwrite** — a stale overlay from a different-geometry
   render (e.g. an 8° vs 1.5° graze) will show the wrong band/line.
6. **LPIPS** crashes on a 1px-tall graze band; the bbox is padded to ≥64px.
7. **cut-eval nerf needs a non-empty train split** (`_cuteval_fix_train.py` mirrors
   test frames) or the reader's scene normalization errors on `np.concatenate([])`.
8. **frame↔render index**: metric scripts pair render `{i:05d}.png` with the i-th
   frame of `sorted(frames, key=file_path)`; sweep frames share one `file_path`, so
   sweep renders are consumed in JSON order (do not re-sort).
