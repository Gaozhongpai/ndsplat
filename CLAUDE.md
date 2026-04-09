# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SplatND** is a unified framework for N-dimensional splatting, supporting multiple kernel types, conditioning parameterizations, rasterization backends, and dimensionalities (3D/6D/7D).

### Model Variants

| Mode | Kernel | Conditioning | Color | Description |
|------|--------|-------------|-------|-------------|
| `3dgs` | Gaussian | None (3D) | SH | Original 3D Gaussian Splatting |
| `ndgs` | Gaussian | Full covariance | SH | N-DGS with conditional slicing |
| `ubs` | Beta | Full covariance | SH | Uncertainty-Based Splatting |
| `dgs` | Gaussian | Direct Cholesky | SH | Direct Gaussian Splatting (dGS) |
| `dbs` | Beta | Direct Cholesky | SH | Direct Beta Splatting (dBS) |

### Two Kernel Types

**Gaussian kernel** (`ndgs`, `dgs`, `3dgs`):
- Standard exponential opacity: `alpha * exp(-0.5 * z^T z)`
- Lambda parameters for per-dimension opacity scaling (dGS)

**Beta kernel** (`ubs`, `dbs`):
- Per-dimension bandwidth control: `alpha * prod((1 - tanh(z_i^2))^{beta_i})`
- Beta parameters `[N, C]` enable adaptive bandwidth per conditioning dimension
- Activation: `4.0 * exp(beta_raw)`

### Two Conditioning Parameterizations

**Full covariance** (`ubs`, `ndgs`):
- Full N-D covariance matrix (6x6 or 7x7) with `_scale` and `_l_triangle` parameters
- Conditional slicing via matrix inversion: `Sigma_cond = Sigma_pp - Sigma_pq @ Sigma_qq^{-1} @ Sigma_qp`
- CUDA kernel: `cond_mean_convariance_opacity()` (UBS), `slice_gaussian_ndgs()` (NDGS)

**Direct Cholesky precision** (`dgs`, `dbs`):
- Direct `_L_22_inv` [N, C*(C+1)/2] (Cholesky of precision) and `_v_12` [N, 3*C] (position displacement)
- No matrix inversion needed: `z = L^T @ delta` directly
- CUDA kernel: `slice_gaussian_full()` (dGS), `slice_dbs()` (dBS)
- 5-6x faster slicing than full covariance

### Three Rasterization Backends

- **gsplat**: Custom fork with CUDA kernels for conditional slicing, SH evaluation, Beta kernel support
- **TCGS** (`tcgs_speedy_rasterizer`): Tile-based rasterizer with cutting plane support (`x_threshold`), tight snugbox culling
- **diff-gaussian-rasterization**: Original 3DGS rasterizer for baseline comparison

Select with `--use_gsplat` flag (gsplat vs TCGS for training).

### Dimensionalities

- **3D** (`--input_dim 3` or `--mode 3dgs`): Standard 3DGS, no conditioning
- **6D** (`--input_dim 6`): Position + view direction conditioning
- **7D** (`--input_dim 7`): Position + view direction + time conditioning

## Environment Setup

```bash
conda env create --file environment.yml
conda activate gaussian_splatting

# Install CUDA extensions
pip install submodules/gsplat
pip install submodules/tcgs_speedy_rasterizer
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

To rebuild gsplat after CUDA kernel changes:
```bash
cd submodules/gsplat && pip install .
```

## Training

```bash
# N-DGS (full covariance, Gaussian kernel)
python train.py -s <dataset> --mode ndgs --input_dim 6

# UBS (full covariance, Beta kernel)
python train.py -s <dataset> --mode ubs --input_dim 6

# dGS (direct Cholesky, Gaussian kernel)
python train.py -s <dataset> --mode dgs --input_dim 6

# dBS (direct Cholesky, Beta kernel)
python train.py -s <dataset> --mode dbs --input_dim 6

# 7D with time
python train.py -s <dataset> --mode ndgs --input_dim 7

# MCMC densification
python train.py -s <dataset> --mode ubs --densification_strategy mcmc --mcmc_cap_max 300000

# Use gsplat rasterizer
python train.py -s <dataset> --mode ubs --use_gsplat
```

### Key Training Parameters

**Model:**
- `--mode`: Model variant (`3dgs`, `ndgs`, `dgs`, `dbs`, `ubs`)
- `--input_dim`: Dimensionality (3, 6, or 7)
- `--sh_degree`: SH degree (default: 3)
- `--use_gsplat`: Use gsplat instead of TCGS rasterizer

**dGS-specific:**
- `--use_view_dependent_pos`: View-dependent position shift (default: True)
- `--use_opacity_pos_decouple`: Decouple position and opacity
- `--l_22_inv_init_scale`: Init scale for L_22_inv diagonal (default: 1.0)
- `--lambda_init`: Initial lambda_view/lambda_time (default: -1.2)

**NDGS-specific:**
- `--learnable_lambda_opc`: Per-Gaussian learnable opacity scaling
- `--use_rot_scale_l_triangle`: UBS-style covariance parameterization
- `--lambda_opc`: Default opacity scaling (default: 0.35)

**Densification:**
- `--densification_strategy`: `standard`, `mcmc`, or `fastgs`
- `--mcmc_cap_max`: Max Gaussians for MCMC (default: 300000)

**Training:**
- `--iterations`: Total iterations (default: 30000)
- `--scale_lr`: Scale LR (default: 0.005)
- `--l_triangle_lr`: L-triangle LR (default: 0.001)
- `--white_background` / `-w`: White background (NeRF Synthetic)
- `--eval`: Train/test split

## Rendering and Evaluation

```bash
python render.py -m <model_path>
python metrics.py -m <model_path>
python view.py -m <model_path> --ply <ply_file> --mode ndgs
```

## Project Structure

```
splatnd/
├── train.py, render.py, metrics.py, view.py   # Main pipeline
├── run.sh                                      # Master experiment runner
├── scene/
│   ├── __init__.py                             # get_gaussian_model() factory
│   ├── gaussian_model.py                       # 3DGS baseline
│   ├── gaussian_model_ndgs.py                  # N-DGS (full cov, Gaussian kernel)
│   ├── gaussian_model_ubs_sh.py                # UBS (full cov, Beta kernel, SH)
│   ├── gaussian_model_dgs_full.py              # dGS (direct Cholesky, Gaussian kernel)
│   ├── gaussian_model_dbs_sh.py                # dBS (direct Cholesky, Beta kernel, SH)
│   ├── gaussian_model_dbs.py                   # dBS with direct RGB (non-SH variant)
│   ├── gaussian_model_ubs.py                   # UBS with direct RGB (non-SH variant)
│   ├── gaussian_viewer.py                      # Viser-based live viewer
│   ├── beta_viewer.py                          # Beta-specific viewer controls
│   ├── cameras.py                              # Camera classes
│   └── dataset_readers.py                      # Dataset loading
├── arguments/__init__.py                       # Command-line arguments
├── utils/
│   ├── loss_utils.py                           # Loss functions (L1, SSIM)
│   ├── ndgs_utils.py                           # N-DGS covariance utilities
│   ├── graphics_utils.py                       # Graphics math
│   ├── sh_utils.py                             # Spherical harmonics
│   └── compress_utils.py                       # PNG compression for models
├── submodules/
│   ├── gsplat/                                 # Custom gsplat fork
│   │   └── gsplat/cuda/csrc/                   # CUDA kernels
│   │       ├── slice_gaussian_full_fwd/bwd.cu  # dGS slicing
│   │       ├── slice_dbs_fwd/bwd.cu            # dBS slicing
│   │       ├── spherical_harmonics_fwd/bwd.cu  # SH evaluation
│   │       └── cond_mean_*_fwd/bwd.cu          # UBS conditioning
│   ├── tcgs_speedy_rasterizer/                 # TCGS with cutting planes
│   ├── diff-gaussian-rasterization/            # Original 3DGS rasterizer
│   ├── simple-knn/                             # KNN utilities
│   └── fused-ssim/                             # Fused SSIM
├── scripts/                                    # Experiment runners
│   ├── benchmarks/                             # Mip-NeRF 360, NeRF Synthetic, etc.
│   ├── ablations/                              # Ablation studies
│   └── tests/                                  # Development scripts
└── tools/                                      # Data preprocessing & evaluation
```

## Code Architecture

### Model Factory

`scene/__init__.py` contains `get_gaussian_model(mode)` which maps `--mode` to the model class. All models share a common interface:
- `create_from_pcd(pcd, spatial_lr_scale, mcmc_cap_max=None, densification_strategy="standard")`
- `render(viewpoint_camera, render_mode="RGB", mask=None)` -- gsplat rendering
- `render_tcgs(viewpoint_camera, ...)` -- TCGS rendering
- `training_setup(training_args)`, `update_learning_rate(iteration)`
- `save_ply(path)`, `load_ply(path)`
- MCMC: `relocate_gs(dead_mask)`, `add_new_gs(cap_max)`
- Standard: `densify_and_prune(...)`, `reset_opacity()`

### Training Loop (`train.py`)

- `render_wrapper()` dispatches to the model's `render()` or `render_tcgs()` based on mode and `--use_gsplat`
- Mode checks use substring matching: `"ubs" in mode`, `"dbs" in mode`, `"ndgs" in mode`
- Model initialization: `if "ubs" in mode or "dbs" in mode` -> `GaussianModel(sh_degree, input_dim)`
- SH degree increases every 1000 iterations
- Opacity reset every 3000 iterations (standard densification only)

### gsplat Custom Fork

The gsplat submodule at `submodules/gsplat/` is a custom fork with additional CUDA kernels:

**Key Python wrappers** (`gsplat/cuda/_wrapper.py`):
- `slice_gaussian_full()` -- dGS conditional slicing
- `slice_dbs()` -- dBS conditional slicing
- `spherical_harmonics()` -- SH evaluation
- `cond_mean_convariance_opacity()` -- UBS full covariance conditioning
- `rasterization()` -- main rendering entry point (supports `shs`/`sh_degree` params)

**Key exports** (`gsplat/__init__.py`):
- `rasterization`, `slice_gaussian_full`, `slice_dbs`, `spherical_harmonics`
- `l_triangle_to_rotmat`, `rot_scale_l_triangle_to_covar`, `quat_scale_to_covar_preci`

### Parameter Naming Convention

All models use unified names:
- `_xyz`: Positions [N, 3]
- `_scale`: Scale parameters (softplus activation for UBS/dBS, exp for NDGS)
- `_l_triangle`: Lower-triangular rotation params
- `_features_dc`, `_features_rest`: SH coefficients (SH models)
- `_rgb`: Direct RGB colors (non-SH models like `gaussian_model_dbs.py`)
- `_opacity`: Base opacity (sigmoid activation)
- `_mean`: Conditioning mean [N, C]
- `_beta`: Beta parameters [N, C] (Beta kernel models)
- `_L_22_inv`: Cholesky precision [N, C*(C+1)/2] (direct models)
- `_v_12`: Position displacement [N, 3*C] (direct models)

### Densification Strategies

**Standard** (`--densification_strategy standard`): Gradient-based clone/split/prune. All models support this.

**MCMC** (`--densification_strategy mcmc`): Relocate dead Gaussians + add new ones up to `cap_max`. Models need `relocate_gs()` and `add_new_gs()` methods (UBS, dBS, dBS-SH have these).

## Dataset Format

### COLMAP
```
<dataset>/images/ + sparse/0/{cameras,images,points3D}.bin
```

### NeRF Synthetic / JSON
```json
{"camera_angle_x": 0.857, "frames": [{"file_path": "...", "transform_matrix": [...], "timestamp": 0.5, "x_threshold": 5.0}]}
```

## Common Issues

- **VRAM**: Reduce `--densify_grad_threshold`, increase `--densification_interval`, use `--test_iterations -1`
- **NaN losses**: Reduce `--scale_lr`, `--l_triangle_lr`
- **Building CUDA**: Ensure CUDA toolkit matches PyTorch CUDA version
- **Rebuilding gsplat**: `cd submodules/gsplat && pip install .` after CUDA kernel changes

## Citation

Based on 6DGS (Gao et al., 2024), 7DGS (Gao et al., ICCV 2025), and UBS (Liu et al., 2025).
