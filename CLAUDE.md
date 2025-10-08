# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **6D Gaussian Splatting (6DGS)**, an extension of the original 3D Gaussian Splatting method for real-time radiance field rendering. The codebase builds on the GRAPHDECO group's 3D Gaussian Splatting foundation and implements multiple Gaussian model variants including:

- **3DGS**: Original 3D Gaussian Splatting
- **NDGS**: N-dimensional Gaussian Splatting with spherical harmonics
- **DDGS**: Double D Gaussian Splatting
- **X-Gaussian**: Alternative Gaussian representation
- **Combined/DDNDGS**: Hybrid models

The current active mode is **NDGS** (configured in [scene/__init__.py:21](scene/__init__.py#L21)).

## Environment Setup

### Installation

```bash
# Create conda environment
conda env create --file environment.yml
conda activate gaussian_splatting

# Install custom CUDA extensions (submodules)
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

**Requirements:**
- CUDA SDK 11.6+ (environment.yml specifies 11.6)
- Python 3.7.13
- PyTorch 1.12.1 with CUDA support
- 24GB+ VRAM recommended for full training

### Custom Rasterizers

The project includes multiple rasterization backends in `submodules/`:
- `diff-gaussian-rasterization`: Original differentiable Gaussian rasterizer
- `diff-ddgs-rasterization`: DDGS-specific rasterizer
- `tcgs_speedy_rasterizer`: Fast TCGS rasterizer with **cutting plane** and **beta splatting** support
- `tcgs_speedy_rasterizer_beta`: Reference implementation for beta splatting (features integrated into main TCGS)

**TCGS Rasterizer Features:**

The TCGS rasterizer has been enhanced with multiple advanced features:

1. **Cutting Plane Support** (`x_threshold` parameter)
   - Truncates Gaussians at a specified world-space X coordinate
   - Uses error function-based truncation for smooth transitions
   - Adjusts mean and covariance to account for truncation
   - Gradients properly backpropagate through cutting plane

2. **Beta Splatting** (`betas` parameter)
   - Alternative to exponential Gaussian falloff using power-law: `α = opacity · (1-σ)^β`
   - Standard mode (when `betas=nullptr`): `α = opacity · exp(-0.5·σ)`
   - Provides learnable per-Gaussian shape control via β parameter
   - Automatic snugbox cutoff computation: `σ_max = 1 - (α_thresh/opacity)^(1/β)`
   - Full gradient support for β parameter

3. **SnugBox Tile Culling**
   - Computes exact ellipse-tile intersections instead of conservative radius bounds
   - Adaptively processes tiles by Y or X slices depending on ellipse aspect ratio
   - Significantly reduces tile over-coverage and improves performance

4. **Optimizations Applied:**
   - **Fixed variance clamping bug** ([backward.cu:242](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/backward.cu#L242)): Corrected `var_expr_clamped` computation for accurate gradients
   - **Early exit optimization** ([backward.cu:137-142](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/backward.cu#L137)): Skips gradient computation when magnitudes < 1e-8
   - **Alpha saturation handling**: Properly zeros gradients when α > 0.99 to prevent numerical instability

**Usage Modes:**
```python
# Create rasterization settings
raster_settings = GaussianRasterizationSettings(
    image_height=512,
    image_width=512,
    tanfovx=1.0,
    tanfovy=1.0,
    bg=bg_color,
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=3,
    campos=camera_center,
    x_threshold=float('inf'),  # Optional: default is inf (no cutting)
    use_tcgs=True,              # Optional: default is True (TCGS fast path)
    prefiltered=False,          # Optional: default is False
    debug=False                 # Optional: default is False
)

# Standard Gaussian splatting (TCGS fast)
output = rasterizer(means3D, means2D, opacities, scores, shs=shs, scales=scales, rotations=rotations)

# Beta splatting mode
output = rasterizer(means3D, means2D, opacities, scores, shs=shs, scales=scales, rotations=rotations, betas=beta_values)

# Use standard rendering (FP32, slower but more accurate)
raster_settings = GaussianRasterizationSettings(..., use_tcgs=False)

# Cutting plane only
raster_settings = GaussianRasterizationSettings(..., x_threshold=threshold_value)

# Combined: cutting plane + beta splatting
output = rasterizer(..., betas=beta_values)  # with x_threshold in settings
```

**Important Notes:**
- **Both standard and TCGS Tensor Core paths** now support beta splatting and cutting plane features
- **Runtime TCGS selection** - You can now switch between TCGS and standard rendering **without recompilation**:
  - Set `use_tcgs=True` in `GaussianRasterizationSettings` for TCGS Tensor Core path (default)
  - Set `use_tcgs=False` for standard rendering path
  - No need to modify [config.h:19](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/config.h#L19) anymore
- **TCGS Tensor Core path** (`use_tcgs=True`)
  - Fully supports beta splatting with FP16 operations and shared memory optimization
  - Uses specialized matrix operations (mma_16x8x8_f16_f16) for maximum performance
  - Implements dual-mode blending: standard exponential vs. power-law falloff
  - Runtime feature detection via nullptr checks (betas=nullptr → standard mode)
- **Standard rendering path** (`use_tcgs=False`)
  - Full-precision (FP32) rendering
  - May be preferred during training for maximum accuracy
- Features are backward compatible: when `betas=nullptr`, the system uses standard Gaussian rendering

**Key Implementation Files:**
- [auxiliary.h:182](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/auxiliary.h#L182): `computeSnugboxCutoff()` - computes cutoff for beta/standard modes
- [forward.cu:157](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/forward.cu#L157): Preprocess kernel with beta support
- [forward.cu:357](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/forward.cu#L357): Render kernel with dual-mode alpha computation
- [backward.cu:714](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/backward.cu#L714): Backward kernel with beta gradients

To rebuild after modifications:
```bash
cd submodules/tcgs_speedy_rasterizer
python setup.py build_ext --inplace
```

## Training

### Basic Training

```bash
# Train on COLMAP or NeRF Synthetic dataset
python train.py -s <path_to_dataset> --eval

# Train with custom output directory
python train.py -s <dataset_path> --model_path <output_path> --eval
```

### Important Training Parameters

- `--eval`: Use train/test split for evaluation (MipNeRF360-style)
- `--iterations`: Total iterations (default: 30,000)
- `--white_background` / `-w`: Use white background (for NeRF Synthetic datasets)
- `--resolution` / `-r`: Image resolution (1=original, 2=half, 4=quarter, 8=eighth)
- `--data_device`: Where to store images (`cuda` or `cpu`, use `cpu` for large datasets to reduce VRAM)

### Training Scripts for Batch Processing

Multiple shell scripts are provided for running experiments on different datasets:

- `run_datasets_6dgs_nerf.sh`: NeRF Synthetic dataset experiments
- `run_datasets_6dgs_360.sh`: MipNeRF360 dataset experiments
- `run_datasets_6dgs_tank.sh`: Tanks & Temples experiments
- `run_datasets_6dgs_shiny.sh`: Shiny dataset experiments
- `run_datasets_ablation.sh`: Ablation studies

These scripts iterate through dataset subdirectories, train models, render at multiple checkpoints (500, 2000, 7000, 15000, 30000 iterations), and compute metrics.

## Rendering and Evaluation

### Render Trained Models

```bash
# Render both train and test sets
python render.py -m <model_path>

# Render specific iteration
python render.py -m <model_path> --iteration 30000

# Skip rendering training set
python render.py -m <model_path> --skip_train
```

**Note:** The codebase uses `render_flash` by default ([render.py:18](render.py#L18)) for faster rendering.

### Compute Metrics

```bash
# Compute PSNR, SSIM, LPIPS on rendered images
python metrics.py -m <model_path>

# For multiple models
python metrics.py -m <path1> <path2> <path3>
```

Alternative metrics implementations:
- `metrics.py`: Standard metrics
- `metrics_ndgs.py`: NDGS-specific metrics

### Full Evaluation Pipeline

```bash
# Complete evaluation (train + render + metrics)
python full_eval.py -m360 <mipnerf360_folder> -tat <tanks_temples_folder> -db <deep_blending_folder>

# Evaluate pre-trained models (skip training)
python full_eval.py -o <pretrained_models_dir> --skip_training -m360 <mipnerf360_folder> ...

# Compute metrics only (skip training and rendering)
python full_eval.py -m <images_dir> --skip_training --skip_rendering
```

## Code Architecture

### Core Components

1. **Scene Management** (`scene/`)
   - `Scene`: Loads datasets, manages cameras, handles train/test splits
   - Multiple `GaussianModel` implementations for different Gaussian representations
   - `dataset_readers.py`: Parsers for COLMAP, NeRF Synthetic, and custom formats
   - `cameras.py`: Camera parameter handling

2. **Gaussian Models** (`scene/gaussian_model_*.py`)
   - `gaussian_model.py`: Base 3DGS model
   - `gaussian_model_ndgs_sh.py`: Current active model (NDGS with spherical harmonics)
   - `gaussian_model_ddgs.py`, `gaussian_model_combined.py`, etc.: Variant implementations
   - `gaussian_model_dgs.py`: DGS-specific implementation

   **To switch models:** Modify the `MODE` variable in [scene/__init__.py:21](scene/__init__.py#L21)

3. **Rendering** (`gaussian_renderer/`)
   - `__init__.py`: Main differentiable rendering functions
   - `network_gui.py`: Real-time viewer communication

4. **Utilities** (`utils/`)
   - `ndgs_utils.py`: NDGS-specific utilities (Cholesky decomposition, covariance handling)
   - `loss_utils.py`: Loss functions (L1, SSIM)
   - `graphics_utils.py`: Graphics math (FOV, projection, point clouds)
   - `sh_utils.py`: Spherical harmonics utilities

5. **Training Arguments** (`arguments/`)
   - `ModelParams`: Dataset/model paths, resolution, background color
   - `PipelineParams`: Rendering pipeline configuration
   - `OptimizationParams`: Learning rates, densification parameters

### Key Training Loop Details

The training loop ([train.py](train.py)) includes:
- **Adaptive densification**: Points are densified based on position gradient threshold (controlled by `--densify_grad_threshold`, `--densification_interval`, `--densify_from_iter`, `--densify_until_iter`)
- **Opacity reset**: Periodic opacity reset every 3000 iterations (by default)
- **Spherical harmonics degree increase**: SH degree increases every 1000 iterations
- **Multi-view training**: Can train with multiple views per iteration (`pipe.mv` parameter)
- **Network viewer support**: Can connect SIBR viewer during training for real-time visualization

### NDGS-Specific Architecture

The NDGS model uses:
- **6D covariance representation**: Stores lower-triangular Cholesky decomposition (`_diags`, `_l_triangs`)
- **Projection vectors**: 8 projection vectors per Gaussian for direction-aware rendering
- **Custom rasterization**: Uses modified rasterization with covariance handling
- Functions in `utils/ndgs_utils.py`: `create_cholesky()`, `create_cholesky_v2()`, `strip_lower_diag()`

## Dataset Structure

### Expected COLMAP Format

```
<dataset>/
├── images/           # Input images
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
```

### NeRF Synthetic Format

```
<dataset>/
├── transforms_train.json
├── transforms_test.json
├── train/            # Training images
└── test/             # Test images
```

### Data Preprocessing

```bash
# Convert images to COLMAP format with undistortion
python convert.py -s <location> [--resize]

# Skip COLMAP matching if COLMAP data already exists
python convert.py -s <location> --skip_matching [--resize]
```

The repository includes a preprocessing script for cloud datasets: [cloud_dataset_preprocssing.py](cloud_dataset_preprocssing.py)

## SIBR Viewer (Interactive Visualization)

### Build SIBR Viewer

```bash
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
```

### Real-time Viewer

```bash
# View trained model
./SIBR_viewers/bin/SIBR_gaussianViewer_app -m <model_path>

# Specify resolution
./SIBR_viewers/bin/SIBR_gaussianViewer_app -m <model_path> --rendering-size 1920 1080
```

### Network Viewer (During Training)

```bash
# Connect to running training process
./SIBR_viewers/bin/SIBR_remoteGaussian_app

# Specify connection
./SIBR_viewers/bin/SIBR_remoteGaussian_app --ip <ip> --port 6009
```

## Development Notes

### Switching Between Model Variants

To switch between different Gaussian model implementations, edit the `MODE` variable in [scene/__init__.py:21](scene/__init__.py#L21):

```python
MODE = "ndgs"  # Options: "3dgs", "ndgs", "ddgs", "ddndgs", "combined", "x-gaussian"
```

This controls which `GaussianModel` class is imported and used throughout the codebase.

### Output Structure

Trained models are saved to `output/<model_name>/` with:
- `point_cloud/iteration_<N>/point_cloud.ply`: Gaussian parameters at iteration N
- `cameras.json`: Camera parameters
- Rendered images in subdirectories when using `render.py`

### Git Submodules

The repository uses git submodules for custom CUDA extensions. When cloning:

```bash
git clone <repo_url> --recursive

# Or if already cloned:
git submodule update --init --recursive
```

**Note:** The `.gitmodules` file has been modified (shown in git status) and some submodules (like `simple-knn`) have been reorganized.

## Common Issues

- **VRAM limitations**: Reduce `--densify_grad_threshold`, increase `--densification_interval`, or decrease `--densify_until_iter`. Set `--test_iterations -1` to avoid testing memory spikes.
- **Large-scale scenes**: Lower `--position_lr_init`, `--position_lr_final`, and `--scaling_lr` (try 0.3x or 0.1x of defaults)
- **Building CUDA extensions**: Ensure CUDA toolkit version matches PyTorch CUDA version and Visual Studio is properly configured (Windows)

## Citation

Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., ACM TOG 2023).
