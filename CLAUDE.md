# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **N-Dimensional Gaussian Splatting (N-DGS)**, an extension of 3D Gaussian Splatting for real-time radiance field rendering with view-dependent effects. The codebase implements multiple Gaussian model variants with focus on:

- **N-DGS**: N-dimensional (6D/7D) Gaussian Splatting with conditional slicing
- **Dual SH Support**: Multi-view consistency via interpolatable spherical harmonics
- **3DGS**: Original 3D Gaussian Splatting baseline
- **DDGS**: Deformable DGS variant
- **UBS**: Uncertainty-Based Splatting with beta parameters for bandwidth control

The current primary modes are **ndgs** and **ndgs-2sh** (configured via `--mode` argument).

### UBS vs NDGS: Key Architectural Differences

Both UBS and NDGS extend 3DGS to N-dimensions (6D/7D) but take fundamentally different approaches to view-dependent rendering:

**NDGS (N-Dimensional Gaussian Splatting)** - [gaussian_model_ndgs.py](scene/gaussian_model_ndgs.py):
- **Color Representation**: Spherical harmonics (DC + rest coefficients) for rich view-dependent appearance
- **View-Dependence Method**: Conditional Gaussian slicing via `slice_gaussian_ndgs()` CUDA kernel
- **Opacity Control**: Lambda opacity parameter (learnable via `--learnable_lambda_opc` or fixed at 0.35)
- **Parametrization**: Flexible - supports both NDGS-style (diagonal-l_triangle) and UBS-style (rot-scale-l_triangle)
- **Covariance Construction**: CUDA-accelerated with test-time optimization (precomputed `v_22_inv`, `v_regr`, `cov3D_precomp`)
- **Mean Parameter**: Normalized direction vector [N, 3] (+ time [N, 1] for 7DGS)
- **Best For**: General N-D scene representation, multi-view consistency, flexible parametrization, SH appearance

**UBS (Uncertainty-Based Splatting)** - [gaussian_model_ubs.py](scene/gaussian_model_ubs.py):
- **Color Representation**: Direct RGB values (no SH encoding) for simpler, faster evaluation
- **View-Dependence Method**: Beta-adjusted conditional covariance with per-dimension bandwidth control
- **Opacity Control**: Beta parameters [N, input_dim-2] for uncertainty modeling (spatial + view dimensions)
- **Parametrization**: Fixed rot-scale-l_triangle with beta-scaled regression
- **Covariance Construction**: Beta adjustment `v_regr * beta_adj` with clamping `torch.clamp_max(beta/4.0, 1.0)`
- **Mean Parameter**: Random uniform [-1, 1] in non-spatial dimensions
- **Best For**: Uncertainty quantification, direct RGB control, bandwidth-aware view synthesis, simplified color model

**Key Implementation Differences:**

1. **Rendering Pipeline**:
   ```python
   # NDGS (ndgs.py lines 869-879)
   m_cond, cov3D_precomp, pdf_cond = self.slice_gaussian(cond_params, c_dim=3, lambda_opc=lambda_opc)
   opacity = self.get_opacity * pdf_cond  # Optional: * self.get_lambda_opc

   # UBS (ubs.py lines 1024-1028)
   means, convs, opacities = self.get_cond_mean_convariance_opacity(query)
   # Beta controls covariance bandwidth via beta-adjusted regression
   ```

2. **Conditional Covariance**:
   ```python
   # NDGS: CUDA kernel (gsplat.slice_gaussian_ndgs)
   # Clean slicing with lambda_opc scaling

   # UBS: Beta-adjusted (ubs.py lines 866-872)
   v_regr = torch.bmm(v_12, v_22_inv)
   v_regr_beta = v_regr * beta_adj.unsqueeze(1)  # Per-dimension beta scaling
   v_cond = v_11 - torch.bmm(v_regr_beta, v_21)
   ```

3. **Viewer Filtering**:
   ```python
   # NDGS: Opacity-based (ndgs.py lines 951-979)
   mask = opacity > opacity_threshold  # Or percentile-based

   # UBS: Beta-based (ubs.py lines 1201-1225)
   mask = quantile_mask(self._beta, b_xyz=(0,100), b_view=(0,100), b_time=(0,100))
   ```

**When to Use Which:**
- **NDGS**: Standard novel view synthesis, need SH appearance modeling, want flexible parametrization options, require test-time optimization
- **UBS**: Uncertainty-aware rendering, direct RGB is sufficient, need per-dimension bandwidth control, want simpler color representation
- **NDGS + `--use_rot_scale_l_triangle`**: Hybrid approach using NDGS slicing with UBS-style covariance parametrization

## Recent Major Changes (v3.0)

### Unified Architecture & Cleanup
- **Consolidated models**: Two main N-DGS implementations
  - `gaussian_model_ndgs.py`: Single SH for standard rendering
  - `gaussian_model_ndgs_2sh.py`: Dual SH for multi-view consistency
- **Removed legacy code**: Cleaned out experimental features (color_net, LSH culling, Taichi imports)
- **Both models support**: 6DGS and 7DGS (with time dimension)

### New Features
- **Learnable Lambda Opacity** (`--learnable_lambda_opc`): Per-Gaussian learnable opacity scaling
- **Full 7DGS Time Support**: Complete temporal dimension with:
  - `_mean_time` parameter
  - Viewer time animation controls (auto-loop, manual slider)
  - Timestamp passing to rendering pipeline
- **Dual SH Blending**: Real-time color interpolation in viewer (ndgs-2sh only)

### Unified Parameter Naming
**IMPORTANT**: Parameters have been renamed for consistency:
- **Old names**: `diags`, `l_triangs`
- **New names**: `_scale`, `_l_triangle`
- **Backward compatible**: PLY files with old names load automatically
- **Learning rates**: `--scale_lr`, `--l_triangle_lr` (old `--diags_lr`, `--l_triangs_lr` still work)

The semantic meaning differs by parametrization:
- **NDGS-style** (`use_rot_scale_l_triangle=False`): `_scale` = diagonal elements with exp() activation
- **UBS-style** (`use_rot_scale_l_triangle=True`): `_scale` = scale parameters with softplus() activation

### Performance Optimizations
- **Viewer masking**: Now uses efficient pointer swaps instead of tensor copying
  - Old approach: Save all tensors, modify `self._xyz`, restore (slow)
  - New approach: Create masked views, swap Python references, restore pointers (fast)
  - Location: `view_tcgs()` method in both model files (lines ~1088-1148 in ndgs.py, ~1146-1205 in ndgs_2sh.py)
- **Test mode**: Precomputed values (`direction`, `v_22_inv`, `v_regr`, `cov3D_precomp`, `shs`) handled correctly
- **No memory allocation per frame** in viewer

## Environment Setup

### Installation

```bash
# Create conda environment
conda env create --file environment.yml
conda activate gaussian_splatting

# Install CUDA extensions (submodules)
pip install submodules/gsplat                 # N-DGS operations
pip install submodules/tcgs-speedy-rasterizer  # TCGS rasterizer
pip install submodules/simple-knn              # KNN utilities
```

**Requirements:**
- CUDA SDK 11 or 12
- Python 3.8+
- PyTorch with CUDA support
- 24GB+ VRAM recommended for full training

### Custom Rasterizers

The project includes multiple rasterization backends in `submodules/`:
- `tcgs_speedy_rasterizer`: Fast TCGS rasterizer with **cutting plane** support
- `gsplat`: N-DGS conditional slicing operations
- `simple-knn`: KNN for initialization

**TCGS Rasterizer Features:**
- Cutting plane support (`x_threshold` parameter)
- Tight snugbox tile culling
- FP16 tensor core operations
- Runtime mode selection (`use_tcgs` parameter)

To rebuild after modifications:
```bash
cd submodules/tcgs_speedy_rasterizer
python setup.py build_ext --inplace
```

## Training

### Basic Training

```bash
# Train N-DGS model (6D)
python train.py -s <path_to_dataset> --mode ndgs --input_dim 6

# Train with 7DGS (time dimension)
python train.py -s <path_to_dataset> --mode ndgs --input_dim 7

# Train with learnable lambda opacity
python train.py -s <path_to_dataset> --mode ndgs --learnable_lambda_opc

# Train with UBS-style parametrization
python train.py -s <path_to_dataset> --mode ndgs --use_rot_scale_l_triangle

# Train dual SH model
python train.py -s <path_to_dataset> --mode ndgs-2sh --input_dim 6
```

### Important Training Parameters

**Model Configuration:**
- `--mode`: Model architecture (`ndgs`, `ndgs-2sh`, `ddgs`, `3dgs`, `ubs`)
- `--input_dim`: Dimensionality (6 for 6DGS, 7 for 7DGS with time)
- `--learnable_lambda_opc`: Enable per-Gaussian learnable opacity scaling
- `--use_rot_scale_l_triangle`: Use UBS-style parametrization instead of NDGS-style

**Training Control:**
- `--iterations`: Total iterations (default: 30,000)
- `--position_lr_init`: Initial position LR (default: 0.00016)
- `--scale_lr`: Scale/diagonal parameters LR (default: 0.005) - replaces `--diags_lr`
- `--l_triangle_lr`: L-triangle parameters LR (default: 0.001) - replaces `--l_triangs_lr`
- `--white_background` / `-w`: Use white background (for NeRF Synthetic)
- `--eval`: Use train/test split for evaluation

**Viewer:**
- `--port`: Viewer port (default: 8080)
- `--disable_viewer`: Disable live viewer

### Training Scripts for Batch Processing

Experiment scripts are organized in the `scripts/` directory by category:

**Use the master run script:**
```bash
./run.sh <category> <script>

# Examples:
./run.sh benchmark mipnerf360       # Run Mip-NeRF 360 benchmark
./run.sh ablation nerf_synthetic    # Run ablation study
./run.sh test ct_data                # Run quick test

# See all options:
./run.sh --help
./run.sh list
```

**Organization:**
- `scripts/benchmarks/` - Standard evaluation scripts (mipnerf360, nerf_synthetic, tanks_temples, shiny_blender)
- `scripts/ablations/` - Ablation study scripts (nerf_synthetic, deepdrr, deepdrr_entangled)
- `scripts/tests/` - Development and debugging scripts (ct_data, dicom, etc.)

See [scripts/README.md](scripts/README.md) for detailed documentation.

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

### Live Viewing

```bash
# View trained model interactively
python view.py -m <model_path> --ply <ply_file> --mode ndgs

# Custom port
python view.py --ply <ply_file> --mode ndgs --port 8080
```

### Compute Metrics

```bash
# Compute PSNR, SSIM, LPIPS on rendered images
python metrics.py -m <model_path>

# For multiple models
python metrics.py -m <path1> <path2> <path3>
```

Alternative metrics implementations:
- `metrics.py`: Standard metrics (root level)
- `tools/evaluation/metrics_ndgs.py`: NDGS-specific metrics

### Full Evaluation Pipeline

```bash
# Complete evaluation (train + render + metrics)
python tools/evaluation/full_eval.py -m360 <mipnerf360_folder> -tat <tanks_temples_folder> -db <deep_blending_folder>

# Evaluate pre-trained models (skip training)
python tools/evaluation/full_eval.py -o <pretrained_models_dir> --skip_training -m360 <mipnerf360_folder> ...

# Compute metrics only (skip training and rendering)
python tools/evaluation/full_eval.py -m <images_dir> --skip_training --skip_rendering
```

See [tools/README.md](tools/README.md) for more utility scripts.

## Project Structure

```
6dgs-iclr/
├── Core Scripts (Root)
│   ├── train.py, render.py, metrics.py, view.py  # Main pipeline
│   └── run.sh                                      # Master experiment runner
│
├── tools/                               # Utility scripts
│   ├── preprocessing/                   # Data preparation
│   │   ├── colmap_convert.py
│   │   └── cloud_dataset_preprocessing.py
│   └── evaluation/                      # Evaluation tools
│       ├── full_eval.py
│       └── metrics_ndgs.py
│
├── scripts/                             # Organized experiment runners
│   ├── benchmarks/                      # Standard evaluations (4 scripts)
│   ├── ablations/                       # Ablation studies (3 scripts)
│   └── tests/                           # Dev/debug (7 scripts)
│
└── Model Code
    ├── scene/                           # Scene & Gaussian models
    ├── gaussian_renderer/               # Rendering engine
    ├── utils/                           # Helper functions
    └── submodules/                      # CUDA extensions
```

See [ORGANIZATION.md](ORGANIZATION.md) for complete documentation.

## Code Architecture

### Core Components

1. **Scene Management** (`scene/`)
   - `Scene`: Loads datasets, manages cameras, handles train/test splits
   - Multiple `GaussianModel` implementations for different representations
   - `dataset_readers.py`: Parsers for COLMAP, NeRF Synthetic, and custom formats
   - `cameras.py`: Camera parameter handling

2. **Gaussian Models** (`scene/gaussian_model_*.py`)
   - **`gaussian_model_ndgs.py`**: Primary N-DGS implementation (single SH)
     - Supports both 6DGS and 7DGS
     - Learnable lambda opacity support
     - Unified naming with dual parametrization
     - Optimized viewer with pointer-swap masking

   - **`gaussian_model_ndgs_2sh.py`**: Dual SH N-DGS implementation
     - Two sets of SH features for multi-view consistency
     - Color interpolation in viewer
     - Same structure as ndgs.py except for dual SH handling

   - **`gaussian_model.py`**: Base 3DGS model
   - **`gaussian_model_ddgs.py`**: DDGS variant
   - **`gaussian_model_ubs.py`**: UBS-specific implementation

   **To switch models:** Use `--mode` argument: `ndgs`, `ndgs-2sh`, `ddgs`, `3dgs`, `ubs`

3. **Live Viewer** (`scene/gaussian_viewer.py`)
   - Viser-based real-time training viewer
   - Features:
     - Time animation controls (7DGS only)
     - Dual SH color blending slider (ndgs-2sh only)
     - Opacity filtering (percentile or absolute threshold)
     - Cutting plane support
     - Render mode switching (RGB, Alpha, Depth, Normal)
     - FPS monitoring
   - **Performance**: Optimized masking with tensor views (no per-frame allocation)

4. **Rendering** (`gaussian_renderer/`)
   - `__init__.py`: Main differentiable rendering functions
   - `network_gui.py`: Real-time viewer communication (legacy SIBR support)

5. **Utilities** (`utils/`)
   - `ndgs_utils.py`: N-DGS-specific utilities (Cholesky decomposition, covariance handling)
   - `loss_utils.py`: Loss functions (L1, SSIM)
   - `graphics_utils.py`: Graphics math (FOV, projection, point clouds)
   - `sh_utils.py`: Spherical harmonics utilities

6. **Training Arguments** (`arguments/__init__.py`)
   - `ModelParams`: Dataset/model paths, resolution, background color, **learnable_lambda_opc flag**
   - `PipelineParams`: Rendering pipeline configuration
   - `OptimizationParams`: Learning rates (updated parameter names), densification parameters

### Key Training Loop Details

The training loop ([train.py](train.py)) includes:
- **Adaptive densification**: Two strategies available (see below)
- **Opacity reset**: Periodic opacity reset every 3000 iterations (standard strategy only)
- **Spherical harmonics degree increase**: SH degree increases every 1000 iterations
- **Live viewer**: Web-based real-time monitoring with Viser
- **7DGS time support**: Timestamp handling throughout pipeline

### Densification Strategies

The codebase now supports two densification strategies controlled by `--densification_strategy`:

**Standard Densification** (`--densification_strategy standard`, default):
- **Method**: Gradient-based clone, split, and prune
- **Clone**: Small Gaussians with high gradients are duplicated
- **Split**: Large Gaussians with high gradients are split into N pieces (default N=2)
- **Prune**: Low opacity and oversized Gaussians are removed
- **Opacity Reset**: Periodic reset every 3000 iterations
- **Parameters**:
  - `--densify_grad_threshold 0.0002`: Gradient threshold for densification
  - `--densification_interval 100`: Frequency of densification checks
  - `--densify_from_iter 500`: Start densification iteration
  - `--densify_until_iter 15000`: Stop densification iteration
  - `--opacity_reset_interval 3000`: Frequency of opacity reset
- **Best For**: Standard 3DGS, NDGS, DDGS models; proven stable convergence
- **Location**: `densify_and_prune()` method in all Gaussian models

**MCMC Densification** (`--densification_strategy mcmc`):
- **Method**: Markov Chain Monte Carlo sampling-based refinement
- **Relocate**: Dead Gaussians (opacity < 0.005) are relocated by sampling from alive ones
- **Add**: New Gaussians are added incrementally up to a maximum cap
- **No Pruning**: Gaussians are relocated rather than removed
- **No Opacity Reset**: Opacity is maintained and used for sampling probabilities
- **Parameters**:
  - `--mcmc_cap_max 300000`: Maximum number of Gaussians
  - `--mcmc_refine_interval 100`: Frequency of MCMC refinement
  - `--mcmc_add_rate 0.25`: Rate of adding new Gaussians (currently unused, hardcoded to 1.02x)
  - `--mcmc_remove_rate 0.1`: Rate of removing Gaussians (currently unused)
- **Best For**: UBS model (currently only UBS has `relocate_gs()` and `add_new_gs()` methods)
- **Location**: `relocate_gs()` and `add_new_gs()` in [gaussian_model_ubs.py](scene/gaussian_model_ubs.py:717-792)

**Usage Examples:**

```bash
# Standard densification (default)
python train.py -s <dataset> --mode ndgs --densification_strategy standard

# MCMC densification with UBS
python train.py -s <dataset> --mode ubs --densification_strategy mcmc --mcmc_cap_max 300000

# MCMC densification with custom parameters
python train.py -s <dataset> --mode ubs --densification_strategy mcmc \
    --mcmc_cap_max 500000 \
    --mcmc_refine_interval 50
```

**Implementation Notes:**

1. **Fallback Behavior**: If MCMC is requested for a model that doesn't support it (e.g., NDGS), the training script falls back to standard densification with a warning.

2. **MCMC Algorithm** (UBS):
   - **Relocation**: `relocate_gs()` samples from alive Gaussians weighted by opacity, adjusts opacity using `1.0 - (1.0 - opacity)^(1/(ratio+1))`, and copies to dead locations
   - **Addition**: `add_new_gs()` grows the Gaussian count to `min(cap_max, 1.02 * current_count)` by sampling from existing Gaussians

3. **Standard Algorithm** (all models):
   - **Clone**: Duplicates small Gaussians (scale < 1% of scene extent) with high gradients
   - **Split**: Samples N new positions from large Gaussians (scale > 1% of scene extent) with high gradients, scales them down by 0.8x
   - **Prune**: Removes Gaussians with opacity < threshold or screen-space radius > threshold

### N-DGS-Specific Architecture

The N-DGS models use:
- **N-D covariance representation**: 6×6 or 7×7 full covariance matrices
- **Unified parameter naming**: `_scale` and `_l_triangle` (replaces old `diags`/`l_triangs`)
- **Dual parametrization support**:
  - NDGS-style: Direct diagonal-l_triangle with exp/sigmoid activations
  - UBS-style: Rotation-scale-l_triangle with softplus/identity activations
- **Conditional slicing**: View-dependent Gaussian slicing via CUDA kernels
- **Custom rasterization**: TCGS rasterizer with cutting plane support
- **Test-time optimization**: Precomputed values (`v_22_inv`, `v_regr`, `direction`, `cov3D_precomp`)

Functions in `utils/ndgs_utils.py`: `create_cholesky()`, `l_triangle_to_covar()`, `strip_lower_diag()`

## Important Implementation Details

### Unified Parameter Naming

**All models now use:**
- `self._scale`: Scale/diagonal parameters (unified name)
- `self._l_triangle`: Lower-triangular elements (unified name)
- Properties: `get_scale`, `get_l_triangle` (apply activation functions)

**Activation functions differ by parametrization:**

**NDGS-style** (`use_rot_scale_l_triangle=False`):
```python
self.scale_activation = lambda x: torch.exp(x)  # Diagonal elements
self.l_triangle_activation = lambda x: torch.sigmoid(x) * 2.0 - 1.0  # Bounded
```

**UBS-style** (`use_rot_scale_l_triangle=True`):
```python
self.scale_activation = torch.nn.functional.softplus  # Smooth positive
self.l_triangle_activation = lambda x: x  # Identity (first 3 encode rotation)
```

### 7DGS Time Dimension Support

**Parameters:**
- `self._mean_time`: Per-Gaussian time parameter [N, 1]
- `viewpoint_camera.timestamp`: Per-frame timestamp value (0.0 - 1.0)

**Key locations:**
- Initialization: `create_from_pcd()` - initialize time to 0.5
- Conditional query: `slice_gaussian()` - concatenate `[view_dir, timestamp]`
- Viewer control: `view_tcgs()` - pass `render_tab_state.timestamp` to camera
- PLY I/O: `save_ply()`/`load_ply()` - handle time parameter with backward compatibility

### Learnable Lambda Opacity (NDGS only)

**Purpose**: Per-Gaussian learnable opacity scaling for view-dependent control

**Implementation:**
```python
# In __init__:
self.learnable_lambda_opc = learnable_lambda_opc
if learnable_lambda_opc:
    self._lambda_opc = nn.Parameter(...)  # Learnable
else:
    self._lambda_opc = torch.ones(...)  # Fixed

# In rendering:
if self.learnable_lambda_opc:
    opacity = self.get_opacity * pdf_cond * self.get_lambda_opc
else:
    opacity = self.get_opacity * pdf_cond
```

**Argument:** `--learnable_lambda_opc` (boolean flag in `ModelParams`)

**Location:** NDGS models only (not applicable to UBS which uses beta parameters instead)

### Beta Parameters (UBS only)

**Purpose**: Uncertainty modeling and per-dimension bandwidth control for view-dependent rendering

**Structure:**
```python
# Beta shape: [N, input_dim-2]
# For 6DGS: [N, 4] = [spatial_beta, view_x_beta, view_y_beta, view_z_beta]
# For 7DGS: [N, 5] = [spatial_beta, view_x_beta, view_y_beta, view_z_beta, time_beta]

# Beta activation (ubs.py line 58)
def beta_activation(betas):
    return 4.0 * torch.exp(betas)
```

**Initialization (ubs.py lines 246-252):**
```python
betas = torch.zeros((N, self.input_dim - 2), device="cuda")
if self.input_dim == 7:
    betas[:, 1:4] -= 3  # Initialize view betas lower for 7DGS
```

**Usage in Conditional Covariance (ubs.py lines 829-843):**
```python
# Get beta parameters (exclude first beta used elsewhere)
beta = self.get_beta[:, 1:]  # [N, C] where C = input_dim-3
beta_adj = torch.clamp_max(beta / 4.0, 1.0)  # Clamp to prevent over-smoothing

# Apply beta adjustment to regression matrix
v_regr = torch.bmm(v_12, v_22_inv)
v_regr_beta = v_regr * beta_adj.unsqueeze(1)  # Per-dimension scaling

# Compute beta-adjusted conditional covariance
v_change = torch.bmm(v_regr_beta, v_21)
v_cond = v_11 - v_change  # Reduced influence = wider bandwidth
```

**Beta Viewer Filtering (ubs.py lines 1201-1225):**
```python
# Quantile-based filtering on beta dimensions
mask = quantile_mask(
    self._beta,
    b_xyz=(0, 100),   # Spatial beta percentile range
    b_view=(0, 100),  # View beta percentile range
    b_time=(0, 100)   # Time beta percentile range (7DGS only)
)
```

**Key Differences from Lambda Opacity:**
- **Beta**: Per-dimension bandwidth control (separate for spatial, view-x, view-y, view-z, time)
- **Lambda**: Single scalar opacity scaling per Gaussian
- **Beta**: Affects conditional covariance computation (bandwidth)
- **Lambda**: Affects final opacity value (visibility)

### Optimized Viewer Masking

**Old approach (inefficient):**
```python
# Save all tensors
original_xyz = self._xyz
# ...
# Modify self
self._xyz = self._xyz[mask]
# Render
# Restore
self._xyz = original_xyz
```

**New approach (efficient):**
```python
# Create masked views (just tensor slices, no copies)
_xyz_masked = self._xyz[mask]
# Swap pointers (O(1) operation)
orig_xyz, self._xyz = self._xyz, _xyz_masked
# Render
# Restore pointers
self._xyz = orig_xyz
```

**Location:** `view_tcgs()` method in:
- `gaussian_model_ndgs.py`: lines 1088-1148
- `gaussian_model_ndgs_2sh.py`: lines 1146-1205

### Dual SH Support (ndgs-2sh only)

**Structure:**
```python
# Single SH (ndgs.py):
self._features_dc = torch.empty(0)
self._features_rest = torch.empty(0)

# Dual SH (ndgs_2sh.py):
self._features_dc = [torch.empty(0), torch.empty(0)]  # Two sets
self._features_rest = [torch.empty(0), torch.empty(0)]
```

**Viewer blending:**
```python
# In viewer (ndgs_2sh only):
shs = (1.0 - color_interpolation) * self.shs[0] + color_interpolation * self.shs[1]
```

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

### JSON with Time/Cutting Plane Support

```json
{
  "camera_angle_x": 0.857,
  "frames": [
    {
      "file_path": "./images/img_001.jpg",
      "transform_matrix": [...],
      "timestamp": 0.5,      // Optional: time value for 7DGS (0.0 - 1.0)
      "x_threshold": 5.0,    // Optional: cutting plane position
      "color_idx": 0         // Optional: color index for dual SH (0 or 1)
    }
  ]
}
```

### Data Preprocessing

```bash
# Convert images to COLMAP format with undistortion
python tools/preprocessing/colmap_convert.py -s <location> [--resize]

# Skip COLMAP matching if COLMAP data already exists
python tools/preprocessing/colmap_convert.py -s <location> --skip_matching [--resize]

# Prepare volumetric/cloud datasets
python tools/preprocessing/cloud_dataset_preprocessing.py
```

See [tools/README.md](tools/README.md) for detailed preprocessing documentation.

## Development Notes

### Switching Between Model Variants

Use the `--mode` argument:

```bash
python train.py --mode ndgs         # N-DGS single SH (default)
python train.py --mode ndgs-2sh     # N-DGS dual SH
python train.py --mode ddgs         # Deformable DGS
python train.py --mode 3dgs         # Standard 3DGS
python train.py --mode ubs          # UBS variant
```

The mode is selected in [scene/__init__.py](scene/__init__.py) via `get_gaussian_model()` factory function.

### Output Structure

Trained models are saved to `output/<model_name>/` with:
- `point_cloud/iteration_<N>/point_cloud.ply`: Gaussian parameters at iteration N
- `cameras.json`: Camera parameters
- Rendered images in subdirectories when using `render.py`

### PLY File Format

**New unified naming** (v3.0+):
- Position: `x`, `y`, `z`
- Normal: `nx`, `ny`, `nz`
- Features: `f_dc_*`, `f_rest_*` (or dual lists for ndgs-2sh)
- Covariance: `scale_*`, `l_triangle_*` (unified names)
- Opacity: `opacity`
- Lambda opacity: `lambda_opc` (if learnable_lambda_opc=True)
- Time: `mean_time` (if input_dim=7)

**Backward compatibility**: Old files with `diags_*`/`l_triangs_*` load automatically.

### Git Submodules

The repository uses git submodules for custom CUDA extensions. When cloning:

```bash
git clone <repo_url> --recursive

# Or if already cloned:
git submodule update --init --recursive
```

## Common Issues

- **VRAM limitations**: Reduce `--densify_grad_threshold`, increase `--densification_interval`, or decrease `--densify_until_iter`. Set `--test_iterations -1` to avoid testing memory spikes.
- **Large-scale scenes**: Lower learning rates: `--position_lr_init 0.000016 --scale_lr 0.001`
- **Building CUDA extensions**: Ensure CUDA toolkit version matches PyTorch CUDA version
- **NaN losses**: Reduce learning rates for covariance parameters (`--scale_lr`, `--l_triangle_lr`)
- **Slow convergence**: Try UBS-style parametrization: `--use_rot_scale_l_triangle`

## Key Files for Common Tasks

### Adding New Model Features
- `scene/gaussian_model_ndgs.py` - Primary model implementation
- `scene/gaussian_model_ndgs_2sh.py` - Dual SH variant (keep consistent with ndgs.py)
- `arguments/__init__.py` - Add new command-line arguments

### Modifying Training Loop
- `train.py` - Main training loop
- `utils/loss_utils.py` - Loss functions
- `gaussian_renderer/__init__.py` - Rendering functions

### Viewer Modifications
- `scene/gaussian_viewer.py` - Viewer state and UI controls
- Model `view_tcgs()` method - Viewer rendering logic

### Rasterization Changes
- `submodules/tcgs_speedy_rasterizer/` - TCGS CUDA kernels
- `submodules/gsplat/` - N-DGS conditional slicing operations

## Citation

Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., ACM TOG 2023).
