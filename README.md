# 6D Gaussian Splatting (6DGS)

Implementation of N-Dimensional Gaussian Splatting for real-time novel view synthesis with view-dependent rendering.

## Overview

This repository implements **N-Dimensional Gaussian Splatting (N-DGS)**, extending traditional 3D Gaussian Splatting with conditional Gaussian slicing for view-dependent appearance modeling. The method represents scenes using N-dimensional Gaussians (6D or 7D) that are conditionally sliced based on viewing direction (and optionally time), enabling efficient capture of view-dependent effects.

### Key Features

- **6D/7D Gaussian Representation**: Extends 3D Gaussians with view-direction (6D) or view-direction + time (7D)
- **Conditional Slicing (NDGS)**: CUDA-accelerated conditional Gaussian slicing based on viewing direction
- **Beta-Based Bandwidth Control (UBS)**: Uncertainty-aware rendering with per-dimension beta parameters
- **Learnable Lambda Opacity (NDGS)**: Per-Gaussian learnable opacity scaling for fine-grained control
- **Dual Parametrization**: Switch between NDGS-style and UBS-style covariance parametrization
- **TCGS Rasterization**: High-performance rasterization with support for cutting planes
- **Live Training Viewer**: Real-time web-based viewer with time animation and dual SH blending
- **Dual SH Support**: Multi-view color consistency with interpolatable SH features (ndgs-2sh model)
- **Optimized Viewer**: Efficient tensor view masking for real-time filtering without copying
- **Multiple Model Modes**: Configurable architecture supporting NDGS, NDGS-2SH, DDGS, 3DGS, and UBS modes

## Installation

### Prerequisites

- CUDA-ready GPU with Compute Capability 7.0+
- CUDA SDK 11 or 12
- Python 3.8+
- Conda (recommended)

### Setup

1. Clone the repository with submodules:
```shell
git clone <repository-url> --recursive
cd 6dgs-iclr
```

2. Create conda environment and install dependencies:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

3. Install CUDA extensions:
```shell
# Install gsplat for N-DGS operations
pip install submodules/gsplat

# Install TCGS rasterizer
pip install submodules/tcgs-speedy-rasterizer

# Install other dependencies
pip install submodules/simple-knn
```

## Project Organization

The repository is organized into three main categories:

- **Core Scripts** (root): `train.py`, `render.py`, `metrics.py`, `view.py` - Main pipeline
- **tools/**: Data preprocessing and evaluation utilities
  - `preprocessing/`: COLMAP conversion, cloud dataset generation
  - `evaluation/`: Automated benchmarking tools
- **scripts/**: Organized experiment runners
  - `benchmarks/`: Standard evaluations (Mip-NeRF 360, NeRF Synthetic, etc.)
  - `ablations/`: Ablation studies
  - `tests/`: Development and debugging scripts

**Run experiments with:**
```shell
./run.sh <category> <script>      # Master entry point
./run.sh --help                    # See all options
./run.sh list                      # List available scripts
```

See [ORGANIZATION.md](ORGANIZATION.md) for complete documentation.

## What's New (v3.0)

### Recent Major Updates

**✨ Unified Model Architecture:**
- Consolidated and cleaned codebase with two main N-DGS models:
  - `gaussian_model_ndgs.py` - Single SH for standard rendering
  - `gaussian_model_ndgs_2sh.py` - Dual SH for multi-view consistency
- Both models support 6DGS and 7DGS (with time dimension)
- Removed legacy experimental code (color_net, LSH culling, Taichi imports)

**🎯 New Features:**
- **Learnable Lambda Opacity** (`--learnable_lambda_opc`): Per-Gaussian learnable opacity scaling parameter
- **Dual Parametrization** (`--use_rot_scale_l_triangle`): Switch between NDGS-style and UBS-style covariance modes
- **Full 7DGS Time Support** (`--input_dim 7`): Complete temporal dimension with viewer animation controls
- **Dual SH Blending**: Real-time color interpolation in viewer for multi-view consistency
- **Optimized Viewer**: Efficient pointer-swap masking instead of tensor copying (significant FPS improvement)

**🚀 Performance Optimizations:**
- Viewer masking now uses tensor views with pointer swaps (no memory allocation per frame)
- Test mode precomputed values properly handled
- Eliminated redundant save/restore operations

**📖 Documentation:**
- See [ORGANIZATION.md](ORGANIZATION.md) for project structure

## Usage

### Training

Train an N-DGS model on your dataset:

```shell
# Basic 6DGS training
python train.py -s <path to COLMAP dataset> --mode ndgs --input_dim 6

# 7DGS with time dimension
python train.py -s <path to dataset> --mode ndgs --input_dim 7

# With learnable lambda opacity
python train.py -s <path to dataset> --mode ndgs --learnable_lambda_opc

# UBS-style parametrization
python train.py -s <path to dataset> --mode ndgs --use_rot_scale_l_triangle

# Dual SH for multi-view consistency
python train.py -s <path to dataset> --mode ndgs-2sh --input_dim 6

# UBS with beta-based bandwidth control
python train.py -s <path to dataset> --mode ubs --input_dim 6

# UBS with MCMC densification strategy
python train.py -s <path to dataset> --mode ubs --densification_strategy mcmc --mcmc_cap_max 300000
```

The training script will automatically launch a live viewer on port 8080 (unless disabled with `--disable_viewer`). Open your browser to `http://localhost:8080` to monitor training in real-time.

**Note:** UBS and NDGS use different approaches for view-dependent rendering. See [Model Modes](#model-modes) section for comparison.

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

#### Dataset Parameters
- `--source_path` / `-s`: Path to COLMAP or NeRF Synthetic dataset
- `--model_path` / `-m`: Output directory for trained model (default: `output/<random>`)
- `--images` / `-i`: Alternative subdirectory for images (default: `images`)
- `--eval`: Use train/test split for evaluation
- `--resolution` / `-r`: Image resolution (1, 2, 4, 8 or specific width)
- `--data_device`: Device for image data (`cuda` or `cpu`)
- `--white_background` / `-w`: Use white background instead of black

#### Model Parameters
- `--mode`: Model architecture (`ndgs`, `ndgs-2sh`, `ddgs`, `3dgs`, `ubs`) - default: `ndgs`
- `--input_dim`: Gaussian dimensionality (6 for 6DGS, 7 for 7DGS with time) - default: `6`
- `--sh_degree`: Spherical harmonics degree (max 3) - default: `3`
- `--learnable_lambda_opc`: Make lambda_opc learnable per Gaussian - default: `False`
- `--use_rot_scale_l_triangle`: Use UBS-style parametrization instead of NDGS-style - default: `False`

#### Training Parameters
- `--iterations`: Total training iterations - default: `30000`
- `--position_lr_init`: Initial position learning rate - default: `0.00016`
- `--position_lr_final`: Final position learning rate - default: `0.0000016`
- `--feature_lr`: Feature learning rate - default: `0.0025`
- `--opacity_lr`: Opacity learning rate - default: `0.05`
- `--scaling_lr`: Normal scaling learning rate - default: `0.005`
- `--scale_lr`: Scale/diagonal parameters learning rate - default: `0.005` (replaces `--diags_lr`)
- `--l_triangle_lr`: L-triangle parameters learning rate - default: `0.001` (replaces `--l_triangs_lr`)
  - Note: Old names (`--diags_lr`, `--l_triangs_lr`) still work for backward compatibility

#### Densification Parameters
- `--densification_strategy`: Strategy for Gaussian densification (`standard` or `mcmc`) - default: `standard`
  - `standard`: Gradient-based clone, split, and prune (all models)
  - `mcmc`: MCMC sampling-based refinement (UBS only)
- `--densify_from_iter`: Start densification iteration - default: `500`
- `--densify_until_iter`: Stop densification iteration - default: `15000`
- `--densify_grad_threshold`: Gradient threshold for densification (standard only) - default: `0.0002`
- `--densification_interval`: Densification frequency (standard only) - default: `100`
- `--opacity_reset_interval`: Opacity reset frequency (standard only) - default: `3000`
- `--percent_dense`: Scene extent percentage for densification - default: `0.01`
- `--mcmc_cap_max`: Maximum number of Gaussians (MCMC only) - default: `300000`
- `--mcmc_refine_interval`: MCMC refinement frequency (MCMC only) - default: `100`
- `--mcmc_add_rate`: Rate of adding new Gaussians (MCMC only, currently unused) - default: `0.25`
- `--mcmc_remove_rate`: Rate of removing Gaussians (MCMC only, currently unused) - default: `0.1`

#### Viewer Parameters
- `--port`: Viewer port - default: `8080`
- `--disable_viewer`: Disable live viewer

#### Checkpointing
- `--test_iterations`: Iterations for evaluation - default: `7000 30000`
- `--save_iterations`: Iterations to save model - default: `7000 30000 <iterations>`
- `--checkpoint_iterations`: Iterations to save checkpoint
- `--start_checkpoint`: Path to checkpoint to continue from

</details>

### Rendering

Render novel views from a trained model:

```shell
python render.py -m <path to trained model> -s <path to dataset>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

- `--model_path` / `-m`: Path to trained model
- `--source_path` / `-s`: Path to source dataset (if not in model config)
- `--skip_train`: Skip rendering training views
- `--skip_test`: Skip rendering test views
- `--mode`: Model architecture to use (overrides saved config)
- `--quiet`: Suppress console output

</details>

### Live Viewing

View a trained model interactively:

```shell
# View trained model
python view.py -m <model_path> --ply <ply_file> --mode ndgs

# Custom port
python view.py -m <model_path> --ply <ply_file> --mode ndgs --port 8080
```

### Evaluation

Compute metrics on rendered images:

```shell
python train.py -s <path to dataset> --eval  # Train with test split
python render.py -m <path to trained model>   # Render test views
python metrics.py -m <path to trained model>  # Compute metrics
```

### Data Preprocessing

Prepare datasets for training:

```shell
# Convert images to COLMAP format
python tools/preprocessing/colmap_convert.py -s <images_directory>

# Generate volumetric/cloud datasets
python tools/preprocessing/cloud_dataset_preprocessing.py

# Run full benchmark evaluation
python tools/evaluation/full_eval.py -m360 <mipnerf360> -tat <tanks> -db <deepblending>
```

See [tools/README.md](tools/README.md) for detailed documentation.

## Live Viewer

The training process includes a real-time web viewer powered by [Viser](https://github.com/nerfstudio-project/viser) that allows you to:

- **Monitor Training**: Watch the scene reconstruction in real-time
- **Interactive Camera Control**: Navigate the scene with mouse/keyboard
- **Time Animation (7DGS)**: Auto-loop and manual time control for temporal models
- **Dual SH Blending**: Smooth interpolation between two SH color representations (ndgs-2sh models)
- **Gaussian Filtering**: Percentile-based or absolute opacity thresholding
- **Cutting Plane**: Enable spatial filtering with x_threshold control
- **Render Modes**: Switch between RGB, Alpha, Depth, Normal, and other visualization modes
- **Performance Monitoring**: Real-time FPS display with smoothing
- **Optimized Rendering**: Efficient masking without memory allocation overhead

The viewer automatically starts when training begins. Access it at `http://localhost:8080` (or custom port specified with `--port`).

### Viewer Controls

**Time Animation (7DGS only):**
- **Auto Loop**: Automatically cycle through time dimension
- **Loop Duration**: Adjust animation speed (0.5 - 10.0 seconds)
- **Time Slider**: Manual time control (0.0 - 1.0)

**Color Interpolation (ndgs-2sh models only):**
- **Color Blend**: Smooth interpolation between SH_0 and SH_1 (0.0 - 1.0)

**Gaussian Filtering:**
- **Use Opacity Percentile**: Toggle between percentile and absolute threshold modes
- **Opacity Percentile**: Show top X% most opaque Gaussians (0 - 100)
- **Opacity Threshold**: Absolute minimum opacity (0.0 - 1.0)

**Cutting Plane:**
- **Enable X Threshold**: Toggle cutting plane on/off
- **X Threshold**: Set cutting plane position along X-axis

**Render Settings:**
- **Render Mode**: RGB, Alpha, Depth, Normal
- **Tight Snugbox**: Enable/disable TCGS optimization
- **Background Color**: RGB color picker
- **FPS Display**: Real-time performance monitoring

## Dataset Format

### COLMAP Dataset Structure

```
<dataset>
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

### JSON Dataset with Cutting Plane

For datasets with cutting plane support, use a `transforms.json` file:

```json
{
  "camera_angle_x": 0.857,
  "frames": [
    {
      "file_path": "./images/image_001.jpg",
      "transform_matrix": [...],
      "x_threshold": 5.0,      // Optional: cutting plane position
      "color_idx": 0,          // Optional: color index for dual SH
      "timestamp": 0.5         // Optional: time value for 7DGS (0.0 - 1.0)
    }
  ]
}
```

## Technical Details

### N-Dimensional Gaussian Representation

Each Gaussian is represented with:
- **6DGS**: `[x, y, z, nx, ny, nz]` - position and normal direction
- **7DGS**: `[x, y, z, nx, ny, nz, t]` - position, normal, and time
- **Covariance** (N×N): Parameterized by scale/diagonal and lower triangular elements
- **Color**:
  - NDGS: Spherical harmonics coefficients (single or dual)
  - UBS: Direct RGB values
- **Opacity**: Sigmoid-activated opacity value
- **Lambda Opacity** (NDGS, optional): Learnable per-Gaussian opacity scaling
- **Beta Parameters** (UBS): Per-dimension bandwidth control [N, input_dim-2]

### Parametrizations

**NDGS-style** (default, `use_rot_scale_l_triangle=False`):
- `_scale`: Exponential activation for diagonal elements
- `_l_triangle`: Sigmoid bounded [-1, 1] for off-diagonal elements
- Direct lower-triangular covariance construction

**UBS-style** (`use_rot_scale_l_triangle=True`):
- `_scale`: Softplus activation for smooth positive scales
- `_l_triangle`: First 3 elements encode 6D rotation matrix
- Rotation-scale-l_triangle covariance construction with KNN initialization

### Conditional Slicing

The N-D Gaussians are conditionally sliced based on viewing direction:
1. Compute view direction from camera to Gaussian center
2. For 7DGS, append timestamp to query vector
3. Perform conditional Gaussian slicing (see `slice_gaussian` in model files)
4. Obtain 3D conditional mean and covariance
5. Scale opacity based on viewing direction alignment

### TCGS Rasterization

The TCGS (Tile-based CUDA Gaussian Splatting) rasterizer provides:
- Efficient tile-based rendering
- Cutting plane support via `x_threshold`
- Precomputed covariance support (for test mode)
- Tight bounding box optimization

## Model Modes

This codebase supports multiple Gaussian splatting variants:

- **ndgs** (default): N-Dimensional GS with conditional slicing (single SH)
- **ndgs-2sh**: N-Dimensional GS with dual SH features for multi-view consistency
- **ddgs**: Deformable DGS variant
- **3dgs**: Standard 3D Gaussian Splatting
- **ubs**: Uncertainty-Based Splatting with beta parameters

Select the mode with `--mode <mode_name>` during training.

### NDGS vs UBS: Choosing the Right Model

Both NDGS and UBS extend 3DGS to N-dimensions (6D/7D) but use different approaches for view-dependent rendering:

| Feature | NDGS (`--mode ndgs`) | UBS (`--mode ubs`) |
|---------|----------------------|---------------------|
| **Color Model** | Spherical harmonics (DC + rest) | Direct RGB values |
| **View-Dependence** | Conditional Gaussian slicing | Beta-adjusted covariance |
| **Opacity Control** | Lambda opacity (learnable/fixed) | Beta parameters per dimension |
| **Parametrization** | Flexible (NDGS/UBS-style) | Fixed rot-scale-l_triangle |
| **Best For** | General scenes, SH appearance | Uncertainty quantification, direct RGB |
| **Complexity** | More complex (SH evaluation) | Simpler (direct RGB) |
| **Test Optimization** | Precomputed values supported | Full computation each frame |

**Usage Examples:**

```shell
# NDGS with default NDGS-style parametrization
python train.py -s <dataset> --mode ndgs --input_dim 6

# NDGS with UBS-style parametrization (hybrid approach)
python train.py -s <dataset> --mode ndgs --use_rot_scale_l_triangle

# NDGS with learnable lambda opacity
python train.py -s <dataset> --mode ndgs --learnable_lambda_opc

# UBS with beta-based bandwidth control
python train.py -s <dataset> --mode ubs --input_dim 6

# UBS with 7D (time dimension)
python train.py -s <dataset> --mode ubs --input_dim 7
```

**Key Differences:**

1. **Color Representation**: NDGS uses SH for rich view-dependent appearance, UBS uses direct RGB for simplicity
2. **View-Dependence**: NDGS slices N-D Gaussians, UBS adjusts covariance bandwidth with beta parameters
3. **Viewer Filtering**: NDGS filters by opacity (percentile/threshold), UBS filters by beta quantiles
4. **Opacity Control**: NDGS uses lambda opacity scalar, UBS uses per-dimension beta vectors

See [CLAUDE.md](CLAUDE.md) for detailed technical comparison and implementation details.

## Code Structure

```
6dgs-iclr/
├── scene/
│   ├── gaussian_model_ndgs.py       # N-DGS single SH implementation
│   ├── gaussian_model_ndgs_2sh.py   # N-DGS dual SH implementation
│   ├── gaussian_model_ddgs.py       # DDGS model
│   ├── gaussian_model.py            # 3DGS model
│   ├── gaussian_model_ubs.py        # UBS model
│   ├── gaussian_viewer.py           # Viser-based live viewer
│   ├── cameras.py                   # Camera classes
│   └── dataset_readers.py           # Dataset loading
├── utils/
│   ├── loss_utils.py                # Loss functions
│   ├── camera_utils.py              # Camera utilities
│   ├── ndgs_utils.py                # N-DGS specific utilities
│   └── ...
├── arguments/
│   └── __init__.py                  # Command-line arguments
├── submodules/
│   ├── gsplat/                      # N-DGS CUDA operations
│   ├── tcgs-speedy-rasterizer/      # TCGS rasterizer
│   └── simple-knn/                  # KNN utilities
├── train.py                         # Training script
├── render.py                        # Rendering script
├── view.py                          # Interactive viewer script
└── metrics.py                       # Evaluation metrics
```

## Key Implementation Files

- [gaussian_model_ndgs.py](scene/gaussian_model_ndgs.py): Core N-DGS implementation with conditional slicing (single SH)
- [gaussian_model_ndgs_2sh.py](scene/gaussian_model_ndgs_2sh.py): N-DGS with dual SH features
- [gaussian_viewer.py](scene/gaussian_viewer.py): Interactive training viewer with optimized masking
- [train.py](train.py): Main training loop with viewer integration
- [render.py](render.py): Rendering script for evaluation
- [view.py](view.py): Standalone interactive viewer

## Performance Tips

- Use `--data_device cpu` for large/high-resolution datasets to reduce VRAM usage
- Adjust densification parameters (`--densify_grad_threshold`, `--densification_interval`) for memory-constrained setups
- Set `--test_iterations -1` to skip testing during training and reduce memory spikes
- For large scenes, reduce learning rates: `--position_lr_init 0.000016 --scale_lr 0.001`
- Enable learnable lambda opacity for scenes with complex view-dependent effects: `--learnable_lambda_opc`
- Use UBS-style parametrization for scenes with wide scale variation: `--use_rot_scale_l_triangle`

## Acknowledgments

This implementation builds upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et al.
- [gsplat](https://github.com/nerfstudio-project/gsplat) for CUDA-accelerated operations
- [Viser](https://github.com/nerfstudio-project/viser) for the interactive viewer
- TCGS rasterizer for high-performance rendering

## License

This software is free for non-commercial, research and evaluation use under the terms of the LICENSE.md file.

## Citation

If you use this code in your research, please cite the original 3D Gaussian Splatting paper:

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
