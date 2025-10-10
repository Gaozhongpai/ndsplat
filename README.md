# 6D Gaussian Splatting (6DGS)

Implementation of 6D Gaussian Splatting for real-time novel view synthesis with view-dependent rendering.

## Overview

This repository implements **6D Gaussian Splatting (6DGS)**, extending traditional 3D Gaussian Splatting with conditional Gaussian slicing for view-dependent appearance modeling. The method represents scenes using 6-dimensional Gaussians that are conditionally sliced based on viewing direction, enabling efficient capture of view-dependent effects.

### Key Features

- **6D Gaussian Representation**: Extends 3D Gaussians with additional dimensions for view-dependent rendering
- **Conditional Slicing**: CUDA-accelerated conditional Gaussian slicing based on viewing direction
- **TCGS Rasterization**: High-performance rasterization with support for cutting planes
- **Live Training Viewer**: Real-time web-based viewer powered by Viser for monitoring training progress
- **Multiple Model Support**: Configurable architecture supporting 6DGS, DDGS, 3DGS, and UBS modes
- **Cutting Plane Functionality**: Support for spatial filtering via x_threshold parameter

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

- **Core Scripts** (root): `train.py`, `render.py`, `metrics.py` - Main pipeline
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

## Usage

### Training

Train a 6DGS model on your dataset:

```shell
python train.py -s <path to COLMAP dataset> --mode 6dgs
```

The training script will automatically launch a live viewer on port 8080 (unless disabled with `--disable_viewer`). Open your browser to `http://localhost:8080` to monitor training in real-time.

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
- `--mode`: Model architecture (`6dgs`, `ddgs`, `3dgs`, `ubs`) - default: `6dgs`
- `--sh_degree`: Spherical harmonics degree (max 3) - default: `3`

#### Training Parameters
- `--iterations`: Total training iterations - default: `30000`
- `--position_lr_init`: Initial position learning rate - default: `0.00016`
- `--position_lr_final`: Final position learning rate - default: `0.0000016`
- `--feature_lr`: Feature learning rate - default: `0.0025`
- `--opacity_lr`: Opacity learning rate - default: `0.05`
- `--diags_lr`: Diagonal covariance learning rate - default: `0.005`
- `--l_triangs_lr`: Lower triangular covariance learning rate - default: `0.001`

#### Densification Parameters
- `--densify_from_iter`: Start densification iteration - default: `500`
- `--densify_until_iter`: Stop densification iteration - default: `15000`
- `--densify_grad_threshold`: Gradient threshold for densification - default: `0.0002`
- `--densification_interval`: Densification frequency - default: `100`
- `--opacity_reset_interval`: Opacity reset frequency - default: `3000`
- `--percent_dense`: Scene extent percentage for densification - default: `0.01`

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
- **Gaussian Filtering**: Adjust opacity and scale thresholds to filter visible Gaussians
- **Cutting Plane**: Enable spatial filtering with x_threshold control
- **Render Modes**: Switch between RGB, Alpha, Depth, and other visualization modes
- **Pause/Resume**: Control training flow interactively

The viewer automatically starts when training begins. Access it at `http://localhost:8080` (or custom port specified with `--port`).

### Viewer Controls

**Gaussian Filtering:**
- **Opacity Threshold**: Filter out low-opacity Gaussians (0.0 - 1.0)
- **Scale Threshold**: Filter out large-scale Gaussians (0.1 - 200.0)

**Cutting Plane:**
- **X Threshold**: Set cutting plane position along X-axis (-100.0 - 100.0)
- **Enable X Threshold**: Toggle cutting plane on/off

**Render Mode:**
- RGB, Alpha, Diffuse, Specular, Depth, Normal
- Total/Rendered Gaussian counts
- Near/Far plane settings
- Background color

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
      "color_idx": 0,          // Optional: color index for labeling
      "label": [1, 0, 0]       // Optional: label vector
    }
  ]
}
```

## Technical Details

### 6D Gaussian Representation

Each Gaussian is represented with:
- **Mean** (6D): `[x, y, z, nx, ny, nz]` - position and normal direction
- **Covariance** (6×6): Parameterized by diagonal and lower triangular elements
- **Color**: Spherical harmonics coefficients
- **Opacity**: Sigmoid-activated opacity value

### Conditional Slicing

The 6D Gaussians are conditionally sliced based on viewing direction:
1. Compute view direction from camera to Gaussian center
2. Perform conditional Gaussian slicing (see `slice_gaussian` in [gaussian_model_6dgs.py](scene/gaussian_model_6dgs.py#L84-L117))
3. Obtain 3D conditional mean and covariance
4. Scale opacity based on viewing direction alignment

### TCGS Rasterization

The TCGS (Tile-based CUDA Gaussian Splatting) rasterizer provides:
- Efficient tile-based rendering
- Cutting plane support via `x_threshold`
- Precomputed covariance support
- Tight bounding box optimization

## Model Modes

This codebase supports multiple Gaussian splatting variants:

- **6dgs** (default): 6D Gaussian Splatting with conditional slicing
- **ddgs**: Deformable DGS variant
- **3dgs**: Standard 3D Gaussian Splatting
- **ubs**: UBS variant

Select the mode with `--mode <mode_name>` during training.

## Code Structure

```
6dgs-iclr/
├── scene/
│   ├── gaussian_model_6dgs.py    # 6DGS model implementation
│   ├── gaussian_model_ddgs.py    # DDGS model
│   ├── gaussian_model.py         # 3DGS model
│   ├── gaussian_model_ubs.py     # UBS model
│   ├── gaussian_viewer.py        # Viser-based live viewer
│   ├── cameras.py                # Camera classes
│   └── dataset_readers.py        # Dataset loading
├── utils/
│   ├── loss_utils.py             # Loss functions
│   ├── camera_utils.py           # Camera utilities
│   └── ...
├── arguments/
│   └── __init__.py               # Command-line arguments
├── submodules/
│   ├── gsplat/                   # N-DGS CUDA operations
│   ├── tcgs-speedy-rasterizer/   # TCGS rasterizer
│   └── simple-knn/               # KNN utilities
├── train.py                      # Training script
├── render.py                     # Rendering script
└── metrics.py                    # Evaluation metrics
```

## Key Implementation Files

- [gaussian_model_6dgs.py](scene/gaussian_model_6dgs.py): Core 6DGS implementation with conditional slicing
- [gaussian_viewer.py](scene/gaussian_viewer.py): Interactive training viewer
- [train.py](train.py): Main training loop with viewer integration
- [render.py](render.py): Rendering script for evaluation

## Performance Tips

- Use `--data_device cpu` for large/high-resolution datasets to reduce VRAM usage
- Adjust densification parameters (`--densify_grad_threshold`, `--densification_interval`) for memory-constrained setups
- Set `--test_iterations -1` to skip testing during training and reduce memory spikes
- For large scenes, reduce learning rates: `--position_lr_init 0.000016 --diags_lr 0.001`

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
