# NDSplat: N-Dimensional Splatting

A unified framework for N-dimensional splatting, supporting multiple kernel types (Gaussian and Beta), conditioning parameterizations, rasterization backends, and dimensionalities (3D/6D/7D).

## Key Features

- **Multiple Kernels**: Gaussian kernel (exponential opacity) and Beta kernel (bandwidth-controlled opacity)
- **Flexible Conditioning**: Full N-D covariance with matrix inversion (UBS/NDGS) or direct Cholesky precision parameterization (dGS/dBS) for faster slicing
- **Multiple Rasterization Backends**: gsplat (CUDA-accelerated N-DGS operations), diff-gaussian-rasterization (original 3DGS backend), and TCGS (tile-based with cutting plane support)
- **3D/6D/7D Support**: Standard 3DGS, 6D with view-dependent conditioning, and 7D with time dimension

## Model Variants

| Mode | Kernel | Conditioning | Color | Description |
|------|--------|-------------|-------|-------------|
| `3dgs` | Gaussian | None (3D) | SH | Original 3D Gaussian Splatting |
| `ndgs` | Gaussian | Full covariance | SH | N-DGS with conditional slicing |
| `ubs` | Beta | Full covariance | SH | Uncertainty-Based Splatting |
| `dgs` | Gaussian | Direct Cholesky | SH | Direct Gaussian Splatting with lambda params |
| `dbs` | Beta | Direct Cholesky | SH | Direct Beta Splatting |

### Kernel Types

**Gaussian kernel** (`ndgs`, `dgs`, `3dgs`):
- Standard exponential opacity: `alpha * exp(-0.5 * z^T z)`
- Lambda parameters for per-dimension opacity scaling (dGS)

**Beta kernel** (`ubs`, `dbs`):
- Per-dimension bandwidth control: `alpha * prod((1 - tanh(z_i^2))^{beta_i})`
- Beta parameters enable adaptive bandwidth per conditioning dimension

### Conditioning Parameterizations

**Full covariance** (`ubs`, `ndgs`):
- Full N-D covariance matrix (6x6 or 7x7)
- Conditional slicing via matrix inversion: `Sigma_cond = Sigma_pp - Sigma_pq @ Sigma_qq^{-1} @ Sigma_qp`
- Richer representation, higher computational cost

**Direct Cholesky precision** (`dgs`, `dbs`):
- Direct `L_22_inv` (Cholesky of precision) and `v_12` (position displacement) parameters
- No matrix inversion needed: `z = L^T @ delta` directly
- 5-6x faster slicing than full covariance

### Rasterization Backends

- **gsplat**: Custom fork with CUDA kernels for conditional slicing, SH evaluation, and Beta kernel support
- **TCGS**: Tile-based rasterizer with cutting plane support (`x_threshold`), tight snugbox culling
- **diff-gaussian-rasterization**: Original 3DGS rasterizer for baseline comparison

Select backend with `--use_gsplat` flag (gsplat vs TCGS for training).

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
cd ndsplat
```

2. Create conda environment and install dependencies:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

3. Install CUDA extensions:
```shell
# Install gsplat for N-DGS/UBS/dBS operations
pip install submodules/gsplat

# Install TCGS rasterizer
pip install submodules/tcgs_speedy_rasterizer

# Install diff-gaussian-rasterization (for 3DGS baseline)
pip install submodules/diff-gaussian-rasterization

# Install other dependencies
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

## Usage

### Training

```shell
# 3D Gaussian Splatting (baseline)
python train.py -s <dataset> --mode 3dgs

# 6D N-DGS with conditional slicing
python train.py -s <dataset> --mode ndgs --input_dim 6

# 7D N-DGS with time dimension
python train.py -s <dataset> --mode ndgs --input_dim 7

# UBS with Beta kernel (full covariance)
python train.py -s <dataset> --mode ubs --input_dim 6

# UBS with MCMC densification
python train.py -s <dataset> --mode ubs --densification_strategy mcmc --mcmc_cap_max 300000

# dGS with direct Cholesky (Gaussian kernel)
python train.py -s <dataset> --mode dgs --input_dim 6

# dBS with direct Cholesky (Beta kernel)
python train.py -s <dataset> --mode dbs --input_dim 6

# Use gsplat rasterizer instead of TCGS
python train.py -s <dataset> --mode ubs --use_gsplat
```

The training script automatically launches a live viewer on port 8080 (disable with `--disable_viewer`).

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
- `--mode`: Model architecture (`3dgs`, `ndgs`, `dgs`, `dbs`, `ubs`) - default: `dgs`
- `--input_dim`: Gaussian dimensionality (3, 6, or 7) - default: `6`
- `--sh_degree`: Spherical harmonics degree (max 3) - default: `3`
- `--use_gsplat`: Use gsplat rasterizer instead of TCGS
- `--use_view_dependent_pos`: Enable view-dependent position shift (dGS) - default: `True`
- `--learnable_lambda_opc`: Make lambda_opc learnable per Gaussian (NDGS)
- `--use_rot_scale_l_triangle`: Use rotation-scale-l_triangle parameterization

#### Training Parameters
- `--iterations`: Total training iterations - default: `30000`
- `--position_lr_init`: Initial position learning rate - default: `0.00016`
- `--feature_lr`: Feature learning rate - default: `0.0025`
- `--opacity_lr`: Opacity learning rate - default: `0.05`
- `--scale_lr`: Scale parameters learning rate - default: `0.005`
- `--l_triangle_lr`: L-triangle parameters learning rate - default: `0.001`

#### Densification Parameters
- `--densification_strategy`: `standard`, `mcmc`, or `fastgs` - default: `standard`
- `--densify_grad_threshold`: Gradient threshold - default: `0.0002`
- `--mcmc_cap_max`: Maximum Gaussians for MCMC - default: `300000`

#### Viewer Parameters
- `--port`: Viewer port - default: `8080`
- `--disable_viewer`: Disable live viewer

</details>

### Rendering

```shell
python render.py -m <path to trained model> -s <path to dataset>
```

### Evaluation

```shell
python train.py -s <dataset> --eval         # Train with test split
python render.py -m <model_path>            # Render test views
python metrics.py -m <model_path>           # Compute PSNR, SSIM, LPIPS
```

### Live Viewing

```shell
python view.py -m <model_path> --ply <ply_file> --mode ndgs
```

## Project Structure

```
ndsplat/
├── train.py, render.py, metrics.py, view.py   # Main pipeline
├── run.sh                                      # Master experiment runner
├── scene/
│   ├── gaussian_model.py                       # 3DGS baseline
│   ├── gaussian_model_ndgs.py                  # N-DGS (full cov, Gaussian kernel, SH)
│   ├── gaussian_model_ubs_sh.py                # UBS (full cov, Beta kernel, SH)
│   ├── gaussian_model_dgs_full.py              # dGS (direct Cholesky, Gaussian kernel, SH)
│   ├── gaussian_model_dbs_sh.py                # dBS (direct Cholesky, Beta kernel, SH)
│   ├── gaussian_viewer.py                      # Viser-based live viewer
│   ├── cameras.py                              # Camera classes
│   └── dataset_readers.py                      # Dataset loading
├── arguments/
│   └── __init__.py                             # Command-line arguments
├── utils/                                      # Helper functions
├── submodules/
│   ├── gsplat/                                 # N-DGS/UBS/dBS CUDA operations
│   ├── tcgs_speedy_rasterizer/                 # TCGS rasterizer with cutting planes
│   ├── diff-gaussian-rasterization/            # Original 3DGS rasterizer
│   ├── simple-knn/                             # KNN utilities
│   └── fused-ssim/                             # Fused SSIM computation
├── scripts/                                    # Experiment runners
│   ├── benchmarks/                             # Standard evaluations
│   ├── ablations/                              # Ablation studies
│   └── tests/                                  # Development scripts
└── tools/                                      # Data preprocessing & evaluation
```

## Dataset Format

### COLMAP Dataset

```
<dataset>/
├── images/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### NeRF Synthetic / JSON Format

```json
{
  "camera_angle_x": 0.857,
  "frames": [
    {
      "file_path": "./images/img_001.jpg",
      "transform_matrix": [...],
      "x_threshold": 5.0,
      "timestamp": 0.5,
      "color_idx": 0
    }
  ]
}
```

### Data Preprocessing

```shell
python tools/preprocessing/colmap_convert.py -s <images_directory>
```

## Live Viewer

Real-time web viewer powered by [Viser](https://github.com/nerfstudio-project/viser):

- Interactive camera navigation
- Time animation controls (7DGS)
- Beta/opacity-based Gaussian filtering
- Cutting plane support
- Multiple render modes (RGB, Alpha, Depth, Normal)
- Real-time FPS monitoring

## Acknowledgments

This implementation builds upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et al.
- [gsplat](https://github.com/nerfstudio-project/gsplat) for CUDA-accelerated operations
- [Viser](https://github.com/nerfstudio-project/viser) for the interactive viewer

## License

This software is free for non-commercial, research and evaluation use under the terms of the LICENSE.md file.

## Citation

```bibtex

@article{gao20246dgs,
  title={6dgs: Enhanced direction-aware gaussian splatting for volumetric rendering},
  author={Gao, Zhongpai and Planche, Benjamin and Zheng, Meng and Choudhuri, Anwesa and Chen, Terrence and Wu, Ziyan},
  journal={arXiv preprint arXiv:2410.04974},
  year={2024}
}

@inproceedings{gao20257dgs,
  title={7DGS: Unified spatial-temporal-angular Gaussian splatting},
  author={Gao, Zhongpai and Planche, Benjamin and Zheng, Meng and Choudhuri, Anwesa and Chen, Terrence and Wu, Ziyan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={26316--26325},
  year={2025}
}

@article{liu2025universal,
  title={Universal Beta Splatting},
  author={Liu, Rong and Gao, Zhongpai and Planche, Benjamin and Chen, Meida and Nguyen, Van Nguyen and Zheng, Meng and Choudhuri, Anwesa and Chen, Terrence and Wang, Yue and Feng, Andrew and others},
  journal={arXiv preprint arXiv:2510.03312},
  year={2025}
}

```
