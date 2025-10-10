# Tools

Utility scripts for data preprocessing and evaluation.

## Directory Structure

```
tools/
├── preprocessing/     # Data preparation and conversion tools
│   ├── cloud_dataset_preprocessing.py
│   └── colmap_convert.py
└── evaluation/        # Evaluation and benchmarking tools
    ├── full_eval.py
    └── metrics_ndgs.py
```

---

## Preprocessing Tools

### Cloud Dataset Preprocessing (`cloud_dataset_preprocessing.py`)

Prepares volumetric cloud/scatter datasets for training.

**Purpose:**
- Generates random point clouds for volumetric scenes
- Splits datasets into train/test/val sets
- Processes camera transforms (NeRF format)
- Organizes images into proper directory structure

**Usage:**
```bash
python tools/preprocessing/cloud_dataset_preprocessing.py
```

**Configuration:**
Edit the script to set:
- `root`: Dataset root path
- `num_pts`: Number of points to generate (default: 100,000)
- Point cloud bounds for your scene
- Test/train split ratio

**Supported Scenes:**
- Cloud volumetrics
- Explosion effects
- Subsurface scattering (suzanne)
- Bunny cloud
- Custom volumetric data

**Output:**
- `points3d.ply` - Initial point cloud
- `train/`, `test/`, `val/` directories with images
- `transforms_train.json`, `transforms_test.json`, `transforms_val.json`

---

### COLMAP Converter (`colmap_convert.py`)

Converts image sets to COLMAP format for 3D reconstruction.

**Purpose:**
- Runs complete COLMAP pipeline (SfM + MVS)
- Extracts features, matches them, runs bundle adjustment
- Generates camera poses and sparse point cloud
- Optionally creates multi-resolution images

**Usage:**
```bash
# Basic usage
python tools/preprocessing/colmap_convert.py -s /path/to/dataset

# With GPU acceleration
python tools/preprocessing/colmap_convert.py -s /path/to/dataset --no_gpu

# Skip feature extraction (if already done)
python tools/preprocessing/colmap_convert.py -s /path/to/dataset --skip_matching

# Generate multi-resolution images
python tools/preprocessing/colmap_convert.py -s /path/to/dataset --resize
```

**Arguments:**
- `-s, --source_path`: Dataset path (must contain `/input` folder with images)
- `--camera`: Camera model (default: OPENCV, options: SIMPLE_PINHOLE, PINHOLE, etc.)
- `--colmap_executable`: Custom COLMAP binary path
- `--no_gpu`: Disable GPU acceleration
- `--skip_matching`: Skip feature extraction/matching
- `--resize`: Generate downsampled images (2x, 4x, 8x)
- `--magick_executable`: Custom ImageMagick path

**Pipeline Steps:**
1. Feature extraction (SIFT)
2. Exhaustive feature matching
3. Sparse reconstruction (Bundle Adjustment)
4. Image undistortion
5. Optional: Multi-resolution generation

**Output:**
```
dataset/
├── input/              # Original images
├── distorted/          # COLMAP sparse model
│   └── sparse/
└── (sparse/ if undistorted)
```

---

## Evaluation Tools

### Full Evaluation (`full_eval.py`)

Automated benchmark evaluation across multiple standard datasets.

**Purpose:**
- Trains on all benchmark scenes
- Renders at multiple checkpoints
- Computes metrics (PSNR, SSIM, LPIPS)
- Reproduces paper results

**Usage:**
```bash
# Full evaluation on all datasets
python tools/evaluation/full_eval.py \
    -m360 /path/to/mipnerf360 \
    -tat /path/to/tanksandtemples \
    -db /path/to/deepblending \
    --output_path ./results

# Skip stages
python tools/evaluation/full_eval.py ... --skip_training
python tools/evaluation/full_eval.py ... --skip_rendering
python tools/evaluation/full_eval.py ... --skip_metrics
```

**Datasets:**
- **Mip-NeRF 360** (9 scenes):
  - Outdoor: bicycle, flowers, garden, stump, treehill
  - Indoor: room, counter, kitchen, bonsai
- **Tanks & Temples** (2 scenes): truck, train
- **Deep Blending** (2 scenes): drjohnson, playroom

**Pipeline:**
1. Training (iterations: 7K, 30K)
2. Rendering on test views
3. Metrics computation

**Output:**
```
./eval/
├── bicycle/
│   ├── train/
│   ├── test/
│   │   ├── ours_7000/
│   │   └── ours_30000/
│   └── results.json
├── flowers/
...
```

---

### NDGS Metrics (`metrics_ndgs.py`)

Specialized metrics computation for NDGS variant.

**Purpose:**
- Computes image quality metrics
- Supports NDGS-specific evaluation
- Batch processing across multiple models

**Usage:**
```bash
# Single model
python tools/evaluation/metrics_ndgs.py -m /path/to/model/output

# Multiple models
python tools/evaluation/metrics_ndgs.py -m /path/to/model1 /path/to/model2
```

**Metrics Computed:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

**Output:**
- `results.json` in each model directory
- Console summary statistics

---

## Integration with Main Scripts

These tools complement the core training/rendering pipeline:

```
Main Pipeline:
1. tools/preprocessing/colmap_convert.py    ← Prepare dataset
2. train.py                                  ← Train model
3. render.py                                 ← Render views
4. metrics.py                                ← Compute metrics
5. tools/evaluation/full_eval.py            ← Full benchmark
```

**For Custom Datasets:**
```bash
# 1. Prepare images in dataset/input/
# 2. Run COLMAP
python tools/preprocessing/colmap_convert.py -s dataset/

# 3. Train
python train.py -s dataset/ -m output/model

# 4. Render
python render.py -m output/model

# 5. Evaluate
python metrics.py -m output/model
```

**For Volumetric/Cloud Data:**
```bash
# 1. Configure and run preprocessing
python tools/preprocessing/cloud_dataset_preprocessing.py

# 2. Train on prepared data
python train.py -s /prepared/dataset -m output/model
```

---

## Tips

### COLMAP Conversion
- Ensure images are properly exposed and sharp
- Use `--camera OPENCV` for standard cameras
- Use `--camera SIMPLE_PINHOLE` for well-calibrated cameras
- Enable `--resize` for faster training on large images

### Cloud Preprocessing
- Adjust point cloud bounds to fit your scene
- Use more points (>100K) for complex volumetric effects
- Verify transforms.json matches your camera setup

### Full Evaluation
- Requires significant compute time (hours per dataset)
- Can run in parallel on multiple GPUs
- Use `--skip_*` flags to resume partial runs

---

## Common Workflows

### New Real-World Dataset
```bash
# 1. Collect images in dataset/input/
# 2. Run COLMAP pipeline
python tools/preprocessing/colmap_convert.py -s dataset/ --resize

# 3. Train and evaluate
./run.sh test my_dataset
```

### Benchmark Reproduction
```bash
# Download standard datasets first
# Then run full evaluation
python tools/evaluation/full_eval.py \
    -m360 /data/mipnerf360 \
    -tat /data/tanksandtemples \
    -db /data/deepblending \
    --output_path ./paper_results
```

### Volumetric Scene Creation
```bash
# 1. Configure scene in cloud_dataset_preprocessing.py
# 2. Generate data
python tools/preprocessing/cloud_dataset_preprocessing.py

# 3. Train
python train.py -s /generated/dataset -m output/volumetric
```

---

For more information, see the main project README and [scripts/README.md](../scripts/README.md).
