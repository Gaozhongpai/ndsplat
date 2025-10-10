# Project Organization Guide

Complete reference for the reorganized 6DGS codebase structure.

---

## Quick Navigation

- [Current Structure](#current-structure)
- [File Locations](#file-locations)
- [Migration Guide](#migration-guide-old--new)
- [Usage Examples](#usage-examples)
- [Documentation Index](#documentation-index)

---

## Current Structure

```
6dgs-iclr/
│
├── 📄 Core Scripts (Root Level)
│   ├── train.py               # Main training script
│   ├── render.py              # Rendering script
│   ├── metrics.py             # Metrics computation
│   ├── __init__.py            # Package marker
│   └── run.sh                 # Master experiment runner
│
├── 🔧 tools/                  # Utility scripts
│   ├── README.md              # Tools documentation
│   ├── preprocessing/         # Data preparation
│   │   ├── cloud_dataset_preprocessing.py  # Volumetric data prep
│   │   └── colmap_convert.py              # COLMAP pipeline
│   └── evaluation/            # Evaluation utilities
│       ├── full_eval.py       # Automated benchmarking
│       └── metrics_ndgs.py    # NDGS metrics
│
├── 🎬 scripts/                # Experiment runners
│   ├── README.md              # Scripts documentation
│   ├── benchmarks/            # Standard evaluations (4 scripts)
│   │   ├── mipnerf360.sh
│   │   ├── nerf_synthetic.sh
│   │   ├── shiny_blender.sh
│   │   └── tanks_temples.sh
│   ├── ablations/             # Ablation studies (3 scripts)
│   │   ├── deepdrr.sh
│   │   ├── deepdrr_entangled.sh
│   │   └── nerf_synthetic.sh
│   └── tests/                 # Dev/debug scripts (7 scripts)
│       ├── ct_data.sh
│       ├── deepdrr.sh
│       ├── dgs_nerf.sh
│       ├── dgs_tanks_temples.sh
│       ├── dicom.sh
│       ├── ljubljana_scaled.sh
│       └── naf_ct.sh
│
├── 📦 Model & Utilities
│   ├── scene/                 # Scene representation
│   ├── gaussian_renderer/     # Rendering engine
│   ├── utils/                 # Helper functions
│   ├── arguments/             # Argument parsers
│   └── submodules/           # Third-party code
│
└── 📚 Documentation
    ├── README.md              # Main project README
    ├── CLAUDE.md              # Development guidance
    ├── LICENSE.md             # License
    ├── ORGANIZATION.md        # This file
    ├── scripts/README.md      # Scripts reference
    └── tools/README.md        # Tools reference
```

---

## File Locations

### Core Pipeline (Root Level)

**Why at root?** Most frequently used - primary interface

| File | Purpose | Usage |
|------|---------|-------|
| `train.py` | Train Gaussian models | `python train.py -s <dataset> -m <output>` |
| `render.py` | Render novel views | `python render.py -m <model>` |
| `metrics.py` | Compute PSNR/SSIM/LPIPS | `python metrics.py -m <model>` |
| `run.sh` | Master experiment runner | `./run.sh <category> <script>` |

### Tools Directory

**Why separate?** Preprocessing/evaluation utilities used before/after main pipeline

#### Preprocessing (`tools/preprocessing/`)

| File | Purpose | Usage |
|------|---------|-------|
| `colmap_convert.py` | Convert images → COLMAP | `python tools/preprocessing/colmap_convert.py -s <dir>` |
| `cloud_dataset_preprocessing.py` | Generate volumetric data | Configure and run directly |

#### Evaluation (`tools/evaluation/`)

| File | Purpose | Usage |
|------|---------|-------|
| `full_eval.py` | Automated benchmarking | `python tools/evaluation/full_eval.py -m360 <path> ...` |
| `metrics_ndgs.py` | NDGS-specific metrics | `python tools/evaluation/metrics_ndgs.py -m <model>` |

### Scripts Directory

**Why separate?** Batch experiment automation, organized by purpose

| Category | Scripts | Purpose |
|----------|---------|---------|
| `benchmarks/` | 4 scripts | Standard dataset evaluations for papers |
| `ablations/` | 3 scripts | Ablation study experiments |
| `tests/` | 7 scripts | Development and debugging |

**Access via:** `./run.sh <category> <script>`

---

## Migration Guide (Old → New)

### Python Scripts

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `cloud_dataset_preprocssing.py` | `tools/preprocessing/cloud_dataset_preprocessing.py` | Typo fixed! |
| `convert.py` | `tools/preprocessing/colmap_convert.py` | More descriptive name |
| `full_eval.py` | `tools/evaluation/full_eval.py` | Moved to tools |
| `metrics_ndgs.py` | `tools/evaluation/metrics_ndgs.py` | Moved to tools |
| `train.py` | `train.py` | Stayed at root |
| `render.py` | `render.py` | Stayed at root |
| `metrics.py` | `metrics.py` | Stayed at root |

### Bash Scripts

| Old Name | New Location | Category |
|----------|--------------|----------|
| `run.sh` | `scripts/tests/ct_data.sh` | test |
| `run_datasets_6dgs_360.sh` | `scripts/benchmarks/mipnerf360.sh` | benchmark |
| `run_datasets_6dgs_nerf.sh` | `scripts/benchmarks/nerf_synthetic.sh` | benchmark |
| `run_datasets_6dgs_tank.sh` | `scripts/benchmarks/tanks_temples.sh` | benchmark |
| `run_datasets_6dgs_shiny.sh` | `scripts/benchmarks/shiny_blender.sh` | benchmark |
| `run_datasets_6dgs_ablation.sh` | `scripts/ablations/nerf_synthetic.sh` | ablation |
| `run_datasets_ablation.sh` | `scripts/ablations/deepdrr.sh` | ablation |
| `run_datasets_ablation_rebuttal.sh` | `scripts/ablations/deepdrr_entangled.sh` | ablation |
| `run_datasets.sh` | `scripts/tests/deepdrr.sh` | test |
| `run_datasets_dicom.sh` | `scripts/tests/dicom.sh` | test |
| `run_datasets_test.sh` | `scripts/tests/naf_ct.sh` | test |
| `run_datasets_scale.sh` | `scripts/tests/ljubljana_scaled.sh` | test |
| `run_datasets_dgs_tank.sh` | `scripts/tests/dgs_tanks_temples.sh` | test |
| `run_datasets_dgs_nerf.sh` | `scripts/tests/dgs_nerf.sh` | test |

### New Master Script

All bash scripts now accessed through unified interface:

**Before:**
```bash
bash run_datasets_6dgs_360.sh
```

**After:**
```bash
./run.sh benchmark mipnerf360
```

---

## Usage Examples

### Core Training Workflow

```bash
# Train a model
python train.py -s /path/to/dataset -m output/my_model --eval

# Render views
python render.py -m output/my_model --skip_train

# Compute metrics
python metrics.py -m output/my_model
```

### Data Preprocessing

```bash
# Convert images to COLMAP format
python tools/preprocessing/colmap_convert.py -s /path/to/images --resize

# Generate volumetric dataset
# (Edit script first to configure scene)
python tools/preprocessing/cloud_dataset_preprocessing.py
```

### Running Experiments

```bash
# Show all available commands
./run.sh --help

# List all scripts
./run.sh list

# Run a benchmark
./run.sh benchmark mipnerf360
./run.sh benchmark nerf_synthetic

# Run an ablation study
./run.sh ablation nerf_synthetic
./run.sh ablation deepdrr

# Run a test
./run.sh test ct_data
./run.sh test dicom
```

### Full Benchmark Evaluation

```bash
# Complete automated evaluation on standard datasets
python tools/evaluation/full_eval.py \
    -m360 /data/mipnerf360 \
    -tat /data/tanksandtemples \
    -db /data/deepblending \
    --output_path ./results
```

---

## Design Principles

### Organization Rules

✅ **Keep at Root:**
- Core pipeline scripts (train, render, metrics)
- Master run.sh
- Main documentation
- Package files

✅ **Move to `tools/`:**
- Data preprocessing utilities
- Evaluation automation
- One-time setup scripts
- Specialized converters

✅ **Move to `scripts/`:**
- Batch experiment runners
- Dataset-specific training scripts
- Benchmark automation
- Ablation study runners

❌ **Don't Move:**
- Model code (scene/, gaussian_renderer/)
- Utility functions (utils/)
- Argument parsers (arguments/)
- Submodules

### Benefits

1. **Clear Hierarchy**
   - Root = Core functionality
   - tools/ = Utilities
   - scripts/ = Automation

2. **Easy Discovery**
   - Logical categorization
   - Self-documenting structure
   - Clear README files

3. **Maintainability**
   - Consistent organization
   - Separation of concerns
   - Easy to extend

4. **User-Friendly**
   - Core scripts at root (easy to find)
   - Advanced tools organized away
   - Single entry point (run.sh)

---

## Documentation Index

### Main Documentation
- **[README.md](README.md)** - Main project documentation
- **[CLAUDE.md](CLAUDE.md)** - Development guidance for AI assistants
- **[LICENSE.md](LICENSE.md)** - License information
- **[ORGANIZATION.md](ORGANIZATION.md)** - This file

### Detailed Guides
- **[scripts/README.md](scripts/README.md)** - Complete scripts reference
- **[tools/README.md](tools/README.md)** - Tools documentation

### Quick References
- `./run.sh --help` - Built-in help for experiment runner
- `./run.sh list` - List all available scripts

---

## Quick Reference Card

```bash
# I want to...

# Train a model
python train.py -s <dataset> -m <output>

# Render views
python render.py -m <model>

# Compute metrics
python metrics.py -m <model>

# Prepare new images
python tools/preprocessing/colmap_convert.py -s <images_dir>

# Run benchmark
./run.sh benchmark <dataset_name>

# Run ablation
./run.sh ablation <experiment_name>

# Run quick test
./run.sh test <test_name>

# See all options
./run.sh --help
./run.sh list
```

---

**Last Updated:** After complete project reorganization  
**Structure Version:** 2.0  
**Total Scripts Organized:** 14 bash scripts + 4 Python utilities
