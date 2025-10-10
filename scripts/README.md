# 6DGS Experiment Scripts

This directory contains organized scripts for running various experiments with the 6D Gaussian Splatting model.

## Directory Structure

```
scripts/
├── benchmarks/     # Standard benchmark evaluations
├── ablations/      # Ablation study experiments
└── tests/          # Test and debug scripts
```

## Usage

From the project root directory, use the master `run.sh` script:

```bash
./run.sh <category> <script>
```

### Quick Start

```bash
# Show help
./run.sh --help

# List all available scripts
./run.sh list

# Run a benchmark
./run.sh benchmark mipnerf360

# Run an ablation study
./run.sh ablation nerf_synthetic

# Run a test
./run.sh test ct_data
```

---

## Benchmarks

Standard benchmark evaluations for paper results.

### Mip-NeRF 360 (`mipnerf360.sh`)
Evaluates on 3 Mip-NeRF 360 scenes:
- room
- stump
- treehill

**Usage:**
```bash
./run.sh benchmark mipnerf360
```

**Iterations:** 500, 2000, 7000, 15000, 30000

---

### NeRF Synthetic (`nerf_synthetic.sh`)
Evaluates on all NeRF synthetic scenes (chair, drums, ficus, hotdog, lego, materials, mic, ship).

**Usage:**
```bash
./run.sh benchmark nerf_synthetic
```

**Iterations:** 500, 2000, 7000, 15000, 30000

---

### Tanks & Temples (`tanks_temples.sh`)
Evaluates on volumetric T&T scenes for rebuttal:
- subsurface_dragon2

**Usage:**
```bash
./run.sh benchmark tanks_temples
```

**Iterations:** 500, 2000, 7000, 15000, 30000

---

### Shiny Blender (`shiny_blender.sh`)
Evaluates on 8 Shiny Blender scenes:
- cd, crest, food, giants, lab, pasta, seasoning, tools

**Usage:**
```bash
./run.sh benchmark shiny_blender
```

**Iterations:** 500, 2000, 7000, 15000, 30000

---

## Ablations

Ablation study experiments for analyzing model components.

### NeRF Synthetic Ablation (`nerf_synthetic.sh`)
Runs ablation on NeRF synthetic dataset (currently lego only).

**Usage:**
```bash
./run.sh ablation nerf_synthetic
```

**Output:** `../output/6dgs_ablation/{scene}_v2`

---

### DeepDRR Ablation (`deepdrr.sh`)
Ablation on specific DeepDRR clinic datasets:
- dataset6_CLINIC_0003_data
- dataset6_CLINIC_0002_data

**Usage:**
```bash
./run.sh ablation deepdrr
```

**Output:** `../output/ablation/init-even/{scene}`

---

### DeepDRR Entangled (`deepdrr_entangled.sh`)
Tests entangled features on all DeepDRR datasets.

**Usage:**
```bash
./run.sh ablation deepdrr_entangled
```

**Output:** `../output/ablation2/ddgs_entangled/{scene}`

**Iterations:** 500, 2000, 7000, 15000, 30000

---

## Tests

Development and debugging scripts.

### CT Data (`ct_data.sh`)
Quick test on 4 CT datasets:
- chest, foot, abdomen, jaw

**Usage:**
```bash
./run.sh test ct_data
```

---

### DeepDRR Test (`deepdrr.sh`)
Test training on all DeepDRR subdirectories.

**Usage:**
```bash
./run.sh test deepdrr
```

**Output:** `../output/ablation/init-even/{scene}`

---

### NAF CT Test (`naf_ct.sh`)
Render and evaluate existing NAF CT models (training commented out):
- abdomen_50, chest_50, foot_50, jaw_50

**Usage:**
```bash
./run.sh test naf_ct
```

**Output:** `../output/orignal-separate/{scene}`

---

### DICOM Test (`dicom.sh`)
Test on single DICOM CT dataset:
- 22022107v2

**Usage:**
```bash
./run.sh test dicom
```

**Output:** `../output/test/3dgs/{scene}`

---

### Ljubljana Scaled (`ljubljana_scaled.sh`)
Train on Ljubljana datasets (only `-scaled` suffixed directories).

**Usage:**
```bash
./run.sh test ljubljana_scaled
```

---

### DGS Tanks & Temples (`dgs_tanks_temples.sh`)
DGS variant on T&T volumetric scenes:
- bunny_cloud_cut, cloud_cut, explosion, smoke, translucent_suzanne_cut, subsurface_dragon2

**Usage:**
```bash
./run.sh test dgs_tanks_temples
```

**Output:** `output/dgs/{scene}`

---

### DGS NeRF Synthetic (`dgs_nerf.sh`)
DGS variant on all NeRF synthetic scenes.

**Usage:**
```bash
./run.sh test dgs_nerf
```

**Output:** `output/dgs/{scene}`

---

## Script Pipeline

Each script typically follows this pipeline:

1. **Training:** `python train.py -s <source> --model_path <output> --eval`
2. **Rendering:** `python render.py -m <output> --skip_train --iteration <iter>`
3. **Metrics:** `python metrics.py -m <output>`

### Common Iterations
- **Full evaluation:** 500, 2000, 7000, 15000, 30000
- **Final only:** 30000

---

## Tips

1. **Check dataset paths** - Ensure your datasets are in the expected locations (`/code/dataset/...` or `../dataset/...`)
2. **Modify dataset lists** - Edit individual scripts to enable/disable specific scenes
3. **Adjust output paths** - Change `--model_path` arguments to organize results differently
4. **Parallel execution** - Run multiple scripts in parallel on different GPUs

---

## Adding New Scripts

1. Create script in appropriate category directory
2. Follow the naming convention: `<dataset_name>.sh`
3. Add shebang and descriptive header comment
4. Use consistent variable names and formatting
5. Test with `./run.sh <category> <script_name>`

---

## Troubleshooting

**Script not found:**
```bash
./run.sh list  # Check available scripts
```

**Dataset path errors:**
- Verify dataset locations match script paths
- Update paths in individual scripts if needed

**Permission denied:**
```bash
chmod +x run.sh
chmod +x scripts/**/*.sh
```

---

For more information, see the main project README.
