# Usage Examples for Merged N-DGS Model

## Quick Start

The merged model is now the default for `--mode ndgs`. You can switch between parametrizations using the `--use_rot_scale_l_triangle` flag.

### Training Examples

#### 1. Basic 6DGS Training (NDGS-style, default)
```bash
python train.py \
    --source_path /path/to/colmap/data \
    --model_path output/my_scene \
    --mode ndgs \
    --input_dim 6
```

#### 2. Training with UBS-style Parametrization
```bash
python train.py \
    --source_path /path/to/colmap/data \
    --model_path output/my_scene_ubs \
    --mode ndgs \
    --input_dim 6 \
    --use_rot_scale_l_triangle
```

#### 3. 7DGS Training (with Time Dimension)
```bash
python train.py \
    --source_path /path/to/dynamic/scene \
    --model_path output/my_dynamic_scene \
    --mode ndgs \
    --input_dim 7
```

#### 4. 7DGS with UBS-style Parametrization
```bash
python train.py \
    --source_path /path/to/dynamic/scene \
    --model_path output/my_dynamic_scene_ubs \
    --mode ndgs \
    --input_dim 7 \
    --use_rot_scale_l_triangle
```

#### 5. Custom Learning Rates
```bash
python train.py \
    --source_path /path/to/data \
    --model_path output/my_scene \
    --mode ndgs \
    --diags_lr 0.01 \
    --l_triangs_lr 0.01 \
    --position_lr_init 0.00016
```

### Viewing Examples

#### 1. View Trained Model
```bash
python view.py \
    --ply output/my_scene/point_cloud/iteration_30000/point_cloud.ply \
    --mode ndgs
```

#### 2. View with Custom Port
```bash
python view.py \
    --ply output/my_scene/point_cloud/iteration_30000/point_cloud.ply \
    --mode ndgs \
    --port 8080
```

#### 3. View with Background Color
```bash
python view.py \
    --ply output/my_scene/point_cloud/iteration_30000/point_cloud.ply \
    --mode ndgs \
    --white_background
```

### Rendering Examples

#### 1. Render Test Views
```bash
python render.py \
    --model_path output/my_scene \
    --mode ndgs \
    --iteration 30000
```

#### 2. Render with Custom Resolution
```bash
python render.py \
    --model_path output/my_scene \
    --mode ndgs \
    --iteration 30000 \
    --resolution 2
```

## Python API Examples

### Example 1: Basic Model Creation
```python
from scene.gaussian_model_ndgs_merged import GaussianModel

# Create NDGS-style model
model = GaussianModel(
    sh_degree=3,
    input_dim=6,
    use_rot_scale_l_triangle=False
)

# Create UBS-style model
model_ubs = GaussianModel(
    sh_degree=3,
    input_dim=6,
    use_rot_scale_l_triangle=True
)
```

### Example 2: Initialize from Point Cloud
```python
from scene.gaussian_model_ndgs_merged import GaussianModel
from utils.graphics_utils import BasicPointCloud
import numpy as np

# Create model
model = GaussianModel(sh_degree=3, input_dim=6)

# Create point cloud
points = np.random.rand(1000, 3)
colors = np.random.rand(1000, 3)
normals = np.random.rand(1000, 3)
pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

# Initialize
model.create_from_pcd(pcd, spatial_lr_scale=1.0)
```

### Example 3: Get Covariance Matrix
```python
# Works for both parametrizations
covar = model.get_pc_v  # [N, D, D] full covariance matrix

# Get 3D scales (for visualization)
scales_3d = model.get_scaling  # [N, 3]
```

### Example 4: Training Setup
```python
from arguments import OptimizationParams
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
opt_params = OptimizationParams(parser)
args = parser.parse_args()
opt = opt_params.extract(args)

# Setup training
model.training_setup(opt)
```

### Example 5: Rendering
```python
# Render with TCGS
render_output = model.render_tcgs(
    viewpoint_camera=camera,
    render_mode="RGB",
    use_tcgs=False,
    is_test=False,
    scaling_modifier=1.0
)

image = render_output["render"]
radii = render_output["radii"]
visibility_filter = render_output["visibility_filter"]
```

## Comparison: When to Use Which Parametrization

### NDGS-style (default, `use_rot_scale_l_triangle=False`)
**Best for:**
- Standard 6DGS/7DGS scenes
- When you want bounded L-triangle values
- Faster convergence on simple scenes
- Default choice for most use cases

**Advantages:**
- Simpler initialization
- Bounded activations (numerical stability)
- Fewer hyperparameters to tune
- Works well out-of-the-box

### UBS-style (`use_rot_scale_l_triangle=True`)
**Best for:**
- Complex scenes with wide range of scales
- When explicit rotation encoding is beneficial
- Scenes with anisotropic structures
- Research comparing parametrizations

**Advantages:**
- KNN-based spatial scale initialization
- Explicit 6D rotation matrix
- Softplus activation (smooth gradients)
- More expressive for complex geometry

## Performance Tips

### 1. Learning Rate Tuning
```bash
# NDGS-style
python train.py --mode ndgs --diags_lr 0.01 --l_triangs_lr 0.01

# UBS-style
python train.py --mode ndgs --use_rot_scale_l_triangle \
    --scale_lr 0.005 --l_triangle_lr 0.001
```

### 2. Densification Parameters
```bash
python train.py --mode ndgs \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --densification_interval 100
```

### 3. Memory Optimization
```bash
# Reduce max points for large scenes
python train.py --mode ndgs --densify_until_iter 10000
```

## Troubleshooting

### Issue: NaN losses during training
**Solution:** Reduce learning rates, especially for covariance parameters
```bash
python train.py --mode ndgs --diags_lr 0.005 --l_triangs_lr 0.005
```

### Issue: Slow convergence
**Solution:** Try UBS-style with KNN initialization
```bash
python train.py --mode ndgs --use_rot_scale_l_triangle
```

### Issue: Artifacts in rendered images
**Solution:** Adjust densification parameters
```bash
python train.py --mode ndgs --percent_dense 0.01 --densify_grad_threshold 0.0002
```

### Issue: Model file incompatibility
**Solution:** Make sure to use the same `use_rot_scale_l_triangle` flag when loading as when training

## Advanced Usage

### Custom Covariance Regularization
```python
# Add custom loss term in training loop
v = model.get_pc_v  # [N, D, D]
regularization = torch.mean(torch.linalg.det(v))
loss += 0.01 * regularization
```

### Switching Parametrizations
```python
# Not directly supported - would need to convert parameters
# Recommended: Train separate models for each parametrization
```

### Export for Deployment
```python
# Save model
model.save_ply("output/final_model.ply")

# Load for inference
model_inference = GaussianModel(sh_degree=3, input_dim=6,
                                use_rot_scale_l_triangle=False)
model_inference.load_ply("output/final_model.ply")
```

## Performance Benchmarks

| Parametrization | Init Time | Training Speed | Memory | Quality |
|-----------------|-----------|----------------|---------|---------|
| NDGS-style      | Fast      | Fast           | Low     | High    |
| UBS-style       | Medium    | Medium         | Low     | Very High |

*Note: Benchmarks are approximate and depend on scene complexity*

## Next Steps

1. **Try both parametrizations** on your scene to see which works better
2. **Tune learning rates** for your specific use case
3. **Experiment with densification** parameters
4. **Monitor training** with TensorBoard or the live viewer
5. **Compare results** quantitatively using metrics.py

## Getting Help

- Check [MERGED_MODEL_README.md](MERGED_MODEL_README.md) for technical details
- Review training logs for errors
- Use `--detect_anomaly` flag for debugging NaN issues
- Enable verbose logging with `--debug_from 0`
