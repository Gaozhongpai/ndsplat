# Merged N-DGS Model with Flexible Covariance Parametrization

## Overview

The merged Gaussian model (`gaussian_model_ndgs_merged.py`) combines two different covariance parametrization approaches into a single unified implementation:

1. **NDGS-style (diagonal-l_triangle)**: Direct parametrization using diagonal and lower-triangular elements
2. **UBS-style (rotation-scale-l_triangle)**: Rotation-scale parametrization with lower-triangular elements

## Key Features

- **Unified Interface**: Both parametrizations share the same API and rendering pipeline
- **Flexible Switching**: Control via `use_rot_scale_l_triangle` flag
- **Full Feature Support**: Both support 6DGS and 7DGS (with time dimension)
- **CUDA-Accelerated**: Both use optimized gsplat kernels for covariance computation

## Usage

### Command Line Arguments

The `use_rot_scale_l_triangle` flag has been added to `ModelParams` in [arguments/__init__.py](arguments/__init__.py):

```python
self.use_rot_scale_l_triangle = False  # Default: NDGS-style
```

### Training

```bash
# Train with NDGS-style (diagonal-l_triangle) - DEFAULT
python train.py --source_path /path/to/data --model_path /path/to/output --mode ndgs

# Train with UBS-style (rotation-scale-l_triangle)
python train.py --source_path /path/to/data --model_path /path/to/output --mode ndgs --use_rot_scale_l_triangle

# Train with 7DGS (with time dimension)
python train.py --source_path /path/to/data --model_path /path/to/output --mode ndgs --input_dim 7

# Train with 7DGS + UBS-style parametrization
python train.py --source_path /path/to/data --model_path /path/to/output --mode ndgs --input_dim 7 --use_rot_scale_l_triangle
```

### Viewing

```bash
# View trained model (parametrization is loaded from saved .ply file)
python view.py --ply /path/to/point_cloud.ply --mode ndgs

# View with specific parametrization flag (must match training)
python view.py --ply /path/to/point_cloud.ply --mode ndgs --use_rot_scale_l_triangle
```

### Python API

```python
from scene.gaussian_model_ndgs_merged import GaussianModel

# NDGS-style (default)
model_ndgs = GaussianModel(
    sh_degree=3,
    input_dim=6,
    use_rot_scale_l_triangle=False  # NDGS-style
)

# UBS-style
model_ubs = GaussianModel(
    sh_degree=3,
    input_dim=6,
    use_rot_scale_l_triangle=True  # UBS-style
)

# 7DGS with time dimension
model_7dgs = GaussianModel(
    sh_degree=3,
    input_dim=7,
    use_rot_scale_l_triangle=False
)
```

## Technical Details

### NDGS-Style Parametrization (`use_rot_scale_l_triangle=False`)

**Parameters:**
- `diags`: [N, D] diagonal elements
- `l_triangs`: [N, D*(D-1)/2] lower-triangular elements

**Activations:**
- `diags_act`: `exp(x)` (ensures positivity)
- `l_triangs_act`: `sigmoid(x) * 2 - 1` (bounded [-1, 1])

**Covariance Construction:**
```python
diag = exp(diags)
l_triang = sigmoid(l_triangs) * 2 - 1
covar = l_triangle_to_covar(diag, l_triang)  # CUDA kernel
```

**Initialization:**
- Uniform covariance bias (`cov_bias = 1e-1`)
- Diagonal: ones × cov_bias
- Lower-triangular: zeros

### UBS-Style Parametrization (`use_rot_scale_l_triangle=True`)

**Parameters:**
- `_scale`: [N, D] scale parameters
- `_l_triangle`: [N, D*(D-1)/2] lower-triangular elements

**Activations:**
- `scale_activation`: `softplus(x)` (smooth, positive)
- `l_triangs_activation`: identity (no activation)

**Covariance Construction:**
```python
rotation = l_triangle_to_rotmat(l_triangle[:, :3])  # First 3 elements → 6D rotation
scale = softplus(_scale)
covar = rot_scale_l_triangle_to_covar(rotation, scale, l_triangle, rest_i, rest_j)  # CUDA kernel
```

**Initialization:**
- Spatial scales (first 3): KNN-based initialization
- Non-spatial scales: small random noise
- Lower-triangular: small random noise

## Learning Rates

The optimizer automatically selects the appropriate learning rates:

**NDGS-style:**
- `diags_lr`: from `training_args.diags_lr` (default: 1e-2)
- `l_triangs_lr`: from `training_args.l_triangs_lr` (default: 1e-2)

**UBS-style:**
- `scale_lr`: from `training_args.scale_lr` or falls back to `diags_lr` (default: 5e-3)
- `l_triangle_lr`: from `training_args.l_triangle_lr` or falls back to `l_triangs_lr` (default: 1e-3)

## File Structure

### Modified Files

1. **[scene/gaussian_model_ndgs_merged.py](scene/gaussian_model_ndgs_merged.py)** (NEW)
   - Merged Gaussian model implementation
   - Supports both parametrizations

2. **[arguments/__init__.py](arguments/__init__.py)**
   - Added `use_rot_scale_l_triangle` flag to `ModelParams`

3. **[train.py](train.py)**
   - Updated to pass `use_rot_scale_l_triangle` flag when mode is "ndgs"

4. **[view.py](view.py)**
   - Updated to pass `use_rot_scale_l_triangle` flag when mode is "ndgs"

5. **[scene/__init__.py](scene/__init__.py)**
   - Updated `get_gaussian_model()` to use merged model for "ndgs" mode
   - Added "ndgs_original" mode for backward compatibility

### Original Files (Preserved)

- **[scene/gaussian_model_ndgs.py](scene/gaussian_model_ndgs.py)**: Original NDGS implementation
- **[scene/gaussian_model_ubs.py](scene/gaussian_model_ubs.py)**: Original UBS implementation
- **[scene/gaussian_model_ndgs_rot.py](scene/gaussian_model_ndgs_rot.py)**: If exists

## Backward Compatibility

The original implementations are preserved:
- Use `--mode ndgs_original` to use the original NDGS implementation
- Old saved models can still be loaded with the appropriate mode

## Comparison: NDGS vs UBS Parametrization

| Aspect | NDGS-style | UBS-style |
|--------|-----------|-----------|
| **Diagonal Activation** | `exp(x)` | `softplus(x)` |
| **L-triangle Activation** | `sigmoid(x)*2-1` | identity |
| **Initialization** | Uniform bias | KNN-based |
| **Rotation Encoding** | Implicit in L | Explicit (first 3 L elements) |
| **Spatial Scale Init** | Same as non-spatial | KNN distances |
| **Numerical Stability** | Good (bounded L) | Excellent (softplus) |
| **Expressiveness** | High | Very High |

## Common Operations

### Densification and Pruning

Both parametrizations share the same densification logic:
- `densify_and_clone()`: Clone small Gaussians with high gradients
- `densify_and_split()`: Split large Gaussians with high gradients
- `densify_and_prune()`: Remove low-opacity and oversized Gaussians

### Conditional Slicing

Both use the same CUDA-accelerated conditional Gaussian slicing:
```python
# Training mode
m_cond, cov3D_precomp, pdf_cond = slice_gaussian(query, c_dim=3, lambda_opc=0.35)

# Test mode (precomputed)
m_cond, pdf_cond = slice_gaussian_test(query, lambda_opc=0.35)
```

### Rendering

Both use the TCGS rasterizer with the same interface:
```python
render_output = gaussians.render_tcgs(
    viewpoint_camera,
    render_mode="RGB",
    use_tcgs=False,
    is_test=False,
    scaling_modifier=1.0
)
```

## Debugging Tips

1. **Check parametrization**: Verify which parametrization is active:
   ```python
   print(f"Using rotation-scale-l_triangle: {model.use_rot_scale_l_triangle}")
   ```

2. **Verify parameters exist**:
   ```python
   if model.use_rot_scale_l_triangle:
       assert hasattr(model, '_scale') and hasattr(model, '_l_triangle')
   else:
       assert hasattr(model, 'diags') and hasattr(model, 'l_triangs')
   ```

3. **Check covariance computation**:
   ```python
   v = model.get_pc_v  # Should work for both parametrizations
   print(f"Covariance shape: {v.shape}")  # [N, D, D]
   ```

## Performance Considerations

- Both parametrizations use CUDA-accelerated covariance computation
- UBS-style may have slightly more computation due to rotation matrix construction
- Memory usage is similar (same number of parameters)
- Training speed depends on initialization and convergence properties

## Future Extensions

The merged model architecture makes it easy to:
1. Add new parametrizations (e.g., quaternion-based)
2. Experiment with different activation functions
3. Implement hybrid approaches
4. Add regularization terms specific to each parametrization

## References

- **N-DGS**: Neural Directional Gaussian Splatting
- **UBS**: Universal Basis Splatting (ICLR 2026)
- **6DGS/7DGS**: 6D/7D Gaussian Splatting with conditional distributions
