# Unified Parameter Naming in Merged N-DGS Model

## Overview

The merged Gaussian model now uses **unified parameter naming** across both parametrizations (NDGS-style and UBS-style), making the code more consistent and easier to maintain.

## Changes Made

### Before (Inconsistent Naming)

**NDGS-style** used:
- `self.diags` - diagonal covariance elements
- `self.l_triangs` - lower-triangular elements
- `self.diags_act()` / `self.diags_act_inv()` - activation functions
- `self.l_triangs_act()` / `self.l_triangs_act_inv()` - activation functions

**UBS-style** used:
- `self._scale` - scale parameters
- `self._l_triangle` - lower-triangular elements
- `self.scale_activation()` / `self.scale_inverse_activation()` - activation functions
- `self.l_triangs_activation()` / `self.l_triangs_inverse_activation()` - activation functions

### After (Unified Naming)

**Both parametrizations** now use:
- `self._scale` - scale/diagonal parameters (unified name)
- `self._l_triangle` - lower-triangular elements (unified name)
- `self.scale_activation()` / `self.scale_inverse_activation()` - activation functions
- `self.l_triangle_activation()` / `self.l_triangle_inverse_activation()` - activation functions

## Key Differences Between Parametrizations

The parametrizations differ only in their **activation functions**:

### NDGS-style (`use_rot_scale_l_triangle=False`)
```python
# _scale represents diagonal elements
self.scale_activation = lambda x: torch.exp(x)  # Exponential activation
self.scale_inverse_activation = lambda x: torch.log(...)

# _l_triangle elements
self.l_triangle_activation = lambda x: torch.sigmoid(x) * 2.0 - 1.0  # Bounded [-1, 1]
self.l_triangle_inverse_activation = lambda x: inverse_sigmoid(...)
```

**Covariance Construction:**
```python
diag = exp(_scale)  # Ensures positive diagonal
l_triang = sigmoid(_l_triangle) * 2 - 1  # Bounded off-diagonal
covar = l_triangle_to_covar(diag, l_triang)
```

### UBS-style (`use_rot_scale_l_triangle=True`)
```python
# _scale represents actual scale parameters
self.scale_activation = torch.nn.functional.softplus  # Smooth, positive
self.scale_inverse_activation = inverse_softplus

# _l_triangle elements (first 3 encode rotation)
self.l_triangle_activation = lambda x: x  # Identity (no activation)
self.l_triangle_inverse_activation = lambda x: x
```

**Covariance Construction:**
```python
rotation = l_triangle_to_rotmat(_l_triangle[:, :3])  # First 3 → 6D rotation
scale = softplus(_scale)
covar = rot_scale_l_triangle_to_covar(rotation, scale, _l_triangle, rest_i, rest_j)
```

## Benefits of Unified Naming

### 1. **Consistent Code Structure**
All methods now work with the same parameter names:
```python
# Before: Different names for different parametrizations
if self.use_rot_scale_l_triangle:
    new_scale = self._scale[mask]
    new_l_triangle = self._l_triangle[mask]
else:
    new_scale = self.diags[mask]
    new_l_triangle = self.l_triangs[mask]

# After: Unified names
new_scale = self._scale[mask]
new_l_triangle = self._l_triangle[mask]
```

### 2. **Simplified Optimizer Setup**
```python
# Always uses same parameter names in optimizer
l.append({'params': [self._scale], 'lr': scale_lr, "name": "scale"})
l.append({'params': [self._l_triangle], 'lr': l_triangle_lr, "name": "l_triangle"})
```

### 3. **Unified Properties**
```python
@property
def get_scale(self):
    """Works for both parametrizations"""
    return self.scale_activation(self._scale)

@property
def get_l_triangle(self):
    """Works for both parametrizations"""
    return self.l_triangle_activation(self._l_triangle)
```

### 4. **Consistent Save/Load**
PLY files now always save with the same attribute names:
- `scale_0`, `scale_1`, ... (instead of `diags_0` or `scale_0`)
- `l_triangle_0`, `l_triangle_1`, ... (instead of `l_triangs_0` or `l_triangle_0`)

### 5. **Backward Compatibility**
The `load_ply()` method can still load old files with `diags_*` and `l_triangs_*` naming:
```python
# Try new naming first, fall back to old naming
scale_names = [p.name for p in properties if p.name.startswith("scale_")]
if not scale_names:  # Backward compatibility
    scale_names = [p.name for p in properties if p.name.startswith("diags_")]
```

## Semantic Interpretation

### NDGS-style: `_scale` = Diagonal Elements
In NDGS mode, `_scale` represents the **diagonal elements** of the covariance decomposition:
- After `exp()` activation, these become positive values
- They control the variance along each dimension
- Combined with `_l_triangle` via `l_triangle_to_covar()`

### UBS-style: `_scale` = Scale Parameters
In UBS mode, `_scale` represents **actual scale** parameters:
- After `softplus()` activation, these become positive smoothly
- First 3 elements control spatial scales (initialized from KNN)
- Remaining elements control non-spatial scales
- Combined with explicit rotation matrix from `_l_triangle[:, :3]`

## Code Locations Updated

All references to old naming have been updated:

1. ✅ **`__init__`** - Initialize `_scale` and `_l_triangle` for both modes
2. ✅ **`get_scale`** / **`get_l_triangle`** - Unified property access
3. ✅ **`get_pc_v`** - Covariance construction
4. ✅ **`create_from_pcd`** - Initialization from point cloud
5. ✅ **`training_setup`** - Optimizer parameter groups
6. ✅ **`construct_list_of_attributes`** - PLY attribute names
7. ✅ **`save_ply`** - Save with unified names
8. ✅ **`load_ply`** - Load with backward compatibility
9. ✅ **`prune_points`** - Pruning operations
10. ✅ **`densification_postfix`** - Adding new Gaussians
11. ✅ **`densify_and_split`** - Splitting large Gaussians
12. ✅ **`densify_and_clone`** - Cloning small Gaussians

## Migration Guide

### For Users
**No action required!** The unified naming is transparent:
- Training with `--use_rot_scale_l_triangle` works the same
- Old PLY files load automatically with backward compatibility
- New PLY files use consistent naming

### For Developers
If you have custom code that accesses parameters directly:

**Update this:**
```python
# Old NDGS-style code
diags = model.diags
l_triangs = model.l_triangs
```

**To this:**
```python
# New unified code (works for both)
scale = model._scale
l_triangle = model._l_triangle
```

**Or use properties:**
```python
# Best practice: use activated values
scale = model.get_scale  # Already activated
l_triangle = model.get_l_triangle  # Already activated
```

## Testing

Run the test to verify unified naming:
```bash
cd /code/workspace/6dgs-iclr
python3 -c "
from scene.gaussian_model_ndgs_merged import GaussianModel

# Test both parametrizations
model_ndgs = GaussianModel(3, 6, use_rot_scale_l_triangle=False)
model_ubs = GaussianModel(3, 6, use_rot_scale_l_triangle=True)

# Verify unified naming
assert hasattr(model_ndgs, '_scale')
assert hasattr(model_ndgs, '_l_triangle')
assert hasattr(model_ubs, '_scale')
assert hasattr(model_ubs, '_l_triangle')

print('✅ Unified naming test passed!')
"
```

## Summary

The unified naming makes the codebase:
- **More consistent** - same parameter names across parametrizations
- **Easier to maintain** - less conditional code
- **Clearer** - semantic meaning preserved through activation functions
- **Backward compatible** - old PLY files still load
- **Future-proof** - easier to add new parametrizations

The key insight: **Both parametrizations use the same underlying parameters (`_scale`, `_l_triangle`), but with different activation functions to achieve their mathematical properties.**
