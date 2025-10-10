# NDGS Conversion Status Report

## Completed Files ✅

### 1. `gaussian_model_ndgs.py` - **COMPLETE** ✓
- Full 7DGS support implemented
- Reference implementation for all patterns
- All methods support both 6D and 7D inputs

### 2. `gaussian_model_ndgs_rot.py` - **COMPLETE** ✓
- Rotation-scale-l_triangle parametrization
- Full 7DGS support with time dimension
- All critical methods updated:
  - `slice_gaussian` - handles 7D m_2
  - `get_xyz_normal` - returns 7D when needed
  - `create_from_pcd` - initializes time parameter
  - `training_setup` - adds time to optimizer
  - `construct_list_of_attributes` - includes time
  - `save_ply` / `load_ply` - persists time parameter
  - `prune_points` - handles time pruning
  - `densification_postfix` - propagates time
  - `densify_and_split` / `densify_and_clone` - clones time
  - `render_tcgs` - appends timestamp to query

## Partially Completed Files 🔄

### 3. `gaussian_model_ndgs_plus.py` - **PARTIAL** 🔄
**Completed:**
- ✅ Constructor with `input_dim` parameter
- ✅ `slice_gaussian` with 7DGS support
- ✅ Time initialization in `create_from_pcd`
- ✅ Time parameter in `training_setup`

**Remaining:**
- ⏳ Update `construct_list_of_attributes` to include 'mean_time'
- ⏳ Update `save_ply` to save time parameter
- ⏳ Update `load_ply` to load time parameter
- ⏳ Update `prune_points` to handle time
- ⏳ Update `densification_postfix` signature and logic
- ⏳ Update `densify_and_split` to clone time
- ⏳ Update `densify_and_clone` to copy time
- ⏳ Update `render_tcgs` to append timestamp

### 4. `gaussian_model_ndgs_2sh.py` - **PARTIAL** 🔄
**Completed:**
- ✅ Constructor with `input_dim` parameter
- ✅ Dual SH features structure maintained

**Remaining:**
- ⏳ All same updates as ndgs_plus (see above list)
- ⏳ Special handling for dual SH in save/load
- ⏳ Special handling for dual SH in densification

## Quick Reference: Key Changes Pattern

For each method that needs updating in remaining files:

### Pattern 1: `slice_gaussian`
```python
# For 7DGS, m_2 includes both normal (3D) and time (1D)
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    normal_normalized = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)
    m_2 = torch.cat([normal_normalized, self._mean_time], dim=-1)  # [N, 4]
else:
    m_2 = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3]
```

### Pattern 2: `get_xyz_normal` (if exists)
```python
@property
def get_xyz_normal(self):
    if self.input_dim == 7 and hasattr(self, '_mean_time'):
        return torch.cat([self._xyz, self._normal, self._mean_time], dim=-1)
    return torch.cat([self._xyz, self._normal], dim=-1)
```

### Pattern 3: `create_from_pcd` - Time Init
```python
# After normal initialization, before print statement:
# For 7DGS, initialize time dimension
if self.input_dim == 7:
    mean_time = torch.empty(init_n_gs, 1, device=device).uniform_(0.0, 1.0)
    self._mean_time = nn.Parameter(mean_time.requires_grad_(True))
```

### Pattern 4: `training_setup` - Optimizer
```python
# Before self.optimizer = torch.optim.Adam(l, ...):
# Add time parameter for 7DGS
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    l.append({'params': [self._mean_time], 'lr': training_args.feature_lr, "name": "mean_time"})
```

### Pattern 5: `construct_list_of_attributes`
```python
l.append('opacity')
# Add time for 7DGS
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    l.append('mean_time')
# Then append diags, l_triangs, etc.
```

### Pattern 6: `save_ply`
```python
# Build attributes list, including time for 7DGS
attr_list = [xyz, normals, f_dc, f_rest, opacities]
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    mean_time = self._mean_time.detach().cpu().numpy()
    attr_list.append(mean_time)
attr_list.extend([diags, l_triangs])  # or [opacityscales, diags, l_triangs] for plus variant

attributes = np.concatenate(attr_list, axis=1)
```

### Pattern 7: `load_ply` - Load Time
```python
# After loading opacities:
# Load time dimension for 7DGS
mean_time = None
if self.input_dim == 7:
    try:
        mean_time = np.asarray(plydata.elements[0]["mean_time"])[..., np.newaxis]
    except:
        mean_time = np.zeros((xyz.shape[0], 1), dtype=np.float32)

# After creating all other parameters:
# Load time parameter for 7DGS
if self.input_dim == 7 and mean_time is not None:
    self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(True))
```

### Pattern 8: `prune_points`
```python
# After assigning all optimizable tensors:
# Handle time parameter for 7DGS
if self.input_dim == 7 and "mean_time" in optimizable_tensors:
    self._mean_time = optimizable_tensors["mean_time"]
```

### Pattern 9: `densification_postfix` - Signature
```python
def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_normal, \
                                new_opacities, new_diags, new_l_triangs, new_mean_time=None):
    # or with new_opacityscales for plus variant
```

### Pattern 10: `densification_postfix` - Logic
```python
d = {"xyz": new_xyz, ...}
# Add time parameter for 7DGS
if self.input_dim == 7 and new_mean_time is not None:
    d["mean_time"] = new_mean_time

# Later, after cat_tensors_to_optimizer:
# Handle time parameter for 7DGS
if self.input_dim == 7 and "mean_time" in optimizable_tensors:
    self._mean_time = optimizable_tensors["mean_time"]
```

### Pattern 11: `densify_and_split` / `densify_and_clone`
```python
# Before calling densification_postfix:
# Handle time parameter for 7DGS
new_mean_time = None
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1)  # for split
    # or without .repeat() for clone: new_mean_time = self._mean_time[selected_pts_mask]

# Update the densification_postfix call to include new_mean_time
```

### Pattern 12: `render_tcgs`
```python
# Replace: cond_params = dir_pp / dir_pp.norm(dim=1, keepdim=True)
# With:
view_dir = dir_pp / dir_pp.norm(dim=1, keepdim=True)

# For 7DGS, append timestamp to query
if self.input_dim == 7:
    timestamp = torch.full(
        (view_dir.shape[0], 1),
        viewpoint_camera.timestamp if hasattr(viewpoint_camera, 'timestamp') else 0.0,
        device=view_dir.device,
        dtype=view_dir.dtype,
    )
    cond_params = torch.cat([view_dir, timestamp], dim=-1)
else:
    cond_params = view_dir
```

## Testing Checklist

Once all files are updated, test:

- [ ] Load 6DGS model with `input_dim=6`
- [ ] Train 6DGS model with `input_dim=6`
- [ ] Save/Load 6DGS model
- [ ] Load 7DGS model with `input_dim=7`
- [ ] Train 7DGS model with `input_dim=7`
- [ ] Save/Load 7DGS model with time parameter
- [ ] Verify densification works with time
- [ ] Verify rendering works with timestamps

## File Cleanup

After verification, remove old 6DGS files:
```bash
rm gaussian_model_6dgs.py
rm gaussian_model_6dgs_rot.py
rm gaussian_model_6dgs_plus.py
rm gaussian_model_6dgs_2sh.py
```

## Summary

**Status:**
- ✅ 2 files complete (ndgs.py, ndgs_rot.py)
- 🔄 2 files partial (ndgs_plus.py, ndgs_2sh.py)

**Estimated remaining work:**
- ~10-15 method updates per file (plus, 2sh)
- Follow the 12 patterns above
- Each pattern appears 1-2 times per file
- Total: ~40-60 small edits across 2 files

**Recommendation:**
Use `gaussian_model_ndgs.py` or `gaussian_model_ndgs_rot.py` as reference and apply patterns systematically to each remaining file.
