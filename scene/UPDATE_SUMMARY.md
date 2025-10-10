# Conversion Summary: 6DGS to NDGS (6DGS+7DGS Support)

## Files Updated:
1. ✅ gaussian_model_6dgs_rot.py → gaussian_model_ndgs_rot.py  
2. ✅ gaussian_model_6dgs_plus.py → gaussian_model_ndgs_plus.py (partial)  
3. ⏳ gaussian_model_6dgs_2sh.py → gaussian_model_ndgs_2sh.py (pending)

## Key Changes for All Files:

### 1. Constructor (`__init__`)
```python
def __init__(self, sh_degree: int, input_dim: int = 6):
    self.input_dim = input_dim  # 6 for 6DGS, 7 for 7DGS
    self.gs_dim = input_dim  # Use input_dim instead of hardcoded 6
    # ... rest of init
```

### 2. Slice Gaussian (`slice_gaussian`)
```python
# For 7DGS, m_2 includes both normal (3D) and time (1D)
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    normal_normalized = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)
    m_2 = torch.cat([normal_normalized, self._mean_time], dim=-1)  # [N, 4]
else:
    m_2 = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3]
```

### 3. Get XYZ Normal Property
```python
@property
def get_xyz_normal(self):
    if self.input_dim == 7 and hasattr(self, '_mean_time'):
        return torch.cat([self._xyz, self._normal, self._mean_time], dim=-1)
    return torch.cat([self._xyz, self._normal], dim=-1)
```

### 4. Create from PCD
```python
# For 7DGS, initialize time dimension
if self.input_dim == 7:
    mean_time = torch.empty(init_n_gs, 1, device=device).uniform_(0.0, 1.0)
    self._mean_time = nn.Parameter(mean_time.requires_grad_(True))
```

### 5. Training Setup
```python
# Add time parameter for 7DGS
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    l.append({'params': [self._mean_time], 'lr': training_args.feature_lr, "name": "mean_time"})
```

### 6. Construct List of Attributes
```python
# Add time for 7DGS
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    l.append('mean_time')
```

### 7. Save PLY
```python
# Build attributes list, including time for 7DGS
attr_list = [xyz, normals, f_dc, f_rest, opacities]
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    mean_time = self._mean_time.detach().cpu().numpy()
    attr_list.append(mean_time)
attr_list.extend([...])  # rest of attributes
```

### 8. Load PLY
```python
# Load time dimension for 7DGS
mean_time = None
if self.input_dim == 7:
    try:
        mean_time = np.asarray(plydata.elements[0]["mean_time"])[..., np.newaxis]
    except:
        mean_time = np.zeros((xyz.shape[0], 1), dtype=np.float32)

# ... later ...
# Load time parameter for 7DGS
if self.input_dim == 7 and mean_time is not None:
    self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(True))
```

### 9. Prune Points
```python
# Handle time parameter for 7DGS
if self.input_dim == 7 and "mean_time" in optimizable_tensors:
    self._mean_time = optimizable_tensors["mean_time"]
```

### 10. Densification Postfix
```python
def densification_postfix(self, ..., new_mean_time=None):
    d = {...}
    # Add time parameter for 7DGS
    if self.input_dim == 7 and new_mean_time is not None:
        d["mean_time"] = new_mean_time
    
    # ... later ...
    # Handle time parameter for 7DGS
    if self.input_dim == 7 and "mean_time" in optimizable_tensors:
        self._mean_time = optimizable_tensors["mean_time"]
```

### 11. Densify and Split/Clone
```python
# Handle time parameter for 7DGS
new_mean_time = None
if self.input_dim == 7 and hasattr(self, '_mean_time'):
    new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1)  # for split
    # or without .repeat() for clone

self.densification_postfix(..., new_mean_time)
```

### 12. Render TCGS
```python
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

## Status:
- ✅ gaussian_model_ndgs.py - Already complete (reference implementation)
- ✅ gaussian_model_ndgs_rot.py - Complete with all 7DGS support
- 🔄 gaussian_model_ndgs_plus.py - Partial (needs manual completion)
- ⏳ gaussian_model_ndgs_2sh.py - Not started

