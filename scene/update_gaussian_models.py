#!/usr/bin/env python3
"""
Script to update gaussian model files from 6DGS to NDGS (supporting both 6DGS and 7DGS).
"""

import re
import os


def add_time_support_to_slice_gaussian(content):
    """Add time dimension support to slice_gaussian method."""
    # Update slice_gaussian to handle 7DGS
    pattern = r'(def slice_gaussian\(self, q, c_dim=3, lambda_opc=0\.35\):.*?""")\s+(m_1 = self\.get_xyz.*?)\s+(m_2 = self\.get_normal / self\.get_normal\.norm\(dim=1, keepdim=True\))'

    replacement = r'''\1
        m_1 = self.get_xyz  # [N, 3]

        # For 7DGS, m_2 includes both normal (3D) and time (1D)
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            normal_normalized = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3]
            m_2 = torch.cat([normal_normalized, self._mean_time], dim=-1)  # [N, 4]
        else:
            m_2 = self.get_normal / self.get_normal.norm(dim=1, keepdim=True)  # [N, 3]'''

    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    return content


def add_get_xyz_normal_time_support(content):
    """Update get_xyz_normal property to handle time."""
    pattern = r'@property\s+def get_xyz_normal\(self\):\s+return torch\.cat\(\[self\._xyz, self\._normal\], dim=-1\)'

    replacement = '''@property
    def get_xyz_normal(self):
        # For 6DGS: [xyz, normal_xyz] -> 6D
        # For 7DGS: [xyz, normal_xyz, time] -> 7D (if _mean_time exists)
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            return torch.cat([self._xyz, self._normal, self._mean_time], dim=-1)
        return torch.cat([self._xyz, self._normal], dim=-1)'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_init_in_create_from_pcd(content):
    """Add time initialization in create_from_pcd."""
    pattern = r'(normal = \(dir / dir\.norm\(dim=1, keepdim=True\)\)\.float\(\)\.cuda\(\))\s+(print\("Number of points at initialisation :)'

    replacement = r'''\1

        # For 7DGS, initialize time dimension
        if self.input_dim == 7:
            mean_time = torch.empty(init_n_gs, 1, device=device).uniform_(0.0, 1.0)
            self._mean_time = nn.Parameter(mean_time.requires_grad_(True))

        \2'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_training_setup(content):
    """Add time parameter to training setup."""
    pattern = r'(\{'params': \[self\.l_triangs\], \'lr\': training_args\.l_triangs_lr, "name": "l_triangs"\},\s*\])\s+(self\.optimizer = torch\.optim\.Adam\(l,)'

    replacement = r'''\1

        # Add time parameter for 7DGS
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            l.append({'params': [self._mean_time], 'lr': training_args.feature_lr, "name": "mean_time"})

        \2'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_construct_attributes(content):
    """Add time to construct_list_of_attributes."""
    pattern = r'(l\.append\(\'opacity\'\))\s+(for i in range\(self\.diags\.shape\[1\]\):)'

    replacement = r'''\1
        # Add time for 7DGS
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            l.append('mean_time')
        \2'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_save_ply(content):
    """Add time handling in save_ply."""
    pattern = r'(elements = np\.empty\(xyz\.shape\[0\], dtype=dtype_full\))\s+(attributes = np\.concatenate\(\(xyz, normals, f_dc, f_rest, opacities,)'

    replacement = r'''\1

        # Build attributes list, including time for 7DGS
        attr_list = [xyz, normals, f_dc, f_rest, opacities]
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            mean_time = self._mean_time.detach().cpu().numpy()
            attr_list.append(mean_time)
        attr_list.extend(['''

    # Find what comes after opacities
    if 'opacityscales' in content:
        pattern = r'(elements = np\.empty\(xyz\.shape\[0\], dtype=dtype_full\))\s+(attributes = np\.concatenate\(\(xyz, normals, f_dc, f_rest, opacities, opacityscales, diags, l_triangs\), axis=1\))'
        replacement = r'''\1

        # Build attributes list, including time for 7DGS
        attr_list = [xyz, normals, f_dc, f_rest, opacities]
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            mean_time = self._mean_time.detach().cpu().numpy()
            attr_list.append(mean_time)
        attr_list.extend([opacityscales, diags, l_triangs])

        attributes = np.concatenate(attr_list, axis=1)'''
    else:
        pattern = r'(elements = np\.empty\(xyz\.shape\[0\], dtype=dtype_full\))\s+(attributes = np\.concatenate\(\(xyz, normals, f_dc, f_rest, opacities, diags, l_triangs\), axis=1\))'
        replacement = r'''\1

        # Build attributes list, including time for 7DGS
        attr_list = [xyz, normals, f_dc, f_rest, opacities]
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            mean_time = self._mean_time.detach().cpu().numpy()
            attr_list.append(mean_time)
        attr_list.extend([diags, l_triangs])

        attributes = np.concatenate(attr_list, axis=1)'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_load_ply(content):
    """Add time handling in load_ply."""
    pattern = r'(opacities = np\.ascontiguousarray\(opacities, dtype=np\.float32\))\s+(features_dc = np\.zeros)'

    replacement = r'''\1

        # Load time dimension for 7DGS
        mean_time = None
        if self.input_dim == 7:
            try:
                mean_time = np.asarray(plydata.elements[0]["mean_time"])[..., np.newaxis]
            except:
                # Initialize with default values if not present
                mean_time = np.zeros((xyz.shape[0], 1), dtype=np.float32)

        \2'''

    content = re.sub(pattern, replacement, content)

    # Add time parameter loading
    pattern = r'(self\.l_triangs = nn\.Parameter\(torch\.tensor\(l_triangs, dtype=torch\.float, device="cuda"\)\.requires_grad_\(True\)\))\s+(### test)'

    replacement = r'''\1

        # Load time parameter for 7DGS
        if self.input_dim == 7 and mean_time is not None:
            self._mean_time = nn.Parameter(torch.tensor(mean_time, dtype=torch.float, device="cuda").requires_grad_(True))

        \2'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_prune_points(content):
    """Add time handling in prune_points."""
    pattern = r'(self\.l_triangs = optimizable_tensors\["l_triangs"\])\s+(self\.xyz_gradient_accum)'

    replacement = r'''\1

        # Handle time parameter for 7DGS
        if self.input_dim == 7 and "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]

        \2'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_densification_postfix(content):
    """Add time handling in densification_postfix."""
    # Update function signature
    pattern = r'def densification_postfix\(self, new_xyz, new_features_dc, new_features_rest, new_normal, \\\s+new_opacities,'

    if 'new_opacityscales' in content:
        content = re.sub(
            r'(def densification_postfix\(self, new_xyz, new_features_dc, new_features_rest, new_normal, \\\s+new_opacities, new_opacityscales, new_diags, new_l_triangs\):)',
            r'def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_normal, \\\n                                    new_opacities, new_opacityscales, new_diags, new_l_triangs, new_mean_time=None):',
            content
        )
    else:
        content = re.sub(
            r'(def densification_postfix\(self, new_xyz, new_features_dc, new_features_rest, new_normal, \\\s+new_opacities, new_diags, new_l_triangs\):)',
            r'def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_normal, \\\n                                    new_opacities, new_diags, new_l_triangs, new_mean_time=None):',
            content
        )

    # Add time to dict
    pattern = r'("l_triangs" : new_l_triangs,\s+\})\s+(optimizable_tensors = self\.cat_tensors_to_optimizer\(d\))'

    replacement = r'''\1

        # Add time parameter for 7DGS
        if self.input_dim == 7 and new_mean_time is not None:
            d["mean_time"] = new_mean_time

        \2'''

    content = re.sub(pattern, replacement, content)

    # Add time tensor assignment
    pattern = r'(self\.l_triangs = optimizable_tensors\["l_triangs"\])\s+(self\.xyz_gradient_accum = torch\.zeros)'

    replacement = r'''\1

        # Handle time parameter for 7DGS
        if self.input_dim == 7 and "mean_time" in optimizable_tensors:
            self._mean_time = optimizable_tensors["mean_time"]

        \2'''

    content = re.sub(pattern, replacement, content)
    return content


def add_time_to_densify_methods(content):
    """Add time handling in densify methods."""
    # For densify_and_split and densify_and_clone
    patterns = [
        # densify_and_split - before densification_postfix call
        (r'(new_l_triangs = self\.l_triangs\[selected_pts_mask\]\.repeat\(N, 1\))\s+(self\.densification_postfix)',
         r'''\1

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask].repeat(N, 1)

        \2'''),

        # densify_and_clone
        (r'(new_l_triangs = self\.l_triangs\[selected_pts_mask\])\s+(self\.densification_postfix)',
         r'''\1

        # Handle time parameter for 7DGS
        new_mean_time = None
        if self.input_dim == 7 and hasattr(self, '_mean_time'):
            new_mean_time = self._mean_time[selected_pts_mask]

        \2'''),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Update all densification_postfix calls to include new_mean_time
    if 'new_opacityscales' in content:
        content = re.sub(
            r'self\.densification_postfix\(new_xyz, new_features_dc, new_features_rest, new_normal, new_opacities, new_opacityscales, new_diags, new_l_triangs\)',
            r'self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal, new_opacities, new_opacityscales, new_diags, new_l_triangs, new_mean_time)',
            content
        )
    else:
        content = re.sub(
            r'self\.densification_postfix\(new_xyz, new_features_dc, new_features_rest, new_normal,\s+new_opacity, new_diags, new_l_triangs\)',
            r'self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_normal,\n                                   new_opacity, new_diags, new_l_triangs, new_mean_time)',
            content
        )

    return content


def add_time_to_render_tcgs(content):
    """Add time handling in render_tcgs."""
    pattern = r'(dir_pp = \(self\.get_xyz - viewpoint_camera\.camera_center\.repeat\(self\.get_normal\.shape\[0\], 1\)\))\s+(cond_params = dir_pp / dir_pp\.norm\(dim=1, keepdim=True\))\s+(lambda_opc = 0\.35)'

    replacement = r'''\1
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

        \3'''

    content = re.sub(pattern, replacement, content)
    return content


def update_plus_file():
    """Update 6dgs_plus to ndgs_plus."""
    print("Updating gaussian_model_ndgs_plus.py...")

    with open('gaussian_model_ndgs_plus.py', 'r') as f:
        content = f.read()

    # Apply all transformations
    content = add_time_support_to_slice_gaussian(content)
    content = add_get_xyz_normal_time_support(content) if 'get_xyz_normal' in content else content
    content = add_time_init_in_create_from_pcd(content)
    content = add_time_to_training_setup(content)
    content = add_time_to_construct_attributes(content)
    content = add_time_to_save_ply(content)
    content = add_time_to_load_ply(content)
    content = add_time_to_prune_points(content)
    content = add_time_to_densification_postfix(content)
    content = add_time_to_densify_methods(content)
    content = add_time_to_render_tcgs(content)

    with open('gaussian_model_ndgs_plus.py', 'w') as f:
        f.write(content)

    print("Updated gaussian_model_ndgs_plus.py ✓")


if __name__ == '__main__':
    os.chdir('/code/workspace/6dgs-iclr/scene')
    update_plus_file()
    print("\nDone! gaussian_model_ndgs_plus.py now supports both 6DGS and 7DGS.")
