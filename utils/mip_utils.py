"""Mip-Splatting (Yu et al., CVPR 2024) support for the TCGS render path.

Two components, both implemented in the tcgs_speedy_rasterizer submodule:
  - 2D screen-space filter: `antialiasing=True` in the rasterization settings
    replaces the fixed 0.3-pixel dilation with a 0.1 dilation plus a detached
    opacity compensation, making the dilation approximately energy-preserving
    so splats keep their trained appearance when the camera zooms.
  - 3D smoothing filter: each Gaussian is convolved with an isotropic filter
    Sigma + v*I where v = 0.2 * (z_min / focal)^2 from the closest TRAINING
    camera that sees it, so no primitive is ever sharper than the maximal
    sampling rate that supervised it (prevents zoom-in erosion artifacts).

`filter_3D` is a per-Gaussian DERIVED buffer, not a learnable parameter: it
is recomputed from the training cameras (every ~100 iterations and after any
densification, since it depends on positions and primitive count), detached,
and never registered with the optimizer. `filter_3D is None` means
Mip-Splatting is disabled and rendering is bit-identical to before.

At inference the filter is frozen at its end-of-training value (persisted as
a `filter_3D` property in the PLY): recomputing it from a novel/zoom camera
would let the model render sharper than it was ever supervised.
"""

import math

import torch


class MipFilterMixin:
    """Mip-Splatting 3D-filter state for models rendered through TCGS.

    Mix into a Gaussian model class. The model must expose `get_xyz`; the
    render path calls `mip_filtered(...)` on the tensors it is about to hand
    to the rasterizer and passes the returned `antialiasing` flag into
    `GaussianRasterizationSettings`.
    """

    # None = Mip-Splatting disabled (the default; class attribute so models
    # that predate the feature and PLYs without the property need no changes).
    filter_3D = None

    @torch.no_grad()
    def update_3d_filter(self, cameras):
        """(Re)compute the per-Gaussian 3D filter variance from the training
        cameras and enable Mip-Splatting rendering. Call after scene setup,
        every ~100 training iterations, and after any densification step."""
        from tcgs_speedy_rasterizer import compute_3d_filter

        views, focals, sizes = [], [], []
        for cam in cameras:
            width = int(cam.image_width)
            height = int(cam.image_height)
            focal_x = width / (2.0 * math.tan(cam.FoVx * 0.5))
            focal_y = height / (2.0 * math.tan(cam.FoVy * 0.5))
            views.append(cam.world_view_transform)
            focals.append(max(focal_x, focal_y))
            sizes.append((width, height))
        self.filter_3D = compute_3d_filter(self.get_xyz, views, focals, sizes)

    def mip_filtered(self, opacity, scales=None, cov6=None):
        """Apply the 3D smoothing filter to what is about to be rasterized.

        Pass exactly one of `scales` [N, 3] or `cov6` [N, 6] (upper-triangular
        3D covariance). Returns (scales_or_cov6, opacity, antialiasing) where
        antialiasing is the flag for the rasterization settings. No-op
        returning the inputs unchanged (antialiasing=False) when disabled.
        Differentiable w.r.t. scales/cov6/opacity; filter_3D is a constant.
        """
        if self.filter_3D is None:
            return (scales if cov6 is None else cov6), opacity, False
        n = (scales if cov6 is None else cov6).shape[0]
        if self.filter_3D.shape[0] != n:
            raise RuntimeError(
                f"filter_3D has {self.filter_3D.shape[0]} entries but the model has "
                f"{n} Gaussians; call update_3d_filter(train_cameras) after any "
                "densification/pruning step.")
        from tcgs_speedy_rasterizer import apply_3d_smoothing_filter

        out, opacity = apply_3d_smoothing_filter(
            opacity, self.filter_3D, scales=scales, cov6=cov6)
        return out, opacity, True

    # --- PLY persistence -------------------------------------------------
    # The filter rides along in the PLY as one extra float property so that
    # render.py / view.py / web export work without access to the training
    # cameras. Absent property = Mip disabled, keeping old PLYs loadable.

    def mip_ply_attributes(self):
        """Extra attribute names for construct_list_of_attributes()."""
        n = self.get_xyz.shape[0]
        if self.filter_3D is not None and self.filter_3D.numel() == n:
            return ["filter_3D"]
        return []

    def mip_ply_columns(self):
        """Extra [N, 1] float32 columns for save_ply(), matching
        mip_ply_attributes() order."""
        if not self.mip_ply_attributes():
            return []
        return [self.filter_3D.detach().reshape(-1, 1).cpu().numpy()]

    def mip_load_from_ply(self, plydata):
        """Restore filter_3D from a PLY element if the property is present."""
        names = {p.name for p in plydata.elements[0].properties}
        if "filter_3D" in names:
            import numpy as np
            self.filter_3D = torch.tensor(
                np.asarray(plydata.elements[0]["filter_3D"]),
                dtype=torch.float, device="cuda")
        else:
            self.filter_3D = None
