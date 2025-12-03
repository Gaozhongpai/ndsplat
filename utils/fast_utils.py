#
# FastGS-style multi-view consistent densification utilities for N-DGS
# Adapted from FastGS (arXiv:2511.04283)
#

import torch
import random


def sampling_cameras(my_viewpoint_stack, num_cams=10):
    """Randomly sample a given number of cameras from the viewpoint stack.

    Args:
        my_viewpoint_stack: List of camera viewpoints
        num_cams: Number of cameras to sample (default: 10)

    Returns:
        List of sampled camera viewpoints
    """
    camlist = []
    stack_copy = my_viewpoint_stack.copy()

    # Limit to available cameras
    num_cams = min(num_cams, len(stack_copy))

    for _ in range(num_cams):
        loc = random.randint(0, len(stack_copy) - 1)
        camlist.append(stack_copy.pop(loc))

    return camlist


def get_loss(reconstructed_image, original_image):
    """Compute per-pixel L1 loss and normalize to [0, 1].

    Args:
        reconstructed_image: Rendered image [C, H, W]
        original_image: Ground truth image [C, H, W]

    Returns:
        Normalized L1 loss [H, W] in range [0, 1]
    """
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), dim=0).detach()

    # Normalize to [0, 1]
    min_val = torch.min(l1_loss)
    max_val = torch.max(l1_loss)
    if max_val - min_val > 1e-8:
        l1_loss_norm = (l1_loss - min_val) / (max_val - min_val)
    else:
        l1_loss_norm = torch.zeros_like(l1_loss)

    return l1_loss_norm


def compute_photometric_loss(viewpoint_cam, image, lambda_dssim=0.2):
    """Compute combined L1 + SSIM loss for a rendered image.

    Args:
        viewpoint_cam: Camera viewpoint with original_image
        image: Rendered image [C, H, W]
        lambda_dssim: Weight for SSIM loss (default: 0.2)

    Returns:
        Scalar photometric loss
    """
    from utils.loss_utils import l1_loss, ssim

    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
    return loss


def compute_gaussian_score_fastgs(camlist, gaussians, background, loss_thresh=0.3, DENSIFY=False):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist: List of viewpoint camera objects to render from
        gaussians: GaussianModel instance with render_tcgs_with_metric method
        background: Background color tensor [3]
        loss_thresh: Threshold for high-error pixel detection (default: 0.3)
        DENSIFY: Whether to compute and return importance_score for densification

    Returns:
        importance_score: Per-Gaussian counts of high-error views (only if DENSIFY=True)
        pruning_score: Normalized per-Gaussian score for pruning guidance [0, 1]
    """
    full_metric_counts = None
    full_metric_score = None

    for view_idx, viewpoint_cam in enumerate(camlist):
        # First render without metric counting to compute loss
        render_pkg = gaussians.render_tcgs_with_metric(
            viewpoint_cam,
            background=background,
            get_flag=False,
            metric_map=None
        )
        render_image = render_pkg["render"]

        # Compute photometric loss
        photometric_loss = compute_photometric_loss(viewpoint_cam, render_image)

        # Compute per-pixel L1 loss and create metric map
        gt_image = viewpoint_cam.original_image.cuda()
        l1_loss_norm = get_loss(render_image, gt_image)

        # Create binary metric map: 1 for high-error pixels, 0 otherwise
        metric_map = (l1_loss_norm > loss_thresh).int().flatten()

        # Second render with metric counting
        render_pkg = gaussians.render_tcgs_with_metric(
            viewpoint_cam,
            background=background,
            get_flag=True,
            metric_map=metric_map
        )

        accum_loss_counts = render_pkg["accum_metric_counts"]

        # Accumulate importance score (for densification)
        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone().float()
            else:
                full_metric_counts += accum_loss_counts.float()

        # Accumulate pruning score (weighted by photometric loss)
        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone().float()
        else:
            full_metric_score += photometric_loss * accum_loss_counts.float()

    # Normalize pruning score to [0, 1]
    min_score = torch.min(full_metric_score)
    max_score = torch.max(full_metric_score)
    if max_score - min_score > 1e-8:
        pruning_score = (full_metric_score - min_score) / (max_score - min_score)
    else:
        pruning_score = torch.zeros_like(full_metric_score)

    # Compute importance score (average across views)
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None

    return importance_score, pruning_score
