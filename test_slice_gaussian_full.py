"""
Gradient verification test for slice_gaussian_full CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- Position: xyz, v_12, L_22_inv, view_mean
- Rotation: rotation, rotation_delta, L_22_inv_diag_rot (time-only, when C=4)

Note: Scale is NOT conditioned in slice_gaussian_full - use get_scaling directly.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import both CUDA and PyTorch implementations
from gsplat import slice_gaussian_full
from gsplat.cuda._torch_impl import _slice_gaussian_full, _quaternion_slerp, _quaternion_multiply


def test_slice_gaussian_full_gradients(N=1000, C=3, use_pos=True, use_rot=True, seed=42):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients for ALL parameters.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view, 4 for view+time)
        use_pos: Whether to use position conditioning
        use_rot: Whether to use rotation conditioning (only meaningful when C=4)
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    device = "cuda"

    # Rotation conditioning only works with C=4 (view+time)
    if use_rot and C != 4:
        print(f"Warning: Rotation conditioning requires C=4, but C={C}. Disabling rotation conditioning.")
        use_rot = False

    print("=" * 70)
    print("slice_gaussian_full Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}, use_rot={use_rot}")
    print()

    # Create test inputs
    xyz = torch.randn(N, 3, device=device, requires_grad=True)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    view_mean.requires_grad_(True)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # Position conditioning
    if use_pos:
        v_12 = torch.randn(N, 3 * C, device=device, requires_grad=True) * 0.1
    else:
        v_12 = None

    # L_22_inv: Full Cholesky [N, C*(C+1)/2]
    n_L_22_inv = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_L_22_inv, device=device, requires_grad=True) * 0.5

    # Rotation conditioning (time-only when C=4)
    rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation.requires_grad_(True)
    if use_rot:
        rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        rotation_delta.requires_grad_(True)
        L_22_inv_diag_rot = torch.randn(N, 1, device=device, requires_grad=True) * 0.5
    else:
        rotation_delta = None
        L_22_inv_diag_rot = None

    lambda_opc = 0.35

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    xyz_pt = xyz.detach().clone().requires_grad_(True)
    view_mean_pt = view_mean.detach().clone().requires_grad_(True)
    v_12_pt = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_pt = L_22_inv.detach().clone().requires_grad_(True)
    rotation_pt = rotation.detach().clone().requires_grad_(True)
    rotation_delta_pt = rotation_delta.detach().clone().requires_grad_(True) if rotation_delta is not None else None
    L_22_inv_diag_rot_pt = L_22_inv_diag_rot.detach().clone().requires_grad_(True) if L_22_inv_diag_rot is not None else None

    # Forward pass with PyTorch
    x_cond_pt, rotation_cond_pt, attention_pt = _slice_gaussian_full(
        xyz_pt, view_mean_pt, query,
        v_12_pt, L_22_inv_pt,
        rotation_pt, rotation_delta_pt, L_22_inv_diag_rot_pt,
        lambda_opc
    )

    # Create random upstream gradients
    grad_x_cond = torch.randn_like(x_cond_pt)
    grad_rotation_cond = torch.randn_like(rotation_cond_pt)
    grad_attention = torch.randn_like(attention_pt)

    # Backward pass with PyTorch
    loss_pt = (
        (x_cond_pt * grad_x_cond).sum() +
        (rotation_cond_pt * grad_rotation_cond).sum() +
        (attention_pt * grad_attention).sum()
    )
    loss_pt.backward()

    # Store PyTorch gradients
    pt_grads = {
        'xyz': xyz_pt.grad.clone(),
        'view_mean': view_mean_pt.grad.clone(),
        'L_22_inv': L_22_inv_pt.grad.clone(),
        'rotation': rotation_pt.grad.clone(),
    }
    if v_12_pt is not None:
        pt_grads['v_12'] = v_12_pt.grad.clone()
    if rotation_delta_pt is not None:
        pt_grads['rotation_delta'] = rotation_delta_pt.grad.clone()
    if L_22_inv_diag_rot_pt is not None:
        pt_grads['L_22_inv_diag_rot'] = L_22_inv_diag_rot_pt.grad.clone()

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    xyz_cuda = xyz.detach().clone().requires_grad_(True)
    view_mean_cuda = view_mean.detach().clone().requires_grad_(True)
    v_12_cuda = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_cuda = L_22_inv.detach().clone().requires_grad_(True)
    rotation_cuda = rotation.detach().clone().requires_grad_(True)
    rotation_delta_cuda = rotation_delta.detach().clone().requires_grad_(True) if rotation_delta is not None else None
    L_22_inv_diag_rot_cuda = L_22_inv_diag_rot.detach().clone().requires_grad_(True) if L_22_inv_diag_rot is not None else None

    # Forward pass with CUDA kernel
    x_cond_cuda, rotation_cond_cuda, attention_cuda = slice_gaussian_full(
        xyz_cuda, view_mean_cuda, query,
        v_12_cuda, L_22_inv_cuda,
        rotation_cuda, rotation_delta_cuda, L_22_inv_diag_rot_cuda,
        lambda_opc
    )

    # Backward pass with same upstream gradients
    loss_cuda = (
        (x_cond_cuda * grad_x_cond).sum() +
        (rotation_cond_cuda * grad_rotation_cond).sum() +
        (attention_cuda * grad_attention).sum()
    )
    loss_cuda.backward()

    # Store CUDA gradients
    cuda_grads = {
        'xyz': xyz_cuda.grad.clone(),
        'view_mean': view_mean_cuda.grad.clone(),
        'L_22_inv': L_22_inv_cuda.grad.clone(),
        'rotation': rotation_cuda.grad.clone(),
    }
    if v_12_cuda is not None:
        cuda_grads['v_12'] = v_12_cuda.grad.clone()
    if rotation_delta_cuda is not None:
        cuda_grads['rotation_delta'] = rotation_delta_cuda.grad.clone()
    if L_22_inv_diag_rot_cuda is not None:
        cuda_grads['L_22_inv_diag_rot'] = L_22_inv_diag_rot_cuda.grad.clone()

    # ============ Compare gradients ============
    print()
    print("Gradient Comparison Results:")
    print("-" * 70)

    def compare_grads(name, grad_pt, grad_cuda):
        """Compare gradients and print statistics."""
        # Absolute error
        abs_err = (grad_pt - grad_cuda).abs()

        # Relative error (with epsilon to avoid division by zero)
        rel_err = abs_err / (grad_pt.abs() + 1e-8)

        # Cosine similarity
        cos_sim = F.cosine_similarity(grad_pt.flatten().unsqueeze(0),
                                       grad_cuda.flatten().unsqueeze(0)).item()

        print(f"\n{name}:")
        print(f"  Shape: {tuple(grad_pt.shape)}")
        print(f"  Absolute Error:  mean={abs_err.mean():.6f}, max={abs_err.max():.6f}, std={abs_err.std():.6f}")
        print(f"  Relative Error:  mean={rel_err.mean():.6f}, max={rel_err.max():.6f}, std={rel_err.std():.6f}")
        print(f"  Cosine Similarity: {cos_sim:.6f}")
        print(f"  PyTorch grad norm: {grad_pt.norm():.6f}")
        print(f"  CUDA grad norm:    {grad_cuda.norm():.6f}")

        return {
            'abs_err_mean': abs_err.mean().item(),
            'abs_err_max': abs_err.max().item(),
            'rel_err_mean': rel_err.mean().item(),
            'rel_err_max': rel_err.max().item(),
            'cos_sim': cos_sim,
        }

    results = {}

    # Position gradients
    print("\n--- POSITION GRADIENTS ---")
    results['xyz'] = compare_grads("xyz (position)", pt_grads['xyz'], cuda_grads['xyz'])
    if 'v_12' in pt_grads:
        results['v_12'] = compare_grads("v_12 (position-view covariance)", pt_grads['v_12'], cuda_grads['v_12'])
    results['L_22_inv'] = compare_grads(f"L_22_inv (full {C}x{C} Cholesky)", pt_grads['L_22_inv'], cuda_grads['L_22_inv'])

    # Rotation gradients
    print("\n--- ROTATION GRADIENTS ---")
    results['rotation'] = compare_grads("rotation (q_base)", pt_grads['rotation'], cuda_grads['rotation'])
    if 'rotation_delta' in pt_grads:
        results['rotation_delta'] = compare_grads("rotation_delta (q_delta)", pt_grads['rotation_delta'], cuda_grads['rotation_delta'])
    if 'L_22_inv_diag_rot' in pt_grads:
        results['L_22_inv_diag_rot'] = compare_grads("L_22_inv_diag_rot (time precision)", pt_grads['L_22_inv_diag_rot'], cuda_grads['L_22_inv_diag_rot'])

    # Shared gradient
    print("\n--- SHARED GRADIENTS ---")
    results['view_mean'] = compare_grads("view_mean (shared canonical view)", pt_grads['view_mean'], cuda_grads['view_mean'])

    # ============ Summary ============
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if gradients are acceptable
    all_good = True
    for name, res in results.items():
        cos_sim = res['cos_sim']
        status = "GOOD" if cos_sim > 0.99 else ("WARNING" if cos_sim > 0.9 else "BAD")
        if cos_sim < 0.99:
            all_good = False
        print(f"{name}: cosine_sim={cos_sim:.4f} {status}")

    print()
    if all_good:
        print("All gradients are accurate (cosine similarity > 0.99)")
    else:
        print("Some gradients have significant error!")
        print("This may affect training convergence.")

    return results


def test_forward_output_match(N=100, C=3, use_pos=True, use_rot=False, seed=42):
    """
    Verify that CUDA and PyTorch forward passes produce identical outputs.
    """
    torch.manual_seed(seed)
    device = "cuda"

    # Rotation conditioning only works with C=4
    if use_rot and C != 4:
        use_rot = False

    print()
    print("=" * 70)
    print("Forward Pass Output Comparison")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}, use_rot={use_rot}")

    # Create test inputs
    xyz = torch.randn(N, 3, device=device)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # Full Cholesky parameters
    v_12 = torch.randn(N, 3 * C, device=device) * 0.1 if use_pos else None
    n_L_22_inv = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_L_22_inv, device=device) * 0.5

    rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1) if use_rot else None
    L_22_inv_diag_rot = torch.randn(N, 1, device=device) * 0.5 if use_rot else None

    lambda_opc = 0.35

    # PyTorch forward
    x_cond_pt, rotation_cond_pt, attention_pt = _slice_gaussian_full(
        xyz, view_mean, query,
        v_12, L_22_inv,
        rotation, rotation_delta, L_22_inv_diag_rot,
        lambda_opc
    )

    # CUDA forward
    x_cond_cuda, rotation_cond_cuda, attention_cuda = slice_gaussian_full(
        xyz, view_mean, query,
        v_12, L_22_inv,
        rotation, rotation_delta, L_22_inv_diag_rot,
        lambda_opc
    )

    # Compare outputs
    print(f"\nx_cond difference:        max={torch.abs(x_cond_pt - x_cond_cuda).max():.8f}")
    print(f"rotation_cond difference: max={torch.abs(rotation_cond_pt - rotation_cond_cuda).max():.8f}")
    print(f"attention difference:     max={torch.abs(attention_pt - attention_cuda).max():.8f}")

    all_match = (
        torch.allclose(x_cond_pt, x_cond_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(rotation_cond_pt, rotation_cond_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(attention_pt, attention_cuda, rtol=1e-4, atol=1e-4)
    )

    if all_match:
        print("\nForward outputs match!")
    else:
        print("\nForward outputs differ!")

    return all_match


def test_different_configurations():
    """
    Test gradient accuracy with different configurations.
    """
    print()
    print("=" * 70)
    print("Testing Different Configurations")
    print("=" * 70)

    configs = [
        (3, True, False, "C=3, pos=True, rot=False (6DGS view-only)"),
        (3, False, False, "C=3, pos=False, rot=False (no position shift)"),
        (4, True, False, "C=4, pos=True, rot=False (7DGS without rotation)"),
        (4, True, True, "C=4, pos=True, rot=True (7DGS full)"),
        (4, False, True, "C=4, pos=False, rot=True (rotation only)"),
    ]

    all_results = {}
    for C, use_pos, use_rot, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_slice_gaussian_full_gradients(N=500, C=C, use_pos=use_pos, use_rot=use_rot, seed=42)
            all_results[desc] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[desc] = None

    return all_results


def test_rotation_at_different_t_values():
    """
    Test gradient accuracy at different interpolation factors (t values) for rotation.

    The approximation should be more accurate when:
    - t ~ 0 (near canonical time): delta gets ~0 gradient
    - t ~ 1 (far from canonical): delta gets full gradient
    """
    print("\n" + "=" * 70)
    print("Testing rotation gradient accuracy at different t values")
    print("=" * 70)

    N = 100
    C = 4  # Must be 4 for rotation conditioning
    device = "cuda"
    torch.manual_seed(42)

    # Test at different time residuals (which determine attention_rot and t)
    time_offsets = [0.01, 0.1, 0.5, 1.0, 2.0]

    for time_offset in time_offsets:
        xyz = torch.randn(N, 3, device=device, requires_grad=True)

        # Create view_mean with specific time component
        view_mean = torch.zeros(N, C, device=device)
        view_mean[:, :3] = F.normalize(torch.randn(N, 3, device=device), dim=-1)
        view_mean[:, 3] = 0.5  # canonical time = 0.5
        view_mean.requires_grad_(True)

        # Create query with offset time
        query = view_mean.detach().clone()
        query[:, 3] = 0.5 + time_offset  # offset from canonical

        # Other parameters - full Cholesky
        v_12 = torch.randn(N, 3 * C, device=device, requires_grad=True) * 0.1
        n_L_22_inv = C * (C + 1) // 2
        L_22_inv = torch.zeros(N, n_L_22_inv, device=device, requires_grad=True)

        rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        rotation.requires_grad_(True)
        rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        rotation_delta.requires_grad_(True)
        L_22_inv_diag_rot = torch.zeros(N, 1, device=device, requires_grad=True)  # exp(0)^2 = 1

        lambda_opc = 0.35

        # PyTorch forward/backward
        rotation_pt = rotation.detach().clone().requires_grad_(True)
        rotation_delta_pt = rotation_delta.detach().clone().requires_grad_(True)

        x_cond_pt, rotation_cond_pt, attention_pt = _slice_gaussian_full(
            xyz.detach(), view_mean.detach(), query,
            v_12.detach(), L_22_inv.detach(),
            rotation_pt, rotation_delta_pt, L_22_inv_diag_rot.detach(),
            lambda_opc
        )

        grad_rotation_cond = torch.randn_like(rotation_cond_pt)
        loss_pt = (rotation_cond_pt * grad_rotation_cond).sum()
        loss_pt.backward()

        # CUDA forward/backward
        rotation_cuda = rotation.detach().clone().requires_grad_(True)
        rotation_delta_cuda = rotation_delta.detach().clone().requires_grad_(True)

        x_cond_cuda, rotation_cond_cuda, attention_cuda = slice_gaussian_full(
            xyz.detach(), view_mean.detach(), query,
            v_12.detach(), L_22_inv.detach(),
            rotation_cuda, rotation_delta_cuda, L_22_inv_diag_rot.detach(),
            lambda_opc
        )

        loss_cuda = (rotation_cond_cuda * grad_rotation_cond).sum()
        loss_cuda.backward()

        # Compute actual t value (t = 1 - attention_rot)
        # attention_rot = exp(-time_residual^2 * L_time^2) where L_time = exp(0) = 1
        actual_attn_rot = np.exp(-time_offset**2)
        actual_t = 1.0 - actual_attn_rot

        # Compare rotation_delta gradients
        cos_sim_delta = F.cosine_similarity(
            rotation_delta_pt.grad.flatten().unsqueeze(0),
            rotation_delta_cuda.grad.flatten().unsqueeze(0)
        ).item()

        cos_sim_base = F.cosine_similarity(
            rotation_pt.grad.flatten().unsqueeze(0),
            rotation_cuda.grad.flatten().unsqueeze(0)
        ).item()

        print(f"time_offset={time_offset:.2f}, t={actual_t:.3f}: "
              f"q_base cos_sim={cos_sim_base:.4f}, q_delta cos_sim={cos_sim_delta:.4f}")


if __name__ == "__main__":
    # Run forward output comparison for different configs
    print("Testing forward output matching...")
    test_forward_output_match(N=100, C=3, use_pos=True, use_rot=False)
    test_forward_output_match(N=100, C=4, use_pos=True, use_rot=True)

    # Run main gradient check
    print("\n\n")
    results = test_slice_gaussian_full_gradients(N=1000, C=3, use_pos=True, use_rot=False)

    # Test with C=4 and rotation
    print("\n\n")
    results_c4 = test_slice_gaussian_full_gradients(N=1000, C=4, use_pos=True, use_rot=True)

    # Test at different rotation interpolation points
    test_rotation_at_different_t_values()

    # Test different configurations
    print("\n\n")
    test_different_configurations()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
