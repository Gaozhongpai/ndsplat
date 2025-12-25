"""
Gradient verification test for slice_gaussian_full CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- Position: xyz, v_12, L_22_inv, view_mean

Note: Rotation conditioning has been removed from this version.
Note: Scale is NOT conditioned in slice_gaussian_full - use get_scaling directly.
"""

import inspect
import torch
import torch.nn.functional as F
import numpy as np

# Import both CUDA and PyTorch implementations
from gsplat import slice_gaussian_full
from gsplat.cuda._torch_impl import _slice_gaussian_full


_HAS_LAMBDA_OPC_TIME_FULL_CUDA = "lambda_opc_time" in inspect.signature(slice_gaussian_full).parameters


def _call_slice_gaussian_full_torch(
    xyz,
    view_mean,
    query,
    v_12,
    L_22_inv,
    lambda_opc,
    lambda_opc_time,  # Kept for API compatibility but not used by torch impl
):
    # Note: _slice_gaussian_full doesn't have lambda_opc_time parameter
    return _slice_gaussian_full(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_opc,
    )


def _call_slice_gaussian_full_cuda(
    xyz,
    view_mean,
    query,
    v_12,
    L_22_inv,
    lambda_opc,
    lambda_opc_time,
):
    if _HAS_LAMBDA_OPC_TIME_FULL_CUDA:
        return slice_gaussian_full(
            xyz,
            view_mean,
            query,
            v_12,
            L_22_inv,
            lambda_opc,
            lambda_opc_time,
        )
    return slice_gaussian_full(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_opc,
    )


def test_slice_gaussian_full_gradients(
    N=1000,
    C=3,
    use_pos=True,
    lambda_opc_time=None,
    seed=42,
):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients for ALL parameters.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view, 4 for view+time)
        use_pos: Whether to use position conditioning
        lambda_opc_time: Optional opacity scaling factor for time (only when C=4). Scalar, tensor, or None.
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    device = "cuda"

    if C == 4:
        if lambda_opc_time is None:
            lambda_opc_time_arg = torch.rand(N, device=device) * 0.3 + 0.2
        else:
            if isinstance(lambda_opc_time, torch.Tensor):
                lambda_opc_time_arg = lambda_opc_time.to(device=device)
            else:
                lambda_opc_time_arg = torch.full((N,), float(lambda_opc_time), device=device)
    else:
        lambda_opc_time_arg = None

    if lambda_opc_time_arg is None:
        lambda_opc_time_desc = "None"
    elif isinstance(lambda_opc_time_arg, torch.Tensor):
        lambda_opc_time_desc = (
            f"tensor[min={lambda_opc_time_arg.min().item():.3f}, "
            f"max={lambda_opc_time_arg.max().item():.3f}]"
        )
    else:
        lambda_opc_time_desc = str(lambda_opc_time_arg)

    print("=" * 70)
    print("slice_gaussian_full Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}")
    print(f"lambda_opc_time={lambda_opc_time_desc}")
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

    lambda_opc = 0.35

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    xyz_pt = xyz.detach().clone().requires_grad_(True)
    view_mean_pt = view_mean.detach().clone().requires_grad_(True)
    v_12_pt = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_pt = L_22_inv.detach().clone().requires_grad_(True)

    # Forward pass with PyTorch
    x_cond_pt, attention_pt = _call_slice_gaussian_full_torch(
        xyz_pt,
        view_mean_pt,
        query,
        v_12_pt,
        L_22_inv_pt,
        lambda_opc,
        lambda_opc_time_arg,
        
    )

    # Create random upstream gradients
    grad_x_cond = torch.randn_like(x_cond_pt)
    grad_attention = torch.randn_like(attention_pt)

    # Backward pass with PyTorch
    loss_pt = (
        (x_cond_pt * grad_x_cond).sum() +
        (attention_pt * grad_attention).sum()
    )
    loss_pt.backward()

    # Store PyTorch gradients
    pt_grads = {
        'xyz': xyz_pt.grad.clone(),
        'view_mean': view_mean_pt.grad.clone(),
        'L_22_inv': L_22_inv_pt.grad.clone(),
    }
    if v_12_pt is not None:
        pt_grads['v_12'] = v_12_pt.grad.clone()

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    xyz_cuda = xyz.detach().clone().requires_grad_(True)
    view_mean_cuda = view_mean.detach().clone().requires_grad_(True)
    v_12_cuda = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_cuda = L_22_inv.detach().clone().requires_grad_(True)

    # Forward pass with CUDA kernel
    x_cond_cuda, attention_cuda = _call_slice_gaussian_full_cuda(
        xyz_cuda,
        view_mean_cuda,
        query,
        v_12_cuda,
        L_22_inv_cuda,
        lambda_opc,
        lambda_opc_time_arg,
        
    )

    # Backward pass with same upstream gradients
    loss_cuda = (
        (x_cond_cuda * grad_x_cond).sum() +
        (attention_cuda * grad_attention).sum()
    )
    loss_cuda.backward()

    # Store CUDA gradients
    cuda_grads = {
        'xyz': xyz_cuda.grad.clone(),
        'view_mean': view_mean_cuda.grad.clone(),
        'L_22_inv': L_22_inv_cuda.grad.clone(),
    }
    if v_12_cuda is not None:
        cuda_grads['v_12'] = v_12_cuda.grad.clone()

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


def test_forward_output_match(
    N=100,
    C=3,
    use_pos=True,
    
    lambda_opc_time=None,
    seed=42,
):
    """
    Verify that CUDA and PyTorch forward passes produce identical outputs.
    """
    torch.manual_seed(seed)
    device = "cuda"

    if C == 4:
        if lambda_opc_time is None:
            lambda_opc_time_arg = torch.rand(N, device=device) * 0.3 + 0.2
        else:
            if isinstance(lambda_opc_time, torch.Tensor):
                lambda_opc_time_arg = lambda_opc_time.to(device=device)
            else:
                lambda_opc_time_arg = torch.full((N,), float(lambda_opc_time), device=device)
    else:
        lambda_opc_time_arg = None

    if lambda_opc_time_arg is None:
        lambda_opc_time_desc = "None"
    elif isinstance(lambda_opc_time_arg, torch.Tensor):
        lambda_opc_time_desc = (
            f"tensor[min={lambda_opc_time_arg.min().item():.3f}, "
            f"max={lambda_opc_time_arg.max().item():.3f}]"
        )
    else:
        lambda_opc_time_desc = str(lambda_opc_time_arg)

    print()
    print("=" * 70)
    print("Forward Pass Output Comparison")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}")
    print(f"lambda_opc_time={lambda_opc_time_desc}")

    # Create test inputs
    xyz = torch.randn(N, 3, device=device)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # Full Cholesky parameters
    v_12 = torch.randn(N, 3 * C, device=device) * 0.1 if use_pos else None
    n_L_22_inv = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_L_22_inv, device=device) * 0.5

    lambda_opc = 0.35

    # PyTorch forward
    x_cond_pt, attention_pt = _call_slice_gaussian_full_torch(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_opc,
        lambda_opc_time_arg,
        
    )

    # CUDA forward
    x_cond_cuda, attention_cuda = _call_slice_gaussian_full_cuda(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_opc,
        lambda_opc_time_arg,
        
    )

    # Compare outputs
    print(f"\nx_cond difference:        max={torch.abs(x_cond_pt - x_cond_cuda).max():.8f}")
    print(f"attention difference:     max={torch.abs(attention_pt - attention_cuda).max():.8f}")

    all_match = (
        torch.allclose(x_cond_pt, x_cond_cuda, rtol=1e-4, atol=1e-4) and
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
        # (C, use_pos,  lambda_opc_time, description)
        (3, True, True, None, "C=3, pos=True, zero_cross=True (6DGS view-only)"),
        (3, True, False, None, "C=3, pos=True, zero_cross=False (view-only with cross-terms)"),
        (3, False, True, None, "C=3, pos=False, zero_cross=True (no position shift)"),
        (4, True, True, 0.45, "C=4, pos=True, zero_cross=True (6DGS view+time)"),
        (4, False, True, 0.45, "C=4, pos=False, zero_cross=True (time only)"),
        # Test with )
        (4, True, False, 0.65, "C=4, pos=True, zero_cross=False (full cross-terms)"),
    ]

    all_results = {}
    for C, use_pos, zero_cross, lambda_time, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_slice_gaussian_full_gradients(
                N=500,
                C=C,
                use_pos=use_pos,
                
                lambda_opc_time=lambda_time,
                seed=42,
            )
            all_results[desc] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[desc] = None

    return all_results


if __name__ == "__main__":
    # Run forward output comparison for different configs
    print("Testing forward output matching...")
    for C in (3, 4):
        for zero_cross in (True, False):
            lambda_time = None if C == 3 else (0.45 if zero_cross else 0.65)
            test_forward_output_match(
                N=100,
                C=C,
                use_pos=True,
                
                lambda_opc_time=lambda_time,
            )

    # Run main gradient checks across configurations
    print("\n\n")
    for C in (3, 4):
        for zero_cross in (True, False):
            print("\n")
            lambda_time = None if C == 3 else (0.45 if zero_cross else 0.65)
            test_slice_gaussian_full_gradients(
                N=1000,
                C=C,
                use_pos=True,
                
                lambda_opc_time=lambda_time,
            )

    # Test different configurations
    print("\n\n")
    test_different_configurations()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
