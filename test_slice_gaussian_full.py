"""
Gradient verification test for slice_gaussian_full CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- Position: xyz, v_12, L_22_inv, view_mean
- Scaling: scale, lambda_view, lambda_time (applied in CUDA)

Note: Rotation conditioning has been removed from this version.
Note: Spatial scaling and lambda modulation are now applied directly in CUDA kernels.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import both CUDA and PyTorch implementations
from gsplat import slice_gaussian_full
from gsplat.cuda._torch_impl import _slice_gaussian_full


def _call_slice_gaussian_full_torch(
    xyz,
    view_mean,
    query,
    v_12,
    L_22_inv,
    lambda_opc,
    scale,
    lambda_view,
    lambda_time,
    use_beta=False,
):
    # PyTorch implementation with scale and lambda parameters
    return _slice_gaussian_full(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_view=lambda_view,
        lambda_time=lambda_time,
        lambda_opc=lambda_opc,
        scale=scale,
        use_beta=use_beta,
    )


def _call_slice_gaussian_full_cuda(
    xyz,
    view_mean,
    query,
    v_12,
    L_22_inv,
    lambda_opc,
    scale,
    lambda_view,
    lambda_time,
    use_beta=False,
):
    # CUDA implementation with scale and lambda parameters
    return slice_gaussian_full(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_opc=lambda_opc,
        scale=scale,
        lambda_view=lambda_view,
        lambda_time=lambda_time,
        use_beta=use_beta,
    )


def test_slice_gaussian_full_gradients(
    N=1000,
    C=3,
    use_pos=True,
    use_scale=True,
    use_lambda=True,
    use_beta=False,
    seed=42,
):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients for ALL parameters.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view, 4 for view+time)
        use_pos: Whether to use position conditioning
        use_scale: Whether to use spatial scaling
        use_lambda: Whether to use lambda modulation
        use_beta: Whether to use beta-based opacity (UBS-style)
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    device = "cuda"

    print("=" * 70)
    print("slice_gaussian_full Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}")
    print(f"use_scale={use_scale}, use_lambda={use_lambda}, use_beta={use_beta}")
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

    # Scaling parameters
    if use_scale:
        scale = torch.rand(N, 3, device=device, requires_grad=True) * 0.5 + 0.5  # [0.5, 1.0]
    else:
        scale = None

    if use_lambda:
        if use_beta:
            # Beta mode: Pass beta values directly (not pre-activated)
            # These values are used as-is for beta parameters in the kernel
            # Range [1.0, 3.0] for beta values (will be clamped to [0,1] for position scaling)
            lambda_view = torch.rand(N, device=device, requires_grad=True) * 2.0 + 1.0  # [1.0, 3.0]
            if C == 4:
                lambda_time = torch.rand(N, device=device, requires_grad=True) * 2.0 + 1.0  # [1.0, 3.0]
            else:
                lambda_time = None
        else:
            # Standard mode: Pass lambda values directly (not pre-activated)
            # These values are used as-is for position scaling in the kernel
            # Range [0.5, 1.0] for lambda values
            lambda_view = torch.rand(N, device=device, requires_grad=True) * 0.5 + 0.5  # [0.5, 1.0]
            if C == 4:
                lambda_time = torch.rand(N, device=device, requires_grad=True) * 0.5 + 0.5  # [0.5, 1.0]
            else:
                lambda_time = None
    else:
        lambda_view = None
        lambda_time = None

    lambda_opc = 0.35

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    xyz_pt = xyz.detach().clone().requires_grad_(True)
    view_mean_pt = view_mean.detach().clone().requires_grad_(True)
    v_12_pt = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_pt = L_22_inv.detach().clone().requires_grad_(True)
    scale_pt = scale.detach().clone().requires_grad_(True) if scale is not None else None
    lambda_view_pt = lambda_view.detach().clone().requires_grad_(True) if lambda_view is not None else None
    lambda_time_pt = lambda_time.detach().clone().requires_grad_(True) if lambda_time is not None else None

    # Forward pass with PyTorch
    x_cond_pt, attention_pt = _call_slice_gaussian_full_torch(
        xyz_pt,
        view_mean_pt,
        query,
        v_12_pt,
        L_22_inv_pt,
        lambda_opc,
        scale_pt,
        lambda_view_pt,
        lambda_time_pt,
        use_beta=use_beta,
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
    if scale_pt is not None and scale_pt.grad is not None:
        pt_grads['scale'] = scale_pt.grad.clone()
    if lambda_view_pt is not None and lambda_view_pt.grad is not None:
        pt_grads['lambda_view'] = lambda_view_pt.grad.clone()
    if lambda_time_pt is not None and lambda_time_pt.grad is not None:
        pt_grads['lambda_time'] = lambda_time_pt.grad.clone()

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    xyz_cuda = xyz.detach().clone().requires_grad_(True)
    view_mean_cuda = view_mean.detach().clone().requires_grad_(True)
    v_12_cuda = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_cuda = L_22_inv.detach().clone().requires_grad_(True)
    scale_cuda = scale.detach().clone().requires_grad_(True) if scale is not None else None
    lambda_view_cuda = lambda_view.detach().clone().requires_grad_(True) if lambda_view is not None else None
    lambda_time_cuda = lambda_time.detach().clone().requires_grad_(True) if lambda_time is not None else None

    # Forward pass with CUDA kernel
    x_cond_cuda, attention_cuda = _call_slice_gaussian_full_cuda(
        xyz_cuda,
        view_mean_cuda,
        query,
        v_12_cuda,
        L_22_inv_cuda,
        lambda_opc,
        scale_cuda,
        lambda_view_cuda,
        lambda_time_cuda,
        use_beta=use_beta,
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
    if scale_cuda is not None and scale_cuda.grad is not None:
        cuda_grads['scale'] = scale_cuda.grad.clone()
    if lambda_view_cuda is not None and lambda_view_cuda.grad is not None:
        cuda_grads['lambda_view'] = lambda_view_cuda.grad.clone()
    if lambda_time_cuda is not None and lambda_time_cuda.grad is not None:
        cuda_grads['lambda_time'] = lambda_time_cuda.grad.clone()

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

    # Scaling gradients
    if 'scale' in pt_grads or 'lambda_view' in pt_grads:
        print("\n--- SCALING GRADIENTS ---")
    if 'scale' in pt_grads:
        results['scale'] = compare_grads("scale (spatial scales)", pt_grads['scale'], cuda_grads['scale'])
    if 'lambda_view' in pt_grads:
        results['lambda_view'] = compare_grads("lambda_view (view lambda modulation)", pt_grads['lambda_view'], cuda_grads['lambda_view'])
    if 'lambda_time' in pt_grads:
        results['lambda_time'] = compare_grads("lambda_time (time lambda modulation)", pt_grads['lambda_time'], cuda_grads['lambda_time'])

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
    use_scale=True,
    use_lambda=True,
    seed=42,
):
    """
    Verify that CUDA and PyTorch forward passes produce identical outputs.
    """
    torch.manual_seed(seed)
    device = "cuda"

    # Create scaling parameters
    if use_scale:
        scale = torch.rand(N, 3, device=device) * 0.5 + 0.75  # Range [0.75, 1.25]
    else:
        scale = None

    if use_lambda:
        lambda_view = torch.rand(N, device=device) * 0.3 + 0.7  # Range [0.7, 1.0]
        if C == 4:
            lambda_time = torch.rand(N, device=device) * 0.3 + 0.7  # Range [0.7, 1.0]
        else:
            lambda_time = None
    else:
        lambda_view = None
        lambda_time = None

    print()
    print("=" * 70)
    print("Forward Pass Output Comparison")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}")
    print(f"use_scale={use_scale}, use_lambda={use_lambda}")

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
        scale,
        lambda_view,
        lambda_time,
    )

    # CUDA forward
    x_cond_cuda, attention_cuda = _call_slice_gaussian_full_cuda(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_opc,
        scale,
        lambda_view,
        lambda_time,
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
        # (C, use_pos, use_scale, use_lambda, description)
        (3, True, True, True, "C=3, pos=True, scale+lambda (6DGS view-only)"),
        (3, False, False, False, "C=3, pos=False, no scale/lambda (minimal)"),
        (4, True, True, True, "C=4, pos=True, scale+lambda (7DGS view+time)"),
        (4, False, True, False, "C=4, pos=False, scale only (time without lambda)"),
    ]

    all_results = {}
    for C, use_pos, use_scale, use_lambda, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_slice_gaussian_full_gradients(
                N=500,
                C=C,
                use_pos=use_pos,
                use_scale=use_scale,
                use_lambda=use_lambda,
                seed=42,
            )
            all_results[desc] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[desc] = None

    return all_results


def test_beta_opacity():
    """Test beta-based opacity (use_beta=True) specifically."""
    print("\n" + "=" * 70)
    print("Testing Beta-Based Opacity (UBS-style)")
    print("=" * 70)

    # Test C=4 (7DGS) with use_beta=True
    print("\n--- C=4, use_beta=True, beta values > 1.0 ---")
    test_slice_gaussian_full_gradients(
        N=500,
        C=4,
        use_pos=True,
        use_scale=True,
        use_lambda=True,
        use_beta=True,
        seed=42,
    )

    # Test C=3 (6DGS) with use_beta=True
    print("\n--- C=3, use_beta=True, beta values > 1.0 ---")
    test_slice_gaussian_full_gradients(
        N=500,
        C=3,
        use_pos=True,
        use_scale=True,
        use_lambda=True,
        use_beta=True,
        seed=42,
    )


if __name__ == "__main__":
    # First test beta-based opacity
    print("Testing beta-based opacity (use_beta=True)...")
    test_beta_opacity()

    # Run forward output comparison for different configs
    print("\n\nTesting forward output matching...")
    for C in (3, 4):
        for use_scale in (True, False):
            for use_lambda in (True, False):
                test_forward_output_match(
                    N=100,
                    C=C,
                    use_pos=True,
                    use_scale=use_scale,
                    use_lambda=use_lambda,
                )

    # Run main gradient checks across configurations
    print("\n\n")
    for C in (3, 4):
        print("\n")
        test_slice_gaussian_full_gradients(
            N=1000,
            C=C,
            use_pos=True,
            use_scale=True,
            use_lambda=True,
        )

    # Test different configurations
    print("\n\n")
    test_different_configurations()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
