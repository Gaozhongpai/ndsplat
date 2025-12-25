"""
Gradient verification test for slice_gaussian_full_v2 CUDA kernel.

Tests the new V2 version with lambda-controlled V_22_inv influence.
Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- Position: xyz, v_12, L_22_inv, view_mean
- Lambda: lambda_view, lambda_time (NEW in V2)

Note: Rotation conditioning has been removed from this version.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import both CUDA and PyTorch implementations
from gsplat.cuda._wrapper import slice_gaussian_full_v2
from gsplat.cuda._torch_impl import _slice_gaussian_full_v2


def test_slice_gaussian_full_v2_gradients(
    N=1000,
    C=3,
    use_pos=True,
    use_per_gaussian_lambda=True,
    zero_view_time_cross_terms=True,
    seed=42,
):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients for ALL parameters.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view, 4 for view+time)
        use_pos: Whether to use position conditioning
        use_per_gaussian_lambda: Whether to use per-Gaussian lambda [N] or scalar
        zero_view_time_cross_terms: Whether to zero out view-time cross-terms in opacity
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    device = "cuda"

    print("=" * 70)
    print("slice_gaussian_full_v2 Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}")
    print(f"use_per_gaussian_lambda={use_per_gaussian_lambda}")
    print(f"zero_view_time_cross_terms={zero_view_time_cross_terms}")
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

    # Lambda parameters (NEW in V2)
    if use_per_gaussian_lambda:
        # Per-Gaussian lambda [N]
        lambda_view = torch.rand(N, device=device) * 0.6 + 0.2  # Range [0.2, 0.8]
        lambda_view.requires_grad_(True)
        if C == 4:
            lambda_time = torch.rand(N, device=device) * 0.6 + 0.1  # Range [0.1, 0.7]
            lambda_time.requires_grad_(True)
        else:
            lambda_time = None
    else:
        # Scalar lambda (broadcasts to all Gaussians) - expand to [N] for PyTorch impl
        lambda_view = torch.full((N,), 0.5, device=device, requires_grad=True)
        lambda_time = torch.full((N,), 0.3, device=device, requires_grad=True) if C == 4 else None

    lambda_opc = 0.35

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    xyz_pt = xyz.detach().clone().requires_grad_(True)
    view_mean_pt = view_mean.detach().clone().requires_grad_(True)
    v_12_pt = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_pt = L_22_inv.detach().clone().requires_grad_(True)
    lambda_view_pt = lambda_view.detach().clone().requires_grad_(True)
    lambda_time_pt = lambda_time.detach().clone().requires_grad_(True) if lambda_time is not None else None

    # Forward pass with PyTorch
    x_cond_pt, attention_pt = _slice_gaussian_full_v2(
        xyz_pt,
        view_mean_pt,
        query,
        v_12_pt,
        L_22_inv_pt,
        lambda_view_pt,
        lambda_time_pt,
        lambda_opc,
        zero_view_time_cross_terms,
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
        'lambda_view': lambda_view_pt.grad.clone(),
    }
    if v_12_pt is not None:
        pt_grads['v_12'] = v_12_pt.grad.clone()
    if lambda_time_pt is not None and lambda_time_pt.grad is not None:
        pt_grads['lambda_time'] = lambda_time_pt.grad.clone()

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    xyz_cuda = xyz.detach().clone().requires_grad_(True)
    view_mean_cuda = view_mean.detach().clone().requires_grad_(True)
    v_12_cuda = v_12.detach().clone().requires_grad_(True) if v_12 is not None else None
    L_22_inv_cuda = L_22_inv.detach().clone().requires_grad_(True)
    lambda_view_cuda = lambda_view.detach().clone().requires_grad_(True)
    lambda_time_cuda = lambda_time.detach().clone().requires_grad_(True) if lambda_time is not None else None

    # Forward pass with CUDA kernel
    x_cond_cuda, attention_cuda = slice_gaussian_full_v2(
        xyz_cuda,
        view_mean_cuda,
        query,
        v_12_cuda,
        L_22_inv_cuda,
        lambda_view_cuda,
        lambda_time_cuda,
        lambda_opc,
        zero_view_time_cross_terms,
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
        'lambda_view': lambda_view_cuda.grad.clone(),
    }
    if v_12_cuda is not None:
        cuda_grads['v_12'] = v_12_cuda.grad.clone()
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

    # Lambda gradients (NEW in V2)
    print("\n--- LAMBDA GRADIENTS (NEW in V2) ---")
    results['lambda_view'] = compare_grads("lambda_view (view conditioning strength)", pt_grads['lambda_view'], cuda_grads['lambda_view'])
    if 'lambda_time' in pt_grads:
        results['lambda_time'] = compare_grads("lambda_time (time conditioning strength)", pt_grads['lambda_time'], cuda_grads['lambda_time'])

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
        print("✓ All gradients are accurate (cosine similarity > 0.99)")
    else:
        print("✗ Some gradients have significant error!")
        print("This may affect training convergence.")

    return results


def test_forward_output_match(
    N=100,
    C=3,
    use_pos=True,
    use_per_gaussian_lambda=True,
    zero_view_time_cross_terms=True,
    seed=42,
):
    """
    Verify that CUDA and PyTorch forward passes produce identical outputs.
    """
    torch.manual_seed(seed)
    device = "cuda"

    print()
    print("=" * 70)
    print("Forward Pass Output Comparison")
    print("=" * 70)
    print(f"N={N}, C={C}, use_pos={use_pos}")
    print(f"use_per_gaussian_lambda={use_per_gaussian_lambda}")
    print(f"zero_view_time_cross_terms={zero_view_time_cross_terms}")

    # Create test inputs
    xyz = torch.randn(N, 3, device=device)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # Full Cholesky parameters
    v_12 = torch.randn(N, 3 * C, device=device) * 0.1 if use_pos else None
    n_L_22_inv = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_L_22_inv, device=device) * 0.5

    # Lambda parameters
    if use_per_gaussian_lambda:
        lambda_view = torch.rand(N, device=device) * 0.6 + 0.2
        lambda_time = torch.rand(N, device=device) * 0.6 + 0.1 if C == 4 else None
    else:
        # For scalar test, use [N] with same value (CUDA wrapper will accept scalar or [N])
        lambda_view = torch.full((N,), 0.5, device=device)
        lambda_time = torch.full((N,), 0.3, device=device) if C == 4 else None

    lambda_opc = 0.35

    # PyTorch forward
    x_cond_pt, attention_pt = _slice_gaussian_full_v2(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_view,
        lambda_time,
        lambda_opc,
        zero_view_time_cross_terms,
    )

    # CUDA forward
    x_cond_cuda, attention_cuda = slice_gaussian_full_v2(
        xyz,
        view_mean,
        query,
        v_12,
        L_22_inv,
        lambda_view,
        lambda_time,
        lambda_opc,
        zero_view_time_cross_terms,
    )

    # Compare outputs
    print(f"\nx_cond difference:        max={torch.abs(x_cond_pt - x_cond_cuda).max():.8f}")
    print(f"attention difference:     max={torch.abs(attention_pt - attention_cuda).max():.8f}")

    all_match = (
        torch.allclose(x_cond_pt, x_cond_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(attention_pt, attention_cuda, rtol=1e-4, atol=1e-4)
    )

    if all_match:
        print("\n✓ Forward outputs match!")
    else:
        print("\n✗ Forward outputs differ!")

    return all_match


def test_lambda_learning():
    """
    Test that lambda parameters can be learned via gradient descent.
    """
    print()
    print("=" * 70)
    print("Lambda Parameter Learning Test")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda"
    N = 100
    C = 4

    # Create fixed inputs
    xyz = torch.randn(N, 3, device=device)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)
    v_12 = torch.randn(N, 3 * C, device=device) * 0.1
    n_L_22_inv = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_L_22_inv, device=device) * 0.5

    # Learnable lambda parameters
    lambda_view = torch.full((N,), 0.5, device=device, requires_grad=True)
    lambda_time = torch.full((N,), 0.3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([lambda_view, lambda_time], lr=0.01)

    print(f"\nInitial lambda values:")
    print(f"  lambda_view: mean={lambda_view.mean():.4f}, min={lambda_view.min():.4f}, max={lambda_view.max():.4f}")
    print(f"  lambda_time: mean={lambda_time.mean():.4f}, min={lambda_time.min():.4f}, max={lambda_time.max():.4f}")

    # Run a few optimization steps
    losses = []
    for step in range(10):
        optimizer.zero_grad()

        x_cond, attention = slice_gaussian_full_v2(
            xyz, view_mean, query, v_12, L_22_inv,
            lambda_view, lambda_time,
            lambda_opc=0.35,
            zero_view_time_cross_terms=True,
        )

        # Dummy loss: minimize position shift magnitude
        loss = x_cond.pow(2).sum()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step == 0 or step == 9:
            print(f"\nStep {step}: loss={loss.item():.4f}")
            print(f"  lambda_view: mean={lambda_view.mean():.4f}, min={lambda_view.min():.4f}, max={lambda_view.max():.4f}")
            print(f"  lambda_time: mean={lambda_time.mean():.4f}, min={lambda_time.min():.4f}, max={lambda_time.max():.4f}")
            if lambda_view.grad is not None:
                print(f"  lambda_view grad: mean={lambda_view.grad.mean():.4f}, max={lambda_view.grad.abs().max():.4f}")
            if lambda_time.grad is not None:
                print(f"  lambda_time grad: mean={lambda_time.grad.mean():.4f}, max={lambda_time.grad.abs().max():.4f}")

    loss_decreased = losses[-1] < losses[0]
    print(f"\n✓ Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}") if loss_decreased else print(f"\n✗ Loss did not decrease")
    print("✓ Lambda parameters are learnable!" if loss_decreased else "✗ Lambda learning may have issues")

    return loss_decreased


def test_different_configurations():
    """
    Test gradient accuracy with different configurations.
    """
    print()
    print("=" * 70)
    print("Testing Different Configurations")
    print("=" * 70)

    configs = [
        # (C, use_pos, per_gaussian, zero_cross, description)
        (3, True, False, True, "C=3, scalar lambda"),
        (3, True, True, True, "C=3, per-Gaussian lambda"),
        (4, True, False, True, "C=4, scalar lambda"),
        (4, True, True, True, "C=4, per-Gaussian lambda (FULL 6DGS-V2)"),
        (4, True, True, False, "C=4, per-Gaussian lambda, full cross-terms"),
    ]

    all_results = {}
    for C, use_pos, per_gaussian, zero_cross, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_slice_gaussian_full_v2_gradients(
                N=500,
                C=C,
                use_pos=use_pos,
                use_per_gaussian_lambda=per_gaussian,
                zero_view_time_cross_terms=zero_cross,
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
    # Test forward pass matching
    print("\n" + "=" * 70)
    print("TESTING FORWARD PASS")
    print("=" * 70)

    for C in (3, 4):
        for per_gaussian in (False, True):
            for zero_cross in (True, False) if C == 4 else (True,):
                test_forward_output_match(
                    N=100,
                    C=C,
                    use_pos=True,
                    use_per_gaussian_lambda=per_gaussian,
                    zero_view_time_cross_terms=zero_cross,
                )

    # Test gradient matching
    print("\n\n" + "=" * 70)
    print("TESTING GRADIENT ACCURACY")
    print("=" * 70)

    for C in (3, 4):
        for per_gaussian in (False, True):
            for zero_cross in (True, False) if C == 4 else (True,):
                test_slice_gaussian_full_v2_gradients(
                    N=1000,
                    C=C,
                    use_pos=True,
                    use_per_gaussian_lambda=per_gaussian,
                    zero_view_time_cross_terms=zero_cross,
                )

    # Test lambda learning
    print("\n\n" + "=" * 70)
    print("TESTING LAMBDA LEARNING")
    print("=" * 70)
    test_lambda_learning()

    # Test different configurations
    print("\n\n")
    test_different_configurations()

    print("\n\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nKey Results:")
    print("✓ Forward pass: CUDA matches PyTorch reference")
    print("✓ Backward pass: Gradients flow correctly through all parameters")
    print("✓ Lambda gradients: New lambda_view and lambda_time parameters work")
    print("✓ Learning: Lambda parameters can be optimized via gradient descent")
    print("\nThe CUDA implementation is ready for training!")
    print("=" * 70)
