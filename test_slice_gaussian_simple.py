"""
Gradient verification test for slice_gaussian_simple CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- value_mean (values to shift)
- key_mean (key for value shifting)
- key_mean_opc (separate key for opacity)
- v_12 (value-key covariance block)
- L_22_inv (Cholesky factor of precision)
- lambda_scale (attention scaling factor)
"""

import torch
import torch.nn.functional as F

# Import both CUDA and PyTorch implementations
from gsplat import slice_gaussian_simple
from gsplat.cuda._torch_impl import _slice_gaussian_simple


def test_slice_gaussian_simple_gradients(N=1000, D=3, C=3, seed=42, use_separate_key_opc=True):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients.

    Args:
        N: Number of Gaussians
        D: Value dimension (e.g., 3 for position or color, 6 for both)
        C: Key/query dimension (e.g., 3 for view direction, 4 for view+time)
        seed: Random seed for reproducibility
        use_separate_key_opc: Whether to use separate key_mean for opacity
    """
    torch.manual_seed(seed)
    device = "cuda"

    print("=" * 70)
    print("slice_gaussian_simple Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, D={D}, C={C}, use_separate_key_opc={use_separate_key_opc}")
    print()

    # Create test inputs
    value_mean = torch.randn(N, D, device=device, requires_grad=True)
    key_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    key_mean.requires_grad_(True)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)  # No grad needed

    # Value-key covariance block
    v_12 = torch.randn(N, D, C, device=device, requires_grad=True) * 0.1

    # Cholesky factor of precision: C*(C+1)/2 elements for lower triangular
    n_chol = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_chol, device=device, requires_grad=True) * 0.5

    # Per-Gaussian lambda scale
    lambda_scale = torch.rand(N, 1, device=device, requires_grad=True) * 0.5 + 0.1

    # Optional separate key_mean for opacity
    if use_separate_key_opc:
        key_mean_opc = F.normalize(torch.randn(N, C, device=device), dim=-1)
        key_mean_opc.requires_grad_(True)
    else:
        key_mean_opc = None

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    value_mean_pt = value_mean.detach().clone().requires_grad_(True)
    key_mean_pt = key_mean.detach().clone().requires_grad_(True)
    v_12_pt = v_12.detach().clone().requires_grad_(True)
    L_22_inv_pt = L_22_inv.detach().clone().requires_grad_(True)
    lambda_scale_pt = lambda_scale.detach().clone().requires_grad_(True)
    if use_separate_key_opc:
        key_mean_opc_pt = key_mean_opc.detach().clone().requires_grad_(True)
    else:
        key_mean_opc_pt = None

    # Forward pass with PyTorch
    value_cond_pt, attention_weight_pt = _slice_gaussian_simple(
        value_mean_pt, key_mean_pt, query,
        v_12_pt, L_22_inv_pt, lambda_scale_pt, key_mean_opc_pt
    )

    # Create random upstream gradients
    grad_value_cond = torch.randn_like(value_cond_pt)
    grad_attention_weight = torch.randn_like(attention_weight_pt)

    # Backward pass with PyTorch
    loss_pt = (
        (value_cond_pt * grad_value_cond).sum() +
        (attention_weight_pt * grad_attention_weight).sum()
    )
    loss_pt.backward()

    # Store PyTorch gradients
    grad_value_mean_pt = value_mean_pt.grad.clone()
    grad_key_mean_pt = key_mean_pt.grad.clone()
    grad_v_12_pt = v_12_pt.grad.clone()
    grad_L_22_inv_pt = L_22_inv_pt.grad.clone()
    grad_lambda_scale_pt = lambda_scale_pt.grad.clone()
    if use_separate_key_opc:
        grad_key_mean_opc_pt = key_mean_opc_pt.grad.clone()

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    value_mean_cuda = value_mean.detach().clone().requires_grad_(True)
    key_mean_cuda = key_mean.detach().clone().requires_grad_(True)
    v_12_cuda = v_12.detach().clone().requires_grad_(True)
    L_22_inv_cuda = L_22_inv.detach().clone().requires_grad_(True)
    lambda_scale_cuda = lambda_scale.detach().clone().requires_grad_(True)
    if use_separate_key_opc:
        key_mean_opc_cuda = key_mean_opc.detach().clone().requires_grad_(True)
    else:
        key_mean_opc_cuda = None

    # Forward pass with CUDA kernel
    value_cond_cuda, attention_weight_cuda = slice_gaussian_simple(
        value_mean_cuda, key_mean_cuda, query,
        v_12_cuda, L_22_inv_cuda, lambda_scale_cuda, key_mean_opc_cuda
    )

    # Backward pass with same upstream gradients
    loss_cuda = (
        (value_cond_cuda * grad_value_cond).sum() +
        (attention_weight_cuda * grad_attention_weight).sum()
    )
    loss_cuda.backward()

    # Store CUDA gradients
    grad_value_mean_cuda = value_mean_cuda.grad.clone()
    grad_key_mean_cuda = key_mean_cuda.grad.clone()
    grad_v_12_cuda = v_12_cuda.grad.clone()
    grad_L_22_inv_cuda = L_22_inv_cuda.grad.clone()
    grad_lambda_scale_cuda = lambda_scale_cuda.grad.clone()
    if use_separate_key_opc:
        grad_key_mean_opc_cuda = key_mean_opc_cuda.grad.clone()

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
    results['value_mean'] = compare_grads("value_mean", grad_value_mean_pt, grad_value_mean_cuda)
    results['key_mean'] = compare_grads("key_mean", grad_key_mean_pt, grad_key_mean_cuda)
    results['v_12'] = compare_grads("v_12 (value-key covariance)", grad_v_12_pt, grad_v_12_cuda)
    results['L_22_inv'] = compare_grads("L_22_inv (Cholesky precision)", grad_L_22_inv_pt, grad_L_22_inv_cuda)
    results['lambda_scale'] = compare_grads("lambda_scale", grad_lambda_scale_pt, grad_lambda_scale_cuda)
    if use_separate_key_opc:
        results['key_mean_opc'] = compare_grads("key_mean_opc", grad_key_mean_opc_pt, grad_key_mean_opc_cuda)

    # ============ Summary ============
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if gradients are acceptable
    all_good = True
    for name, res in results.items():
        cos_sim = res['cos_sim']
        status = "✓ GOOD" if cos_sim > 0.99 else ("⚠ WARNING" if cos_sim > 0.9 else "✗ BAD")
        if cos_sim < 0.99:
            all_good = False
        print(f"{name}: cosine_sim={cos_sim:.4f} {status}")

    print()
    if all_good:
        print("All gradients are accurate (cosine similarity > 0.99)")
    else:
        print("Some gradients have significant error!")

    return results


def test_forward_output_match(N=100, D=3, C=3, seed=42):
    """
    Verify that CUDA and PyTorch forward passes produce identical outputs.
    """
    torch.manual_seed(seed)
    device = "cuda"

    print()
    print("=" * 70)
    print("Forward Pass Output Comparison")
    print("=" * 70)

    # Create test inputs
    value_mean = torch.randn(N, D, device=device)
    key_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)
    v_12 = torch.randn(N, D, C, device=device) * 0.1
    n_chol = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_chol, device=device) * 0.5
    lambda_scale = torch.rand(N, 1, device=device) * 0.5 + 0.1
    key_mean_opc = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # PyTorch forward
    value_cond_pt, attention_weight_pt = _slice_gaussian_simple(
        value_mean, key_mean, query, v_12, L_22_inv, lambda_scale, key_mean_opc
    )

    # CUDA forward
    value_cond_cuda, attention_weight_cuda = slice_gaussian_simple(
        value_mean, key_mean, query, v_12, L_22_inv, lambda_scale, key_mean_opc
    )

    # Compare outputs
    print(f"\nvalue_cond difference:      max={torch.abs(value_cond_pt - value_cond_cuda).max():.8f}")
    print(f"attention_weight difference: max={torch.abs(attention_weight_pt - attention_weight_cuda).max():.8f}")

    all_match = (
        torch.allclose(value_cond_pt, value_cond_cuda, rtol=1e-5, atol=1e-5) and
        torch.allclose(attention_weight_pt, attention_weight_cuda, rtol=1e-5, atol=1e-5)
    )

    if all_match:
        print("\n✓ Forward outputs match!")
    else:
        print("\n✗ Forward outputs differ!")

    return all_match


def test_different_dimensions():
    """
    Test gradient accuracy with different dimension combinations.
    """
    print()
    print("=" * 70)
    print("Testing Different Dimension Combinations")
    print("=" * 70)

    configs = [
        (3, 3, "D=3, C=3 (color-only with view)"),
        (3, 4, "D=3, C=4 (color-only with view+time)"),
        (6, 3, "D=6, C=3 (position+color with view)"),
        (6, 4, "D=6, C=4 (position+color with view+time)"),
    ]

    all_results = {}
    for D, C, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_slice_gaussian_simple_gradients(N=500, D=D, C=C, seed=42, use_separate_key_opc=True)
            all_results[desc] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[desc] = None

    return all_results


if __name__ == "__main__":
    # Run forward output comparison
    test_forward_output_match(N=100, D=3, C=3)

    # Run main gradient check with default dimensions
    print("\n\n")
    results = test_slice_gaussian_simple_gradients(N=1000, D=3, C=3, use_separate_key_opc=True)

    # Test without separate key_mean_opc
    print("\n\n")
    print("Testing without separate key_mean_opc:")
    results_no_opc = test_slice_gaussian_simple_gradients(N=1000, D=3, C=3, use_separate_key_opc=False)

    # Test different dimension combinations
    print("\n\n")
    test_different_dimensions()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
