"""
Gradient verification test for slice_gaussian_ndgs CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- m_1 (spatial means)
- m_2 (conditional dimensions)
- covars (full covariance matrices)
"""

import torch
import torch.nn.functional as F

# Import both CUDA and PyTorch implementations
from gsplat import slice_gaussian_ndgs
from gsplat.cuda._torch_impl import _slice_gaussian_ndgs


def test_slice_gaussian_ndgs_gradients(N=1000, C=3, seed=42):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view direction)
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    device = "cuda"
    D = 3 + C  # Total dimensions (spatial + conditional)

    print("=" * 70)
    print("slice_gaussian_ndgs Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, C={C}, D={D}")
    print()

    # Create test inputs
    m_1 = torch.randn(N, 3, device=device, requires_grad=True)  # Spatial means
    m_2 = F.normalize(torch.randn(N, C, device=device), dim=-1)  # Conditional means (normalized)
    m_2.requires_grad_(True)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)  # Query (no grad needed)

    # Create valid positive semi-definite covariance matrices
    # Use L @ L^T construction to ensure PSD
    L = torch.randn(N, D, D, device=device) * 0.5
    # Make it lower triangular
    L = torch.tril(L)
    # Add small diagonal to ensure positive definiteness
    L = L + torch.eye(D, device=device).unsqueeze(0) * 0.5
    covars = torch.bmm(L, L.transpose(-1, -2))
    covars.requires_grad_(True)

    lambda_opc = 0.35

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    m_1_pt = m_1.detach().clone().requires_grad_(True)
    m_2_pt = m_2.detach().clone().requires_grad_(True)
    covars_pt = covars.detach().clone().requires_grad_(True)

    # Forward pass with PyTorch
    m_cond_pt, cov3D_pt, scale_pt = _slice_gaussian_ndgs(
        m_1_pt, m_2_pt, query, covars_pt, lambda_opc
    )

    # Create random upstream gradients
    grad_m_cond = torch.randn_like(m_cond_pt)
    grad_cov3D = torch.randn_like(cov3D_pt)
    grad_scale = torch.randn_like(scale_pt)

    # Backward pass with PyTorch
    loss_pt = (
        (m_cond_pt * grad_m_cond).sum() +
        (cov3D_pt * grad_cov3D).sum() +
        (scale_pt * grad_scale).sum()
    )
    loss_pt.backward()

    # Store PyTorch gradients
    grad_m_1_pt = m_1_pt.grad.clone()
    grad_m_2_pt = m_2_pt.grad.clone()
    grad_covars_pt = covars_pt.grad.clone()

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    m_1_cuda = m_1.detach().clone().requires_grad_(True)
    m_2_cuda = m_2.detach().clone().requires_grad_(True)
    covars_cuda = covars.detach().clone().requires_grad_(True)

    # Forward pass with CUDA kernel
    m_cond_cuda, cov3D_cuda, scale_cuda = slice_gaussian_ndgs(
        m_1_cuda, m_2_cuda, query, covars_cuda, lambda_opc
    )

    # Backward pass with same upstream gradients
    loss_cuda = (
        (m_cond_cuda * grad_m_cond).sum() +
        (cov3D_cuda * grad_cov3D).sum() +
        (scale_cuda * grad_scale).sum()
    )
    loss_cuda.backward()

    # Store CUDA gradients
    grad_m_1_cuda = m_1_cuda.grad.clone()
    grad_m_2_cuda = m_2_cuda.grad.clone()
    grad_covars_cuda = covars_cuda.grad.clone()

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
    results['m_1'] = compare_grads("m_1 (spatial means)", grad_m_1_pt, grad_m_1_cuda)
    results['m_2'] = compare_grads("m_2 (conditional means)", grad_m_2_pt, grad_m_2_cuda)
    results['covars'] = compare_grads("covars (covariance matrices)", grad_covars_pt, grad_covars_cuda)

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


def test_forward_output_match(N=100, C=3, seed=42):
    """
    Verify that CUDA and PyTorch forward passes produce identical outputs.
    """
    torch.manual_seed(seed)
    device = "cuda"
    D = 3 + C

    print()
    print("=" * 70)
    print("Forward Pass Output Comparison")
    print("=" * 70)

    # Create test inputs
    m_1 = torch.randn(N, 3, device=device)
    m_2 = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # Create valid PSD covariance matrices
    L = torch.randn(N, D, D, device=device) * 0.5
    L = torch.tril(L)
    L = L + torch.eye(D, device=device).unsqueeze(0) * 0.5
    covars = torch.bmm(L, L.transpose(-1, -2))

    lambda_opc = 0.35

    # PyTorch forward
    m_cond_pt, cov3D_pt, scale_pt = _slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc)

    # CUDA forward
    m_cond_cuda, cov3D_cuda, scale_cuda = slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc)

    # Compare outputs
    print(f"\nm_cond difference: max={torch.abs(m_cond_pt - m_cond_cuda).max():.8f}")
    print(f"cov3D difference:  max={torch.abs(cov3D_pt - cov3D_cuda).max():.8f}")
    print(f"scale difference:  max={torch.abs(scale_pt - scale_cuda).max():.8f}")

    all_match = (
        torch.allclose(m_cond_pt, m_cond_cuda, rtol=1e-5, atol=1e-5) and
        torch.allclose(cov3D_pt, cov3D_cuda, rtol=1e-5, atol=1e-5) and
        torch.allclose(scale_pt, scale_cuda, rtol=1e-5, atol=1e-5)
    )

    if all_match:
        print("\n✓ Forward outputs match!")
    else:
        print("\n✗ Forward outputs differ!")

    return all_match


if __name__ == "__main__":
    # Run forward output comparison
    test_forward_output_match(N=100, C=3)

    # Run main gradient check
    results = test_slice_gaussian_ndgs_gradients(N=1000, C=3)

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
