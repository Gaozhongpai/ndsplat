"""
Gradient verification test for cond_mean_covariance_opacity_cholesky CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- means [N, D]
- covars [N, D, D]
- opacities [N, 1]
- betas [N, C]

Also verifies forward pass output match between CUDA and PyTorch implementations.
"""

import torch
import torch.nn.functional as F

from gsplat import cond_mean_covariance_opacity_cholesky
from gsplat.cuda._torch_impl import _cond_mean_covariance_opacity_cholesky


def make_spd_batch(N, C, device, scale=1.0):
    """Create a batch of symmetric positive definite matrices [N, C, C]."""
    A = torch.randn(N, C, C, device=device) * scale
    return torch.bmm(A, A.transpose(-1, -2)) + 0.1 * torch.eye(C, device=device).unsqueeze(0)


def make_test_inputs(N, D, device, seed=42):
    """Create test inputs with valid covariance matrices."""
    torch.manual_seed(seed)
    C = D - 3

    # Full covariance: build from blocks to ensure SPD
    # Generate full D-dim SPD matrix
    A = torch.randn(N, D, D, device=device) * 0.5
    covars = torch.bmm(A, A.transpose(-1, -2)) + 0.1 * torch.eye(D, device=device).unsqueeze(0)

    means = torch.randn(N, D, device=device)
    opacities = torch.rand(N, 1, device=device) * 0.8 + 0.1
    betas = torch.rand(N, C, device=device) * 4.0 + 0.5  # positive betas
    query = torch.randn(N, C, device=device)

    return means, covars, opacities, betas, query


def test_forward_output_match(N=500, D=6, seed=42):
    """Verify that CUDA and PyTorch forward passes produce identical outputs."""
    device = "cuda"
    C = D - 3

    print("=" * 70)
    print(f"Forward Pass Output Comparison (N={N}, D={D}, C={C})")
    print("=" * 70)

    means, covars, opacities, betas, query = make_test_inputs(N, D, device, seed)

    # PyTorch forward
    m_pt, v_pt, o_pt = _cond_mean_covariance_opacity_cholesky(
        means, covars, opacities, betas, query
    )

    # CUDA forward
    m_cuda, v_cuda, o_cuda = cond_mean_covariance_opacity_cholesky(
        means, covars, opacities, betas, query
    )

    diff_m = (m_pt - m_cuda).abs().max().item()
    diff_v = (v_pt - v_cuda).abs().max().item()
    diff_o = (o_pt - o_cuda).abs().max().item()

    print(f"  cond_means max diff:      {diff_m:.8f}")
    print(f"  cond_covars max diff:     {diff_v:.8f}")
    print(f"  cond_opacities max diff:  {diff_o:.8f}")

    all_match = (
        torch.allclose(m_pt, m_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(v_pt, v_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(o_pt, o_cuda, rtol=1e-4, atol=1e-4)
    )

    if all_match:
        print("  PASS: Forward outputs match!")
    else:
        print("  FAIL: Forward outputs differ!")

    return all_match


def compare_grads(name, grad_pt, grad_cuda):
    """Compare gradients and print statistics."""
    abs_err = (grad_pt - grad_cuda).abs()
    rel_err = abs_err / (grad_pt.abs() + 1e-8)
    cos_sim = F.cosine_similarity(
        grad_pt.flatten().unsqueeze(0),
        grad_cuda.flatten().unsqueeze(0)
    ).item()

    print(f"\n  {name}:")
    print(f"    Shape: {tuple(grad_pt.shape)}")
    print(f"    Abs Error:  mean={abs_err.mean():.6f}, max={abs_err.max():.6f}")
    print(f"    Rel Error:  mean={rel_err.mean():.6f}, max={rel_err.max():.6f}")
    print(f"    Cosine Sim: {cos_sim:.6f}")
    print(f"    PT norm:    {grad_pt.norm():.6f},  CUDA norm: {grad_cuda.norm():.6f}")

    return {
        'abs_err_mean': abs_err.mean().item(),
        'abs_err_max': abs_err.max().item(),
        'rel_err_mean': rel_err.mean().item(),
        'rel_err_max': rel_err.max().item(),
        'cos_sim': cos_sim,
    }


def test_gradients(N=1000, D=6, seed=42):
    """Compare CUDA kernel gradients against PyTorch autograd gradients."""
    device = "cuda"
    C = D - 3

    print("=" * 70)
    print(f"Gradient Verification (N={N}, D={D}, C={C})")
    print("=" * 70)

    means, covars, opacities, betas, query = make_test_inputs(N, D, device, seed)

    # ========== PyTorch autograd (ground truth) ==========
    print("  Computing PyTorch autograd gradients...")
    means_pt = means.detach().clone().requires_grad_(True)
    covars_pt = covars.detach().clone().requires_grad_(True)
    opacities_pt = opacities.detach().clone().requires_grad_(True)
    betas_pt = betas.detach().clone().requires_grad_(True)

    m_pt, v_pt, o_pt = _cond_mean_covariance_opacity_cholesky(
        means_pt, covars_pt, opacities_pt, betas_pt, query
    )

    # Random upstream gradients
    grad_m = torch.randn_like(m_pt)
    grad_v = torch.randn_like(v_pt)
    grad_o = torch.randn_like(o_pt)

    loss_pt = (m_pt * grad_m).sum() + (v_pt * grad_v).sum() + (o_pt * grad_o).sum()
    loss_pt.backward()

    g_means_pt = means_pt.grad.clone()
    g_covars_pt = covars_pt.grad.clone()
    g_opacities_pt = opacities_pt.grad.clone()
    g_betas_pt = betas_pt.grad.clone()

    # ========== CUDA kernel ==========
    print("  Computing CUDA kernel gradients...")
    means_cuda = means.detach().clone().requires_grad_(True)
    covars_cuda = covars.detach().clone().requires_grad_(True)
    opacities_cuda = opacities.detach().clone().requires_grad_(True)
    betas_cuda = betas.detach().clone().requires_grad_(True)

    m_cuda, v_cuda, o_cuda = cond_mean_covariance_opacity_cholesky(
        means_cuda, covars_cuda, opacities_cuda, betas_cuda, query
    )

    loss_cuda = (m_cuda * grad_m).sum() + (v_cuda * grad_v).sum() + (o_cuda * grad_o).sum()
    loss_cuda.backward()

    g_means_cuda = means_cuda.grad.clone()
    g_covars_cuda = covars_cuda.grad.clone()
    g_opacities_cuda = opacities_cuda.grad.clone()
    g_betas_cuda = betas_cuda.grad.clone()

    # ========== Compare ==========
    print("\n  Gradient Comparison:")
    print("  " + "-" * 60)

    results = {}
    results['means'] = compare_grads("means", g_means_pt, g_means_cuda)
    results['covars'] = compare_grads("covars", g_covars_pt, g_covars_cuda)
    results['opacities'] = compare_grads("opacities", g_opacities_pt, g_opacities_cuda)
    results['betas'] = compare_grads("betas", g_betas_pt, g_betas_cuda)

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY:")
    all_good = True
    for name, res in results.items():
        cos_sim = res['cos_sim']
        if cos_sim > 0.999:
            status = "PASS"
        elif cos_sim > 0.99:
            status = "WARN"
            all_good = False
        else:
            status = "FAIL"
            all_good = False
        print(f"    {name:12s}: cosine_sim={cos_sim:.6f}  [{status}]")

    if all_good:
        print("\n  All gradients PASS (cosine similarity > 0.999)")
    else:
        print("\n  Some gradients have significant error!")

    return results


def test_different_dimensions():
    """Test with various D values."""
    print("\n" + "=" * 70)
    print("Testing Different Dimension Combinations")
    print("=" * 70)

    configs = [
        (6, "D=6 (3 spatial + 3 view)"),
        (7, "D=7 (3 spatial + 3 view + 1 time)"),
        (5, "D=5 (3 spatial + 2 cond)"),
        (4, "D=4 (3 spatial + 1 cond, minimal)"),
    ]

    all_results = {}
    for D, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_gradients(N=500, D=D, seed=42)
            all_results[desc] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[desc] = None

    return all_results


def test_numerical_gradcheck(N=10, D=6, seed=42):
    """Use torch.autograd.gradcheck for rigorous numerical gradient verification."""
    device = "cuda"

    print("\n" + "=" * 70)
    print(f"Numerical Gradcheck (N={N}, D={D})")
    print("=" * 70)

    means, covars, opacities, betas, query = make_test_inputs(N, D, device, seed)

    # Use float64 for numerical gradcheck
    means = means.double().requires_grad_(True)
    covars = covars.double().requires_grad_(True)
    opacities = opacities.double().requires_grad_(True)
    betas = betas.double().requires_grad_(True)
    query = query.double()

    # Test PyTorch implementation with gradcheck
    print("  Running gradcheck on PyTorch implementation...")
    try:
        result = torch.autograd.gradcheck(
            _cond_mean_covariance_opacity_cholesky,
            (means, covars, opacities, betas, query),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print(f"  PyTorch gradcheck: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"  PyTorch gradcheck FAILED: {e}")

    # Test CUDA implementation with gradcheck (if on CUDA)
    print("  Running gradcheck on CUDA implementation...")
    try:
        result = torch.autograd.gradcheck(
            cond_mean_covariance_opacity_cholesky,
            (means, covars, opacities, betas, query),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print(f"  CUDA gradcheck: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"  CUDA gradcheck FAILED: {e}")


if __name__ == "__main__":
    # 1. Forward output comparison
    for D in [4, 5, 6, 7]:
        test_forward_output_match(N=500, D=D)
        print()

    # 2. Gradient comparison (CUDA vs PyTorch autograd)
    print("\n")
    test_gradients(N=1000, D=6)

    # 3. Different dimensions
    print("\n")
    test_different_dimensions()

    # 4. Numerical gradcheck (small N, double precision)
    print("\n")
    test_numerical_gradcheck(N=10, D=6)

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
