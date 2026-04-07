"""
Gradient verification test for slice_dbs CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- xyz [N, 3]
- cond_mean [N, C]
- v_12 [N, 3*C]
- L_22_inv [N, C*(C+1)/2]
- betas [N, C]

Also verifies forward pass output match between CUDA and PyTorch implementations.
"""

import torch
import torch.nn.functional as F

from gsplat import slice_dbs, _slice_dbs


def make_test_inputs(N, C, device, seed=42):
    """Create test inputs for slice_dbs."""
    torch.manual_seed(seed)

    xyz = torch.randn(N, 3, device=device)
    cond_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    v_12 = torch.randn(N, 3 * C, device=device) * 0.1
    n_L = C * (C + 1) // 2
    L_22_inv = torch.randn(N, n_L, device=device) * 0.5

    # Activated betas: 4.0 * exp(raw), so raw=0 gives beta=4.0
    raw_betas = torch.randn(N, C, device=device) * 0.5
    betas = 4.0 * torch.exp(raw_betas)

    return xyz, cond_mean, query, v_12, L_22_inv, betas


def test_forward_output_match(N=500, C=3, seed=42):
    """Verify that CUDA and PyTorch forward passes produce identical outputs."""
    device = "cuda"

    print("=" * 70)
    print(f"Forward Pass Output Comparison (N={N}, C={C})")
    print("=" * 70)

    xyz, cond_mean, query, v_12, L_22_inv, betas = make_test_inputs(N, C, device, seed)

    # PyTorch forward
    m_pt, o_pt = _slice_dbs(xyz, cond_mean, query, v_12, L_22_inv, betas)

    # CUDA forward
    m_cuda, o_cuda = slice_dbs(xyz, cond_mean, query, v_12, L_22_inv, betas)

    diff_m = (m_pt - m_cuda).abs().max().item()
    diff_o = (o_pt - o_cuda).abs().max().item()

    print(f"  m_cond max diff:         {diff_m:.8f}")
    print(f"  opacity_scale max diff:  {diff_o:.8f}")

    all_match = (
        torch.allclose(m_pt, m_cuda, rtol=1e-4, atol=1e-4) and
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
        'cos_sim': cos_sim,
    }


def test_gradients(N=1000, C=3, seed=42):
    """Compare CUDA kernel gradients against PyTorch autograd gradients."""
    device = "cuda"

    print("=" * 70)
    print(f"Gradient Verification (N={N}, C={C})")
    print("=" * 70)

    xyz, cond_mean, query, v_12, L_22_inv, betas = make_test_inputs(N, C, device, seed)

    # ========== PyTorch autograd (ground truth) ==========
    print("  Computing PyTorch autograd gradients...")
    xyz_pt = xyz.detach().clone().requires_grad_(True)
    cm_pt = cond_mean.detach().clone().requires_grad_(True)
    v12_pt = v_12.detach().clone().requires_grad_(True)
    L_pt = L_22_inv.detach().clone().requires_grad_(True)
    b_pt = betas.detach().clone().requires_grad_(True)

    m_pt, o_pt = _slice_dbs(xyz_pt, cm_pt, query, v12_pt, L_pt, b_pt)

    grad_m = torch.randn_like(m_pt)
    grad_o = torch.randn_like(o_pt)

    loss_pt = (m_pt * grad_m).sum() + (o_pt * grad_o).sum()
    loss_pt.backward()

    g_xyz_pt = xyz_pt.grad.clone()
    g_cm_pt = cm_pt.grad.clone()
    g_v12_pt = v12_pt.grad.clone()
    g_L_pt = L_pt.grad.clone()
    g_b_pt = b_pt.grad.clone()

    # ========== CUDA kernel ==========
    print("  Computing CUDA kernel gradients...")
    xyz_cuda = xyz.detach().clone().requires_grad_(True)
    cm_cuda = cond_mean.detach().clone().requires_grad_(True)
    v12_cuda = v_12.detach().clone().requires_grad_(True)
    L_cuda = L_22_inv.detach().clone().requires_grad_(True)
    b_cuda = betas.detach().clone().requires_grad_(True)

    m_cuda, o_cuda = slice_dbs(xyz_cuda, cm_cuda, query, v12_cuda, L_cuda, b_cuda)

    loss_cuda = (m_cuda * grad_m).sum() + (o_cuda * grad_o).sum()
    loss_cuda.backward()

    g_xyz_cuda = xyz_cuda.grad.clone()
    g_cm_cuda = cm_cuda.grad.clone()
    g_v12_cuda = v12_cuda.grad.clone()
    g_L_cuda = L_cuda.grad.clone()
    g_b_cuda = b_cuda.grad.clone()

    # ========== Compare ==========
    print("\n  Gradient Comparison:")
    print("  " + "-" * 60)

    results = {}
    results['xyz'] = compare_grads("xyz", g_xyz_pt, g_xyz_cuda)
    results['cond_mean'] = compare_grads("cond_mean", g_cm_pt, g_cm_cuda)
    results['v_12'] = compare_grads("v_12", g_v12_pt, g_v12_cuda)
    results['L_22_inv'] = compare_grads("L_22_inv", g_L_pt, g_L_cuda)
    results['betas'] = compare_grads("betas", g_b_pt, g_b_cuda)

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


def test_different_dims():
    """Test with C=3 and C=4."""
    print("\n" + "=" * 70)
    print("Testing Different Dimensions")
    print("=" * 70)

    for C, desc in [(3, "C=3 (view only, 6DGS)"), (4, "C=4 (view+time, 7DGS)")]:
        print(f"\n--- {desc} ---")
        try:
            test_forward_output_match(N=500, C=C)
            print()
            test_gradients(N=500, C=C)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_numerical_gradcheck(N=10, C=3, seed=42):
    """Use torch.autograd.gradcheck for rigorous numerical gradient verification."""
    device = "cuda"

    print("\n" + "=" * 70)
    print(f"Numerical Gradcheck (N={N}, C={C})")
    print("=" * 70)

    xyz, cond_mean, query, v_12, L_22_inv, betas = make_test_inputs(N, C, device, seed)

    # Float64 for numerical precision
    xyz = xyz.double().requires_grad_(True)
    cond_mean = cond_mean.double().requires_grad_(True)
    query = query.double()
    v_12 = v_12.double().requires_grad_(True)
    L_22_inv = L_22_inv.double().requires_grad_(True)
    betas = betas.double().requires_grad_(True)

    print("  Running gradcheck on PyTorch implementation...")
    try:
        result = torch.autograd.gradcheck(
            _slice_dbs,
            (xyz, cond_mean, query, v_12, L_22_inv, betas),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print(f"  PyTorch gradcheck: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"  PyTorch gradcheck FAILED: {e}")


if __name__ == "__main__":
    # 1. Forward output comparison
    for C in [3, 4]:
        test_forward_output_match(N=500, C=C)
        print()

    # 2. Gradient comparison (CUDA vs PyTorch autograd)
    print("\n")
    test_gradients(N=1000, C=3)

    # 3. Different dimensions
    print("\n")
    test_different_dims()

    # 4. Numerical gradcheck (small N, double precision, PyTorch only)
    print("\n")
    test_numerical_gradcheck(N=10, C=3)
    test_numerical_gradcheck(N=10, C=4)

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
