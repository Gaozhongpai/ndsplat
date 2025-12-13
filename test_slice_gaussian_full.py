"""
Gradient verification test for slice_gaussian_full CUDA kernel.

Compares CUDA kernel gradients against PyTorch autograd gradients for all parameters:
- Position: xyz, v_12, L_22_inv_diag, view_mean
- Scale: scaling, scale_deviation, L_22_inv_diag_scale
- Rotation: rotation, rotation_delta, L_22_inv_diag_rot
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import both CUDA and PyTorch implementations
from gsplat import slice_gaussian_full
from gsplat.cuda._torch_impl import _slice_gaussian_full, _quaternion_slerp, _quaternion_multiply


def test_slice_gaussian_full_gradients(N=1000, C=3, seed=42):
    """
    Compare CUDA kernel gradients against PyTorch autograd gradients for ALL parameters.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view, 4 for view+time)
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    device = "cuda"

    print("=" * 70)
    print("slice_gaussian_full Gradient Verification Test")
    print("=" * 70)
    print(f"N={N}, C={C}")
    print()

    # Create test inputs
    xyz = torch.randn(N, 3, device=device, requires_grad=True)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    view_mean.requires_grad_(True)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)

    # Position conditioning
    v_12 = torch.randn(N, 3 * C, device=device, requires_grad=True) * 0.1
    L_22_inv_diag = torch.randn(N, C, device=device, requires_grad=True) * 0.5

    # Scale conditioning
    scaling = torch.exp(torch.randn(N, 3, device=device) * 0.5)
    scaling.requires_grad_(True)
    scale_deviation = torch.randn(N, 3, device=device, requires_grad=True) * 0.1
    L_22_inv_diag_scale = torch.randn(N, C, device=device, requires_grad=True) * 0.5

    # Rotation conditioning
    rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation.requires_grad_(True)
    rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation_delta.requires_grad_(True)
    L_22_inv_diag_rot = torch.randn(N, C, device=device, requires_grad=True) * 0.5

    # ============ Test with PyTorch autograd (ground truth) ============
    print("Computing PyTorch autograd gradients (ground truth)...")

    # Clone inputs for PyTorch path
    xyz_pt = xyz.detach().clone().requires_grad_(True)
    view_mean_pt = view_mean.detach().clone().requires_grad_(True)
    v_12_pt = v_12.detach().clone().requires_grad_(True)
    L_22_inv_diag_pt = L_22_inv_diag.detach().clone().requires_grad_(True)
    scaling_pt = scaling.detach().clone().requires_grad_(True)
    scale_deviation_pt = scale_deviation.detach().clone().requires_grad_(True)
    L_22_inv_diag_scale_pt = L_22_inv_diag_scale.detach().clone().requires_grad_(True)
    rotation_pt = rotation.detach().clone().requires_grad_(True)
    rotation_delta_pt = rotation_delta.detach().clone().requires_grad_(True)
    L_22_inv_diag_rot_pt = L_22_inv_diag_rot.detach().clone().requires_grad_(True)

    # Forward pass with PyTorch
    x_cond_pt, scale_cond_pt, rotation_cond_pt, attention_pt = _slice_gaussian_full(
        xyz_pt, view_mean_pt, query,
        v_12_pt, L_22_inv_diag_pt,
        scaling_pt, scale_deviation_pt, L_22_inv_diag_scale_pt,
        rotation_pt, rotation_delta_pt, L_22_inv_diag_rot_pt
    )

    # Create random upstream gradients
    grad_x_cond = torch.randn_like(x_cond_pt)
    grad_scale_cond = torch.randn_like(scale_cond_pt)
    grad_rotation_cond = torch.randn_like(rotation_cond_pt)
    grad_attention = torch.randn_like(attention_pt)

    # Backward pass with PyTorch
    loss_pt = (
        (x_cond_pt * grad_x_cond).sum() +
        (scale_cond_pt * grad_scale_cond).sum() +
        (rotation_cond_pt * grad_rotation_cond).sum() +
        (attention_pt * grad_attention).sum()
    )
    loss_pt.backward()

    # Store PyTorch gradients
    pt_grads = {
        'xyz': xyz_pt.grad.clone(),
        'view_mean': view_mean_pt.grad.clone(),
        'v_12': v_12_pt.grad.clone(),
        'L_22_inv_diag': L_22_inv_diag_pt.grad.clone(),
        'scaling': scaling_pt.grad.clone(),
        'scale_deviation': scale_deviation_pt.grad.clone(),
        'L_22_inv_diag_scale': L_22_inv_diag_scale_pt.grad.clone(),
        'rotation': rotation_pt.grad.clone(),
        'rotation_delta': rotation_delta_pt.grad.clone(),
        'L_22_inv_diag_rot': L_22_inv_diag_rot_pt.grad.clone(),
    }

    # ============ Test with CUDA kernel ============
    print("Computing CUDA kernel gradients...")

    # Clone inputs for CUDA path
    xyz_cuda = xyz.detach().clone().requires_grad_(True)
    view_mean_cuda = view_mean.detach().clone().requires_grad_(True)
    v_12_cuda = v_12.detach().clone().requires_grad_(True)
    L_22_inv_diag_cuda = L_22_inv_diag.detach().clone().requires_grad_(True)
    scaling_cuda = scaling.detach().clone().requires_grad_(True)
    scale_deviation_cuda = scale_deviation.detach().clone().requires_grad_(True)
    L_22_inv_diag_scale_cuda = L_22_inv_diag_scale.detach().clone().requires_grad_(True)
    rotation_cuda = rotation.detach().clone().requires_grad_(True)
    rotation_delta_cuda = rotation_delta.detach().clone().requires_grad_(True)
    L_22_inv_diag_rot_cuda = L_22_inv_diag_rot.detach().clone().requires_grad_(True)

    # Forward pass with CUDA kernel
    x_cond_cuda, scale_cond_cuda, rotation_cond_cuda, attention_cuda = slice_gaussian_full(
        xyz_cuda, view_mean_cuda, query,
        v_12_cuda, L_22_inv_diag_cuda,
        scaling_cuda, scale_deviation_cuda, L_22_inv_diag_scale_cuda,
        rotation_cuda, rotation_delta_cuda, L_22_inv_diag_rot_cuda
    )

    # Backward pass with same upstream gradients
    loss_cuda = (
        (x_cond_cuda * grad_x_cond).sum() +
        (scale_cond_cuda * grad_scale_cond).sum() +
        (rotation_cond_cuda * grad_rotation_cond).sum() +
        (attention_cuda * grad_attention).sum()
    )
    loss_cuda.backward()

    # Store CUDA gradients
    cuda_grads = {
        'xyz': xyz_cuda.grad.clone(),
        'view_mean': view_mean_cuda.grad.clone(),
        'v_12': v_12_cuda.grad.clone(),
        'L_22_inv_diag': L_22_inv_diag_cuda.grad.clone(),
        'scaling': scaling_cuda.grad.clone(),
        'scale_deviation': scale_deviation_cuda.grad.clone(),
        'L_22_inv_diag_scale': L_22_inv_diag_scale_cuda.grad.clone(),
        'rotation': rotation_cuda.grad.clone(),
        'rotation_delta': rotation_delta_cuda.grad.clone(),
        'L_22_inv_diag_rot': L_22_inv_diag_rot_cuda.grad.clone(),
    }

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
    results['v_12'] = compare_grads("v_12 (position-view covariance)", pt_grads['v_12'], cuda_grads['v_12'])
    results['L_22_inv_diag'] = compare_grads("L_22_inv_diag (position precision)", pt_grads['L_22_inv_diag'], cuda_grads['L_22_inv_diag'])

    # Scale gradients
    print("\n--- SCALE GRADIENTS ---")
    results['scaling'] = compare_grads("scaling (base scale)", pt_grads['scaling'], cuda_grads['scaling'])
    results['scale_deviation'] = compare_grads("scale_deviation", pt_grads['scale_deviation'], cuda_grads['scale_deviation'])
    results['L_22_inv_diag_scale'] = compare_grads("L_22_inv_diag_scale (scale precision)", pt_grads['L_22_inv_diag_scale'], cuda_grads['L_22_inv_diag_scale'])

    # Rotation gradients
    print("\n--- ROTATION GRADIENTS ---")
    results['rotation'] = compare_grads("rotation (q_base)", pt_grads['rotation'], cuda_grads['rotation'])
    results['rotation_delta'] = compare_grads("rotation_delta (q_delta)", pt_grads['rotation_delta'], cuda_grads['rotation_delta'])
    results['L_22_inv_diag_rot'] = compare_grads("L_22_inv_diag_rot (rotation precision)", pt_grads['L_22_inv_diag_rot'], cuda_grads['L_22_inv_diag_rot'])

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
        status = "✓ GOOD" if cos_sim > 0.99 else ("⚠ WARNING" if cos_sim > 0.9 else "✗ BAD")
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


def test_forward_output_match(N=100, C=3, seed=42):
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
    xyz = torch.randn(N, 3, device=device)
    view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
    query = F.normalize(torch.randn(N, C, device=device), dim=-1)
    v_12 = torch.randn(N, 3 * C, device=device) * 0.1
    L_22_inv_diag = torch.randn(N, C, device=device) * 0.5
    scaling = torch.exp(torch.randn(N, 3, device=device) * 0.5)
    scale_deviation = torch.randn(N, 3, device=device) * 0.1
    L_22_inv_diag_scale = torch.randn(N, C, device=device) * 0.5
    rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    L_22_inv_diag_rot = torch.randn(N, C, device=device) * 0.5

    # PyTorch forward
    x_cond_pt, scale_cond_pt, rotation_cond_pt, attention_pt = _slice_gaussian_full(
        xyz, view_mean, query,
        v_12, L_22_inv_diag,
        scaling, scale_deviation, L_22_inv_diag_scale,
        rotation, rotation_delta, L_22_inv_diag_rot
    )

    # CUDA forward
    x_cond_cuda, scale_cond_cuda, rotation_cond_cuda, attention_cuda = slice_gaussian_full(
        xyz, view_mean, query,
        v_12, L_22_inv_diag,
        scaling, scale_deviation, L_22_inv_diag_scale,
        rotation, rotation_delta, L_22_inv_diag_rot
    )

    # Compare outputs
    print(f"\nx_cond difference:        max={torch.abs(x_cond_pt - x_cond_cuda).max():.8f}")
    print(f"scale_cond difference:    max={torch.abs(scale_cond_pt - scale_cond_cuda).max():.8f}")
    print(f"rotation_cond difference: max={torch.abs(rotation_cond_pt - rotation_cond_cuda).max():.8f}")
    print(f"attention difference:     max={torch.abs(attention_pt - attention_cuda).max():.8f}")

    all_match = (
        torch.allclose(x_cond_pt, x_cond_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(scale_cond_pt, scale_cond_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(rotation_cond_pt, rotation_cond_cuda, rtol=1e-4, atol=1e-4) and
        torch.allclose(attention_pt, attention_cuda, rtol=1e-4, atol=1e-4)
    )

    if all_match:
        print("\n✓ Forward outputs match!")
    else:
        print("\n✗ Forward outputs differ!")

    return all_match


def test_gradient_at_different_t_values():
    """
    Test gradient accuracy at different interpolation factors (t values).

    The approximation should be more accurate when:
    - t ≈ 0 (near canonical view): delta gets ~0 gradient
    - t ≈ 1 (far from canonical): delta gets full gradient
    """
    print("\n" + "=" * 70)
    print("Testing gradient accuracy at different t values (rotation)")
    print("=" * 70)

    N = 100
    C = 3
    device = "cuda"
    torch.manual_seed(42)

    # Test at different attention levels (which determine t = 1 - attention)
    attention_targets = [0.99, 0.9, 0.5, 0.1, 0.01]  # t = 0.01, 0.1, 0.5, 0.9, 0.99

    for target_attn in attention_targets:
        # Create inputs that produce specific attention values
        xyz = torch.randn(N, 3, device=device, requires_grad=True)
        view_mean = F.normalize(torch.randn(N, C, device=device), dim=-1)
        view_mean.requires_grad_(True)

        # Set query to be close/far from view_mean based on desired attention
        # attention = exp(-||x||^2) where x = (query - view_mean) * L_diag
        # So ||x||^2 = -log(attention)
        target_x_norm_sq = -np.log(target_attn)

        # Create query that achieves target attention
        query = view_mean.detach().clone()
        noise = F.normalize(torch.randn(N, C, device=device), dim=-1)
        query = query + noise * np.sqrt(target_x_norm_sq / C)
        query = F.normalize(query, dim=-1)

        # Other parameters
        v_12 = torch.randn(N, 3 * C, device=device, requires_grad=True) * 0.1
        L_22_inv_diag = torch.zeros(N, C, device=device, requires_grad=True)  # exp(0)^2 = 1
        scaling = torch.ones(N, 3, device=device, requires_grad=True)
        scale_deviation = torch.zeros(N, 3, device=device, requires_grad=True)
        L_22_inv_diag_scale = torch.zeros(N, C, device=device, requires_grad=True)
        rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        rotation.requires_grad_(True)
        rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        rotation_delta.requires_grad_(True)
        L_22_inv_diag_rot = torch.zeros(N, C, device=device, requires_grad=True)

        # PyTorch forward/backward
        rotation_pt = rotation.detach().clone().requires_grad_(True)
        rotation_delta_pt = rotation_delta.detach().clone().requires_grad_(True)

        x_cond_pt, scale_cond_pt, rotation_cond_pt, attention_pt = _slice_gaussian_full(
            xyz.detach(), view_mean.detach(), query,
            v_12.detach(), L_22_inv_diag.detach(),
            scaling.detach(), scale_deviation.detach(), L_22_inv_diag_scale.detach(),
            rotation_pt, rotation_delta_pt, L_22_inv_diag_rot.detach()
        )

        grad_rotation_cond = torch.randn_like(rotation_cond_pt)
        loss_pt = (rotation_cond_pt * grad_rotation_cond).sum()
        loss_pt.backward()

        # CUDA forward/backward
        rotation_cuda = rotation.detach().clone().requires_grad_(True)
        rotation_delta_cuda = rotation_delta.detach().clone().requires_grad_(True)

        x_cond_cuda, scale_cond_cuda, rotation_cond_cuda, attention_cuda = slice_gaussian_full(
            xyz.detach(), view_mean.detach(), query,
            v_12.detach(), L_22_inv_diag.detach(),
            scaling.detach(), scale_deviation.detach(), L_22_inv_diag_scale.detach(),
            rotation_cuda, rotation_delta_cuda, L_22_inv_diag_rot.detach()
        )

        loss_cuda = (rotation_cond_cuda * grad_rotation_cond).sum()
        loss_cuda.backward()

        # Compute actual t value
        actual_attn = attention_cuda.mean().item()
        actual_t = 1.0 - actual_attn

        # Compare rotation_delta gradients
        cos_sim_delta = F.cosine_similarity(
            rotation_delta_pt.grad.flatten().unsqueeze(0),
            rotation_delta_cuda.grad.flatten().unsqueeze(0)
        ).item()

        cos_sim_base = F.cosine_similarity(
            rotation_pt.grad.flatten().unsqueeze(0),
            rotation_cuda.grad.flatten().unsqueeze(0)
        ).item()

        print(f"t ≈ {actual_t:.3f} (attention ≈ {actual_attn:.3f}): "
              f"q_base cos_sim={cos_sim_base:.4f}, q_delta cos_sim={cos_sim_delta:.4f}")


def test_different_dimensions():
    """
    Test gradient accuracy with different conditioning dimensions.
    """
    print()
    print("=" * 70)
    print("Testing Different Conditioning Dimensions")
    print("=" * 70)

    configs = [
        (3, "C=3 (view direction only)"),
        (4, "C=4 (view + time)"),
    ]

    all_results = {}
    for C, desc in configs:
        print(f"\n--- {desc} ---")
        try:
            results = test_slice_gaussian_full_gradients(N=500, C=C, seed=42)
            all_results[desc] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[desc] = None

    return all_results


if __name__ == "__main__":
    # Run forward output comparison
    test_forward_output_match(N=100, C=3)

    # Run main gradient check
    print("\n\n")
    results = test_slice_gaussian_full_gradients(N=1000, C=3)

    # Test at different interpolation points
    test_gradient_at_different_t_values()

    # Test different conditioning dimensions
    print("\n\n")
    test_different_dimensions()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
