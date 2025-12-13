"""
Test rotation gradient specifically to debug the rotation_delta gradient mismatch.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/code/workspace/6dgs-iclr/submodules/gsplat')

def _quaternion_slerp(q0, q1, t):
    """Reference SLERP implementation."""
    # Normalize inputs
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    # Compute dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # [N, 1]

    # If dot < 0, negate q1 to take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # Clamp dot
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angle between quaternions
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Handle near-parallel case
    use_lerp = sin_theta.abs() < 1e-6

    t_theta = t * theta
    one_minus_t_theta = (1.0 - t) * theta

    s0 = torch.sin(one_minus_t_theta) / (sin_theta + 1e-8)
    s1 = torch.sin(t_theta) / (sin_theta + 1e-8)

    s0_lerp = 1.0 - t
    s1_lerp = t

    s0 = torch.where(use_lerp, s0_lerp, s0)
    s1 = torch.where(use_lerp, s1_lerp, s1)

    result = s0 * q0 + s1 * q1
    return F.normalize(result, dim=-1)

def _quaternion_multiply(q1, q2):
    """Reference quaternion multiplication."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def test_cuda_vs_pytorch_rotation_gradient():
    """Compare CUDA kernel gradient against PyTorch reference."""
    from gsplat import slice_gaussian_full

    torch.manual_seed(42)
    N = 100
    C = 4  # Need C=4 for rotation
    device = "cuda"

    # Create inputs
    xyz = torch.randn(N, 3, device=device, requires_grad=True)
    view_mean = torch.randn(N, C, device=device, requires_grad=True)
    query = torch.randn(N, C, device=device)

    # Position conditioning
    v_12 = torch.randn(N, 3, C, device=device, requires_grad=True)
    L_22_inv = torch.randn(N, C * (C + 1) // 2, device=device, requires_grad=True)

    # Rotation conditioning
    rotation = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation.requires_grad_(True)
    rotation_delta = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    rotation_delta.requires_grad_(True)
    L_22_inv_diag_rot = torch.randn(N, 1, device=device, requires_grad=True)

    lambda_opc = 0.35

    # CUDA forward
    x_cond, rotation_cond, attention_weight = slice_gaussian_full(
        xyz, view_mean, query,
        v_12, L_22_inv,
        rotation, rotation_delta, L_22_inv_diag_rot,
        lambda_opc
    )

    # Random gradients
    grad_x_cond = torch.randn_like(x_cond)
    grad_rotation_cond = torch.randn_like(rotation_cond)
    grad_attention = torch.randn_like(attention_weight)

    # CUDA backward
    torch.autograd.backward(
        [x_cond, rotation_cond, attention_weight],
        [grad_x_cond, grad_rotation_cond, grad_attention]
    )

    cuda_grad_rotation_delta = rotation_delta.grad.clone()
    cuda_grad_rotation = rotation.grad.clone()
    cuda_grad_L_22_inv_diag_rot = L_22_inv_diag_rot.grad.clone()

    # Reset gradients
    rotation_delta.grad = None
    rotation.grad = None
    L_22_inv_diag_rot.grad = None
    xyz.grad = None
    view_mean.grad = None
    v_12.grad = None
    L_22_inv.grad = None

    # PyTorch reference forward
    from gsplat.cuda._torch_impl import _slice_gaussian_full

    x_cond_ref, rotation_cond_ref, attention_weight_ref = _slice_gaussian_full(
        xyz, view_mean, query,
        v_12, L_22_inv,
        rotation, rotation_delta, L_22_inv_diag_rot,
        lambda_opc
    )

    # PyTorch backward
    torch.autograd.backward(
        [x_cond_ref, rotation_cond_ref, attention_weight_ref],
        [grad_x_cond, grad_rotation_cond, grad_attention]
    )

    pytorch_grad_rotation_delta = rotation_delta.grad.clone()
    pytorch_grad_rotation = rotation.grad.clone()
    pytorch_grad_L_22_inv_diag_rot = L_22_inv_diag_rot.grad.clone()

    # Compare
    print("CUDA vs PyTorch rotation gradients:")

    cos_sim_delta = F.cosine_similarity(
        cuda_grad_rotation_delta.flatten(),
        pytorch_grad_rotation_delta.flatten(),
        dim=0
    )
    print(f"  rotation_delta: cosine_sim={cos_sim_delta.item():.6f}")

    cos_sim_rot = F.cosine_similarity(
        cuda_grad_rotation.flatten(),
        pytorch_grad_rotation.flatten(),
        dim=0
    )
    print(f"  rotation: cosine_sim={cos_sim_rot.item():.6f}")

    cos_sim_L = F.cosine_similarity(
        cuda_grad_L_22_inv_diag_rot.flatten(),
        pytorch_grad_L_22_inv_diag_rot.flatten(),
        dim=0
    )
    print(f"  L_22_inv_diag_rot: cosine_sim={cos_sim_L.item():.6f}")

    # Detailed comparison for rotation_delta
    print("\nDetailed rotation_delta gradient comparison (first 5 samples):")
    for i in range(min(5, N)):
        print(f"  Sample {i}:")
        print(f"    CUDA:    {cuda_grad_rotation_delta[i].cpu().numpy()}")
        print(f"    PyTorch: {pytorch_grad_rotation_delta[i].cpu().numpy()}")
        diff = (cuda_grad_rotation_delta[i] - pytorch_grad_rotation_delta[i]).abs()
        print(f"    Diff:    {diff.cpu().numpy()}")

    return cuda_grad_rotation_delta, pytorch_grad_rotation_delta


if __name__ == "__main__":
    test_cuda_vs_pytorch_rotation_gradient()
