import torch
import torch.nn.functional as F
from submodules.gsplat.gsplat.cuda._torch_impl import _slice_gaussian_full_v2  
from submodules.gsplat.gsplat.cuda._wrapper import slice_gaussian_full_v2

torch.manual_seed(42)
N, C = 2, 3
device = "cuda"

# Simple test case
xyz = torch.randn(N, 3, device=device, requires_grad=True)
view_mean = torch.zeros(N, C, device=device, requires_grad=True)
query = torch.ones(N, C, device=device) * 0.1
v_12 = torch.randn(N, 3 * C, device=device, requires_grad=True)
L_22_inv = torch.randn(N, 6, device=device, requires_grad=True) * 0.1
lambda_view = torch.ones(N, device=device, requires_grad=True) * 0.5

# PyTorch forward + backward
x_cond_pt, attn_pt = _slice_gaussian_full_v2(
    xyz, view_mean, query, v_12, L_22_inv, 
    lambda_view, None, 0.35, True
)
loss_pt = x_cond_pt.sum()
loss_pt.backward()

print("PyTorch L_22_inv gradient:")
print(L_22_inv.grad)
print(f"Norm: {L_22_inv.grad.norm():.6f}")

# Reset gradients
xyz2 = xyz.detach().clone().requires_grad_(True)
view_mean2 = view_mean.detach().clone().requires_grad_(True)
v_122 = v_12.detach().clone().requires_grad_(True)
L_22_inv2 = L_22_inv.detach().clone().requires_grad_(True)
lambda_view2 = lambda_view.detach().clone().requires_grad_(True)

# CUDA forward + backward
x_cond_cuda, attn_cuda = slice_gaussian_full_v2(
    xyz2, view_mean2, query, v_122, L_22_inv2,
    lambda_view2, None, 0.35, True
)
loss_cuda = x_cond_cuda.sum()
loss_cuda.backward()

print("\nCUDA L_22_inv gradient:")
print(L_22_inv2.grad)
print(f"Norm: {L_22_inv2.grad.norm():.6f}")

print("\nDifference:")
print(L_22_inv.grad - L_22_inv2.grad)
print(f"Max abs diff: {(L_22_inv.grad - L_22_inv2.grad).abs().max():.6f}")
print(f"Cosine sim: {F.cosine_similarity(L_22_inv.grad.flatten(), L_22_inv2.grad.flatten(), dim=0):.6f}")
