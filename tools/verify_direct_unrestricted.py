"""Pre-training verification for the Direct-Unrestricted ablation (NeurIPS 26 rebuttal).

Checks that (a) the opacity path is bit-identical to dGS, (b) the position update is
exactly mu_p + M delta, and (c) setting M = V_pq diag(Lambda) V_qq reproduces dGS,
which establishes that Direct-Unrestricted strictly contains dGS.

Usage: python tools/verify_direct_unrestricted.py
"""
import torch, sys; sys.path.insert(0,"/code")
from scene.gaussian_model_dgs import GaussianModel
torch.manual_seed(0)
N, C = 1000, 3

xyz  = torch.randn(N,3,device="cuda")
mv   = torch.nn.functional.normalize(torch.randn(N,3,device="cuda"),dim=1)
Lraw = torch.randn(N,C*(C+1)//2,device="cuda")*0.3
sc   = torch.log(torch.rand(N,3,device="cuda")*0.1+0.01)
v12  = torch.randn(N,3*C,device="cuda")*0.05
lamv = torch.randn(N,device="cuda")*0.5

def mk(du, v12_override=None):
    g = GaussianModel(3, input_dim=3+C, direct_unrestricted=du)
    g._xyz, g._mean_view, g._L_22_inv, g._scaling = xyz, mv, Lraw, sc
    g._v_12_direction = v12 if v12_override is None else v12_override
    g._lambda_view = lamv
    return g

q = torch.nn.functional.normalize(torch.randn(N,C,device="cuda"),dim=1)
dgs, du = mk(False), mk(True)
x_dgs, op_dgs = dgs.slice_gaussian_full_method(q)
x_du,  op_du  = du.slice_gaussian_full_method(q)

print(f"1. opacity identical (same L, same delta): {torch.allclose(op_dgs,op_du,atol=1e-6)}  max err {(op_dgs-op_du).abs().max():.2e}")

delta = q - du.get_cond_mean
M = du._v_12_direction.reshape(N,3,C)
print(f"2. DU position == mu_p + M@delta         : {torch.allclose(x_du, xyz+torch.bmm(M,delta.unsqueeze(-1)).squeeze(-1), atol=1e-6)}")

# Build dGS's effective operator and feed it to DU -> must reproduce dGS exactly
L = torch.zeros(N,C,C,device="cuda"); i_=0
for i in range(C):
    for j in range(i+1):
        L[:,i,j] = torch.exp(Lraw[:,i_]) if i==j else Lraw[:,i_]; i_+=1
Vqq  = torch.bmm(L, L.transpose(1,2))
Meff = torch.bmm(dgs.get_v_12.reshape(N,3,C) * torch.sigmoid(lamv).view(N,1,1), Vqq)
x_eq,_ = mk(True, Meff.reshape(N,3*C).contiguous()).slice_gaussian_full_method(q)
print(f"3. DU with M = V_pq diag(Lam) V_qq == dGS: {torch.allclose(x_dgs,x_eq,atol=1e-5)}  max err {(x_dgs-x_eq).abs().max():.2e}")

# M is unconstrained: dGS row norms are capped by s-bar, DU's are not
sbar = dgs.get_scaling.mean(dim=1)
print(f"\n4. dGS row-norm/s-bar max = {(dgs.get_v_12.reshape(N,3,C).norm(dim=2).max(1).values/sbar).max():.3f}  (bounded by 1 by construction)")
print(f"   DU  raw M row-norm max  = {M.norm(dim=2).max():.3f}  (free)")

# gradients flow
g = mk(True); g._v_12_direction = torch.nn.Parameter(v12.clone())
xx,oo = g.slice_gaussian_full_method(q); (xx.sum()+oo.sum()).backward()
print(f"5. grad wrt M finite/nonzero: {torch.isfinite(g._v_12_direction.grad).all().item()} / {(g._v_12_direction.grad.abs().sum()>0).item()}")
