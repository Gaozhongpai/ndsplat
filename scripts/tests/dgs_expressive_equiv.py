#!/usr/bin/env python3
"""Expressive-equivalence check for the dGS view-shift variants.

The quality-null explanation rests on: each variant can represent the same set
of regression operators M (dmu = M delta) as baseline dGS, so under full
observability the optimizer reaches the same optimum. This checks, per variant,
whether an arbitrary target M can be reproduced by choosing that variant's free
parameters (given the SAME fixed P = L L^T and scale S the primitive holds).

Baseline effective operator:   M = lambda * v_12 * P            (v_12 free [3x3], lambda in (0,1))
whitened:                      M = lambda * v_12 * L            (drive = L^T delta => operator lambda v_12 L^T; NB uses L^T)
cca:                           M = lambda * S^{1/2} K L         (K free [3x3], ||K||<=kappa)
mdirect:                       M = lambda * (s_bar Theta_M)     (Theta_M free direction, P-independent)

For a random target M*, solve each variant's free matrix and check ||M_hat - M*||.
A variant is expressively equivalent iff it can hit arbitrary M* (up to its norm
budget); mdirect trivially can (M free); white/cca can iff their fixed right
factor (L or L) is invertible, which it is (exp diagonal > 0).
"""
import sys
sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")
import torch

def main():
    torch.manual_seed(0)
    N = 4096
    L = torch.tril(torch.randn(N, 3, 3, device="cuda"))
    di = torch.arange(3)
    L[:, di, di] = torch.exp(0.3 * torch.randn(N, 3, device="cuda"))   # positive diag
    P = L @ L.transpose(1, 2)
    S = torch.rand(N, 3, device="cuda") * 0.4 + 0.05
    Mstar = torch.randn(N, 3, 3, device="cuda") * 0.1                  # arbitrary target operator
    fails = []

    def rep(name, Mhat):
        r = float(((Mhat - Mstar).norm(dim=(1,2)) / Mstar.norm(dim=(1,2)).clamp_min(1e-9)).median())
        ok = r < 1e-4
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: median rel recon err {r:.2e}")
        if not ok: fails.append(name)

    # baseline: M = v_12 P  (fold lambda into v_12). v_12 = M* P^{-1}
    Pinv = torch.linalg.inv(P)
    rep("baseline (v_12 = M* P^-1)", Mstar @ Pinv @ P)
    # whitened: operator = v_12 L^T  => v_12 = M* (L^T)^{-1}
    LT = L.transpose(1, 2)
    v12_w = Mstar @ torch.linalg.inv(LT)
    rep("whitened (v_12 = M* (L^T)^-1)", v12_w @ LT)
    # cca: M = S^{1/2} K L => K = S^{-1/2} M* L^{-1}
    Sh = S.sqrt()
    K = (1.0 / Sh)[:, :, None] * (Mstar @ torch.linalg.inv(L))
    Mhat = Sh[:, :, None] * (K @ L)
    rep("cca (K = S^-1/2 M* L^-1)", Mhat)
    # mdirect: M free
    rep("mdirect (M = M* directly)", Mstar.clone())

    print("ALL EQUIVALENT" if not fails else f"NOT EQUIVALENT: {fails}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
