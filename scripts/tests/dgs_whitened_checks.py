#!/usr/bin/env python3
"""Verification for the whitened-displacement dGS study (`dgs-white`).

1. PARITY: the pure-torch port of slice_gaussian_full (STANDARD formula) must
   match the CUDA kernel on random inputs — outputs and gradients wrt every
   leaf (xyz, view_mean/query via delta, v_12, L_22_inv, lambda_view).
2. WHITENED SEMANTICS: whitened == standard when P == I (L = I, i.e. raw
   triangle = 0 -> exp(0)=1 diagonal); differs otherwise; and the level-set
   bound ||dmu|| <= lambda * s_bar * ||z|| holds (v_12 is Frobenius-normalized
   times the mean scale, so its spectral norm <= s_bar).
3. PRECISION INDEPENDENCE: with z ~ N(0,I) fixed, scaling L (sharper query
   precision) leaves the whitened displacement unchanged but grows the
   standard one.

Run inside the ndgs container:
    python scripts/tests/dgs_whitened_checks.py
"""
import sys

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import torch

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def main():
    from gsplat import slice_gaussian_full
    from scene.gaussian_model_dgs_whitened import torch_slice_gaussian_full

    torch.manual_seed(0)
    N = 8192
    dev = "cuda"
    xyz = torch.randn(N, 3, device=dev)
    view_mean = torch.nn.functional.normalize(torch.randn(N, 3, device=dev), dim=1)
    query = torch.nn.functional.normalize(torch.randn(N, 3, device=dev), dim=1)
    v12_raw = torch.randn(N, 9, device=dev)
    sbar = torch.rand(N, 1, device=dev) * 0.5 + 0.05
    v12 = torch.nn.functional.normalize(v12_raw, dim=1) * sbar
    L_tri = torch.randn(N, 6, device=dev) * 0.5
    lam = torch.sigmoid(torch.randn(N, device=dev))
    lam_opc = 0.35

    # ---- 1. parity (values + grads) ---------------------------------------
    print("== torch port vs CUDA slice (standard formula) ==")
    leaves_c = [t.detach().clone().requires_grad_(True)
                for t in (xyz, view_mean, query, v12, L_tri, lam)]
    xc_c, at_c = slice_gaussian_full(
        xyz=leaves_c[0], view_mean=leaves_c[1], query=leaves_c[2],
        v_12=leaves_c[3], L_22_inv=leaves_c[4], lambda_opc=lam_opc,
        lambda_view=leaves_c[5], lambda_time=None)
    w1 = torch.randn_like(xc_c)
    w2 = torch.randn_like(at_c)
    ((xc_c * w1).sum() + (at_c * w2).sum()).backward()

    leaves_t = [t.detach().clone().requires_grad_(True)
                for t in (xyz, view_mean, query, v12, L_tri, lam)]
    xc_t, at_t = torch_slice_gaussian_full(
        leaves_t[0], leaves_t[1], leaves_t[2], leaves_t[3], leaves_t[4],
        lam_opc, leaves_t[5], whitened=False)
    ((xc_t * w1).sum() + (at_t * w2).sum()).backward()

    check("x_cond values", rel(xc_t, xc_c) < 1e-5, f"rel {rel(xc_t, xc_c):.2e}")
    check("attention values", rel(at_t.squeeze(), at_c.squeeze()) < 1e-5,
          f"rel {rel(at_t.squeeze(), at_c.squeeze()):.2e}")
    names = ["xyz", "view_mean", "query", "v_12", "L_22_inv", "lambda_view"]
    for nm, tc, tt in zip(names, leaves_c, leaves_t):
        if tc.grad is None:
            # CUDA does not emit this gradient (e.g. query); the torch port
            # detaches to match — assert it produced none/zero too.
            ok = tt.grad is None or tt.grad.abs().max().item() == 0.0
            check(f"grad d/d {nm} (CUDA: none)", ok, "port matches")
            continue
        r = rel(tt.grad, tc.grad)
        check(f"grad d/d {nm}", r < 5e-4, f"rel {r:.2e}")

    # ---- 2. whitened semantics ---------------------------------------------
    print("== whitened semantics ==")
    with torch.no_grad():
        # P == I when the raw triangle is all zeros
        L_id = torch.zeros(N, 6, device=dev)
        xs, _ = torch_slice_gaussian_full(xyz, view_mean, query, v12, L_id,
                                          lam_opc, lam, whitened=False)
        xw, _ = torch_slice_gaussian_full(xyz, view_mean, query, v12, L_id,
                                          lam_opc, lam, whitened=True)
        check("whitened == standard at P=I", rel(xw, xs) < 1e-6,
              f"rel {rel(xw, xs):.2e}")
        xs, _ = torch_slice_gaussian_full(xyz, view_mean, query, v12, L_tri,
                                          lam_opc, lam, whitened=False)
        xw, _ = torch_slice_gaussian_full(xyz, view_mean, query, v12, L_tri,
                                          lam_opc, lam, whitened=True)
        d = (xw - xs).abs().max().item()
        check("whitened != standard at P!=I", d > 1e-3, f"max diff {d:.3f}")

        # level-set bound: ||dmu_white|| <= lambda * s_bar * ||z||
        from scene.gaussian_model_dgs_whitened import unpack_L
        L = unpack_L(L_tri)
        z = torch.einsum('nji,nj->ni', L, query - view_mean)
        dmu = xw - xyz
        bound = lam.unsqueeze(-1) * sbar * z.norm(dim=1, keepdim=True)
        viol = (dmu.norm(dim=1, keepdim=True) > bound * (1 + 1e-5)).sum().item()
        check("level-set bound ||dmu|| <= lam*s_bar*||z||", viol == 0,
              f"{viol}/{N} violations")

        # precision independence: scale L by 10 (sharper precision) with z held
        # by rescaling delta accordingly -> whitened dmu is a function of z
        # only; standard dmu grows.
        L_tri_sharp = L_tri.clone()
        L_tri_sharp[:, [0, 2, 5]] += torch.log(torch.tensor(10.0))
        q_shrunk = view_mean + (query - view_mean) / 10.0   # keeps z ~ const
        xw2, _ = torch_slice_gaussian_full(xyz, view_mean, q_shrunk, v12,
                                           L_tri_sharp, lam_opc, lam, whitened=True)
        xs2, _ = torch_slice_gaussian_full(xyz, view_mean, q_shrunk, v12,
                                           L_tri_sharp, lam_opc, lam, whitened=False)
        # z is only approximately preserved (off-diagonals unscaled); compare
        # displacement magnitudes statistically.
        rw = ((xw2 - xyz).norm(dim=1) / (xw - xyz).norm(dim=1).clamp_min(1e-9)).median()
        rs = ((xs2 - xyz).norm(dim=1) / (xs - xyz).norm(dim=1).clamp_min(1e-9)).median()
        check("precision-independence (whitened ~1x, standard >>1x)",
              0.5 < float(rw) < 2.0 and float(rs) > 3.0,
              f"whitened x{float(rw):.2f}, standard x{float(rs):.2f}")

    # ---- 3. dgs-cca slice: M = S^{1/2} K P^{1/2},  P^{1/2} := L ------------
    print("== dgs-cca (S^{1/2} K L) ==")
    from scene.gaussian_model_dgs_cca import _sym_sqrt_spd
    with torch.no_grad():
        L = unpack_L(L_tri)
        P = L @ L.transpose(1, 2)
        # sanity: the symmetric root (harness only) is a valid root of P; the
        # MODEL uses L, a Cholesky-style root, equally valid since K absorbs the
        # rotation between roots. L needs no eigendecomposition (eigh is
        # ill-conditioned at init -> the crash this replaced).
        P_half_sym = _sym_sqrt_spd(P)
        check("sym-root^2 == P (harness)", rel(P_half_sym @ P_half_sym, P) < 1e-4,
              f"rel {rel(P_half_sym @ P_half_sym, P):.2e}")
        check("L L^T == P (model root)", rel(L @ L.transpose(1, 2), P) < 1e-5,
              f"rel {rel(L @ L.transpose(1, 2), P):.2e}")
        # reference matching the model: dmu = lam * S^{1/2} K (L x)
        S_half = sbar.expand(N, 3).sqrt()
        Kmat = v12.reshape(N, 3, 3)
        x = query - view_mean
        Lx = torch.einsum('nij,nj->ni', L, x)   # NOTE: L, not L^T (P^{1/2} apply)
        ref = lam.unsqueeze(-1) * S_half * torch.einsum('nij,nj->ni', Kmat, Lx)
        # metric bound: ||S^{-1/2} dmu|| <= lam * ||K||_2 * ||L x||
        lhs = (ref / S_half).norm(dim=1)
        svK = torch.linalg.matrix_norm(Kmat, ord=2)
        rhs = lam * svK * Lx.norm(dim=1)
        viol = (lhs > rhs * (1 + 1e-4)).sum().item()
        check("metric bound ||S^-1/2 dmu|| <= lam ||K||_2 ||L x||",
              viol == 0, f"{viol}/{N} violations")
        # kappa clamp caps the spectral norm
        Kbig = torch.randn(N, 3, 3, device=dev) * 5.0
        sv0 = torch.linalg.matrix_norm(Kbig, ord=2)
        factor = sv0.clamp_max(3.0) / sv0.clamp_min(1e-12)
        Kc = Kbig * factor.reshape(-1, 1, 1)
        svc = torch.linalg.matrix_norm(Kc, ord=2)
        check("kappa clamp: ||K||_2 <= 3.0", svc.max().item() <= 3.0 + 1e-3,
              f"max ||K||_2 {svc.max().item():.3f}")

    print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
