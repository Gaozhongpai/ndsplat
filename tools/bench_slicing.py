"""Benchmark the per-primitive slicing CUDA kernels for the 5 methods used in
Table~\\ref{tab:comparison_overview}.

Reports forward-only slicing time (no autograd, no rasterization) at N
Gaussians, separately for C=3 (6DGS) and C=4 (7DGS).

Usage:
    python tools/bench_slicing.py [--n 1000000] [--trials 100] [--warmup 20]

The script synthesizes random valid inputs for each method and times only the
slicing call. Speedup is reported relative to N-DGS for each C.
"""
import argparse
import math
import sys
import time

import torch

from gsplat import (
    cond_mean_convariance_opacity,
    slice_gaussian_ndgs,
    slice_gaussian_full,
    slice_dbs,
)


def randn(*shape, device="cuda", dtype=torch.float32):
    return torch.randn(*shape, device=device, dtype=dtype)


def make_spd_covar(N, D, device="cuda"):
    """Random symmetric positive-definite covariance, shape [N, D, D]."""
    A = torch.randn(N, D, D, device=device) * 0.1
    cov = torch.bmm(A, A.transpose(-1, -2)) + torch.eye(D, device=device).unsqueeze(0) * 0.5
    return cov


def time_call(fn, trials, warmup):
    """Time fn() forward only. Returns median ms over `trials` after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def bench_for_C(C, N, trials, warmup, device="cuda"):
    """Benchmark all 5 methods at conditioning dim C and N Gaussians."""
    D = 3 + C
    n_L = C * (C + 1) // 2

    # Shared inputs
    xyz = randn(N, 3, device=device)
    mean_full = randn(N, D, device=device)    # [N, D] — full (xyz, cond_mean) for UBS
    cond_mean = mean_full[:, 3:].contiguous() # [N, C] for NDGS/dGS/dBS
    query = randn(N, C, device=device)
    opacity = torch.sigmoid(randn(N, 1, device=device))
    betas_C = torch.rand(N, C, device=device) * 4.0       # for dBS slicing and UBS

    # NDGS / UBS use full (3+C)x(3+C) covariance
    covar_full = make_spd_covar(N, D, device=device)

    # dGS / dBS direct parameterization
    L_22_inv = randn(N, n_L, device=device) * 0.1
    # diagonal log-precision so exp() in kernel gives ~1
    for i in range(C):
        diag = i * (i + 1) // 2 + i
        L_22_inv[:, diag] = 0.0
    v_12 = randn(N, 3 * C, device=device) * 0.01

    # Per-primitive lambdas for dGS (1 = full coupling; 0 = opacity-only)
    lambda_view_dgs = torch.ones(N, device=device)
    lambda_time_dgs = torch.ones(N, device=device) if C == 4 else None
    lambda_view_zero = torch.zeros(N, device=device)
    lambda_time_zero = torch.zeros(N, device=device) if C == 4 else None

    methods = {}

    def f_ndgs():
        return slice_gaussian_ndgs(xyz, cond_mean, query, covar_full, 0.35)
    methods["N-DGS"] = f_ndgs

    def f_ubs():
        return cond_mean_convariance_opacity(mean_full, covar_full, opacity, betas_C, query)
    methods["UBS"] = f_ubs

    # dGS-O: opacity-only — pass v_12=None (skip position shift)
    def f_dgso():
        return slice_gaussian_full(xyz, cond_mean, query, None, L_22_inv,
                                   0.35, lambda_view_zero, lambda_time_zero)
    methods["dGS-O"] = f_dgso

    # dGS: full position+opacity, with per-primitive lambdas (=1 here for full coupling)
    def f_dgs():
        return slice_gaussian_full(xyz, cond_mean, query, v_12, L_22_inv,
                                   0.35, lambda_view_dgs, lambda_time_dgs)
    methods["dGS"] = f_dgs

    def f_dbs():
        return slice_dbs(xyz, cond_mean, query, v_12, L_22_inv, betas_C)
    methods["dBS"] = f_dbs

    print(f"\n=== C={C}, N={N:,} (D={D}) ===")
    results = {}
    for name, fn in methods.items():
        try:
            ms = time_call(fn, trials=trials, warmup=warmup)
            results[name] = ms
            print(f"  {name:<8s} {ms:7.3f} ms")
        except Exception as e:
            print(f"  {name:<8s} ERROR: {type(e).__name__}: {e}")
            results[name] = None

    base = results.get("N-DGS")
    if base is None:
        return results
    print(f"  -- speedups vs N-DGS --")
    for name in ("UBS", "dGS-O", "dGS", "dBS"):
        v = results.get(name)
        if v is None:
            continue
        print(f"  {name:<8s} {base / v:5.2f}x")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable.")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"N={args.n:,}  trials={args.trials}  warmup={args.warmup}")

    for C in (3, 4):
        bench_for_C(C, args.n, args.trials, args.warmup)


if __name__ == "__main__":
    main()
