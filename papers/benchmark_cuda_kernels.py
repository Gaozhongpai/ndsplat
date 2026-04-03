"""Benchmark: Actual CUDA kernels — N-DGS vs dGS-O vs dGS.

Uses the real gsplat CUDA kernels from the codebase.
Calls the low-level C++ functions directly to avoid autograd overhead.
"""

import torch
import time
import sys
sys.path.insert(0, "/code/workspace/6dgs-iclr/submodules/gsplat")

from gsplat.cuda._wrapper import _make_lazy_cuda_func

torch.backends.cudnn.benchmark = True


def benchmark_fn(fn, warmup=20, repeats=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / repeats * 1000  # ms


def build_spd_covars(N, D, device):
    """Build valid full covariance matrices [N, D, D] for N-DGS."""
    L = torch.randn(N, D, D, device=device) * 0.3
    L = torch.tril(L)
    L[:, range(D), range(D)] = L[:, range(D), range(D)].abs() + 0.5
    return torch.bmm(L, L.transpose(-1, -2))


def run_benchmarks():
    device = torch.device("cuda")
    gaussian_counts = [200_000, 500_000, 1_000_000, 2_000_000, 3_000_000]

    for C in [3, 4]:
        D = 3 + C
        n_L = C * (C + 1) // 2

        print(f"\n{'='*90}")
        print(f"  C = {C} ({'6DGS' if C == 3 else '7DGS'}) — CUDA Kernel Benchmark")
        print(f"{'='*90}")
        print(f"{'N':>12} | {'N-DGS (ms)':>10} | {'dGS-O (ms)':>10} | {'dGS (ms)':>10} | {'N-DGS/dGS-O':>12} | {'N-DGS/dGS':>10}")
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

        for N in gaussian_counts:
            try:
                # === N-DGS inputs ===
                m_1 = torch.randn(N, 3, device=device).contiguous()
                m_2 = torch.randn(N, C, device=device).contiguous()
                query_ndgs = torch.randn(N, C, device=device).contiguous()
                covars = build_spd_covars(N, D, device).contiguous()
                lambda_opc_ndgs = torch.full((N,), 0.35, device=device).contiguous()
                lambda_opc_time_ndgs = torch.full((N,), 0.35, device=device).contiguous()

                # === dGS inputs ===
                xyz = torch.randn(N, 3, device=device).contiguous()
                view_mean = torch.randn(N, C, device=device).contiguous()
                query_dgs = torch.randn(N, C, device=device).contiguous()
                v_12 = (torch.randn(N, 3 * C, device=device) * 0.1).contiguous()
                L_22_inv = (torch.randn(N, n_L, device=device) * 0.5).contiguous()
                lambda_view = torch.sigmoid(torch.randn(N, device=device)).contiguous()
                lambda_time = torch.sigmoid(torch.randn(N, device=device)).contiguous() if C == 4 else None

                # N-DGS: call C++ forward directly (no autograd overhead)
                def fn_ndgs():
                    return _make_lazy_cuda_func("slice_gaussian_ndgs_fwd")(
                        m_1, m_2, query_ndgs, covars,
                        lambda_opc_ndgs, lambda_opc_time_ndgs, True,  # zero_view_time_cross_terms
                    )

                # dGS-O: call C++ forward directly (opacity only, no v_12)
                def fn_dgs_o():
                    return _make_lazy_cuda_func("slice_gaussian_full_fwd")(
                        xyz, view_mean, query_dgs,
                        None,  # v_12 = None
                        L_22_inv,
                        0.35,  # lambda_opc
                        False,  # use_beta
                        None,  # lambda_view
                        None,  # lambda_time
                        None,  # spatial_beta
                    )

                # dGS: call C++ forward directly (with position shift)
                def fn_dgs():
                    return _make_lazy_cuda_func("slice_gaussian_full_fwd")(
                        xyz, view_mean, query_dgs,
                        v_12,
                        L_22_inv,
                        0.35,  # lambda_opc
                        False,  # use_beta
                        lambda_view,
                        lambda_time,
                        None,  # spatial_beta
                    )

                ms_ndgs = benchmark_fn(fn_ndgs)
                ms_dgs_o = benchmark_fn(fn_dgs_o)
                ms_dgs = benchmark_fn(fn_dgs)

                print(f"{N:>12,} | {ms_ndgs:>10.3f} | {ms_dgs_o:>10.3f} | {ms_dgs:>10.3f} | {ms_ndgs/ms_dgs_o:>11.2f}x | {ms_ndgs/ms_dgs:>9.2f}x")

            except (torch.cuda.OutOfMemoryError, Exception) as e:
                print(f"{N:>12,} | Error: {e}")
                break

            del m_1, m_2, query_ndgs, covars, lambda_opc_ndgs, lambda_opc_time_ndgs
            del xyz, view_mean, query_dgs, v_12, L_22_inv, lambda_view
            if lambda_time is not None:
                del lambda_time
            torch.cuda.empty_cache()

    print(f"\n{'='*90}")
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"{'='*90}")


if __name__ == "__main__":
    run_benchmarks()
