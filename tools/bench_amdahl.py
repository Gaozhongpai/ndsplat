"""Decompose end-to-end frame time into slicing vs. everything-else.

Answers Reviewer HuLf's W2 (NeurIPS 2026 Submission 8371): why is end-to-end
rendering only 2.66x faster when the slicing kernel alone is 6.9-7.7x faster?

The claim in the rebuttal is Amdahl's law: slicing is one stage of the frame,
and the remainder (projection, tile binning, depth sort, alpha-composited
rasterization, SH evaluation) is identical across methods and untouched by our
change. This script measures that split on TRAINED models rather than asserting
it, so the predicted end-to-end speedup can be checked against the observed one.

For each model it times, per frame, over real test cameras:
    t_slice  -- the conditional slicing call alone
    t_total  -- the full render call
    t_rest   -- t_total - t_slice

and then reports, for a (baseline, direct) pair:
    observed  end-to-end speedup = t_total_base / t_total_direct
    predicted end-to-end speedup = 1 / ((1-f) + f/s)
        where f = t_slice_base / t_total_base   (slicing fraction of baseline)
              s = t_slice_base / t_slice_direct (slicing-only speedup)

If predicted ~= observed, Amdahl's law explains the gap and the answer to W2 is
quantitative rather than hand-waved.

Usage (inside the container, cwd /code):
    python tools/bench_amdahl.py \
        --base   output/mcmc/ndgs/nerf_synthetic/lego --base_mode ndgs \
        --direct output/mcmc/dgs/nerf_synthetic/lego  --direct_mode dgs \
        --source /code/dataset/nerf_synthetic/lego -w

    # Mip-NeRF 360 (slicing-dominated, expect the largest end-to-end gain)
    python tools/bench_amdahl.py \
        --base   output/mcmc/ndgs/360_v2/bicycle --base_mode ndgs \
        --direct output/mcmc/dgs/360_v2/bicycle  --direct_mode dgs \
        --source /code/dataset/360_v2/bicycle
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams
from scene import Scene, get_gaussian_model
from utils.general_utils import safe_state


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, trials, warmup):
    """Median ms over `trials` after `warmup`, with explicit syncs."""
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(trials):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def cond_params_for(gaussians, cam, mode):
    """Build the query tensor the slicing call expects, as render_tcgs does."""
    N = gaussians.get_xyz.shape[0]
    campos = cam.camera_center
    dirs = gaussians.get_xyz - campos.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if getattr(gaussians, "input_dim", 6) == 7:
        t = getattr(cam, "timestamp", None)
        if t is None:
            t = 0.0
        tcol = torch.full((N, 1), float(t), device=dirs.device, dtype=dirs.dtype)
        return torch.cat([dirs, tcol], dim=-1)
    return dirs


def slice_call(gaussians, cam, mode):
    """Just the conditional slicing step, no rasterization."""
    q = cond_params_for(gaussians, cam, mode)
    if "ndgs" in mode:
        # Same arguments render_tcgs passes, so this times the identical call.
        lo = gaussians.get_lambda_opc.squeeze(-1)
        lot = (gaussians.get_lambda_opc_time.squeeze(-1)
               if getattr(gaussians, "input_dim", 6) == 7 else None)
        return lambda: gaussians.slice_gaussian(q, c_dim=3, lambda_opc=lo, lambda_opc_time=lot)
    if "dgs" in mode:
        return lambda: gaussians.slice_gaussian_full_method(q)
    raise SystemExit(f"slice_call: unsupported mode {mode!r} (expected ndgs/dgs)")


def load(model_path, mode, source, white_bg, sh_degree, input_dim, iteration):
    """Load a trained model + its cameras."""
    parser = argparse.ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    args.model_path = model_path
    args.source_path = source
    args.mode = mode
    args.sh_degree = sh_degree
    args.input_dim = input_dim
    args.white_background = white_bg
    args.eval = True
    args.data_device = "cuda"
    args.resolution = -1
    args.images = "images"

    dataset = lp.extract(args)
    GaussianModel = get_gaussian_model(mode)
    gaussians = GaussianModel(sh_degree, input_dim)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return gaussians, scene, pp.extract(args)


def measure(tag, model_path, mode, source, white_bg, sh, dim, iters, n_views, trials, warmup):
    gaussians, scene, pipe = load(model_path, mode, source, white_bg, sh, dim, iters)
    cams = scene.getTestCameras() or scene.getTrainCameras()
    cams = cams[:n_views]
    bg = torch.tensor([1, 1, 1] if white_bg else [0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians.background = bg

    n_gauss = gaussians.get_xyz.shape[0]
    tot = sli = 0.0
    for cam in cams:
        tot += time_fn(lambda c=cam: gaussians.render_tcgs(c, use_tcgs=True), trials, warmup)
        sli += time_fn(slice_call(gaussians, cam, mode), trials, warmup)
    tot /= len(cams)
    sli /= len(cams)

    print(f"  {tag:<10} N={n_gauss:>9,}  total={tot:7.3f} ms  slice={sli:7.3f} ms  "
          f"rest={tot - sli:7.3f} ms  slice_frac={sli / tot:6.1%}  ({1000/tot:6.1f} FPS)")
    return dict(tag=tag, n=n_gauss, total=tot, slice=sli, rest=tot - sli)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base_mode", default="ndgs")
    ap.add_argument("--direct", required=True)
    ap.add_argument("--direct_mode", default="dgs")
    ap.add_argument("--source", required=True)
    ap.add_argument("-w", "--white_background", action="store_true")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--input_dim", type=int, default=6)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    # safe_state(silent=True) replaces sys.stdout with a swallowing wrapper,
    # so seed the RNG the same way but keep stdout for our own reporting.
    safe_state(False, 0)
    sys.stdout = sys.__stdout__
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"views={a.views}  trials={a.trials}  warmup={a.warmup}\n")

    b = measure("baseline", a.base, a.base_mode, a.source, a.white_background,
                a.sh_degree, a.input_dim, a.iteration, a.views, a.trials, a.warmup)
    d = measure("direct", a.direct, a.direct_mode, a.source, a.white_background,
                a.sh_degree, a.input_dim, a.iteration, a.views, a.trials, a.warmup)

    f = b["slice"] / b["total"]          # slicing fraction of the baseline frame
    s = b["slice"] / d["slice"]          # slicing-only speedup
    predicted = 1.0 / ((1.0 - f) + f / s)
    observed = b["total"] / d["total"]

    print("\n  --- Amdahl decomposition ---")
    print(f"  slicing fraction of baseline frame  f = {f:.3f}")
    print(f"  slicing-only speedup                s = {s:.2f}x")
    print(f"  predicted end-to-end  1/((1-f)+f/s)   = {predicted:.2f}x")
    print(f"  observed  end-to-end                  = {observed:.2f}x")
    print(f"  agreement                             = {observed/predicted:.3f} "
          f"({'consistent' if 0.85 <= observed/predicted <= 1.15 else 'DISCREPANT -- investigate'})")
    rest_ratio = b["rest"] / d["rest"]
    print(f"\n  rasterization+rest: baseline {b['rest']:.3f} ms vs direct {d['rest']:.3f} ms "
          f"(ratio {rest_ratio:.2f}x)")
    if not (0.9 <= rest_ratio <= 1.1):
        print("  NOTE: the non-slicing remainder is NOT equal across methods, so the simple")
        print("  Amdahl prediction above does not apply. The two models are trained")
        print("  separately: at an equal Gaussian budget they still learn different scale")
        print("  and opacity distributions, and larger/more-opaque primitives cost the")
        print("  rasterizer more per frame. Report the measured split, not the prediction.")
        adj = (b["slice"] + b["rest"]) / (d["slice"] + b["rest"])
        print(f"  Slicing-only effect, holding the remainder at the baseline's "
              f"{b['rest']:.3f} ms: {adj:.2f}x")


if __name__ == "__main__":
    main()
