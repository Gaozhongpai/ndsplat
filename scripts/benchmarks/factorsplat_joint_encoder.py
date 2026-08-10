#!/usr/bin/env python3
"""Joint two-scene training with ONE shared TF encoder.

phi is shared across scenes and produces local z_{i,T} codes; geometry, base appearance,
per-Gaussian factors, lookup descriptors, and lookup gains stay scene-specific.
One batch per scene per step, so each scene keeps the same 30k-update budget as
its independent-encoder counterpart. The encoder is LOCAL and label-order invariant:
phi maps one raw-RGBA delta sample to a code, evaluated once over the
(L*B, 4) delta
table per preset, and each primitive's packed (label, bin) samples gather and
average those codes -> z_{i,T}. Numeric label ids never enter the network and
label counts do not affect phi's width (heart 12 curves, vascular 15).
"""
import os, sys, uuid, numpy as np, torch
from argparse import ArgumentParser, Namespace
from random import randint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, get_gaussian_model
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim
from utils.image_utils import psnr
from tqdm import tqdm


def build(dataset, opt, source, model_path):
    GaussianModel = get_gaussian_model("factorsplat")
    g = GaussianModel(
        dataset.sh_degree, input_dim=dataset.input_dim,
        use_view_dependent_pos=False, l_22_inv_init_scale=dataset.l_22_inv_init_scale,
        tf_rank=dataset.tf_rank, tf_hidden=dataset.tf_hidden,
        tf_samples=dataset.tf_samples, tf_color_scale=dataset.tf_color_scale,
        tf_opacity_scale=dataset.tf_opacity_scale,
        tf_condition_color=True, tf_condition_opacity=True,
        tf_color_sh_degree=0,                      # Hybrid (DC)
        tf_aware_prune=True, tf_use_lookup=True,
        tf_encoder_local=True,
        tf_lookup_bins=dataset.tf_lookup_bins,
    )
    d = Namespace(**vars(dataset)); d.source_path = source; d.model_path = model_path
    os.makedirs(model_path, exist_ok=True)
    scene = Scene(d, g, opt_params=opt)
    g.load_ply(os.path.join(source, "points3d.ply"))
    g.load_lookup_descriptors(os.path.join(source, "points3d_lookup.npz"))
    grid = os.path.join(source, "points3d_refresh_grid.npz")
    if os.path.exists(grid):
        g.load_refresh_grid(grid)
    with open(os.path.join(model_path, "cfg_args"), "w") as f:
        f.write(str(d))
    g.training_setup(opt)
    g.background = torch.tensor([0., 0., 0.], device="cuda")
    return g, scene


if __name__ == "__main__":
    parser = ArgumentParser()
    mp = ModelParams(parser); pp = PipelineParams(parser); op = OptimizationParams(parser)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--preset", default="full")
    args = parser.parse_args()
    dataset = mp.extract(args); opt = op.extract(args)

    models, scenes = {}, {}
    for name in args.scenes:
        src = os.path.join(args.data_root, f"{name}_factorsplat_{args.preset}")
        out = os.path.join(args.out_root, "shared_encoder", f"rank{dataset.tf_rank}",
                           name, args.preset)
        models[name], scenes[name] = build(dataset, opt, src, out)

    # SHARE the encoder: one module, one optimizer group (kept in scene 0's
    # optimizer); every other scene points at the same parameters.
    first = args.scenes[0]
    shared = models[first].tf_encoder          # phi only; no psi in local mode
    for name in args.scenes[1:]:
        models[name].tf_encoder = shared
        models[name].optimizer.param_groups = [
            gp for gp in models[name].optimizer.param_groups if gp["name"] != "tf_encoder"]
    print(f"Shared local phi: {sum(p.numel() for p in shared.parameters())} params "
          f"across {args.scenes}; per-scene factors kept separate", flush=True)

    cams = {n: scenes[n].getTrainCameras().copy() for n in args.scenes}
    stacks = {n: [] for n in args.scenes}
    bar = tqdm(range(1, opt.iterations + 1), desc="Joint training")
    for it in bar:
        losses = {}
        for name in args.scenes:                    # one batch per scene per step
            g = models[name]
            g.update_learning_rate(it)
            if it % 1000 == 0:
                g.oneupSHdegree()
            if not stacks[name]:
                stacks[name] = cams[name].copy()
            cam = stacks[name].pop(randint(0, len(stacks[name]) - 1))
            out = g.render_tcgs(cam, render_mode="RGB", use_tcgs=False)
            img, gt = out["render"], cam.original_image.cuda()[:3]
            loss = 0.8 * l1_loss(img, gt) + 0.2 * (1.0 - fused_ssim(img[None], gt[None]))
            (loss / len(args.scenes)).backward()    # averaged objective
            losses[name] = float(loss)
            with torch.no_grad():
                if it < opt.densify_until_iter:
                    g.max_radii2D[out["visibility_filter"]] = torch.max(
                        g.max_radii2D[out["visibility_filter"]],
                        out["radii"][out["visibility_filter"]])
                    g.add_densification_stats(out["viewspace_points"], out["visibility_filter"])
                    if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                        size_threshold = 20 if it > opt.opacity_reset_interval else None
                        g.densify_and_prune(opt.densify_grad_threshold, 0.01,
                                            scenes[name].cameras_extent,
                                            size_threshold, it)
                        if hasattr(g, "refresh_descriptors"):
                            g.refresh_descriptors()
                    if it % opt.opacity_reset_interval == 0:
                        g.reset_opacity()
        for name in args.scenes:                    # step AFTER both backwards
            models[name].optimizer.step()
            models[name].optimizer.zero_grad(set_to_none=True)
        if it % 200 == 0:
            bar.set_postfix({n: f"{v:.4f}" for n, v in losses.items()})
        if it in (opt.iterations,):
            for name in args.scenes:
                g = models[name]
                d = os.path.join(args.out_root, "shared_encoder",
                                 f"rank{dataset.tf_rank}", name, args.preset)
                g.save_ply(os.path.join(d, "point_cloud", f"iteration_{it}",
                                        "point_cloud.ply"))
                print(f"saved {name} -> {d}", flush=True)
    print("JOINT TRAINING DONE", flush=True)
