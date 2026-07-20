#!/usr/bin/env python3
"""Smoke test: the residual Gabor band bootstraps from amp==0.

Two things must hold for the additive-residual design to be trainable:
  (1) FORWARD PARITY: with amp==0 the render is byte-identical to plain dGS
      (the residual is off), regardless of omega/phase.
  (2) AMP BOOTSTRAP: dL/d(amp) is NONZERO at amp==0 (so one optimizer step
      lifts amp off zero and the cosine turns on). This is the whole reason
      the backward must run its gradient block at amp==0, not gate on amp!=0.

We test the raw tcgs rasterizer directly with a tiny random splat set, so the
test is independent of the model/training plumbing.
"""
import sys
import torch

sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import tcgs_speedy_rasterizer as tg

torch.manual_seed(0)
dev = "cuda"
N = 2000
W = H = 128


def make_settings():
    # Minimal identity-ish camera; values only need to be self-consistent.
    fov = 1.0
    tanx = tany = torch.tan(torch.tensor(fov / 2)).item()
    view = torch.eye(4, device=dev)
    view[2, 3] = 5.0  # push scene in front of camera
    proj = view.clone()
    return tg.GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanx, tanfovy=tany,
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=view, projmatrix=view @ proj, sh_degree=0,
        campos=torch.zeros(3, device=dev), prefiltered=False, debug=False,
        # use_tcgs=False so ALL renders share the standard forward: the gabor
        # path forces it anyway, and the parity check must compare like paths.
        use_tcgs=False,
    )


def render(gabor):
    means = (torch.randn(N, 3, device=dev) * 0.5)
    torch.manual_seed(0)  # identical splats across calls
    means = (torch.randn(N, 3, device=dev) * 0.5)
    means[:, 2] = 0.0
    # Larger footprints + moderate opacity so many splat/pixel pairs clear the
    # alpha>1/255 cutoff WITHOUT saturating at 0.99 -- the regime where the
    # gabor amp gradient must be nonzero. (Tiny/opaque splats either get culled
    # below the alpha floor or saturate, both of which skip the gradient block
    # and would give a false BOOTSTRAP FAIL.)
    scales = torch.full((N, 3), 0.15, device=dev)
    rots = torch.zeros(N, 4, device=dev); rots[:, 0] = 1.0
    opac = torch.full((N, 1), 0.3, device=dev)
    colors = torch.rand(N, 3, device=dev)
    r = tg.GaussianRasterizer(make_settings())
    scores = torch.zeros(N, device=dev)  # informed-pruning scores; unused here
    out = r(means3D=means, means2D=torch.zeros_like(means, requires_grad=True),
            opacities=opac, scores=scores, shs=None, colors_precomp=colors,
            scales=scales, rotations=rots, cov3D_precomp=None, gabor=gabor)
    return out[0]  # (color, radii, timing, accum_metric_counts)


# ---- (1) forward parity at amp==0 -------------------------------------------
gab_zero = torch.zeros(N, 4, device=dev)              # omega=phase=amp=0
gab_omega = torch.zeros(N, 4, device=dev)             # omega!=0 but amp==0
gab_omega[:, 0] = 0.4
img_none = render(None)
img_zero = render(gab_zero)
img_omega = render(gab_omega)
d0 = (img_none - img_zero).abs().max().item()
d1 = (img_none - img_omega).abs().max().item()
print(f"[parity] max|dGS - gabor(amp=0,omega=0)|   = {d0:.3e}")
print(f"[parity] max|dGS - gabor(amp=0,omega=0.4)| = {d1:.3e}")
parity_ok = d0 < 1e-6 and d1 < 1e-6

# ---- (2) amp gradient is nonzero at amp==0 ----------------------------------
def amp_grad_for(amp0):
    gab = torch.zeros(N, 4, device=dev)
    gab[:, 0] = 0.4                     # omega_x
    gab[:, 1] = 0.2                     # omega_y
    gab[:, 3] = amp0                    # amp
    gab.requires_grad_(True)
    img = render(gab)
    target = torch.rand_like(img)
    loss = ((img - target) ** 2).mean()
    loss.backward()
    g = gab.grad
    return g

# Discriminator: does the amp gradient appear at amp=0 (bootstrap) vs amp=0.1
# (residual already active)? If it's zero at 0 but nonzero at 0.1, the gate is
# wrong (blocks bootstrap). If zero at BOTH, the whole gabor gradient is unwired.
g0 = amp_grad_for(0.0)
g1 = amp_grad_for(0.1)
print(f"[grad@amp=0.0] any nonzero in WHOLE gabor grad? {(g0.abs()>1e-12).sum().item()}/{N*4}  "
      f"max|d/d(amp)|={g0[:,3].abs().max().item():.3e}  max|d/d(omega)|={g0[:,:2].abs().max().item():.3e}")
print(f"[grad@amp=0.1] any nonzero in WHOLE gabor grad? {(g1.abs()>1e-12).sum().item()}/{N*4}  "
      f"max|d/d(amp)|={g1[:,3].abs().max().item():.3e}  max|d/d(omega)|={g1[:,:2].abs().max().item():.3e}")
n_nonzero = (g0[:, 3].abs() > 1e-9).sum().item()
boot_ok = n_nonzero > 0

print()
print(f"PARITY:    {'PASS' if parity_ok else 'FAIL'}")
print(f"BOOTSTRAP: {'PASS' if boot_ok else 'FAIL'}")
sys.exit(0 if (parity_ok and boot_ok) else 1)
