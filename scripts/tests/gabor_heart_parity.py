#!/usr/bin/env python3
"""Sanity gate: dgs-gabor with a zero-amplitude residual renders the heart dGS
checkpoint numerically identically to plain dGS through the same tcgs path.

Loads the SAME checkpoint into both model classes, renders the same test view
with use_tcgs=False on both sides (the wrapper forces the standard forward
whenever a gabor buffer is bound, so both must go through it), and compares
bit-for-bit. Also checks the use_gabor=False null-buffer escape hatch.

Run inside the ndgs container (heart data mounted at /data):
    python scripts/tests/gabor_heart_parity.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/ndsplat")
sys.path.insert(0, "/workspace/ndsplat/submodules/tcgs_speedy_rasterizer")
sys.path.insert(0, "/workspace/ndsplat/submodules/gsplat")

import torch

DATA = "/data/nerf_dataset/heart_900"
PLY = "/data/output/xclipgs/ours/heart_900/point_cloud/iteration_30000/point_cloud.ply"
# Flags the heart checkpoint was trained with (its cfg_args).
MODEL_KW = dict(input_dim=6, use_view_dependent_pos=False,
                use_opacity_pos_decouple=False, l_22_inv_init_scale=2.0,
                lambda_init=-1.2, lambda_opc=0.35)
SH_DEGREE = 3


def load_camera(idx=0):
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    args = SimpleNamespace(resolution=-1, data_device="cuda",
                           white_background=False, use_jpeg_compression=False)
    cam_infos = readCamerasFromTransforms(DATA, "transforms_test.json", False)
    return loadCam(args, idx, cam_infos[idx], 1.0), len(cam_infos)


def make_model(mode):
    from scene import get_gaussian_model
    cls = get_gaussian_model(mode)
    m = cls(SH_DEGREE, **MODEL_KW)
    m.load_ply(PLY)
    m.active_sh_degree = m.max_sh_degree
    m.background = torch.zeros(3, dtype=torch.float32, device="cuda")
    return m


def main():
    cam, n_cams = load_camera(0)
    print(f"[cam] test view 0 of {n_cams}: {cam.image_name} "
          f"{cam.image_width}x{cam.image_height} clip={getattr(cam, 'clip_plane', None)}")

    dgs = make_model("dgs")
    with torch.no_grad():
        img_dgs = dgs.render_tcgs(cam, use_tcgs=False)["render"]
    n_dgs = dgs.get_xyz.shape[0]
    del dgs
    torch.cuda.empty_cache()

    gab = make_model("dgs-gabor")
    amp_max = gab.get_gabor_amp.abs().max().item()
    om = gab.get_gabor_omega
    print(f"[gabor-init] N={gab.get_xyz.shape[0]} (dgs N={n_dgs})  "
          f"max|amp|={amp_max:.1e}  omega range=[{om.min().item():.3f},{om.max().item():.3f}]")
    assert amp_max == 0.0, "checkpoint has no gabor columns; amp must load as 0"
    with torch.no_grad():
        img_gab = gab.render_tcgs(cam, use_tcgs=False)["render"]     # buffer bound, amp=0
        gab.use_gabor = False
        img_off = gab.render_tcgs(cam, use_tcgs=False)["render"]     # null buffer path
        gab.use_gabor = True

    d_buf = (img_dgs - img_gab).abs().max().item()
    d_off = (img_dgs - img_off).abs().max().item()
    same_buf = torch.equal(img_dgs, img_gab)
    same_off = torch.equal(img_dgs, img_off)
    rng = (img_dgs.min().item(), img_dgs.max().item(), img_dgs.mean().item())
    print(f"[render] dGS image min/max/mean = {rng[0]:.4f}/{rng[1]:.4f}/{rng[2]:.4f}")
    print(f"[parity] dgs vs dgs-gabor(amp=0, buffer bound): max|diff|={d_buf:.3e} "
          f"bitwise_equal={same_buf}")
    print(f"[parity] dgs vs dgs-gabor(use_gabor=False):     max|diff|={d_off:.3e} "
          f"bitwise_equal={same_off}")

    out = os.environ.get("PARITY_DUMP", "")
    if out:
        import torchvision
        os.makedirs(out, exist_ok=True)
        torchvision.utils.save_image(img_dgs.clamp(0, 1), os.path.join(out, "parity_dgs.png"))
        torchvision.utils.save_image(img_gab.clamp(0, 1), os.path.join(out, "parity_gabor.png"))

    ok = same_buf and same_off and rng[1] > 0.01
    print("PARITY GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
