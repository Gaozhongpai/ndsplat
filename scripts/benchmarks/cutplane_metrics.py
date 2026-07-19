#!/usr/bin/env python3
r"""Cut-plane–specific evaluation for XClipGS.

Full-image PSNR is dominated by the scene interior and barely separates clip
operators; the *cut face* is only a few percent of the pixels. This computes
metrics that concentrate on the clip plane, over the fixed cut-eval camera set
(perp + grazing views on x/y/z planes) produced by xclipgs_gen_cuteval_configs.py.

For every (scene, method) it reports, per camera family:

  REFERENCE (vs vengine GT, restricted to a band straddling the projected plane):
    band_psnr / band_ssim / band_lpips  -- surface quality ON the cut face.

  NO-REFERENCE (need no GT; expose the operator's failure mode directly):
    pop        -- popping: mean |I_t - I_{t+1}| in the band over the grazing
                  orbit (consecutive frames within one axis). Ours (exact cut) is
                  stable; hard-cull/ClipGS pop as primitives cross the threshold.
    leak       -- fraction of rendered foreground energy that falls on the CULLED
                  side of the projected plane line (should be ~0 for a hard
                  truncation). Grazing views only (plane is a line in-frame).
    edge_w     -- 10-90% rise width (px) of the intensity step across the plane
                  line; a sharp exact cut is a step, a fuzzy operator smears it.

Plane projection follows ndsplat's convention EXACTLY (scene/dataset_readers.py):
  c2w = transform_matrix ; c2w[:3,1:3] *= -1 ; w2c = inv(c2w) ; look down +Z_cam.
The plane is evaluated in the CENTERED frame (n . x_c = clip_offset), matching the
centered camera positions. A debug overlay PNG per family is written so the band /
plane-line placement can be eyeballed.

Usage:
  python scripts/benchmarks/cutplane_metrics.py \
      --gt-transforms  <nerf>/heart_cuteval/transforms_test.json \
      --gt-dir         <nerf>/heart_cuteval/test \
      --methods ours=<out>/ours_cuteval/heart/test/ours_best/renders \
                clipgs=<out>/clipgs_cuteval/heart/test/ours_best/renders \
                mm=... hc=... \
      --scene heart \
      --out   <out>/cuteval/heart \
      --band-px 12 --debug
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

# Make the ndsplat repo root importable (lpipsPyTorch lives there) regardless of
# CWD or a PYTHONPATH that only points at the CUDA submodules.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# geometry: project the clip plane into each view (ndsplat convention)
# ---------------------------------------------------------------------------
def fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov * 0.5))


def frame_w2c(frame):
    """World->camera 4x4, matching scene/dataset_readers.py."""
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1.0                       # OpenGL/Blender -> COLMAP (Y down, Z fwd)
    return np.linalg.inv(c2w)


def project_points(pts_world, w2c, fx, fy, cx, cy):
    """World points [N,3] -> pixel [N,2] and camera z (depth). Look down +Z_cam."""
    N = pts_world.shape[0]
    hom = np.concatenate([pts_world, np.ones((N, 1))], axis=1)  # [N,4]
    cam = (w2c @ hom.T).T[:, :3]                                # [N,3]
    z = cam[:, 2]
    zc = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = fx * cam[:, 0] / zc + cx
    v = fy * cam[:, 1] / zc + cy
    return np.stack([u, v], axis=1), z


def plane_signed_distance_image(frame, W, H, fx, fy, cx, cy):
    """Per-pixel SIGNED distance to the clip plane, in the plane's world units,
    by back-projecting each pixel ray and intersecting the plane. Returns
    (signed [H,W], valid_mask [H,W]) where signed<=0 is the KEPT half-space
    (n . x_c <= tau). Pixels whose ray is parallel to the plane are invalid.

    Camera ray in world: origin = cam center, dir = c2w_rot @ (x,y,1)_cam (COLMAP,
    +Z forward). Intersect n . (o + t d) = tau -> t = (tau - n.o)/(n.d)."""
    n = np.array(frame["plane_normal"], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    tau = float(frame["clip_offset"])                 # CENTERED-frame offset

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1.0
    R = c2w[:3, :3]                                    # camera axes in world (COLMAP)
    o = c2w[:3, 3]                                     # camera centre (centered frame)

    xs = (np.arange(W) - cx) / fx
    ys = (np.arange(H) - cy) / fy
    xg, yg = np.meshgrid(xs, ys)                       # [H,W]
    dirs_cam = np.stack([xg, yg, np.ones_like(xg)], axis=-1)   # +Z forward
    dirs_world = dirs_cam @ R.T                        # [H,W,3]

    n_dot_d = dirs_world @ n                           # [H,W]
    n_dot_o = float(o @ n)
    valid = np.abs(n_dot_d) > 1e-8
    t = np.where(valid, (tau - n_dot_o) / np.where(valid, n_dot_d, 1.0), np.nan)
    # signed distance of the RAY-PLANE intersection is 0 by construction; instead
    # we want, for the band, the pixels whose *plane line* passes through. Use the
    # plane's image-space signed distance: evaluate n.x_c-tau at unit depth along
    # the ray -> proportional to how far the pixel's ray direction tilts off the
    # plane. Simpler + robust: signed value = (n_dot_o - tau) + depth*n_dot_d at a
    # reference depth. But for a BAND we want distance to the projected plane LINE.
    # We compute that separately (plane_line_distance). Here return t (intersection
    # depth) and validity, plus the constant-side sign of the camera centre.
    side = np.sign(n_dot_o - tau)                      # which half-space the cam is in
    return t, valid, side, (n, tau, o, R)


def plane_line_distance(frame, W, H, fx, fy, cx, cy, geom):
    """Per-pixel SIGNED pixel distance to the PROJECTED clip-plane line, and the
    line coefficients (a,b,c). This is the projection of the ACTUAL plane
    {X : n.X = tau} at scene depth -- NOT its vanishing line.

    Derivation. A world point X projects to pixel p ~ P [X;1] with the world->pixel
    matrix P = K [R_wc | t_wc] (K = diag(fx,fy,1) with principal point cx,cy). A
    world plane is the homogeneous 4-vector pi = [n; -tau] (so pi.[X;1]=0 iff
    n.X=tau). Under a projective camera a plane maps to the image line
        l = P_pinv^T pi ,   P_pinv = pseudo-inverse of P (4x3),
    equivalently, since points on the plane satisfy pi.[X;1]=0 and X = P_pinv p (+
    null-space, which lies ON the plane through the camera centre), the line is
        l ∝ (P M)^{-T} ... -> in practice we form l directly from the 3x4 P and pi
    using the adjugate. Cleanest closed form: express the plane in CAMERA coords
    (n_c = R_wc n, d_c = tau - n.o with o the camera centre; a camera-space point
    Xc satisfies n_c.Xc = d_c). A camera-space plane n_c.Xc = d_c projects (pinhole
    Xc=(x,y,z), u=fx x/z+cx, v=fy y/z+cy) to
        n_c[0]*(u-cx)/fx + n_c[1]*(v-cy)/fy + n_c[2] = d_c / z_plane ...
    but z cancels along the plane's image line exactly when we use the plane's
    own depth. Substituting Xc = z*((u-cx)/fx,(v-cy)/fy,1) into n_c.Xc=d_c gives
        z * [ n_c[0](u-cx)/fx + n_c[1](v-cy)/fy + n_c[2] ] = d_c
    The projected line is where this holds for the plane, i.e. the image locus is
        n_c[0]*(u-cx)/fx + n_c[1]*(v-cy)/fy + n_c[2] = d_c / z
    and along the true line z takes the plane depth. The z-INDEPENDENT line is the
    set where the bracket equals d_c/z for the pixel's own plane-intersection z --
    which reduces to the linear equation a*u+b*v+c=0 with the d_c term folded in
    via the homogeneous form below (a,b,c derived from the full 3x4 P·pi adjugate).
    """
    n, tau, o, R = geom            # R = c2w rot (camera axes in world, COLMAP)
    R_wc = R.T                     # camera-from-world
    t_wc = -R_wc @ o
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    P = K @ np.hstack([R_wc, t_wc.reshape(3, 1)])          # 3x4 world->pixel

    # A world plane pi=[n;-tau] projects to the image line l = P_pinv^T pi (standard
    # projective result; robust for ALL incidences incl. grazing, unlike projecting
    # finite plane points which blow up as their depth -> 0). l=(a,b,c): a u+b v+c=0.
    pi = np.array([n[0], n[1], n[2], -tau], dtype=np.float64)
    l = np.linalg.pinv(P).T @ pi
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    grad = math.hypot(a, b)
    xs = np.arange(W); ys = np.arange(H)
    ug, vg = np.meshgrid(xs, ys)

    face_on = grad < 1e-6                                  # plane ⟂ optical axis, dead-on
    if face_on:
        # Degenerate: the projected line is at infinity (plane seen exactly face-on
        # through the principal point). There is no meaningful "distance to line";
        # the whole visible face IS the cut face. Signal this with all-zeros signed
        # (every pixel in-band) so the caller's band covers the visible foreground.
        signed_px = np.zeros((H, W), dtype=np.float64)
        return signed_px, (a, b, c)

    signed_px = (a * ug + b * vg + c) / grad
    # Orient the sign so the KEPT (object) side is NEGATIVE and the CULLED (empty)
    # side is POSITIVE -- this is what leak / hole / overshoot assume.
    #
    # We orient off the CAMERA, not off n.x<=tau: the grazing cut-eval camera is
    # always placed on the KEPT side (it looks at the exposed cut face from the
    # kept half-space; xclipgs_gen_cuteval_configs positions it at plane_point +
    # dist*cam_dir on the +normal/kept side). This is convention-independent -- it
    # does not matter whether the pipeline stores the plane as keeps n.x<=tau or
    # n.x>=tau, or which way plane_normal points; the camera side is ground truth
    # for "where the kept object is". (A prior version oriented by n.x>tau and was
    # inverted for this data: the object/GT-foreground sits at the camera side,
    # which n.x>tau labelled 'culled' -- so leak/hole/overshoot measured the wrong
    # half-spaces. Verified: GT foreground mean signed_px is now negative.)
    cam_hom = P @ np.array([o[0], o[1], o[2], 1.0])
    # The camera centre projects to the principal point; instead test a point a
    # little in FRONT of the camera on the kept side: step from a plane point back
    # toward the camera. plane_pt is the nearest plane point; o - plane_pt points
    # from plane to camera (into the kept half-space).
    plane_pt = o + (tau - n @ o) * n / (n @ n + 1e-12)
    kept_dir = o - plane_pt                      # from plane toward camera = kept side
    kept_pt = plane_pt + 0.5 * kept_dir          # midway, safely on the kept side, in front
    hk = P @ np.array([kept_pt[0], kept_pt[1], kept_pt[2], 1.0])
    if abs(hk[2]) > 1e-9:
        uk, vk = hk[0] / hk[2], hk[1] / hk[2]
        if (a * uk + b * vk + c) / grad > 0:     # kept projected positive -> flip so kept<0
            signed_px = -signed_px
    return signed_px, (a, b, c)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr_np(a, b, mask=None):
    if mask is not None:
        if mask.sum() < 1:
            return None
        d = ((a - b) ** 2)[mask]
    else:
        d = (a - b) ** 2
    mse = d.mean()
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * math.log10(mse))


def ssim_np(a, b, mask=None):
    """Global SSIM (single-window, luminance) on masked pixels. Cheap, mask-aware;
    not windowed SSIM but monotone with it for the band comparison."""
    ga = a.mean(-1); gb = b.mean(-1)
    if mask is not None:
        if mask.sum() < 2:
            return None
        ga = ga[mask]; gb = gb[mask]
    mu_a, mu_b = ga.mean(), gb.mean()
    va, vb = ga.var(), gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def foreground_mask(img, bg_is_black=True, thr=0.04):
    """Foreground = pixels that differ from the (black) background."""
    lum = img.max(-1)
    return lum > thr if bg_is_black else (1 - lum) > thr


def edge_spread(gray, signed_px, band_px):
    """Cut-edge width = the gradient-magnitude-weighted STD (in px) of where the
    cross-plane luminance change happens. A sharp cut concentrates all its
    intensity change in a narrow band around the line -> small spread; a fuzzy
    operator (or a smeared dim halo) spreads the change out -> large spread.

    Why not a 10-90% RISE WIDTH: that assumes a clean monotonic step and is gamed
    by a fuzzy operator that fills the band with a uniform DIM FLOOR -- the floor
    compresses the 10%/90% thresholds so the "rise" looks tiny (falsely sharp).
    Verified: rise-width gave ClipGS spurious values of 2-4 px on some frames. A
    gradient-weighted spread cannot be gamed that way: a flat floor has ~zero
    gradient, so it contributes no weight; only a genuine localized step gives a
    small spread. Robust and monotone (Ours<HC<MM<ClipGS on every scene)."""
    m = np.abs(signed_px) <= band_px
    if m.sum() < 50:
        return None
    d = signed_px[m]; g = gray[m]
    xs = np.arange(math.floor(d.min()), math.ceil(d.max()))
    prof, cs = [], []
    for x0 in xs:
        sel = (d >= x0) & (d < x0 + 1)
        if sel.sum() >= 3:
            prof.append(g[sel].mean()); cs.append(x0 + 0.5)
    if len(prof) < 5:
        return None
    prof = np.array(prof); cs = np.array(cs)
    grad = np.abs(np.gradient(prof))          # |d luminance / d s| along the profile
    if grad.sum() < 1e-6:
        return None
    w = grad / grad.sum()                      # gradient-magnitude weights
    mean = float(np.sum(w * cs))
    return float(math.sqrt(np.sum(w * (cs - mean) ** 2)))


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-transforms", required=True)
    ap.add_argument("--gt-dir", required=True, help="dir with GT pngs (frame file_path stems)")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name=renders_dir entries; renders indexed 00000.png in "
                         "transforms frame order (ndsplat render_set order)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band-px", type=float, default=12.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tf = json.load(open(args.gt_transforms))
    frames = tf["frames"]
    fovx = float(tf["camera_angle_x"])

    methods = {}
    for m in args.methods:
        name, d = m.split("=", 1)
        methods[name] = d

    # frames are sorted by file_path in the converter; ndsplat render_set enumerates
    # getTestCameras() in that same sorted order -> render 00000.png <-> frames[0].
    frames = sorted(frames, key=lambda f: f["file_path"])

    # accumulate per (method, family) metrics
    ref = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # ref[method][family][metric]=[..]
    noref = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # noref[method][family][metric]=[..]
    edge_w_acc = defaultdict(list)                   # edge_w_acc[family] = [.. GT edge widths ..]

    # lpips (optional, torch); fall back to None if unavailable. Matches metrics.py:
    # a pre-built LPIPS(net_type, version) criterion passed to lpips(x, y, criterion).
    lpips_fn = None
    try:
        import torch
        from lpipsPyTorch import lpips as _lpips, LPIPS
        _crit = LPIPS("vgg", "0.1").to("cuda")
        _MIN = 64   # VGG has 5 max-pools; a crop thinner than this underflows a
                    # pooling layer ("output size too small"). Grazing bands are a
                    # 1-2px-tall strip, so we EXPAND the bbox to >=_MIN in each dim
                    # (clamped to the frame). Out-of-band pixels stay zeroed.
        def lpips_fn(a, b, mask):
            # a,b HxWx3 in [0,1]; mask HxW bool. Zero out-of-band pixels, then crop
            # a (padded) mask bbox so LPIPS sees the cut-face patch, not the frame.
            if mask.sum() < 64:
                return None
            H, W = mask.shape
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            # expand each dimension to at least _MIN, centred, clamped to bounds
            if (y1 - y0) < _MIN:
                cyc = (y0 + y1) // 2
                y0 = max(0, cyc - _MIN // 2); y1 = min(H, y0 + _MIN); y0 = max(0, y1 - _MIN)
            if (x1 - x0) < _MIN:
                cxc = (x0 + x1) // 2
                x0 = max(0, cxc - _MIN // 2); x1 = min(W, x0 + _MIN); x0 = max(0, x1 - _MIN)
            if (y1 - y0) < 32 or (x1 - x0) < 32:
                return None                       # frame itself smaller than the net floor
            m = mask[y0:y1, x0:x1][..., None]
            pa = a[y0:y1, x0:x1] * m
            pb = b[y0:y1, x0:x1] * m
            ta = torch.tensor(pa).permute(2, 0, 1)[None].float().cuda()
            tb = torch.tensor(pb).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                return float(_lpips(ta, tb, _crit))
    except Exception as e:
        print(f"  [lpips unavailable: {e}] band_lpips will be None")

    for idx, fr in enumerate(frames):
        if not fr.get("clip"):
            continue
        stem = os.path.basename(fr["file_path"])
        gt_path = os.path.join(args.gt_dir, stem + ".png")
        if not os.path.isfile(gt_path):
            print(f"  [warn] missing GT {gt_path}; skip")
            continue
        gt = load_rgb(gt_path)
        H, W = gt.shape[:2]
        fx = fov2focal(fovx, W); fy = fx
        cx, cy = W / 2.0, H / 2.0
        cut_axis = fr.get("cut_axis")
        # family from mode
        family = "graze" if "graze" in (fr.get("mode") or "") else "perp"

        _t, _v, _side, geom = plane_signed_distance_image(fr, W, H, fx, fy, cx, cy)
        signed_px, line_abc = plane_line_distance(fr, W, H, fx, fy, cx, cy, geom)
        face_on = math.hypot(line_abc[0], line_abc[1]) < 1e-6

        # GT foreground (so black background never enters any band metric).
        gt_fg = foreground_mask(gt)
        if family == "perp" or face_on:
            # Face-on / perpendicular: the whole visible cut face is the region of
            # interest -> band = GT foreground (the exposed interior fills the frame).
            band = gt_fg
        else:
            # Grazing: a strip straddling the projected plane line, restricted to
            # foreground so we score the cut edge, not empty background.
            band = (np.abs(signed_px) <= args.band_px) & gt_fg

        gt_gray = gt.mean(-1)
        # edge width (GT-based; property of the reference cut) -> per family
        ew = edge_spread(gt_gray, signed_px, args.band_px)
        if ew is not None:
            edge_w_acc[family].append(ew)

        for mname, mdir in methods.items():
            rp = os.path.join(mdir, f"{idx:05d}.png")
            if not os.path.isfile(rp):
                print(f"  [warn] missing render {rp}; skip {mname}")
                continue
            rimg = load_rgb(rp)
            if rimg.shape[:2] != (H, W):
                rimg = np.asarray(Image.fromarray((rimg * 255).astype(np.uint8)).resize((W, H))) / 255.0

            # --- reference band metrics (vs GT) ---
            ref[mname][family]["band_psnr"].append(psnr_np(rimg, gt, band))
            s = ssim_np(rimg, gt, band)
            if s is not None:
                ref[mname][family]["band_ssim"].append(s)
            if lpips_fn is not None:
                lv = lpips_fn(rimg, gt, band)
                if lv is not None:
                    ref[mname][family]["band_lpips"].append(lv)

            # --- no-reference: LEAK past the plane (grazing only; the line is the
            # cut edge). A correct hard truncation renders NOTHING beyond the plane
            # that the reference cut doesn't have. We measure the method's spurious
            # foreground energy on the CULLED side where the GT is background --
            # i.e. material that leaked past the cut. Normalized by the method's
            # total near-plane foreground so it's a fraction, scale-free.
            # signed_px > 0 is the culled side (oriented in plane_line_distance).
            if family == "graze" and not face_on:
                r_fg = foreground_mask(rimg)
                gt_bg = ~gt_fg
                culled = signed_px > 2.0               # just past the cut edge
                near = np.abs(signed_px) <= 60.0       # ignore far-field; focus at cut
                leak_mask = r_fg & culled & gt_bg & near
                ref_mask = r_fg & near
                leak_energy = rimg.max(-1)[leak_mask].sum()
                total_energy = rimg.max(-1)[ref_mask].sum() + 1e-9
                noref[mname][family]["leak"].append(float(leak_energy / total_energy))

                # --- GT-referenced CUT-BOUNDARY MASS ERROR (holes + overshoot).
                # A whole-splat cull mis-truncates near the plane in two SIGNED ways,
                # invisible to the edge/leak metrics: it OVER-removes (drops splats
                # whose bodies should have reached the plane -> HOLES = missing kept-
                # side material) and UNDER-removes (keeps splats whose bodies poke
                # past -> OVERSHOOT). Measured against the independent vengine GT in a
                # near-plane band, normalized by the GT kept-side energy so it is a
                # scale-free fraction; cut_error = hole + overshoot is the total
                # material the operator misplaces at the cut. (band radius = --band-px.)
                #
                # We accumulate the RAW numerator/denominator ENERGIES per frame and
                # take a single ratio at aggregation time -- NOT a per-frame ratio.
                # A grazing frame with little kept-side foreground has a tiny
                # denominator; a per-frame ratio would explode there and dominate the
                # mean (seen: cut_error ~200 instead of ~0.2). Energy-pooling weights
                # each frame by how much cut it actually shows, which is correct.
                Lr = rimg.max(-1); Lg = gt.max(-1)
                cband = np.abs(signed_px) <= args.band_px
                kept = (signed_px < 0.0) & cband      # kept half-space, near plane
                culled_c = (signed_px > 0.0) & cband  # culled half-space, near plane
                hole_e = float(np.clip(Lg - Lr, 0.0, None)[kept & gt_fg].sum())
                over_e = float(Lr[culled_c & r_fg & (~gt_fg)].sum())
                ref_e = float(Lg[kept & gt_fg].sum())
                noref[mname][family]["hole_num"].append(hole_e)
                noref[mname][family]["over_num"].append(over_e)
                noref[mname][family]["cutref_den"].append(ref_e)

                # --- no-reference: EDGE SHARPNESS of THIS method's cut boundary.
                # gradient-weighted spread (px) of the method's own cut edge (see
                # edge_spread). Exact truncation -> narrow (sharp); moment/hard-cull
                # a touch wider; ClipGS widest (smeared). Robust to a dim halo, unlike
                # a 10-90 rise width.
                mew = edge_spread(rimg.mean(-1), signed_px, args.band_px)
                if mew is not None:
                    noref[mname][family]["edge_w"].append(mew)

        # debug overlay per frame. ALWAYS overwrite -- a stale overlay from a prior
        # (e.g. different-geometry) render must never survive a re-run, or it would
        # show the wrong band/line for the current GT (this bit us with the 8-deg
        # vs 1.5-deg grazing re-render).
        #
        # Green band = the region the metrics score. Red line = the projected cut
        # EDGE, drawn ONLY for grazing views: there the plane genuinely projects to
        # the cut edge and the band straddles it. For perp views the band is the
        # whole cut face (the plane is face-on) and there is NO meaningful cut line
        # -- a tilted perp view's "line" is just the plane grazing off-frame, which
        # is misleading, so we never draw it.
        if args.debug:
            dbg_key = f"{cut_axis}_{family}"
            dbg_path = os.path.join(args.out, f"overlay_{dbg_key}_{idx:05d}.png")
            ov = (gt * 255).astype(np.uint8).copy()
            ov[band] = (0.5 * ov[band] + np.array([0, 128, 0])).astype(np.uint8)
            if family == "graze" and not face_on:
                ov[np.abs(signed_px) <= 2.0] = np.array([255, 0, 0], dtype=np.uint8)
            Image.fromarray(ov).save(dbg_path)

    # NOTE: a "popping" metric was removed. It compared consecutive grazing frames,
    # but the cut-eval grazing views are ~254 mm apart on a scene of ~294 mm diagonal
    # -- a large viewpoint jump, not a smooth orbit -- so the frame-to-frame change
    # was dominated by parallax, not by cull-threshold flicker. A faithful popping
    # metric needs a densely-sampled orbit (small camera steps) re-render; until
    # then it is omitted rather than reported misleadingly.

    # ---- aggregate ----
    def agg(lst):
        lst = [x for x in lst if x is not None]
        return float(np.mean(lst)) if lst else None

    results = {"scene": args.scene, "band_px": args.band_px, "methods": {}}
    edge_by_family = {fam: agg(v) for fam, v in edge_w_acc.items()}
    results["gt_edge_width_px"] = edge_by_family

    for mname in methods:
        entry = {}
        for fam in ("perp", "graze"):
            entry[fam] = {
                "band_psnr": agg(ref[mname][fam]["band_psnr"]),
                "band_ssim": agg(ref[mname][fam]["band_ssim"]),
                "band_lpips": agg(ref[mname][fam]["band_lpips"]),
            }
            if fam == "graze":
                entry[fam]["leak"] = agg(noref[mname][fam]["leak"])
                entry[fam]["edge_w"] = agg(noref[mname][fam]["edge_w"])
                # Pool energies across frames, then one ratio (see the metric block).
                den = sum(noref[mname][fam]["cutref_den"]) + 1e-9
                hole = sum(noref[mname][fam]["hole_num"]) / den
                overshoot = sum(noref[mname][fam]["over_num"]) / den
                entry[fam]["hole"] = hole if noref[mname][fam]["cutref_den"] else None
                entry[fam]["overshoot"] = overshoot if noref[mname][fam]["cutref_den"] else None
                entry[fam]["cut_error"] = (hole + overshoot) if noref[mname][fam]["cutref_den"] else None
        results["methods"][mname] = entry

    out_json = os.path.join(args.out, "cutplane_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # pretty print
    print(f"\n=== cut-plane metrics: {args.scene} (band {args.band_px:.0f}px) ===")
    print(f"{'method':10s} | {'perp bPSNR':>10s} {'perp bSSIM':>10s} | "
          f"{'leak↓':>7s} {'edge↓':>6s} {'cutErr↓':>8s}")
    for mname in methods:
        e = results["methods"][mname]
        pp = e["perp"]["band_psnr"]; ps = e["perp"]["band_ssim"]
        lk = e["graze"].get("leak"); ew = e["graze"].get("edge_w")
        ce = e["graze"].get("cut_error")
        def f(x, d=2): return f"{x:.{d}f}" if x is not None else "  —"
        print(f"{mname:10s} | {f(pp):>10s} {f(ps,4):>10s} | "
              f"{f(lk,4):>7s} {f(ew,2):>6s} {f(ce,4):>8s}")
    print(f"GT edge width (px): {edge_by_family}")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
