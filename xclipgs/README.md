# XClipGS — real Gaussian-splatting clip demo (PlayCanvas)

A standalone web page (not a claude.ai artifact) that renders a **genuine 3D
Gaussian-Splatting scene** with the PlayCanvas engine and clips it with a plane,
comparing the exact analytic operator against hard-cull and moment surrogates —
**per pixel, through the real splat rasterizer**.

This is the high-fidelity companion to the self-contained WebGL artifact
(hand-rolled splatter). The artifact is shareable but can't load an engine (CSP);
this page can, so it uses the production pipeline and can load real `.ply` captures.

## Run

It's static — serve the folder and open it:

```bash
cd pages/XClipGS-demo
python3 -m http.server 8000
# open http://localhost:8000
```

The PlayCanvas engine loads from its CDN (`code.playcanvas.com`), so the page needs
network access the first time. A procedural demo scene is generated in-browser so it
works with no assets; **drag a `.ply` / `.compressed.ply` onto the canvas** to view
your own Gaussian scene (e.g. one exported from the XClipGS pipeline).

## Controls

- **Ours / MM / HC** — the clip operator, injected into the gsplat shader.
- **Plane position** slider, **Axis** (X/Y/Z), **Sweep** (animate the plane — watch HC pop).
- Drag to orbit, scroll to zoom.

## How the clip is injected

The engine is **pinned to 2.7.4**. Before the gsplat material builds,
`installShaderClip()` patches the engine's **global chunk registry**
(`pc.shaderChunks.gsplatVS` / `.gsplatPS`) — NOT `material.chunks`, which is empty
for the 2.7.x ShaderMaterial. Three global uniforms (`uClipN`, `uClipTau`,
`uClipMode`) are set on `graphicsDevice.scope` every frame.

Per operator:

- **Ours — exact per-ray truncation (the paper's operator).** The vertex shader
  reconstructs each quad corner's exact world position (`corner.offset` is a pure
  clip-space offset added to `center.proj`, so `Δview = (offset.x/P00,
  offset.y/P11, 0)` and `world = Rᵀ(view − t)` from `matrix_view`; the
  interpolated varying is exact since the quad has constant view depth), and ships
  the splat's world centre `μ` and inverse covariance `Σ⁻¹` (6 floats). The
  fragment then evaluates the **closed-form CDF of the kept ray segment**: the 3D
  Gaussian restricted to the view ray `o + t·d` is a 1D Gaussian with peak
  `t* = −dᵀΣ⁻¹(o−μ)/(dᵀΣ⁻¹d)` and `σₜ = 1/√(dᵀΣ⁻¹d)`; the half-space `n·x ≤ τ`
  becomes `t ≶ t₀ = (τ − n·o)/(n·d)`, so
  `alpha *= Φ(sign(n·d)·(t₀ − t*)/σₜ)`. **Why not just cut the billboard at the
  plane?** A ray past the projected plane line can still cross the kept half of
  the volume — the billboard cut leaves hard diagonal cut-lines in side/oblique
  views. The per-ray CDF is exact from every viewpoint.
- **MM — moment surrogate (per splat).** `σ_N = √(nᵀΣn)` from `readCovariance()`;
  the centre shifts toward the kept side by `σ_N·0.9·e^{−u²/2}` and alpha fades by
  the kept mass `Φ(−u)`. The untruncated shape still renders → the tail **leaks**.
- **HC — hard cull (per splat).** `sd > 0` at the centre → the whole splat is
  discarded in the vertex shader; otherwise it renders untouched → **overshoot,
  holes, popping** under Sweep.

Do NOT redeclare `matrix_model/view/projection` in the injected header — they are
already declared by the included `gsplatCenterVS` chunk, and the redefinition
silently kills the whole splat shader (invisible splats).

## Verification

The injection is verified headlessly against the real 2.7.4 chunk sources
(compile + link on an NVIDIA driver via EGL, then a single-splat render test):
Ours matches a **numpy ray-integration reference** of the truncated 3D Gaussian
to max error 1.4e-3 (the erf approximation's accuracy), with the kept side
bit-identical and the culled side extinguished; HC overshoots / vanishes whole;
MM fades to 0.5 at the plane, shifts, and still leaks. Harness: scratchpad
`pcsrc/{patch_chunks.js, compile_check.py, render_check.py}` (12/12 checks pass).

Engine quirk worth knowing: a splat whose **screen-space covariance is exactly
axis-aligned** (identity rotation viewed head-on) hits `normalize(vec2(0,0))` in
the engine's `initCorner` → NaN → invisible. The procedural scene uses random
rotations, so this never occurs in practice.

## Files
- `index.html` — page shell + controls + PlayCanvas CDN script.
- `assets/demo.js` — app, orbit camera, procedural `.ply`, clip-shader injection, UI.
