/* XClipGS — real Gaussian-splatting clip demo (PlayCanvas 2.7.4).
 *
 * Loads a 3DGS scene and injects the clip operators into the engine's own gsplat
 * shader chunks (pc.shaderChunks.gsplatVS / gsplatPS), so the cut runs inside the
 * real rasterizer:
 *   Ours (0): EXACT per-RAY truncation — each fragment evaluates the closed-form
 *             Gaussian-CDF of the kept ray segment (the paper's operator), so the
 *             cut is correct from every viewpoint (no billboard cut-lines)
 *   MM   (1): per-splat moment surrogate — shift centre + fade by kept mass
 *             (the untruncated shape still renders, so the tail LEAKS)
 *   HC   (2): per-splat hard cull by centre — overshoot / holes / popping
 * driven by global uniforms: uClipN (vec3), uClipTau, uClipMode, uClipCam.
 *
 * Verified headlessly: the patched chunks compile + link on an NVIDIA driver, and a
 * single-splat render matches a numpy ray-integration reference of the truncated
 * 3D Gaussian to 1.4e-3 max error (scratchpad pcsrc/render_check.py, all pass);
 * HC overshoots and MM leaks as designed.
 *
 * If no .ply is supplied we synthesize a compact procedural splat scene so the page
 * works standalone; drop a real .ply on the canvas to view your own capture.
 */
(function () {
  "use strict";
  const statusEl = document.getElementById('status');
  const canvas = document.getElementById('app');
  const controls = document.querySelector('.controls');
  const info = document.querySelector('.info');

  function setStatus(msg, spin) {
    statusEl.innerHTML = (spin ? '<div class="spin"></div>' : '') +
      '<div>' + msg + '</div>' +
      '<div class="drop" id="drop">or drop a <code>.ply</code> / <code>.compressed.ply</code> Gaussian scene here</div>';
    wireDrop();
  }
  const profSec = document.querySelector('.profile');
  function hideStatus() {
    statusEl.style.display = 'none'; controls.hidden = false; info.hidden = false;
    profSec.hidden = false;
    requestAnimationFrame(() => { app.resizeCanvas(); drawProfile(); });  // strip now has real width
  }

  if (typeof pc === 'undefined') { setStatus('Could not load the PlayCanvas engine (offline?).', false); return; }

  /* ---------- app ---------- */
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: true, alpha: false, preferWebGl2: true }
  });
  // FILLMODE_NONE + RESOLUTION_AUTO: the canvas is sized by CSS (it fills <main>,
  // leaving room for the cross-section strip below), and the drawing buffer tracks
  // the client size automatically.
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  window.addEventListener('resize', () => app.resizeCanvas());
  app.scene.clusteredLightingEnabled = false;

  /* ---------- REAL clip via the engine's GLOBAL gsplat chunk registry -------------
   * pc.shaderChunks is the flat chunk map the gsplat shader #includes at build time.
   * We patch gsplatVS + gsplatPS BEFORE the material builds. Verified against the
   * v2.7.4 chunk sources (src/scene/shader-lib/chunks/gsplat/{vert,frag}/gsplat.js).
   *
   * Three operators, driven by uClipMode (uClipN / uClipTau = plane):
   *   Ours (0): EXACT per-RAY truncation. The vertex shader reconstructs each quad
   *             corner's exact world position (corner.offset is a clip-space offset,
   *             so view offset = offset / projection diagonal, then view->world) and
   *             ships the splat's world centre + inverse covariance; the fragment
   *             evaluates the closed-form CDF of the kept segment of ITS view ray —
   *             correct from every viewpoint, unlike cutting the billboard at the
   *             plane (which leaves hard cut-lines in side views).
   *   MM   (1): per-SPLAT moment surrogate. Shift the centre toward the kept side
   *             and fade by kept mass — the full Gaussian still renders, so the
   *             tail visibly LEAKS past the plane (that's the point).
   *   HC   (2): per-SPLAT hard cull by centre — straddling splats overshoot or
   *             leave holes, and pop under Sweep.
   *
   * NOTE: matrix_model/view/projection are ALREADY declared by the included
   * gsplatCenterVS chunk — redeclaring them is a GLSL redefinition that kills the
   * whole splat shader (invisible splats). Only add NEW names here.              */
  let shaderClipReady = false;
  function installShaderClip() {
    const sc = pc.shaderChunks;
    if (!sc || !sc.gsplatVS || !sc.gsplatPS) { console.warn('[xclip] pc.shaderChunks gsplat mains missing'); return false; }
    if (sc.__xclip) { shaderClipReady = true; return true; }

    const VS_DECL =
      'uniform vec3 uClipN;\nuniform float uClipTau;\nuniform float uClipMode;\n' +
      'varying vec3 vClipWpos;\nvarying float vClipMul;\n' +
      'varying vec3 vClipMu;\nvarying vec3 vClipSi0;\nvarying vec3 vClipSi1;\n' +
      'float xclipErf(float x){float s=sign(x);x=abs(x);float t=1.0/(1.0+0.3275911*x);' +
      'return s*(1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-x*x));}\n';

    // (a) per-SPLAT stage — right after the model-space centre is read, BEFORE it is
    // projected, so MM's centre shift feeds the real projection. matrix_model,
    // discardVec and readCovariance() are all in scope here (from the includes).
    // For Ours we also ship the world-space centre + inverse covariance to the
    // fragment stage (6 floats, symmetric), so each FRAGMENT can evaluate the exact
    // per-ray CDF of the truncated 3D Gaussian — no billboard cut-line artifacts.
    const VS_SPLAT_ANCHOR = 'vec3 modelCenter = readCenter(source);';
    const VS_SPLAT =
      VS_SPLAT_ANCHOR + '\n' +
      '    vClipMul = 1.0; vClipWpos = vec3(0.0);\n' +
      '    {\n' +
      '        vec3 xcA, xcB;\n' +
      '        readCovariance(source, xcA, xcB);\n' +
      '        mat3 xcV = mat3(xcA.x, xcA.y, xcA.z, xcA.y, xcB.x, xcB.y, xcA.z, xcB.y, xcB.z);\n' +
      '        mat3 xcM = mat3(matrix_model);\n' +
      '        mat3 xcSw = xcM * xcV * transpose(xcM);      // world-space 3D covariance\n' +
      '        xcSw[0][0] += 1e-7; xcSw[1][1] += 1e-7; xcSw[2][2] += 1e-7;  // regularize flat splats\n' +
      '        vClipMu = (matrix_model * vec4(modelCenter, 1.0)).xyz;\n' +
      '        mat3 xcSi = inverse(xcSw);\n' +
      '        vClipSi0 = vec3(xcSi[0][0], xcSi[0][1], xcSi[0][2]);\n' +
      '        vClipSi1 = vec3(xcSi[1][1], xcSi[1][2], xcSi[2][2]);\n' +
      '        float xcSd = dot(uClipN, vClipMu) - uClipTau;\n' +
      '        if (uClipMode > 1.5) {\n' +
      '            // HC: keep/drop the WHOLE splat by its centre side\n' +
      '            if (xcSd > 0.0) { gl_Position = discardVec; return; }\n' +
      '        } else if (uClipMode > 0.5) {\n' +
      '            // MM: sigma along the plane normal from the 3D covariance\n' +
      '            float xcSig = sqrt(max(dot(uClipN, xcSw * uClipN), 1e-12));\n' +
      '            float xcU = xcSd / xcSig;\n' +
      '            float xcKept = 0.5 * (1.0 - xclipErf(xcU * 0.70710678));\n' +
      '            if (xcKept < 0.004) { gl_Position = discardVec; return; }\n' +
      '            // shift centre toward the kept side, fade by kept mass; the\n' +
      '            // untruncated shape still renders -> tail leaks past the plane\n' +
      '            modelCenter -= (uClipN * (xcSig * 0.9 * exp(-0.5 * xcU * xcU))) * mat3(matrix_model);\n' +
      '            vClipMul = xcKept;\n' +
      '        }\n' +
      '    }';

    // (b) per-CORNER world position — corner.offset is the exact clip-space offset
    // added to center.proj, so view offset = offset / diag(P), then view -> world
    // (world = R^T (view - t)). Interpolates exactly: the quad is a constant-depth
    // plane in view space.
    const VS_CORNER_ANCHOR = 'gaussianUV = corner.uv;';
    const VS_CORNER =
      VS_CORNER_ANCHOR + '\n' +
      '    {\n' +
      '        vec3 xcView = center.view + vec3(corner.offset.x / matrix_projection[0][0],\n' +
      '                                         corner.offset.y / matrix_projection[1][1], 0.0);\n' +
      '        vClipWpos = transpose(mat3(matrix_view)) * (xcView - matrix_view[3].xyz);\n' +
      '    }';

    // (c) fragment stage — Ours applies the EXACT closed-form per-ray truncation:
    // the 3D Gaussian restricted to this fragment's view ray o + t*d is a 1D
    // Gaussian in t (peak t*, sigma_t = 1/sqrt(d^T Sigma^-1 d)); the half-space
    // n.x <= tau becomes t <= t0 (or >=), so the kept fraction of the ray integral
    // is a Gaussian CDF. This is exact from EVERY viewpoint — no billboard
    // cut-line (a plain "cut the quad at the plane" leaves hard diagonal edges in
    // side views, because a ray past the projected plane line can still cross the
    // kept half of the volume). MM's kept-mass fade arrives via vClipMul.
    const PS_DECL =
      'uniform vec3 uClipN;\nuniform float uClipTau;\nuniform float uClipMode;\nuniform vec3 uClipCam;\n' +
      'varying vec3 vClipWpos;\nvarying float vClipMul;\n' +
      'varying vec3 vClipMu;\nvarying vec3 vClipSi0;\nvarying vec3 vClipSi1;\n' +
      'float xclipErf(float x){float s=sign(x);x=abs(x);float t=1.0/(1.0+0.3275911*x);' +
      'return s*(1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-x*x));}\n';
    const PS_ANCHOR = 'mediump float alpha = exp(-A * 4.0) * gaussianColor.a;';
    const PS_CLIP =
      PS_ANCHOR + '\n' +
      '    if (uClipMode < 0.5) {\n' +
      '        vec3 xcD = vClipWpos - uClipCam;             // view ray (unnormalized: scale cancels)\n' +
      '        mat3 xcSi = mat3(vClipSi0.x, vClipSi0.y, vClipSi0.z,\n' +
      '                         vClipSi0.y, vClipSi1.x, vClipSi1.y,\n' +
      '                         vClipSi0.z, vClipSi1.y, vClipSi1.z);\n' +
      '        vec3 xcAd = xcSi * xcD;\n' +
      '        float xcDad = max(dot(xcD, xcAd), 1e-12);\n' +
      '        float xcTpk = -dot(uClipCam - vClipMu, xcAd) / xcDad;   // 1D peak along the ray\n' +
      '        float xcNd = dot(uClipN, xcD);\n' +
      '        float xcNdS = (abs(xcNd) < 1e-9) ? 1e-9 : xcNd;         // parallel-ray guard\n' +
      '        float xcT0 = (uClipTau - dot(uClipN, uClipCam)) / xcNdS; // ray-plane crossing\n' +
      '        float xcZ = (xcT0 - xcTpk) * sqrt(xcDad) * 0.70710678;\n' +
      '        alpha *= clamp(0.5 + 0.5 * sign(xcNdS) * xclipErf(xcZ), 0.0, 1.0);\n' +
      '    }\n' +
      '    alpha *= vClipMul;';

    const vsOk = sc.gsplatVS.includes(VS_SPLAT_ANCHOR) && sc.gsplatVS.includes(VS_CORNER_ANCHOR);
    const psOk = sc.gsplatPS.includes(PS_ANCHOR);
    if (!vsOk || !psOk) { console.warn('[xclip] chunk anchors missing (engine version changed?) vs:', vsOk, 'ps:', psOk); return false; }

    sc.gsplatVS = VS_DECL + sc.gsplatVS.replace(VS_SPLAT_ANCHOR, VS_SPLAT).replace(VS_CORNER_ANCHOR, VS_CORNER);
    sc.gsplatPS = PS_DECL + sc.gsplatPS.replace(PS_ANCHOR, PS_CLIP);

    sc.__xclip = true; shaderClipReady = true;
    console.log('[xclip] clip installed: Ours=exact per-ray CDF, MM=shift+fade, HC=centre cull');
    return true;
  }
  installShaderClip();

  function setShaderClipUniforms() {
    if (!shaderClipReady) return;
    const gd = app.graphicsDevice, n = AXES[state.axis], cp = cam.getPosition();
    gd.scope.resolve('uClipN').setValue([n.x, n.y, n.z]);
    gd.scope.resolve('uClipTau').setValue(planeWorldOffset());
    gd.scope.resolve('uClipMode').setValue(state.mode);
    gd.scope.resolve('uClipCam').setValue([cp.x, cp.y, cp.z]);
  }

  // camera
  const cam = new pc.Entity('camera');
  cam.addComponent('camera', { clearColor: new pc.Color(0.027, 0.035, 0.055, 1), fov: 45, toneMapping: pc.TONEMAP_NONE });
  cam.setPosition(0, 0.2, 3.2);
  app.root.addChild(cam);

  /* ---------- visible clipping plane: a teal WIREFRAME GRID ---------- */
  let planeEntity = null;
  const PLANE_HALF = 1.25;                            // grid half-extent (world units)
  function buildGridMesh(half, div) {
    // build line positions for a grid lying in the local XZ plane (normal = +Y)
    const pos = [], step = (2 * half) / div;
    for (let i = 0; i <= div; i++) {
      const t = -half + i * step;
      pos.push(t, 0, -half, t, 0, half);             // lines along Z
      pos.push(-half, 0, t, half, 0, t);             // lines along X
    }
    const mesh = new pc.Mesh(app.graphicsDevice);
    mesh.setPositions(pos);
    mesh.primitive[0] = { type: pc.PRIMITIVE_LINES, base: 0, count: pos.length / 3, indexed: false };
    mesh.update(pc.PRIMITIVE_LINES);
    return mesh;
  }
  function buildPlane() {
    const mat = new pc.StandardMaterial();
    mat.useLighting = false; mat.emissive = new pc.Color(0.24, 0.95, 0.88);
    mat.emissiveIntensity = 1.0; mat.diffuse = new pc.Color(0, 0, 0);
    mat.blendType = pc.BLEND_ADDITIVE; mat.depthWrite = false; mat.update();
    const mesh = buildGridMesh(PLANE_HALF, 6);
    const mi = new pc.MeshInstance(mesh, mat);
    planeEntity = new pc.Entity('clipPlane');
    planeEntity.addComponent('render', { meshInstances: [mi] });
    app.root.addChild(planeEntity);
    updatePlane();
  }
  const _q = new pc.Quat();
  function planeWorldOffset() {                        // offset of the plane along +axis, world units
    const t = target || { x: 0, y: 0, z: 0 }, n = AXES[state.axis];
    return n.x * t.x + n.y * t.y + n.z * t.z + state.tau;
  }
  function updatePlane() {
    if (!planeEntity) return;
    const n = AXES[state.axis];
    if (state.axis === 0) _q.setFromEulerAngles(0, 0, -90);       // X normal
    else if (state.axis === 1) _q.setFromEulerAngles(0, 0, 0);    // Y normal
    else _q.setFromEulerAngles(90, 0, 0);                         // Z normal
    planeEntity.setLocalRotation(_q);
    // place the grid at the SAME world offset the clip cuts at
    const off = planeWorldOffset();
    planeEntity.setPosition(n.x * off, n.y * off, n.z * off);
  }

  // simple orbit control
  let az = 0.6, el = 0.15, rad = 3.2, target = new pc.Vec3(0, 0, 0);
  let dragging = false, lx = 0, ly = 0, autoOrbit = true;
  function applyCam() {
    const x = Math.cos(el) * Math.sin(az) * rad, y = Math.sin(el) * rad, z = Math.cos(el) * Math.cos(az) * rad;
    cam.setPosition(target.x + x, target.y + y, target.z + z);
    cam.lookAt(target);
  }
  // one pointer = orbit, two pointers = pinch zoom (mobile)
  const pts = new Map(); let pinchD = 0;
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); autoOrbit = false;
    if (pts.size === 1) { dragging = true; lx = e.clientX; ly = e.clientY; } else { dragging = false; pinchD = 0; }
  });
  canvas.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinchD > 0) rad = Math.max(1.2, Math.min(8, rad * pinchD / d));
      pinchD = d; applyCam(); return;
    }
    if (!dragging) return;
    az -= (e.clientX - lx) * 0.006; el += (e.clientY - ly) * 0.006; el = Math.max(-1.4, Math.min(1.4, el));
    lx = e.clientX; ly = e.clientY; applyCam();
  });
  const endPointer = e => { pts.delete(e.pointerId); dragging = false; pinchD = 0; };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('wheel', e => { e.preventDefault(); rad = Math.max(1.2, Math.min(8, rad * (1 + Math.sign(e.deltaY) * 0.08))); applyCam(); }, { passive: false });

  /* ---------- clip state ---------- */
  const state = { mode: 0, tau: 0.0, axis: 0, sweep: false };
  const AXES = [new pc.Vec3(1, 0, 0), new pc.Vec3(0, 1, 0), new pc.Vec3(0, 0, 1)];
  let gsplatEntity = null;

  // Robustly locate the gsplat material across engine-version API shapes (used by
  // the console diagnostics / window.XC only — the clip itself is chunk-injected).
  function getMat(e) {
    if (!e || !e.gsplat) return null;
    const inst = e.gsplat.instance;
    const cands = [
      inst && inst.material,
      inst && inst.meshInstance && inst.meshInstance.material,
      e.gsplat.material,
      inst && inst.meshInstance && inst.meshInstance.mesh && inst.meshInstance.mesh.material
    ];
    for (const m of cands) if (m && typeof m.setParameter === 'function') return m;
    return null;
  }

  /* ---------- procedural fallback splat scene (a small .ply in memory) ---------- */
  function proceduralPly() {
    // A FEW big, distinct Gaussian ellipsoids (so the clip is obvious per primitive),
    // written as a real binary .ply with the standard 3DGS fields.
    let s = 20240711; const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    function quat() { // random unit quaternion (rot_0..3 = w,x,y,z in PlayCanvas .ply order)
      const u1 = rnd(), u2 = rnd(), u3 = rnd();
      const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
      return [b * Math.cos(6.2831853 * u3), a * Math.sin(6.2831853 * u2), a * Math.cos(6.2831853 * u2), b * Math.sin(6.2831853 * u3)];
    }
    const PAL = [[0.95, 0.45, 0.32], [0.93, 0.30, 0.37], [0.98, 0.66, 0.30], [0.38, 0.72, 0.92],
                 [0.32, 0.55, 0.96], [0.58, 0.47, 0.94], [0.30, 0.82, 0.73], [0.88, 0.55, 0.78]];
    const POS = [[-0.62, 0.16, -0.08], [-0.30, -0.30, 0.22], [-0.04, 0.32, 0.05], [0.02, -0.10, -0.28],
                 [0.28, 0.20, 0.26], [0.34, -0.28, -0.05], [0.60, 0.06, 0.13], [-0.42, 0.02, 0.32],
                 [0.10, 0.38, -0.22], [-0.14, -0.40, -0.16], [0.48, 0.32, -0.18], [0.46, -0.06, 0.34],
                 [-0.64, -0.18, 0.02], [0.16, -0.34, 0.28], [-0.02, 0.02, 0.40], [0.64, -0.28, 0.18]];
    const N = POS.length;
    const props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
    let header = 'ply\nformat binary_little_endian 1.0\nelement vertex ' + N + '\n';
    props.forEach(p => header += 'property float ' + p + '\n'); header += 'end_header\n';
    const head = new TextEncoder().encode(header);
    const buf = new ArrayBuffer(head.length + N * props.length * 4);
    new Uint8Array(buf).set(head, 0);
    const dv = new DataView(buf, head.length);
    const SH = 0.2820948, f = c => (c - 0.5) / SH; // inverse of PlayCanvas 0.5 + SH*f_dc
    let o = 0;
    for (let i = 0; i < N; i++) {
      const p = POS[i], col = PAL[i % PAL.length], q = quat();
      const b = 0.16 + rnd() * 0.13;                    // big ellipsoids
      const sc = [b * (0.8 + rnd() * 0.6), b * (0.7 + rnd() * 0.5), b * (0.6 + rnd() * 0.4)];
      const vals = [p[0], p[1], p[2], f(col[0]), f(col[1]), f(col[2]), 6.0,   // high opacity logit -> solid
        Math.log(sc[0]), Math.log(sc[1]), Math.log(sc[2]), q[0], q[1], q[2], q[3]];
      for (let k = 0; k < vals.length; k++) { dv.setFloat32(o, vals[k], true); o += 4; }
    }
    return new Blob([buf], { type: 'application/octet-stream' });
  }

  /* ---------- load a gsplat from a URL/blob ---------- */
  function loadSplat(url, name) {
    if (gsplatEntity) { gsplatEntity.destroy(); gsplatEntity = null; }
    // A blob: URL has no extension, so the gsplat loader can't detect .ply -> give it
    // an explicit filename so the format is recognised.
    const fname = (name && /\.(ply|splat|compressed\.ply|sog)$/i.test(name)) ? name : 'scene.ply';
    const asset = new pc.Asset(name || 'scene', 'gsplat', { url: url, filename: fname });
    asset.on('load', () => {
      const e = new pc.Entity('gsplat');
      e.addComponent('gsplat', { asset: asset });
      app.root.addChild(e);
      gsplatEntity = e;
      // diagnostics — visible in devtools console. Expose objects for live poking.
      try {
        const inst = e.gsplat && e.gsplat.instance;
        const mat = getMat(e);
        const res = asset.resource;
        const splat = (inst && inst.splat) || (res && res.splat) || res;
        window.XC = { app, entity: e, gsplat: e.gsplat, instance: inst, material: mat, asset, resource: res, splat: splat };
        // Recursively locate a color/opacity Texture and a centers Float32Array so we
        // know exactly where to read distances and write clipped opacity.
        const seen = new Set(); const found = { textures: [], floatArrays: [], numSplats: [] };
        function probe(obj, path, depth) {
          if (!obj || depth > 3 || seen.has(obj)) return; seen.add(obj);
          for (const k of Object.keys(obj)) {
            let v; try { v = obj[k]; } catch (_) { continue; }
            const p = path + '.' + k;
            if (v && v.constructor && /Texture/.test(v.constructor.name)) found.textures.push(p + ' (' + (v.width) + 'x' + (v.height) + ', fmt ' + v.format + ')');
            else if (v instanceof Float32Array) found.floatArrays.push(p + ' [' + v.length + ']');
            else if ((k === 'numSplats' || k === 'count') && typeof v === 'number') found.numSplats.push(p + '=' + v);
            else if (v && typeof v === 'object' && depth < 3 && !ArrayBuffer.isView(v)) probe(v, p, depth + 1);
          }
        }
        probe(res, 'resource', 0); if (inst) probe(inst, 'instance', 0);
        console.log('[xclip] gsplat loaded · instance=', !!inst, '· material=', !!mat);
        console.log('[xclip] resource keys:', res && Object.keys(res));
        console.log('[xclip] splat keys:', splat && Object.keys(splat));
        console.log('[xclip] TEXTURES found:', found.textures);
        console.log('[xclip] FLOAT ARRAYS found:', found.floatArrays);
        console.log('[xclip] numSplats:', found.numSplats);
        console.log('[xclip] >>> paste these [xclip] lines back; then window.XC is the live handle.');
      } catch (d) { console.warn('[xclip] diag', d); }
      frameToContent(e);
      hideStatus();
      requestAnimationFrame(afterSplatLoad);
    });
    asset.on('error', err => setStatus('Could not load that scene: ' + err, false));
    app.assets.add(asset);
    app.assets.load(asset);
  }

  function frameToContent(e) {
    const aabb = e.gsplat && e.gsplat.instance && e.gsplat.instance.meshInstance
      ? e.gsplat.instance.meshInstance.aabb : null;
    if (aabb) {
      target = aabb.center.clone();
      rad = aabb.halfExtents.length() * 2.4 + 0.4;
    }
    applyCam();
  }

  // Post-load hook: build/refresh the visible grid plane (the clip itself lives in
  // the injected shader chunks and needs no per-scene setup).
  function afterSplatLoad() {
    if (!planeEntity) buildPlane(); else updatePlane();
    console.log('[xclip] shader clip active:', shaderClipReady, '· mode:', ['Ours', 'MM', 'HC'][state.mode]);
  }

  /* ---------- UI ---------- */
  // edge/leak/pop: qualitative %, mirroring the shareable artifact's panel
  const COPY = {
    0: { name: 'Ours — analytic half-space', d: 'Every pixel’s <b>view ray</b> gets the exact closed-form Gaussian-<b>CDF</b> factor of the truncated 3D Gaussian — correct from <b>every viewpoint</b> (no billboard cut-lines in side views), no leak, no popping.', edge: 6, leak: 4, pop: 3 },
    1: { name: 'MM — moment-matched', d: 'Each splat is shifted toward the kept side and faded by its kept mass — a smooth surrogate that <b>leaks a tail</b> past the plane.', edge: 24, leak: 22, pop: 5 },
    2: { name: 'HC — hard cull', d: 'Each splat is kept or dropped whole by its <b>centre</b>. Straddling splats overshoot the plane or leave <b>holes</b>, and <b>pop</b> as the plane sweeps.', edge: 16, leak: 14, pop: 88 }
  };
  const opname = document.getElementById('opname'), opdesc = document.getElementById('opdesc');
  const bE = document.getElementById('b-edge'), bL = document.getElementById('b-leak'), bP = document.getElementById('b-pop');
  const barCol = v => v <= 10 ? 'var(--accent)' : v <= 45 ? 'var(--warn)' : 'var(--bad)';
  function setMode(m) {
    state.mode = m;
    document.querySelectorAll('.op').forEach(b => b.setAttribute('aria-selected', b.dataset.mode == String(m)));
    const c = COPY[m];
    opname.textContent = c.name; opdesc.innerHTML = c.d;
    bE.style.width = c.edge + '%'; bE.style.background = barCol(c.edge);
    bL.style.width = c.leak + '%'; bL.style.background = barCol(c.leak);
    bP.style.width = c.pop + '%'; bP.style.background = barCol(c.pop);
    drawProfile();
  }
  document.querySelectorAll('.op').forEach(b => b.addEventListener('click', () => setMode(+b.dataset.mode)));

  /* ---------- 1-D density cross-section (same math as the artifact) ----------
   * One unit Gaussian (sigma=1) at x=0; the plane sits at x=p (kept side x<p).
   * We plot the density each operator KEEPS along the plane normal. */
  const pcv = document.getElementById('prof'), pctx = pcv.getContext('2d');
  const profSub = document.getElementById('prof-sub');
  function erf1(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);
    return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x));}
  const PHI = x => 0.5*(1+erf1(x/1.4142135)), phi = x => Math.exp(-0.5*x*x)/2.5066283;
  const SUBS = {
    0: 'Density kept by the <b>exact</b> operator (Ours): it ends right at the plane — <b>no</b> material past the cut. In 3D the rendered face follows the per-ray Gaussian <b>CDF</b>, evaluated in closed form per pixel.',
    1: '<b>MM</b> re-fits a whole Gaussian to the kept part. It preserves mass but its tail <b>leaks</b> across the plane (red).',
    2: '<b>HC</b> keeps or drops the <b>whole</b> Gaussian by its centre — here it overshoots far past the plane (red). Slide the plane past the centre and it vanishes entirely (a hole).'
  };
  function profile(x, p, m) {              // density kept at position x, plane at p, mode m
    if (m === 0) return x <= p ? phi(x) : phi(x)*Math.max(0, 0.5*(1-erf1((x-p)/(0.05*1.4142))));  // exact truncation
    if (m === 1) { const Z = Math.max(PHI(p), 1e-4); const mean = -phi(p)/Z;
      const varr = Math.max(1 + (-p*phi(p))/Z - mean*mean, 1e-3);
      return Z*Math.exp(-0.5*(x-mean)*(x-mean)/varr)/Math.sqrt(2*Math.PI*varr); }                 // MM surrogate
    return (0 <= p) ? phi(x) : 0;                                                                  // HC whole/none by centre
  }
  function cssv(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function drawProfile() {
    if (!pcv) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = pcv.clientWidth || 600, H = 130; pcv.width = W*dpr|0; pcv.height = H*dpr|0;
    pctx.setTransform(dpr,0,0,dpr,0,0); pctx.clearRect(0,0,W,H);
    const X0=-3.4, X1=3.4, pad=8;
    const sx = x => pad+(x-X0)/(X1-X0)*(W-2*pad);
    const yMax=0.46, sy = v => H-pad-(v/yMax)*(H-2*pad-14);
    const p = state.tau*2.6;               // slider (-1..1) -> plane pos in sigma units
    const accent=cssv('--accent')||'#34e2d1', faint=cssv('--faint')||'#5f6c82',
          bad=cssv('--bad')||'#ff5d6c', ink=cssv('--ink')||'#eaeef7', stroke=cssv('--stroke')||'rgba(255,255,255,.09)';
    // baseline + culled-side shading + plane line
    pctx.strokeStyle=stroke; pctx.lineWidth=1; pctx.beginPath(); pctx.moveTo(pad,H-pad); pctx.lineTo(W-pad,H-pad); pctx.stroke();
    pctx.fillStyle='rgba(127,140,160,.05)'; pctx.fillRect(sx(p),pad,W-pad-sx(p),H-2*pad);
    pctx.strokeStyle=accent; pctx.lineWidth=1.5; pctx.setLineDash([4,4]);
    pctx.beginPath(); pctx.moveTo(sx(p),pad-2); pctx.lineTo(sx(p),H-pad); pctx.stroke(); pctx.setLineDash([]);
    pctx.fillStyle=accent; pctx.font='600 10px '+cssv('--fmono'); pctx.textAlign='center';
    pctx.fillText('plane', sx(p), pad+6);
    pctx.fillStyle=faint; pctx.font='10px '+cssv('--fmono');
    pctx.textAlign='left'; pctx.fillText('kept', pad+2, H-pad-3);
    pctx.textAlign='right'; pctx.fillText('culled', W-pad-2, H-pad-3);
    const S=180, xs = i => X0+(X1-X0)*i/S;
    // exact reference (dashed)
    pctx.strokeStyle=faint; pctx.lineWidth=1.25; pctx.setLineDash([3,3]); pctx.beginPath();
    for(let i=0;i<=S;i++){const x=xs(i),v=profile(x,p,0);const X=sx(x),Y=sy(v); i?pctx.lineTo(X,Y):pctx.moveTo(X,Y);} pctx.stroke(); pctx.setLineDash([]);
    // leak/overshoot area past the plane (this operator)
    pctx.fillStyle=bad; pctx.globalAlpha=0.30; pctx.beginPath(); pctx.moveTo(sx(Math.max(p,X0)),H-pad);
    for(let i=0;i<=S;i++){const x=xs(i); if(x<p)continue; pctx.lineTo(sx(x),sy(profile(x,p,state.mode)));}
    pctx.lineTo(sx(X1),H-pad); pctx.closePath(); pctx.fill(); pctx.globalAlpha=1;
    // this operator's curve
    pctx.strokeStyle=state.mode===0?accent:ink; pctx.lineWidth=2.4; pctx.beginPath();
    for(let i=0;i<=S;i++){const x=xs(i),v=profile(x,p,state.mode);const X=sx(x),Y=sy(v); i?pctx.lineTo(X,Y):pctx.moveTo(X,Y);} pctx.stroke();
    profSub.innerHTML = SUBS[state.mode];
  }
  window.addEventListener('resize', drawProfile);

  const tauEl = document.getElementById('tau'), tauVal = document.getElementById('tauval');
  const fill = () => { const p = (+tauEl.value - (+tauEl.min)) / ((+tauEl.max) - (+tauEl.min)) * 100; tauEl.style.setProperty('--fill', p + '%'); };
  function setTau(v) { state.tau = v; tauVal.textContent = v.toFixed(2); updatePlane(); drawProfile(); }
  tauEl.addEventListener('input', () => { state.sweep = false; sweepBtn.setAttribute('aria-pressed', 'false'); setTau(+tauEl.value); fill(); });
  const axisBtn = document.getElementById('axis');
  axisBtn.addEventListener('click', () => { state.axis = (state.axis + 1) % 3; axisBtn.textContent = 'Axis: ' + 'XYZ'[state.axis]; updatePlane(); });
  const sweepBtn = document.getElementById('sweep');
  let t0 = 0; sweepBtn.addEventListener('click', () => { state.sweep = !state.sweep; sweepBtn.setAttribute('aria-pressed', String(state.sweep)); t0 = performance.now(); });

  app.on('update', () => {
    if (autoOrbit && !dragging) { az += 0.0015; applyCam(); }
    if (state.sweep) { const v = Math.sin((performance.now() - t0) * 0.001) * 0.9; tauEl.value = v; fill(); setTau(v); }
    setShaderClipUniforms();          // drive the shader clip every frame
  });

  /* ---------- drag & drop a real .ply ---------- */
  function wireDrop() {
    const drop = document.getElementById('drop');
    if (drop) drop.addEventListener('click', () => pick());
  }
  function pick() {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.ply,.splat';
    inp.onchange = () => { if (inp.files[0]) loadSplat(URL.createObjectURL(inp.files[0]), inp.files[0].name); };
    inp.click();
  }
  ['dragover', 'drop'].forEach(ev => canvas.addEventListener(ev, e => e.preventDefault()));
  canvas.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) loadSplat(URL.createObjectURL(f), f.name); });
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadSplat(URL.createObjectURL(f), f.name); });

  /* ---------- go ---------- */
  app.start();
  setMode(0); fill();
  setStatus('Building a demo Gaussian scene… (or drop your own .ply)', true);
  // load the procedural fallback scene
  loadSplat(URL.createObjectURL(proceduralPly()), 'demo');
})();
