# NDSplat — Project Website

Project page for **NDSplat: N-Dimensional Splatting**, a unified research line for
N-dimensional Gaussian and Beta splatting.

The site introduces the research line and links to each paper's own project page:

| Work | Venue | Page |
|------|-------|------|
| DDGS-CT — Direction-Disentangled Gaussian Splatting | NeurIPS 2024 | https://gaozhongpai.github.io/ddgs/ |
| 6DGS — Enhanced Direction-Aware Gaussian Splatting | ICLR 2025 | https://gaozhongpai.github.io/6dgs/ |
| 7DGS — Unified Spatial-Temporal-Angular Gaussian Splatting | ICCV 2025 | https://gaozhongpai.github.io/7dgs/ |
| Universal Beta Splatting (UBS) | ICLR 2026 | https://rongliu-leo.github.io/universal-beta-splatting/ |
| Render-FM — Feedforward Volumetric Rendering | ECCV 2026 | https://gaozhongpai.github.io/renderfm/ |
| XClipGS — Exact Half-Space Clipping for Medical Volumes | arXiv | https://gaozhongpai.github.io/XClipGS/ |
| FactorSplat — Appearance-Controllable Gaussian Proxies for Medical Volume Rendering | under review | https://gaozhongpai.github.io/FactorSplat/ |
| *d*GS / *d*BS — Direct Conditional Parameterization | NeurIPS 2026 | https://gaozhongpai.github.io/dgs/ |

## Structure

```
ndsplat-web/
├── index.html              # single-page site
├── assets/video/           # explainer video (Manim, narrated) + poster
└── assets/
    ├── css/style.css       # styles
    └── js/main.js          # nav, scroll reveal, copy-to-clipboard
```

The site is fully static with no build step — open `index.html` directly, or serve the
folder with any static server:

```bash
python3 -m http.server 8000   # then visit http://localhost:8000
```

## Deploying with GitHub Pages

Push to the `Gaozhongpai/ndsplat` repository, then in **Settings → Pages** select the
branch and root folder. The page will be served at
`https://gaozhongpai.github.io/ndsplat/`.
