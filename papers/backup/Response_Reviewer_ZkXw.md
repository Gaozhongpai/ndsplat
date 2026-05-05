We thank Reviewer ZkXw for the detailed review and recognition of our fair CUDA baseline implementation and the value of learnable Lambda.

---

## W1 / Q1: True source of the speedup

> *"The true source of the speedup is unclear... A simple controlled experiment would definitively clarify this."*

The speedup comes from dGS requiring **fundamentally fewer operations per Gaussian**, not just eliminating a small matrix inversion:

| Operation | N-DGS | dGS | dGS-O |
|---|---|---|---|
| Invert Sigma_qq (CxC) | Yes | No | No |
| Regression (3xC x CxC) | Yes | No | No |
| **Cov correction (3xC x Cx3)** | **Yes** | **No** | **No** |
| Position shift | regr x delta | V x Lambda x delta | No |
| Opacity quadratic | delta^T Sigma^{-1} delta | \|\|L^T delta\|\|^2 | \|\|L^T delta\|\|^2 |

The **covariance correction** is the most expensive single step -- a 3xC times Cx3 matrix product producing 9 output values per Gaussian. N-DGS performs three matrix multiplications plus one inversion; dGS replaces all of this with a fused triangular matvec + scaled displacement that fits entirely in GPU registers with no intermediate memory allocations.

We benchmarked raw CUDA kernels (C++ forward, no autograd) on A100:

| N Gauss | N-DGS (ms) | dGS-O (ms) | dGS (ms) | N-DGS/dGS-O | N-DGS/dGS |
|---|---|---|---|---|---|
| **C=3** | | | | | |
| 500K | 0.44 | 0.05 | 0.09 | **9.5x** | **5.1x** |
| 1M | 0.85 | 0.11 | 0.16 | **8.0x** | **5.2x** |
| 2M | 1.63 | 0.20 | 0.31 | **8.1x** | **5.3x** |
| **C=4** | | | | | |
| 500K | 0.77 | 0.07 | 0.16 | **11.1x** | **4.8x** |
| 1M | 1.40 | 0.15 | 0.23 | **9.4x** | **6.2x** |

Slicing is **5-6x faster** (dGS) and **8-11x faster** (dGS-O). Both methods call the same C++ entry point with everything else identical, isolating the slicing computation. End-to-end speedup (up to 2.66x) is lower due to shared rasterization. This directly addresses the suggested controlled experiment -- the benchmark compares the full N-DGS kernel vs the full dGS kernel with all other pipeline components held constant. The consistent speedup across C=3/C=4 and 500K-2M Gaussians confirms this is structural, not a measurement artifact.

---

## W2: Comparison scope

> *"Comparing against only one method family is insufficient."*

N-DGS is the most direct baseline since dGS targets its conditioning. That said, we provide broader context:

1. **Scaffold-GS** (Lu et al., CVPR 2024) uses anchor-based MLPs for view dependence. On Mip-NeRF 360, Scaffold-GS reports 27.72 dB at 102 FPS, while dGS (MCMC) achieves **27.67 dB at 201 FPS** -- comparable quality with **~2x faster rendering**, demonstrating competitiveness beyond the N-DGS family.

2. **Deformable 3DGS and SC-GS** model dynamics through deformation fields -- a different paradigm from 4DGS/7DGS's conditional slicing which does not require a separate deformation network.

3. **3DGS and 4DGS baselines** will be added to Table 3 in the revised paper.

---

## W3 / Q2: Primitive count inflation

> *"dGS uses far more primitives... stump: N-DGS 644K vs dGS 3343K."*

1. **Under equal budgets, dGS still wins.** MCMC (Table 2) caps all methods at the same count. dGS exceeds N-DGS by +0.79 dB on Mip-NeRF 360, +0.30 dB on NeRF Synthetic, +0.28 dB on 6DGS-PBR. **Quality gains are not from more primitives.**

2. **N-DGS over-prunes on challenging scenes.** On `stump`, N-DGS prunes to 644K / 23.32 dB; dGS retains 3343K / 26+ dB -- a ~3 dB gap. The additional primitives meaningfully contribute to quality, not compensate for lower expressiveness.

3. **dGS is faster despite more primitives.** On Mip-NeRF 360, dGS-O uses 86% more primitives yet renders 89% faster (279 vs 147 FPS). Per-primitive compute savings dominate the cost of additional primitives. This also highlights that N-DGS's lower primitive count is not necessarily an advantage -- it reflects over-pruning that hurts reconstruction quality.

We will add a discussion of this tradeoff in the Limitations section.

---

## Q3: Alternative approaches

> *"Could you add comparisons with a non-high-dimensional method?"*

As noted in W2, Scaffold-GS achieves 27.72 dB at 102 FPS on Mip-NeRF 360, while dGS (MCMC) reaches 27.67 dB at 201 FPS -- comparable quality with ~2x faster rendering, despite using a fundamentally different approach (conditional slicing vs MLP-based view dependence).

---

## Limitations

> *"The paper lacks a Limitations section."*

We will add a dedicated Limitations section:

> **Limitations.** Under standard densification, dGS tends to retain more Gaussian primitives than N-DGS. As demonstrated in our results, these additional primitives contribute positively to reconstruction quality while still maintaining faster rendering, but they do increase memory consumption. Additionally, dGS is currently evaluated only within the N-DGS Gaussian framework. Combining our direct parameterization with alternative kernels (e.g., Beta kernels in UBS) is a promising direction for future work.
