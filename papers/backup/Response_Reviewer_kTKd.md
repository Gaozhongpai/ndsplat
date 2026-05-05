We thank Reviewer kTKd for the constructive feedback. We address each concern with new experiments and revisions.

---

## W1: Relaxing probabilistic consistency

> *"The method no longer preserves the strict semantics of a single underlying conditional Gaussian model."*

The departure from strict probabilistic semantics is a **deliberate design choice** validated empirically: dGS matches or exceeds N-DGS across all six datasets (+0.53 dB NeRF Synthetic, +0.97 dB Mip-NeRF 360, +0.46 dB D-NeRF, +0.09 dB NeRF-DS). The relaxation provides three benefits:

1. **Independent control.** In N-DGS, opacity and position conditioning are rigidly coupled through the shared covariance -- a Gaussian cannot modulate its opacity strongly without also affecting its position shift (and vice versa). dGS decouples these effects, allowing each to be optimized independently.

2. **Learnable Lambda.** Per-Gaussian Lambda enables adaptive interpolation between no coupling (Lambda=0) and full coupling (Lambda=1). Our new ablation (W3) confirms learned Lambda outperforms both fixed extremes, validating that per-Gaussian adaptivity is strictly more expressive than N-DGS's rigid coupling.

3. **Dropping covariance correction.** dGS uses Sigma_pp directly instead of Sigma_cond = Sigma_pp - Sigma_pq Sigma_qq^{-1} Sigma_pq^T. We observe no quality degradation -- the position displacement mechanism already captures the dominant view/time-dependent geometric effects, and the covariance adjustment provides diminishing returns.

We will discuss this in the revised paper.

---

## W2: Efficiency claim; "expensive cubic-complexity" overstated

> *"Full dGS is often similar to or slower than N-DGS in training time... C is small (3 or 4)."*

We will revise the O(C^3) language. The speedup comes from dGS requiring **fewer operations per Gaussian**, not just eliminating the inversion:

| Operation | N-DGS | dGS | dGS-O |
|---|---|---|---|
| Invert Sigma_qq (CxC) | Yes | No | No |
| Regression (3xC x CxC) | Yes | No | No |
| **Cov correction (3xC x Cx3)** | **Yes** | **No** | **No** |
| Position shift | regr x delta | V x Lambda x delta | No |
| Opacity quadratic form | delta^T Sigma^{-1} delta | \|\|L^T delta\|\|^2 | \|\|L^T delta\|\|^2 |

The **covariance correction** is the most expensive single step -- a 3xC times Cx3 matrix product per Gaussian. N-DGS performs three matrix multiplications plus one inversion; dGS replaces all with a fused triangular matvec that fits entirely in GPU registers.

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

Slicing is **5-6x faster** (dGS) and **8-11x faster** (dGS-O), consistently across scales. End-to-end rendering speedup (up to 2.66x) is lower because rasterization time is shared.

**On training time:** Our primary efficiency claim is **rendering speed**, not training time. Training includes many components beyond slicing (backpropagation, densification). dGS-O consistently reduces training time (6-9% across datasets). Full dGS has comparable training time to N-DGS -- note that N-DGS tends to over-prune (e.g., `stump`: 644K Gaussians / 23.32 dB vs dGS's 3343K / 26+ dB), so its faster training partly reflects fewer primitives at the cost of quality, not inherently faster computation.

---

## W3: Ablation on Lambda (learned vs fixed)

> *"Results do not disentangle learned Lambda vs fixed Lambda=1 vs Lambda=0."*

We have conducted this ablation:

| Config | NeRF Syn PSNR | 6DGS-PBR PSNR |
|---|---|---|
| Lambda=0 (= dGS-O) | 33.84 | 36.54 |
| Lambda=1 (full coupling) | 33.78 | 37.31 |
| **Lambda learned (= dGS)** | **33.91** | **37.75** |

**Learned Lambda outperforms both fixed extremes** on both datasets. On 6DGS-PBR (view-dependent scenes): +1.21 dB over Lambda=0, +0.44 dB over Lambda=1. The optimal coupling is scene-dependent: on diffuse scenes (NeRF Syn) Lambda=0 slightly beats Lambda=1; on specular scenes (PBR) Lambda=1 wins by +0.77 dB. This validates that per-Gaussian adaptive coupling is the right design. On D-NeRF (dynamic), the effect is even stronger: dGS-O (Lambda=0) drops -2.21 dB vs N-DGS, while learned Lambda exceeds N-DGS by +0.46 dB. We will include these results in the revised paper.

---

## UBS comparison

> *"Universal Beta Splatting... its strong performance makes this paper less convincing."*

UBS proposes a different kernel (Beta) for N-D splatting; dGS focuses on **parameterization of conditioning effects**. These are orthogonal -- dGS can be combined with UBS's Beta kernel, potentially bringing the same efficiency gains. We will discuss this in the revised paper.
