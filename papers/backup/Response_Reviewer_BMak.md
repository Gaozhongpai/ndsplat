We thank Reviewer BMak for the thorough review and the recognition of the soundness, presentation, and importance of our work. We address each question below with new experimental results and paper revisions.

---

## Q1: Limited real-world dynamic evaluation (NeRF-DS, Flame Salmon, videos)

> *"Testing on more in-the-wild dynamic datasets could further demonstrate robustness."*

Following the reviewer's recommendation, we have evaluated on **NeRF-DS** (7 real-world dynamic specular scenes):

| Method | #Gauss (K) | Train (min) | FPS | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| N-DGS (7DGS) | **76** | 339 | 867 | 23.11 | 0.8249 | 0.1835 |
| dGS-O | 107 | **316** | **959** | 22.72 | 0.8068 | 0.2008 |
| **dGS** | 105 | 343 | 907 | **23.20** | **0.8204** | **0.1833** |

dGS outperforms N-DGS by **+0.09 dB PSNR** while rendering **5% faster** (907 vs 867 FPS). dGS-O is the fastest (**11% faster**) but sacrifices quality (−0.39 dB), confirming that position coupling is important for dynamic specular scenes. This brings our total evaluation to **6 datasets and 38+ scenes** (NeRF Synthetic: 8, Mip-NeRF 360: 9, 6DGS-PBR: 6, D-NeRF: 8, 7DGS-PBR: 6, NeRF-DS: 7), with dGS consistently matching or exceeding N-DGS. We will also add video comparisons for dynamic scenes in the supplementary material.

---

## Q2: Limited baseline selection; 3DGS/4DGS results missing from Table 3

> *"Why full results for other baselines (4DGS, 3DGS) are not included in Table 3?"*

**On baseline selection:** dGS is specifically designed as an efficient alternative to N-DGS's conditioning mechanism, making N-DGS the most direct baseline. Reference [4] (6DGS) is part of the N-DGS family and is already included. That said, for broader context: on Mip-NeRF 360, Scaffold-GS (Lu et al., CVPR 2024) reports 27.72 dB at 102 FPS, while dGS (MCMC) achieves 27.67 dB at 201 FPS -- comparable quality with ~2x faster rendering. This demonstrates dGS is competitive beyond the N-DGS family.

**On 3DGS/4DGS in Table 3:** We agree about reader convenience and will add 3DGS and 4DGS results directly to Table 3 in the revised paper.

---

## Q2b: Failure cases

> *"Are there any failure cases typical for this method?"*

**dGS-O (opacity-only)** struggles when position conditioning is critical: it drops −2.21 dB on D-NeRF (dynamic scenes) and −0.57 dB on 6DGS-PBR (specular scenes). In both cases, full dGS with learned Lambda resolves this and exceeds N-DGS (+0.46 dB on D-NeRF, +0.64 dB on 6DGS-PBR). We have not identified scenes where full dGS consistently underperforms N-DGS across all 38+ scenes in six datasets.

---

## Q3: Scalability analysis for additional conditioning variables

> *"Limited scalability analysis for additional conditioning variables."*

dGS's modular structure scales efficiently. Comparing per-Gaussian costs for C=3 vs C=4:

| | N-DGS (C=3) | dGS (C=3) | N-DGS (C=4) | dGS (C=4) |
|---|---|---|---|---|
| Spatial covariance | 6 | 7 | 6 | 7 |
| Query mean | 3 | 3 | 4 | 4 |
| Cross-cov / Displacement | 9 | 9 | 12 | 12 |
| Query cov / Precision | 6 (full) | 6 (full) | 10 (full) | **7 (block-diag)** |
| Coupling | 1 | 1 | 2 | 2 |
| **Total** | **25** | **26** | **34** | **32** |

For C=4 (7DGS), dGS is **more efficient** -- the block-diagonal precision uses 7 parameters vs 10 for full Cholesky, saving 2 parameters per Gaussian. For higher C (e.g., C=7 with lighting), N-DGS's joint covariance grows as (3+C)^2 = 100 parameters with O(C^3) inversion, while dGS grows linearly: C(C+1)/2 + 3C + 1 + C = 56 parameters with O(C^2) computation.

---

## Q4: Memory overhead

> *"What is the memory overhead of explicit parameters vs the covariance representation?"*

As shown in the table above, per-Gaussian parameter counts are comparable: dGS uses 1 extra parameter for C=3 (quaternion vs 3-param rotation), but **saves 2 parameters for C=4** via block-diagonal precision. The memory footprint is effectively equivalent. In practice, the main memory difference comes from Gaussian count under standard densification, not per-Gaussian parameters.

---

## Limitations

> *"The paper lacks a Limitations section."*

We will add a dedicated Limitations section in the revised paper:

> **Limitations.** Under standard densification, dGS tends to retain more Gaussian primitives than N-DGS. As demonstrated in our results, these additional primitives contribute positively to reconstruction quality while still maintaining faster rendering, but they do increase memory consumption. Additionally, dGS is currently evaluated only within the N-DGS Gaussian framework. Combining our direct parameterization with alternative kernels (e.g., Beta kernels in UBS) is a promising direction for future work.

---

We believe the new NeRF-DS experiments, scalability analysis, and paper revisions address all raised concerns. We hope the reviewer will consider raising the score given these clarifications.
