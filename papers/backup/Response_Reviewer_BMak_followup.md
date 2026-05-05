We thank the reviewer for considering our rebuttal. We respectfully address the concern about marginal improvement on real-world scenes.

---

## On "marginal improvement over 7DGS" on real-world scenes

> *"The provided results on real world scenes show only marginal improvement over 7DGS."*

**1. The improvements are consistent across all benchmarks, not just real-world scenes.**

dGS improves over N-DGS on every dataset evaluated, spanning 38+ scenes with zero regressions:

| Dataset | Scenes | PSNR gain | Setting |
|---|---|---|---|
| NeRF Synthetic | 8 | +0.53 dB | Static, diffuse |
| Mip-NeRF 360 | 9 | +0.97 dB | Static, real-world |
| 6DGS-PBR | 6 | +0.64 dB | Static, specular |
| D-NeRF | 8 | +0.46 dB | Dynamic, synthetic |
| 7DGS-PBR | 6 | +0.28 dB | Dynamic, specular |
| NeRF-DS | 7 | +0.09 dB | Dynamic, real specular |

We acknowledge that NeRF-DS shows the smallest gain (+0.09 dB). However, NeRF-DS is a particularly challenging benchmark where all methods cluster tightly (23.11 vs 23.20 dB). Importantly, the largest gain is on **Mip-NeRF 360** (+0.97 dB), which is both real-world and the most widely used benchmark in this field. A method that never regresses across six diverse benchmarks demonstrates a strictly better parameterization.

**2. The primary contribution is rendering efficiency, not just quality.**

We want to emphasize that dGS is fundamentally an efficiency contribution. Even on the NeRF-DS scenes where quality gains are modest, dGS renders **5% faster** (907 vs 867 FPS). More broadly:

- Slicing kernel: **5-6x faster** (dGS), **8-11x faster** (dGS-O)
- End-to-end: up to **2.66x speedup**
- dGS (MCMC) on Mip-NeRF 360: 27.67 dB at **201 FPS** vs Scaffold-GS's 27.72 dB at 102 FPS

Even if dGS achieved identical quality to N-DGS, the rendering speedup alone would be valuable for practical deployment. dGS delivers this speedup while also improving quality across all benchmarks.

**3. Theoretical support for the parameterization exists in the supplementary.**

Our supplementary (Appendix A, "Comparison to N-DGS Displacement Parameterization") provides formal analysis showing that dGS offers tighter displacement bounds than N-DGS. Specifically, N-DGS's implicit Cholesky coupling has bound $\sqrt{C}(i-1 + s_i)$ with cross-term offset and dimension-ordering asymmetry, while dGS provides a tight bound of $\bar{s}$ (mean spatial scale) with uniform treatment across all dimensions. The ablation confirms this is essential: removing spatial scaling degrades PSNR by **-2.59 dB** on NeRF Synthetic (Table 5).

We believe the combination of consistent quality improvements (6 datasets, zero regressions), substantial rendering speedup (5-6x kernel, 2.66x end-to-end), and theoretical justification (Appendix A) constitutes a meaningful contribution. We hope the reviewer will reconsider the score given these clarifications.

---

## New SOTA: dBS (dGS + UBS)

Since the rebuttal, we have applied dGS to UBS (MCMC), yielding dBS (dGS + UBS):

| Dataset | Method | PSNR | SSIM | LPIPS | FPS | Train (min) |
|:--------|:-------|-----:|-----:|------:|----:|------------:|
| NeRF Synthetic | **dBS (Ours)** | **34.96** | **0.975** | **0.026** | **423.82** | **8.4** |
| | UBS | 34.85 | 0.974 | 0.026 | 315.47 | 9.1 |
| Mip-NeRF 360 | **dBS (Ours)** | **28.74** | **0.842** | **0.184** | **158.84** | **24.4** |
| | UBS | 28.63 | 0.842 | 0.184 | 76.94 | 28.6 |

dBS outperforms UBS on both benchmarks (+0.11 dB on NeRF Synthetic, +0.11 dB on Mip-NeRF 360), with 1.3-2.1x faster rendering and 8-15% shorter training. This confirms that the dGS direct parameterization is not limited to the Gaussian kernel; it generalizes to the Beta kernel as well, improving both quality and speed.
