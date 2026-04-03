We thank the reviewer for the follow-up. We respectfully disagree that the improvements are marginal and would like to clarify.

---

## On the characterization of improvements as "nearly negligible"

> *"The improvements over the baseline are too marginal, nearly negligible."*

**1. The improvements are consistent across all benchmarks.**

dGS improves over N-DGS on every dataset evaluated, spanning 38+ scenes:

| Dataset | Scenes | PSNR gain | Setting |
|---|---|---|---|
| NeRF Synthetic | 8 | +0.53 dB | Static, diffuse |
| **Mip-NeRF 360** | 9 | +0.97 dB | Static, real-world |
| 6DGS-PBR | 6 | +0.64 dB | Static, specular |
| D-NeRF | 8 | +0.46 dB | Dynamic, synthetic |
| 7DGS-PBR | 6 | +0.28 dB | Dynamic, specular |
| NeRF-DS | 7 | +0.09 dB | Dynamic, real specular |

Zero regressions across six diverse benchmarks. The +0.97 dB on Mip-NeRF 360 (the most challenging real-world benchmark) is not negligible by any standard in this field.

**2. The primary contribution is rendering efficiency.**

We emphasize that dGS is fundamentally an *efficiency* contribution. The slicing kernel is **5-6x faster** (dGS) and **8-11x faster** (dGS-O), yielding up to **2.66x end-to-end speedup**. Even if quality were identical to N-DGS, a method that renders at 279 FPS vs 147 FPS (Mip-NeRF 360) while maintaining the same quality would be a meaningful contribution for real-time applications. dGS delivers this speedup *and* improves quality on top.

As shown in our rebuttal, dGS (MCMC) achieves 27.67 dB at 201 FPS on Mip-NeRF 360, comparable to Scaffold-GS's 27.72 dB at 102 FPS (CVPR 2024), demonstrating competitiveness with ~2x faster rendering against a fundamentally different approach.

**3. Theoretical justification exists in the supplementary.**

Our supplementary (Appendix A) provides formal analysis showing dGS's displacement parameterization offers tighter theoretical bounds than N-DGS:

- N-DGS displacement bound: $\sqrt{C}(i-1 + s_i)$, with cross-term offset and dimension-ordering asymmetry from the implicit Cholesky structure.
- dGS displacement bound: $\bar{s}$ (mean spatial scale), with no offset and uniform treatment across all spatial dimensions.

This is not an arbitrary relaxation of probabilistic consistency. dGS replaces implicit Cholesky-derived coupling with explicit, tighter-bounded parameterization. The ablation removing spatial scaling confirms this: -2.59 dB on NeRF Synthetic (Table 5).

**4. Quality improvements and efficiency gains are complementary.**

We believe the reviewer's original concerns (W1: speedup source, W2: comparison scope, W3: primitive count) were addressed in our rebuttal with controlled CUDA benchmarks, Scaffold-GS comparison, and MCMC equal-budget evidence. The quality gains, while secondary to the efficiency contribution, are consistently positive and complement the core speedup. A method that is both faster and better should not be penalized for not being *sufficiently* better.
