Dear Area Chair,

We respectfully request a closer examination of the reviewer concerns, which we believe do not accurately reflect the contributions of this work.

## The "marginal improvement" characterization is not supported by the evidence

Three reviewers (kTKd, BMak, ZkXw) adopted the language of "marginal" or "negligible" improvements after the discussion phase. However, dGS improves over N-DGS on **every single dataset** evaluated, spanning 6 benchmarks and 38+ scenes with zero regressions:

| Dataset | Scenes | PSNR gain | Setting |
|---|---|---|---|
| NeRF Synthetic | 8 | +0.53 dB | Static, diffuse |
| Mip-NeRF 360 | 9 | +0.97 dB | Static, real-world |
| 6DGS-PBR | 6 | +0.64 dB | Static, specular |
| D-NeRF | 8 | +0.46 dB | Dynamic, synthetic |
| 7DGS-PBR | 6 | +0.28 dB | Dynamic, specular |
| NeRF-DS | 7 | +0.09 dB | Dynamic, real specular |

A +0.97 dB improvement on Mip-NeRF 360, the most widely used real-world benchmark in this field, is not negligible. More importantly, a method that never regresses across six diverse benchmarks demonstrates a strictly superior parameterization.

## The primary contribution is rendering efficiency, which reviewers underweighted

The reviewers focused almost exclusively on PSNR improvements while underweighting the core contribution: rendering efficiency. As shown in our revised Table 1, dGS eliminates 3 of 5 per-Gaussian operations required by N-DGS during conditional slicing:

| Operation | N-DGS | dGS-O | dGS |
|---|---|---|---|
| Invert $\Sigma_{qq}$ | $\Sigma_{qq}^{-1}$ | — | — |
| Regression matrix $K$ | $\Sigma_{pq}\Sigma_{qq}^{-1}$ | — | — |
| Covariance correction | $K \Sigma_{pq}^\top$ | — | — |
| Position shift | $K \delta$ | — | $V_{pq} \tilde{V}_{qq} \delta$ |
| Opacity | $\delta^\top \Sigma_{qq}^{-1} \delta$ | $\|L^\top \delta\|^2$ | $\|L^\top \delta\|^2$ |

No matrix inversion, no regression matrix, no covariance correction. This directly translates to measured speedups on A100:

- Slicing kernel: **5-6x faster** (dGS), **8-11x faster** (dGS-O)
- End-to-end rendering: up to **2.66x speedup** (279 vs 147 FPS on Mip-NeRF 360)
- dGS (MCMC) achieves 27.67 dB at 201 FPS on Mip-NeRF 360, comparable to Scaffold-GS's 27.72 dB at 102 FPS (CVPR 2024), with ~2x faster rendering

Even if dGS achieved identical quality to N-DGS, eliminating three of five per-Gaussian operations—yielding a 5-6x faster slicing kernel with up to 2.66x end-to-end speedup—would constitute a significant contribution for real-time Gaussian Splatting applications. dGS achieves this speedup while *also* improving quality across all benchmarks.

## Theoretical analysis exists but was overlooked

Reviewer kTKd specifically stated "small gains are not sufficient without theoretical support." However, our supplementary (Appendix A) provides formal analysis of the displacement parameterization:

- **N-DGS**: displacement bound $\sqrt{C}(i-1 + s_i)$ with cross-term offset and dimension-ordering asymmetry from implicit Cholesky structure
- **dGS**: tight displacement bound $\bar{s}$ (mean spatial scale) with no offset and uniform dimensional treatment

This proves dGS provides tighter theoretical bounds on displacement, not an arbitrary relaxation. The ablation validates this: removing spatial scaling degrades PSNR by -2.59 dB (Table 5). We believe this addresses the concern about lacking theoretical support.

## Summary of contributions

1. **Rendering efficiency**: Eliminates 3 of 5 per-Gaussian operations (inversion, regression matrix, covariance correction), yielding 5-6x faster slicing kernel (8-11x for dGS-O) and up to 2.66x end-to-end speedup, critical for real-time deployment
2. **Consistent quality gains**: improvements on all 6 datasets (38+ scenes), zero regressions, up to +0.97 dB
3. **Theoretically justified**: formal displacement bound analysis (Appendix A) proving tighter bounds than N-DGS
4. **New ablations**: Lambda learned vs fixed, CUDA kernel benchmarks, MCMC equal-budget comparisons, NeRF-DS evaluation
5. **Future direction**: concrete dBS formulation combining dGS efficiency with UBS Beta kernels (developed during reviewer discussion with hVDr)

We believe a method that is both faster and better, with theoretical justification and comprehensive evaluation, constitutes a clear contribution to the community. We respectfully ask the AC to weigh the full scope of evidence presented above.
