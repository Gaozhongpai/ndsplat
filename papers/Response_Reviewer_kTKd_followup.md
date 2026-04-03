We thank the reviewer for the continued engagement. We respectfully address the remaining concern.

---

## On the sufficiency of empirical gains without theoretical support

> *"The reported improvements seem relatively modest, and such small gains are not sufficient to convince me without theoretical support to loose probabilistic consistency."*

We would like to highlight two points:

**1. The improvements are consistent, not modest.**

dGS improves over N-DGS on *every single dataset* evaluated: six datasets spanning 38+ scenes across static, dynamic, and specular settings:

| Dataset | Scenes | PSNR gain | Setting |
|---|---|---|---|
| NeRF Synthetic | 8 | +0.53 dB | Static, diffuse |
| Mip-NeRF 360 | 9 | +0.97 dB | Static, real-world |
| 6DGS-PBR | 6 | +0.64 dB | Static, specular |
| D-NeRF | 8 | +0.46 dB | Dynamic, synthetic |
| 7DGS-PBR | 6 | +0.28 dB | Dynamic, specular |
| NeRF-DS | 7 | +0.09 dB | Dynamic, real specular |

A method that consistently outperforms its baseline across all six diverse benchmarks without a single regression is not a marginal result: it demonstrates that the parameterization is strictly better in practice. The +0.97 dB on Mip-NeRF 360 (the most challenging real-world benchmark) is a substantial improvement in this field.

**2. The primary contribution is rendering efficiency, not just quality.**

Even if dGS achieved *identical* quality to N-DGS, the **5-6x faster slicing kernel** (up to 2.66x end-to-end speedup) would represent a significant contribution. Real-time rendering speed is critical for practical deployment of Gaussian Splatting. dGS achieves both faster rendering *and* better quality, making the efficiency-quality tradeoff strictly favorable.

**3. Theoretical support exists in the supplementary.**

We appreciate the reviewer's emphasis on theoretical grounding. In fact, our supplementary material (Appendix A, "Comparison to N-DGS Displacement Parameterization") provides a formal analysis of the displacement parameterization:

- **N-DGS** displacement emerges *implicitly* from the Cholesky structure with bound $\sqrt{C}(i-1 + s_i)$. Cross-terms from spatial correlations can allow disproportionately large displacements for small Gaussians and introduce asymmetry across spatial dimensions depending on dimension ordering.

- **dGS** *explicitly* parameterizes displacement with tight bound $\bar{s}$ (mean spatial scale), ensuring proper scaling across all Gaussian sizes with uniform treatment of all spatial dimensions.

This is not merely an engineering simplification: dGS provides **tighter theoretical bounds** on displacement magnitude. N-DGS's implicit coupling introduces a constant offset $(i-1)\sqrt{C}$ that can dominate for small Gaussians, and an asymmetry where identical spatial scales produce different displacement bounds depending on their position in the Cholesky ordering. dGS eliminates both artifacts through direct parameterization with mean-scale normalization.

Furthermore, our ablation confirms this is critical: removing spatial scaling ($\bar{s}$) degrades PSNR by **-2.59 dB** on NeRF Synthetic (Table 5), validating that the explicit coupling is essential, not just convenient.

**In summary:** dGS does not arbitrarily discard probabilistic consistency: it replaces implicit Cholesky-derived coupling with explicit, tighter-bounded parameterization that is theoretically justified (Appendix A), empirically validated (6 datasets, 38+ scenes, zero regressions), and practically impactful (5-6x faster slicing). We believe consistent improvement across all benchmarks combined with formal displacement analysis constitutes sufficient theoretical and empirical support.
