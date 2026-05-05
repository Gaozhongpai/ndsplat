We thank the reviewer for the thoughtful follow-up. We address each point below.

---

## Q1 follow-up: dGS vs alternative kernels from an application perspective

> *"From the application aspect, dGS/NDGS is just another way to enhance the representation power... it is still interesting to compare them directly."*

We agree with the reviewer's perspective. While the formulations are orthogonal (conditioning parameterization vs kernel shape), from an application standpoint they share the same goal: enhancing representation power for view-dependent and dynamic rendering.

We first compare dGS and UBS directly on shared static benchmarks (both using MCMC densification):

| Dataset | Method | PSNR | SSIM | LPIPS | FPS |
|---|---|---|---|---|---|
| NeRF Synthetic | dGS | 34.34 | 0.973 | 0.026 | **594** |
| | UBS | **34.92** | **0.975** | **0.026** | 346 |
| Mip-NeRF 360 | dGS | 27.67 | 0.794 | 0.227 | **201** |
| | UBS | **28.66** | **0.840** | **0.184** | 100 |
| 6DGS-PBR | dGS | **40.34** | 0.976 | **0.038** | **551** |
| | UBS | 40.10 | **0.979** | 0.039 | 290 |

UBS achieves higher quality on NeRF Synthetic and Mip-NeRF 360, while dGS is comparable on 6DGS-PBR. dGS is consistently **~2x faster** in rendering. This motivates combining both: UBS's Beta kernel quality with dGS's efficient conditioning. We call this **dBS** (direct Beta Splatting, dGS + UBS):

**UBS** (current):

Distance: $d_i = \tanh((\hat{L}_q^{-1} \delta)_i^2 / 2)$

Opacity: $o_{cond} = o \prod_{i=1}^{C} (1 - d_i)^{4\beta_i}$

Position: $\mu_{x|q} = \mu_x + \Sigma_{xq} \Sigma_q^{-1} diag(\beta_q) \delta$

Covariance: $\Sigma_{x|q} = \Sigma_x - \Sigma_{xq} \Sigma_q^{-1} diag(\beta_q) \Sigma_{qx}$

Rendering: $\sigma(x, q) = B(x; \mu_{x|q}, \Sigma_{x|q}, b_x) \cdot o_{cond}$

**dBS** (dGS + UBS) replaces covariance-derived conditioning with dGS's direct parameterization, keeping UBS's Beta kernel:

Distance: $d_i = \tanh((L^\top \delta)_i^2 / 2)$

Opacity: $o_{cond} = o \prod_{i=1}^{C} (1 - d_i)^{4\beta_i}$

Position: $\mu_{cond} = \mu_p + V_{pq} \cdot diag(\beta_q) \cdot V_{qq} \cdot \delta$

Covariance: $\Sigma_x$ (no correction)

Rendering: $\sigma(x, q) = B(x; \mu_{cond}, \Sigma_x, b_x) \cdot o_{cond}$

Key differences: 
- $L$ is directly parameterized as a Cholesky precision factor, replacing $\hat{L}_q^{-1}$, no inversion needed; 
- Position reuses per-dimension $\beta_q$ as coupling strength, replacing covariance regression with $\Sigma_q^{-1}$. This unifies opacity bandwidth and position coupling into a single learned parameter per dimension; 
- 3D covariance uses query-independent $\Sigma_x$ instead of conditional $\Sigma_{x|q}$ -- no correction needed, as validated in our dGS paper.

**Benefits of dBS:**
- UBS's per-dimension Beta bandwidth control (diffuse vs specular, static vs dynamic)
- dGS's efficient direct conditioning (5-6x faster slicing kernel)
- Unified $\beta_q$: controls both opacity bandwidth and position coupling per dimension
- Interpretable decomposition: spatial shape via $b_x$, conditioning via $\beta_q$
- No conditional covariance correction (further simplification over UBS)

---

## Q3 & Q4: Including discussions in the paper

> *"The discussions should also be included in the paper somewhere if possible."*

We agree. We will include the dGS-O vs dGS usage guidance (Q2), the analysis of why dGS achieves better quality via learnable Lambda (Q3), and the Gaussian count discussion with MCMC equal-budget evidence (Q4) in the revised paper, specifically in an expanded Discussion/Analysis section and the Limitations section.

---

## New SOTA: dBS (dGS + UBS)

We have now implemented and evaluated dBS. Results on the full NeRF Synthetic benchmark (8 scenes, 30K iterations, MCMC, 300K cap):

| Dataset | Method | PSNR | SSIM | LPIPS | FPS | Train (min) |
|:--------|:-------|-----:|-----:|------:|----:|------------:|
| NeRF Synthetic | **dBS (Ours)** | **34.96** | **0.975** | **0.026** | **423.82** | **8.4** |
| | UBS | 34.85 | 0.974 | 0.026 | 315.47 | 9.1 |
| Mip-NeRF 360 | **dBS (Ours)** | **28.74** | **0.842** | **0.184** | **158.84** | **24.4** |
| | UBS | 28.63 | 0.842 | 0.184 | 76.94 | 28.6 |

dBS outperforms UBS on both benchmarks (+0.11 dB on NeRF Synthetic, +0.11 dB on Mip-NeRF 360), with 1.3-2.1x faster rendering and 8-15% shorter training. This validates the dBS formulation proposed during our discussion and confirms that the dGS direct parameterization is not limited to the Gaussian kernel; it generalizes to the Beta kernel as well.
