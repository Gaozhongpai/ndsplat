We thank Reviewer hVDr for the positive assessment and insightful questions. We address each point below.

---

## Issue 1: Figure 3 zoom-in region selection

> *"Why are only a few regions zoomed in?"*

The zoomed regions were selected to highlight areas with **visible artifacts** in the baselines, which are the most informative for quality comparison. In the revised paper, we will update the figure to include zoomed-in views of the **same regions for ground truth and dGS**, so readers can directly compare all methods against the reference. We confirm dGS performs well across the full image, not just in selected crops.

---

## Issue 2: Table layout inconsistency

> *"Table 2 puts methods vertically, while table 3 puts methods horizontally."*

We will update Table 3 in the revised paper to follow the same vertical layout as Table 2 for consistency, and add 3DGS and 4DGS baselines directly to Table 3 so all comparisons are visible in a single table.

---

## Q1: Alternative kernels (Beta Splatting, Spherical Voronoi, Textured Gaussians)

> *"Would we have a comparison against more expressive kernels?"*

dGS and these kernels address **orthogonal aspects**: dGS focuses on how conditioning effects are parameterized for N-D Gaussians; Beta/Voronoi/Textured Gaussians propose alternative 3D spatial kernels. These are complementary -- dGS's direct parameterization can be combined with alternative kernels (e.g., replacing UBS's conditioning with our Cholesky precision and displacement). We will discuss this as future work in the revised paper.

---

## Q2: dGS-O vs dGS -- which is recommended?

> *"dGS-O appears in all main tables. What is the recommended method?"*

dGS-O serves as both an ablation and a **practical method**, targeting different use cases:

| Setting | Recommended | Rationale |
|---|---|---|
| Static, speed-critical | **dGS-O** | Maximum speed (up to 2.66x over N-DGS), strong quality on diffuse scenes |
| Static, quality-critical | **dGS** | Best quality across all benchmarks (+0.53 to +0.97 dB over N-DGS) |
| Dynamic scenes | **dGS** | Lambda essential for temporal motion (dGS-O drops -2.21 dB on D-NeRF) |
| View-dependent (PBR) | **dGS** | Learned coupling recovers quality (+0.64 dB on 6DGS-PBR vs dGS-O's -0.57 dB) |

The key insight is that dGS-O provides a compelling speed-quality tradeoff when opacity modulation alone suffices, while full dGS is needed for dynamic scenes and strong view-dependent effects. We will clarify this guidance in the revised paper.

---

## Q3: Why does dGS often achieve better quality than N-DGS?

> *"One might expect dGS to be less expressive than N-DGS."*

Two factors contribute:

1. **Per-Gaussian Lambda provides adaptive coupling.** N-DGS uses rigid coupling -- every Gaussian's position shift is dictated by the same covariance structure. In dGS, each Gaussian learns its own coupling strength via Lambda. Diffuse surfaces set Lambda near 0 (no position shift), specular surfaces near 1. Our ablation confirms learned Lambda outperforms both Lambda=0 and Lambda=1 on all tested datasets, showing adaptive coupling is more expressive.

2. **Independent optimization of opacity and position.** In N-DGS, both are derived from the same covariance matrix, so gradient updates to improve one effect can inadvertently affect the other. dGS parameterizes them independently, allowing each to converge to its optimum without interference.

In short, dGS trades theoretical expressiveness (covariance correction) for practical flexibility (independent, learnable conditioning), and the latter proves more valuable empirically across all six evaluated datasets.

---

## Q4: Increased Gaussian count

> *"dGS and dGS-O produce more Gaussians in some settings."*

1. **Under equal budgets, dGS still wins.** MCMC (Table 2) caps all methods at the same Gaussian count; dGS exceeds N-DGS by +0.79 dB on Mip-NeRF 360. Quality gains are not from more primitives.

2. **N-DGS over-prunes on challenging scenes.** On `stump`, N-DGS prunes to 644K / 23.32 dB while dGS reaches 3343K / 26+ dB -- a ~3 dB gap showing the extra primitives meaningfully contribute.

3. **dGS is faster despite more primitives.** dGS-O uses 86% more yet renders 89% faster (279 vs 147 FPS on Mip-NeRF 360). Per-primitive compute savings dominate; N-DGS's lower count reflects over-pruning, not efficiency.

---

## Limitations

> *"The paper does not discuss limitations directly."*

We will add a dedicated Limitations section:

> **Limitations.** Under standard densification, dGS tends to retain more Gaussian primitives than N-DGS. As demonstrated in our results, these additional primitives contribute positively to reconstruction quality while still maintaining faster rendering, but they do increase memory consumption. Additionally, dGS is currently evaluated only within the N-DGS Gaussian framework. Combining our direct parameterization with alternative kernels (e.g., Beta kernels in UBS) is a promising direction for future work.
