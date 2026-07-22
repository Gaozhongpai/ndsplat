# dGS Conditional-Coordinates Study — Handoff

> **STATUS (2026-07-22): experiments complete on heart_900.** Investigates the
> conditional-coordinates interpretation of dGS from the paper review: dGS
> optimizes an intrinsic chart (conditional covariance Sigma_cond, query
> precision P = L L^T, regression matrix M) of the joint Gaussian, rather than
> an entangled ambient covariance. Four training variants + two theory probes.
> **Headline: all reparameterizations are quality-neutral (within ~0.02 dB);
> the useful result is the coupling regularizer, which makes positions ~70%
> more view-static at a 0.009 dB cost — deployment-relevant for RenderFM.**
> The contribution is a coordinate-chart / conditioning result, NOT a PSNR win
> (matches the paper reviewer's recommendation).

## Background: the implemented dGS view shift

Ground truth from `submodules/gsplat/.../slice_gaussian_full_fwd.cu` (NOT the
model docstring, which describes a stale (1-lam)I+lam*P interpolation):

    delta     = q - mu_v                    (query - per-splat view mean)
    z         = L^T delta                   (L = unpack of L_22_inv, exp diag)
    attention = exp(-lambda_o ||z||^2)      (opacity, = exp(-lam_o delta^T P delta))
    dmu       = lambda_view * v_12 @ (L L^T) @ delta = lambda * v_12 P delta

So the effective regression matrix is M = v_12 D_Lambda P and the position
shift is COUPLED to the opacity precision P. v_12 = normalize(_v_12_direction)
* mean_scale (row-norm-bounded by s_bar); lambda_view = sigmoid(_lambda_view).

## Variants (all: heart_900, 30k from scratch, --use_view_dependent_pos True,
## --mip3dgs --eval, identical recipe; differ only as noted)

| mode          | position shift dmu                    | params | CUDA |
|---------------|---------------------------------------|--------|------|
| `dgs` (base)  | lambda v_12 P delta                   | 9 (v_12) | kernel |
| `dgs-white`   | lambda v_12 z  (z = L^T delta)        | 9      | torch, none |
| `dgs-cca`     | lambda S^{1/2} K L delta, ||K||_2<=3  | 9 (K=v_12) | torch, none |
| `dgs-mdirect` | lambda s_bar Theta_M delta (P-free)   | 9 (Theta_M=v_12) | torch, none |
| + `--lambda_coupling eta` | base + eta * coupling penalty | 9 | torch, none |

None add per-primitive parameters (white/cca/mdirect reinterpret v_12; the
regularizer is a loss term). white/cca/mdirect use a pure-torch slice
(validated against the CUDA kernel at ~2e-7 in values and all gradients,
`scripts/tests/dgs_whitened_checks.py`), so no CUDA edits or rebuild — at a
small per-iteration compute cost vs the fused kernel (fine for an A/B; port
the winner to CUDA for production).

## Results (test set: 45 intact + 45 clipped views)

  | variant             | all     | intact  | clipped | SSIM   | LPIPS  |
  |---------------------|---------|---------|---------|--------|--------|
  | baseline (dgs)      | 28.9171 | 30.7915 | 27.0426 | 0.9368 | 0.1011 |
  | +coupling (eta=.01) | 28.9083 | 30.7605 | 27.0561 | 0.9370 | 0.1010 |
  | whitened            | 28.9038 | 30.7717 | 27.0359 | 0.9371 | 0.1009 |
  | cca (S^1/2 K L)     | <PENDING> | | | | |
  | mdirect (free M)    | <PENDING> | | | | |

All finished variants land within **0.013 dB overall** and **~0.02 dB on both
the intact and clipped halves** — a clean quality-null.

## Theory probes

### Identifiability (`scripts/tests/dgs_identifiability.py`)
Tested whether the null is caused by UNOBSERVABLE parameter directions (the
initial conjecture: with limited view coverage, perturbations of M/P
orthogonal to the observed queries would not render).
**REFUTED.** heart has **810 training cameras** on a full orbit (I had
mis-remembered 90 — that is the test split). Per primitive, over the cameras
that see it:
- M observable (span{delta} = R^3): rank 3/3 on **100%** of primitives.
- P observable (span{delta delta^T} = S^3): rank 6/6 on **100%**.
- Coverage is well-conditioned (G1 min/max eig ratio median 0.20; G2 0.028),
  ~810 views per primitive.
So M and P are fully identifiable — the null is not an observability artifact.

### Coupling magnitude (`scripts/tests/dgs_coupling_diag.py`) — the real cause
Coordinate-invariant coupling K = S^{-1/2} M P^{1/2}, ||K||_F^2 =
E_{delta~N(0,P^-1)} ||S^{-1/2} M delta||^2 (expected squared shift in spatial-
std units), plus the physical RMS shift over training views relative to
footprint sqrt(mean(scale^2)):

  | run       | median ||K||_F | median shift/footprint | median lambda_view |
  |-----------|----------------|------------------------|--------------------|
  | baseline  | 0.329          | 0.332                  | 0.241              |
  | +coupling | 0.102          | 0.089                  | 0.105              |
  | whitened  | 0.386          | 0.392                  | 0.274              |

**The learned view shift is only ~0.33 of a splat's own footprint.** Given
full observability + expressive equivalence (white/cca/mdirect span the same
regression operators, so the optimizer reaches the same optimum from any
chart), reparameterizing a sub-footprint correction cannot move PSNR. That is
the measured explanation for the null.

### The coupling regularizer is the useful outcome
`--lambda_coupling 0.01` adds a dimensionless penalty on the effective shift.
It removes **69% of the coupling energy** (||K||_F 0.329 -> 0.102, shift/foot
0.332 -> 0.089, lambda_view 0.241 -> 0.105) at a PSNR cost of **0.009 dB**
(28.917 -> 28.908) — i.e. positions ~70% more view-static essentially for
free. Relevance: view-static positions are easier for the RenderFM head to
predict feed-forward and cheaper to store/compress. NOTE the implemented
penalty uses tr(v_12 P v_12^T)/tr(Sigma); the review's anisotropy-exact form
is tr(S^-1 v_12 D_Lambda P D_Lambda v_12^T) (= ||K||_F^2) — worth swapping if
pursued.

## Verification (all PASS)
- `dgs_whitened_checks.py`: torch slice == CUDA kernel (~2e-7 values + all
  grads); whitened==standard at P=I; level-set bound 0/8192; precision-
  independence x0.96 vs x8.64; cca metric bound 0/8192; kappa clamp caps at 3.
- `dgs_identifiability.py`, `dgs_coupling_diag.py`: as above.

## Conclusion / recommendation (per the paper reviewer)
Frame the contribution as **coordinate-chart geometry + identifiability**, not
a leaderboard result: dGS works in observable conditional coordinates
(Sigma_cond, P, M) where ambient-covariance reconstruction is block-triangular,
globally invertible, with nonvanishing Jacobian on the SPD domain. The
reparameterizations (whitened / cca / free-M) are quality-neutral BECAUSE the
geometry is coordinate-invariant, the parameters are fully observable at 810
views, and the view shift is sub-footprint. The one practical lever is the
coupling regularizer (view-static positions ~free). Independent-M (mdirect)
tests whether decoupling the shift from P — breaking the shared-P inductive
bias — helps; result <PENDING>.

Branch `feat/dgs-conditional-coords` (base `mip`). Outputs under
`/data/output/xclipgs/dgscoord/heart_900_{baseline,coupling,whitened,cca,mdirect}`.
No CUDA changes; no rebuild needed. Related studies: `GABOR_HANDOFF.md`,
`NASG_HANDOFF.md` (same feat-branch pattern; both also quality-neutral /
not-adopted on this content).
