#
# CCA-geometry dGS (`dgs-cca`): the position shift uses the fully whitened,
# anisotropy-aware regression operator from the conditional-coordinates
# analysis,
#     M = S^{1/2} K P^{1/2},          dmu = lambda * M @ (q - mu_v),
# so the displacement obeys the exact metric bound
#     || S^{-1/2} dmu || <= lambda * ||K||_2 * || P^{1/2} (q-mu_v) ||.
#
# NOT a new parameter: the existing _v_12_direction tensor ([N, 3C]) stores K
# (same shape / optimizer group / PLY column / init as v_12). Only the slice
# interpretation changes, so dgs-cca loads plain dGS checkpoints and warm-starts
# unchanged. A spectral-norm clamp ||K||_2 <= kappa is applied after each
# optimizer step (analogous to the gabor frequency clamp) to realise the bound.
#
# Factor sources (all already stored, C=3):
#   S^{1/2} = diag(sqrt(get_scaling))          spatial metric (per axis)
#   P^{1/2} : symmetric sqrt of P = L L^T      query precision metric
#             (L from L_22_inv, exp-activated diagonal)
# K = reshape(_v_12_direction, [N,3,3]).  With unconstrained K this spans the
# same regression operators as dGS; the value is in measuring/clamping the
# budget in the primitive's own metric rather than isotropically in pixels.
#
# Pure-torch slice (no CUDA). Reduces to dGS-white's query-side whitening when
# S = I; the extra piece is the S^{1/2} output-side whitening + symmetric P^{1/2}
# (vs L, which differ by a rotation K absorbs, but the symmetric root makes the
# bound exact and rotation-equivariant).
#

import torch

from scene.gaussian_model_dgs_whitened import GaussianModel as WhitenedDGS, unpack_L


def _sym_sqrt_spd(P):
    """Symmetric PSD square root of a batch of SPD 3x3 matrices via eigh.
    Kept for the verification harness; the model uses L directly (see below)."""
    evals, evecs = torch.linalg.eigh(P)
    s = evals.clamp_min(1e-12).sqrt()
    return evecs @ torch.diag_embed(s) @ evecs.transpose(1, 2)


class GaussianModel(WhitenedDGS):
    """dGS with the M = S^{1/2} K P^{1/2} regression parameterization."""

    KAPPA = 3.0   # spectral-norm bound on K (rad/sigma-scale, matches the
                  #  whitened-frequency bound used in the gabor study)

    def slice_gaussian_full_method(self, query, lambda_opc=None):
        assert self.input_dim == 6, "dgs-cca torch slice implements C=3 only"
        if lambda_opc is None:
            lambda_opc = self.default_lambda_opc
        x = query.detach() - self.get_cond_mean                  # [N,3]
        L = unpack_L(self.get_L_22_inv)                          # [N,3,3]
        # opacity attention (identical to dGS): exp(-lam_opc ||L^T x||^2)
        z = torch.einsum('nji,nj->ni', L, x)
        attention = torch.exp(-lambda_opc * (z * z).sum(-1, keepdim=True))
        if not self.use_view_dependent_pos:
            return self._xyz, attention

        S_half = self.get_scaling.sqrt()                         # [N,3] diag of S^{1/2}
        # P^{1/2} := L (Cholesky root of P = L L^T). Any root works because K
        # absorbs the rotation between roots; L is what unpack_L already gives
        # and needs no eigendecomposition (the symmetric-root eigh is
        # ill-conditioned at init, where all L diagonals are equal -> repeated
        # eigenvalues -> cuSOLVER non-convergence; L sidesteps it entirely).
        P_half = L                                               # [N,3,3]
        # NOTE: _v_12_direction is used RAW here (K), without the
        # F.normalize/mean-scale of get_v_12 — the S^{1/2} factor now carries
        # the spatial scaling that get_v_12 baked in, and the KAPPA clamp bounds
        # K's spectral norm instead.
        K = self._v_12_direction.reshape(-1, 3, 3)               # [N,3,3]
        if self.use_opacity_pos_decouple:
            lam = self._lambda_view
        else:
            lam = self.lambda_activation(self._lambda_view)
        # dmu = lam * S^{1/2} K P^{1/2} x
        Px = torch.einsum('nij,nj->ni', P_half, x)               # P^{1/2} x
        KPx = torch.einsum('nij,nj->ni', K, Px)
        dmu = lam.unsqueeze(-1) * S_half * KPx                   # S^{1/2} applied as diag
        return self._xyz + dmu, attention

    @torch.no_grad()
    def clamp_cca_kappa(self):
        """Rescale each K so its spectral norm <= KAPPA (realises the bound)."""
        if self._v_12_direction.numel() == 0:
            return
        K = self._v_12_direction.reshape(-1, 3, 3)
        sv = torch.linalg.matrix_norm(K, ord=2)                  # [N] top singular value
        factor = sv.clamp_max(self.KAPPA) / sv.clamp_min(1e-12)
        self._v_12_direction.mul_(factor.reshape(-1, 1).repeat(1, 9))
