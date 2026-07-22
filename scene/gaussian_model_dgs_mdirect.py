#
# Independent-M dGS (`dgs-mdirect`): the position regression matrix M is a FREE
# parameter, decoupled from the opacity precision P.
#
#     dmu = lambda * (s_bar * Theta_M) @ (q - mu_v),   Theta_M = normalize(v_12)
#     attention = exp(-lambda_o (q-mu_v)^T P (q-mu_v))   (P = L L^T, unchanged)
#
# Standard dGS ties the shift to precision: M = v_12 D_Lambda P, so a sharply
# view-selective primitive (large P) also gets a large position swing. The
# conditional-coordinates theorem shows (Sigma_cond, M, P) is a complete chart
# for arbitrary joint Gaussians, so M need NOT have that P-coupled form to stay
# probabilistically coherent. This ablation makes opacity selectivity and
# position regression genuinely INDEPENDENT (the logical endpoint of directly
# parameterizing the sliced quantities). It is the one variant that changes the
# model's inductive bias rather than only the coordinate chart, so it is the
# one that could actually move PSNR (up OR down: the shared-P coupling may be a
# useful prior).
#
# Reuses _v_12_direction as Theta_M (F.normalize'd, x mean-scale via get_v_12),
# so NO new per-primitive parameters, same PLY schema, warm-start compatible.
# Pure-torch slice (no CUDA); reduces the drive from P@delta to just delta.
#

import torch

from scene.gaussian_model_dgs_whitened import GaussianModel as WhitenedDGS, unpack_L


class GaussianModel(WhitenedDGS):
    """dGS with a free (P-independent) position regression matrix M."""

    def slice_gaussian_full_method(self, query, lambda_opc=None):
        assert self.input_dim == 6, "dgs-mdirect torch slice implements C=3 only"
        if lambda_opc is None:
            lambda_opc = self.default_lambda_opc
        x = query.detach() - self.get_cond_mean                  # [N,3]
        L = unpack_L(self.get_L_22_inv)                          # [N,3,3]
        z = torch.einsum('nji,nj->ni', L, x)                    # opacity unchanged
        attention = torch.exp(-lambda_opc * (z * z).sum(-1, keepdim=True))
        if not self.use_view_dependent_pos:
            return self._xyz, attention
        # M = s_bar * Theta_M applied DIRECTLY to delta (no P coupling).
        # get_v_12 already returns normalize(v_12_direction) * mean_scale.
        M = self.get_v_12.reshape(-1, 3, 3)
        if self.use_opacity_pos_decouple:
            lam = self._lambda_view
        else:
            lam = self.lambda_activation(self._lambda_view)
        dmu = lam.unsqueeze(-1) * torch.einsum('nic,nc->ni', M, x)
        return self._xyz + dmu, attention
