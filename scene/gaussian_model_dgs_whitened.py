#
# Whitened-displacement dGS (`dgs-white`): identical parameters to dGS, but the
# view-dependent position shift uses the WHITENED query offset.
#
# Implemented dGS (ground truth = gsplat slice_gaussian_full_fwd.cu):
#     z         = L^T (q - mu_v)                 (whitened offset; L lower-tri,
#                                                 exp-activated diagonal)
#     attention = exp(-lambda_opc * ||z||^2)
#     dmu       = lambda_view * v_12 @ (L L^T) (q - mu_v)
#
# Whitened variant (conditional-coordinates analysis of dGS):
#     dmu_white = lambda_view * v_12 @ z
#
# Motivation (from the joint-Gaussian coordinate interpretation): v_12 rows are
# norm-bounded and lambda in [0,1], so on an opacity level set ||z|| <= r the
# displacement obeys ||dmu_white|| <= s_bar * r, and under the query kernel
# z ~ N(0, I) the expected squared displacement is precision-INDEPENDENT
# (<= s_bar^2). The standard form dmu = v_12 P delta grows with the eigenvalues
# of P: sharper view-dependence implies larger position swings. The whitened
# form also reuses the opacity feature z (one less matrix apply).
#
# The slice here is a pure-torch port (C=3 only) so the variant needs ZERO CUDA
# changes; the port is validated against the CUDA kernel in
# scripts/tests/dgs_whitened_checks.py (standard formula, values + grads).
# No new parameters: dgs-white loads/saves plain dGS PLYs unchanged.
#

import torch

from scene.gaussian_model_dgs import GaussianModel as DGSGaussianModel

# Row-major lower-triangle layout of L_22_inv for C=3 (see get_L_22_inv):
#   (0,0)->0, (1,0)->1, (1,1)->2, (2,0)->3, (2,1)->4, (2,2)->5
_TRIL_ROWS = torch.tensor([0, 1, 1, 2, 2, 2])
_TRIL_COLS = torch.tensor([0, 0, 1, 0, 1, 2])
_DIAG_SLOTS = torch.tensor([0, 2, 5])


def unpack_L(L_tri):
    """[N,6] row-major lower triangle -> [N,3,3] with exp-activated diagonal
    (exactly as slice_gaussian_full_fwd.cu unpacks it)."""
    n = L_tri.shape[0]
    dev = L_tri.device
    diag = _DIAG_SLOTS.to(dev)
    vals = L_tri.clone()
    vals[:, diag] = torch.exp(L_tri[:, diag])
    L = torch.zeros(n, 3, 3, device=dev, dtype=L_tri.dtype)
    L[:, _TRIL_ROWS.to(dev), _TRIL_COLS.to(dev)] = vals
    return L


def torch_slice_gaussian_full(xyz, view_mean, query, v_12, L_tri, lambda_opc,
                              lambda_view, whitened=False):
    """Pure-torch port of slice_gaussian_full (C=3), with the optional
    whitened displacement. Returns (x_cond [N,3], attention [N,1])."""
    # The CUDA kernel does not backpropagate into the query direction (it
    # returns no query gradient); detach to match, so the whitened variant
    # differs from the baseline ONLY in the displacement formula.
    x = query.detach() - view_mean                          # [N,3]
    L = unpack_L(L_tri)                                     # [N,3,3]
    z = torch.einsum('nji,nj->ni', L, x)                    # L^T x
    attention = torch.exp(-lambda_opc * (z * z).sum(-1, keepdim=True))
    if v_12 is None:
        return xyz, attention
    V12 = v_12.reshape(-1, 3, 3)
    if lambda_view is not None:
        V12 = V12 * lambda_view.reshape(-1, 1, 1)
    if whitened:
        drive = z                                           # L^T x
    else:
        drive = torch.einsum('nij,nj->ni', L, z)            # L L^T x = P x
    dmu = torch.einsum('nic,nc->ni', V12, drive)
    return xyz + dmu, attention


class GaussianModel(DGSGaussianModel):
    """dGS with the whitened view-dependent position shift (torch slice).

    Parameter set, PLY format, densification, and rendering are inherited
    unchanged — only slice_gaussian_full_method is overridden.
    """

    def slice_gaussian_full_method(self, query, lambda_opc=None):
        assert self.input_dim == 6, "dgs-white torch slice implements C=3 only"
        if lambda_opc is None:
            lambda_opc = self.default_lambda_opc
        if self.use_view_dependent_pos:
            if self.use_opacity_pos_decouple:
                lambda_view = self._lambda_view
            else:
                lambda_view = self.lambda_activation(self._lambda_view)
            v_12 = self.get_v_12
        else:
            lambda_view = None
            v_12 = None
        return torch_slice_gaussian_full(
            self._xyz, self.get_cond_mean, query, v_12, self.get_L_22_inv,
            lambda_opc, lambda_view, whitened=True)
