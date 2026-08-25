"""FactorSplat: functional TF-conditioned appearance on opacity-only dGS.

Geometry follows the XClipGS backbone exactly: view-dependent position is
disabled by the experiment launcher, while the existing dGS view query still
scales opacity. A sampled label/intensity transfer function is encoded once per
camera condition and applied through low-rank per-Gaussian color/opacity factors.
"""

import os

import numpy as np
import torch
from torch import nn

from scene.gaussian_model_dgs import GaussianModel as DGSModel


class _TFOnlyCamera:
    """Minimal stand-in carrying just the preset index: the conditional
    branches are view-independent, so no camera geometry is needed."""

    def __init__(self, tf_index):
        self.tf_index = tf_index


class GaussianModel(DGSModel):
    TF_ALPHA_EPS = 1.0 / 255.0     # floor for the log-ratio opacity coordinate
    def __init__(self, sh_degree: int, input_dim: int = 6,
                 use_view_dependent_pos: bool = False,
                 use_opacity_pos_decouple: bool = False,
                 l_22_inv_init_scale: float = 2.0, lambda_init: float = -1.2,
                 lambda_opc: float = 0.35, direct_unrestricted: bool = False,
                 tf_rank: int = 8, tf_hidden: int = 64, tf_samples: int = 32,
                 tf_color_scale: float = 0.25, tf_opacity_scale: float = 4.0,
                 tf_condition_color: bool = True,
                 tf_condition_opacity: bool = True,
                 tf_color_sh_degree: int = 1,
                 tf_encoder_type: str = "functional",
                 tf_embedding_fallback: str = "nearest",
                 tf_aware_prune: bool = True,
                 tf_use_lookup: bool = False,
                 tf_veg_packed: bool = False,
                 tf_veg_max_gaussians: int = 0,
                 tf_encoder_pooled: bool = False,
                 tf_encoder_local: bool = False,
                 tf_global_context_rank: int = 0,
                 tf_opacity_alpha_only: bool = False,
                 tf_opacity_alpha_identity_gate: bool = True,
                 tf_opacity_train_envelope: bool = False,
                 tf_opacity_residual_clip: float = 0.0,
                 tf_refresh_label_locked: bool = False,
                 tf_opacity_log_ratio: bool = False,
                 tf_log_ratio_encoder: bool = False,
                 tf_lookup_mode: str = "joint",
                 tf_lookup_bins: int = 64,
                 tf_lookup_color_scale: float = 1.0,
                 tf_lookup_opacity_scale: float = 4.0,
                 tf_exact_visibility_gate: bool = True,
                 tf_soft_visibility_gate: bool = False,
                 tf_gate_removed_mass: float = 0.5):
        super().__init__(
            sh_degree=sh_degree,
            input_dim=input_dim,
            use_view_dependent_pos=use_view_dependent_pos,
            use_opacity_pos_decouple=use_opacity_pos_decouple,
            l_22_inv_init_scale=l_22_inv_init_scale,
            lambda_init=lambda_init,
            lambda_opc=lambda_opc,
            direct_unrestricted=direct_unrestricted,
        )
        if use_view_dependent_pos:
            raise ValueError("FactorSplat uses the XClipGS opacity-only backbone; "
                             "set --use_view_dependent_pos False")
        self.tf_rank = int(tf_rank)
        self.tf_hidden = int(tf_hidden)
        self.tf_samples = int(tf_samples)
        self.tf_color_scale = float(tf_color_scale)
        self.tf_opacity_scale = float(tf_opacity_scale)
        self.tf_condition_color = bool(tf_condition_color)
        self.tf_condition_opacity = bool(tf_condition_opacity)
        if not 0 <= int(tf_color_sh_degree) <= sh_degree:
            raise ValueError(f"tf_color_sh_degree must be in [0, sh_degree={sh_degree}]")
        self.tf_color_sh_degree = int(tf_color_sh_degree)
        # SH color coefficients conditioned by the residual (DC + lower bands).
        self.tf_color_coeffs = (self.tf_color_sh_degree + 1) ** 2
        if tf_encoder_type not in ("functional", "embedding"):
            raise ValueError(f"tf_encoder_type must be functional|embedding, got {tf_encoder_type}")
        if tf_embedding_fallback not in ("nearest", "zero"):
            raise ValueError(f"tf_embedding_fallback must be nearest|zero, got {tf_embedding_fallback}")
        self.tf_encoder_type = tf_encoder_type
        self.tf_embedding_fallback = tf_embedding_fallback
        self._tf_train_rows = []
        self._tf_nearest_train = None
        self.tf_aware_prune = bool(tf_aware_prune)
        self.tf_use_lookup = bool(tf_use_lookup)
        # Adapted VEG reference.  Each Gaussian keeps a fixed categorical
        # region and learns one bounded within-region intensity coordinate;
        # DC color and opacity are read from that region's TF curve.  Fixing
        # the region prevents optimization from crossing discontinuous packed
        # label blocks.  This excludes factors/encoder/lookup.
        self.tf_veg_packed = bool(tf_veg_packed)
        self.tf_veg_max_gaussians = int(tf_veg_max_gaussians)
        if self.tf_veg_max_gaussians < 0:
            raise ValueError("tf_veg_max_gaussians must be >= 0")
        if self.tf_veg_packed and (tf_condition_color or tf_condition_opacity
                                   or tf_use_lookup):
            raise ValueError("tf_veg_packed excludes the FactorSplat branches")
        self._tf_veg_v = None            # [N,1] raw logit -> within-region bin
        self._tf_veg_label = None        # [N] fixed TF-bank region column
        self._tf_veg_has_support = None  # [N] descriptor contains foreground
        self._tf_baked_state = None       # pre-bake tensors + branch flags
        self._tf_baked_index = None       # preset currently baked in, if any
        if tf_lookup_mode not in ("joint", "separable"):
            raise ValueError(f"tf_lookup_mode must be joint|separable, got {tf_lookup_mode}")
        # Label-order-invariant encoder: bias-free phi per TF CURVE, mean pool
        # over labels, then psi. Works for any label count without a positional
        # code, and keeps z_{T0} = 0 exactly. Preferred over a fixed per-label
        # axis: the numeric label carries no physical meaning, it only routes a
        # curve to a region, and the packed lookup already does that routing.
        self.tf_encoder_pooled = bool(tf_encoder_pooled)
        self.tf_encoder_psi = None
        # LOCAL functional encoder: phi is applied to the TF *delta table*
        # (L x B x 4 -> L x B x r), then each Gaussian's packed (label, bin)
        # samples gather and average those codes:
        #   z_{i,T} = (1/K_i) sum_k phi(R_T(l_ik,h_ik) - R_T0(l_ik,h_ik)).
        # phi never sees a label id, costs one small table pass per preset, and
        # is bias-free so z_{i,T0} = 0 exactly. Shareable across scenes.
        self.tf_encoder_local = bool(tf_encoder_local)
        # Concatenate a local material code with a compact global TF summary
        # while keeping the total factor rank fixed. This tests non-local
        # context without increasing per-Gaussian factor storage.
        self.tf_global_context_rank = int(tf_global_context_rank)
        if not 0 <= self.tf_global_context_rank < self.tf_rank:
            raise ValueError("tf_global_context_rank must be in [0, tf_rank)")
        if self.tf_global_context_rank and not self.tf_encoder_local:
            raise ValueError("tf_global_context_rank requires tf_encoder_local")
        if self.tf_global_context_rank and self.tf_encoder_type != "functional":
            raise ValueError("global TF context requires the functional encoder")
        self.tf_local_rank = self.tf_rank - self.tf_global_context_rank
        self.tf_context_encoder = None
        self.tf_context_encoder_psi = None
        self._tf_context_descriptors = None
        self.tf_opacity_alpha_only = bool(tf_opacity_alpha_only)
        self.tf_opacity_alpha_identity_gate = bool(
            tf_opacity_alpha_identity_gate)
        self.tf_opacity_train_envelope = bool(tf_opacity_train_envelope)
        self._tf_opacity_envelope = None
        self.tf_opacity_residual_clip = float(tf_opacity_residual_clip)
        if self.tf_opacity_residual_clip < 0:
            raise ValueError("tf_opacity_residual_clip must be >= 0")
        self.tf_refresh_label_locked = bool(tf_refresh_label_locked)
        # Opacity conditioning in LOG-RATIO coordinates: the logit is log-odds,
        # so ell = log((a_T+eps)/(a_T0+eps)) makes the authored family
        # near-linear -- gamma becomes (gamma-1)*log a, scale becomes log s,
        # hide becomes ~-6 (subsuming exact visibility gating), reveal ~+6.
        # With the gain initialised at 1 the model STARTS at the exact
        # multiplicative rule sigma(o+ell) ~ sigma(o)*(a_T/a_T0) (low-alpha
        # regime) and learns deviations where the renderer's response is not
        # proportional. The additive delta stays in use for the color channel.
        self.tf_opacity_log_ratio = bool(tf_opacity_log_ratio)
        # Feed the log-ratio coordinate to the ENCODER as well, so the learned
        # residual reasons in the same coordinates as the physical lookup. Without
        # it phi sees only the additive delta, whose opacity channel is ~0 exactly
        # where the multiplicative change is large (a faint sample needing a 21x
        # amplification has delta_alpha ~ 0.19), so the correction term is
        # poorly conditioned in the regime the lookup was moved to log space to
        # fix. Changes the encoder input width 4 -> 5.
        self.tf_log_ratio_encoder = bool(tf_log_ratio_encoder)
        if self.tf_encoder_local and self.tf_encoder_pooled:
            raise ValueError("tf_encoder_local and tf_encoder_pooled are exclusive")
        if self.tf_opacity_alpha_only and not self.tf_encoder_local:
            raise ValueError("tf_opacity_alpha_only requires tf_encoder_local")
        self.tf_lookup_mode = tf_lookup_mode
        if self.tf_encoder_local and self.tf_lookup_mode != "joint":
            raise ValueError("tf_encoder_local requires packed joint descriptors")
        self.tf_lookup_bins = int(tf_lookup_bins)
        self.tf_lookup_color_scale = float(tf_lookup_color_scale)
        self.tf_lookup_opacity_scale = float(tf_lookup_opacity_scale)
        self.tf_exact_visibility_gate = bool(tf_exact_visibility_gate)
        self.tf_soft_visibility_gate = bool(tf_soft_visibility_gate)
        # Gate a primitive when >= this fraction of its descriptor mass lies on
        # removed labels. 1.0 reproduces the keep-if-any-survives rule; 0.5 is
        # a majority-mask rule; ~0 removes any primitive touching a hidden label.
        self.tf_gate_removed_mass = float(tf_gate_removed_mass)
        self._tf_label_visible = None    # [T, L] authored IsVisible flags
        self._tf_bank_lookup = None      # [T, L, B, 4] raw RGBA, downsampled bins
        self._tf_bank_log_alpha = None   # [T, L, B] log(alpha + TF_ALPHA_EPS)
        self._tf_gather_op = None        # cached sparse [N, L*B] contraction
        self._tf_gather_sig = None       # fingerprint of the descriptors it encodes
        self._tf_base_index = 0
        self._tf_label_ids = None
        # Separable mode: dense label distribution + intensity histogram.
        self._tf_lookup_p = None         # [N, L] per-Gaussian label distribution
        self._tf_lookup_q = None         # [N, B] per-Gaussian intensity histogram
        # Joint mode (packed, lossless): each window voxel is one flat id
        # l*B + b (int16 storage; uint16 in the npz), valid entries first;
        # uniform weights 1/count are reconstructed at lookup time.
        self._tf_lookup_ids = None       # [N, K] int16 flat (label, bin) ids
        self._tf_lookup_counts = None    # [N] uint8 valid-sample counts
        self._tf_lookup_w = None         # [N, K] per-sample weights, or None
        self._refresh_id_vol = None       # train-time refresh grid (label*S+bin)
                                         # (None => uniform 1/K_i, legacy npz)
        self._tf_lookup_gain = None      # learned global RGBA gain (4,)

        self.tf_encoder = None
        self._tf_descriptors = None
        self._tf_ids = []
        self._tf_color_factors = torch.empty(0)
        self._tf_opacity_factors = torch.empty(0)

    @property
    def tf_factors_active(self):
        return self.tf_condition_color or self.tf_condition_opacity

    @property
    def tf_lookup_ready(self):
        return self._tf_lookup_ids is not None or self._tf_lookup_p is not None

    def set_tf_bank(self, bank):
        rgba = np.asarray(bank["rgba"], dtype=np.float32)
        if rgba.ndim != 4 or rgba.shape[-1] != 4:
            raise ValueError(f"expected TF bank rgba [T,L,K,4], got {rgba.shape}")
        if self.tf_use_lookup or self.tf_encoder_local or self.tf_veg_packed:
            # The lookup consumes the bank as exported (raw RGB + alpha): color
            # deltas come from the RGB channels, opacity deltas from alpha, so
            # premultiplying here would double-count alpha edits in the color path.
            raw = rgba.copy()
            bins = min(self.tf_lookup_bins, raw.shape[2])
            if raw.shape[2] % bins == 0:
                raw = raw.reshape(raw.shape[0], raw.shape[1], bins,
                                  raw.shape[2] // bins, 4).mean(axis=3)
            else:
                keep = np.linspace(0, raw.shape[2] - 1, bins).round().astype(int)
                raw = raw[:, :, keep, :]
            self._tf_bank_lookup = torch.tensor(raw, dtype=torch.float32, device="cuda")
            # Precompute log(alpha + eps) for every preset once. The
            # log-ratio opacity coordinate is a difference of two rows of
            # this table, so recomputing the logs per call (per training
            # iteration, and per TF switch at inference) is pure waste.
            self._tf_bank_log_alpha = torch.log(
                self._tf_bank_lookup[..., 3] + self.TF_ALPHA_EPS)     # [T,L,B]
            self._tf_label_ids = np.asarray(bank["label_ids"]).astype(int).tolist() \
                if "label_ids" in bank else None
            # Authored per-preset label visibility (IsVisible in the bookmark),
            # exported by the bank generator as visible[T, L].
            self._tf_label_visible = (torch.tensor(
                np.asarray(bank["visible"]).astype(bool), device="cuda")
                if "visible" in bank else None)
            if self.tf_use_lookup and self._tf_lookup_gain is None:
                self._tf_lookup_gain = nn.Parameter(
                    torch.ones(4, device="cuda").requires_grad_(True))
        # RGB is irrelevant where the TF is transparent. Premultiplication keeps
        # hidden-label and zero-alpha colors from becoming spurious conditions.
        rgba = rgba.copy()
        rgba[..., :3] *= rgba[..., 3:4]
        sample_count = min(self.tf_samples, rgba.shape[2])
        sample_indices = np.linspace(0, rgba.shape[2] - 1, sample_count).round().astype(int)
        sampled = rgba[:, :, sample_indices, :]                      # [T,L,S',4]
        # Preserve intensity order within each curve, then pool across curves.
        # This makes the global summary independent of region ordering and
        # compatible with different label counts.
        context_descriptors = sampled.reshape(sampled.shape[0],
                                               sampled.shape[1], -1)
        if self.tf_encoder_pooled:
            descriptors = sampled.reshape(sampled.shape[0], sampled.shape[1], -1)
        else:
            descriptors = sampled.reshape(sampled.shape[0], -1)
        tf_ids = np.asarray(bank["tf_ids"]).astype(str).tolist()
        base_index = next((i for i, value in enumerate(tf_ids)
                           if value == "train_00_base"), 0)
        self._tf_base_index = base_index
        # Centering gives the authored base TF an exact zero code. Bias-free
        # layers consequently preserve the input dGS checkpoint at T_base.
        descriptors = descriptors - descriptors[base_index:base_index + 1]
        context_descriptors = (
            context_descriptors
            - context_descriptors[base_index:base_index + 1])
        self._tf_descriptors = torch.tensor(descriptors, dtype=torch.float32, device="cuda")
        self._tf_context_descriptors = torch.tensor(
            context_descriptors, dtype=torch.float32, device="cuda")
        self._tf_ids = tf_ids
        input_dim = int(descriptors.shape[-1])
        # Seen-only baseline bookkeeping: rows whose id marks a training preset,
        # and each row's nearest training row in descriptor space (used when the
        # embedding encoder must answer for an unseen preset).
        self._tf_train_rows = [i for i, t in enumerate(tf_ids) if t.startswith("train")]
        if self._tf_train_rows:
            # flatten any label axis: this table only needs preset-to-preset
            # distance (used by the seen-only embedding fallback).
            flat = self._tf_descriptors.reshape(self._tf_descriptors.shape[0], -1)
            train = flat[self._tf_train_rows]                           # [Ttr, D]
            dist = torch.cdist(flat, train)                             # [T, Ttr]
            self._tf_nearest_train = torch.tensor(
                [self._tf_train_rows[j] for j in dist.argmin(dim=1).tolist()],
                device="cuda")
        if self.tf_encoder is None and self.tf_factors_active:
            if self.tf_encoder_type == "embedding":
                # Zero init: presets whose rows never receive gradients (all
                # held-out presets) keep an exactly-zero code, i.e. render the
                # base appearance unless the nearest-training fallback is used.
                self.tf_encoder = nn.Embedding(len(tf_ids), self.tf_rank).cuda()
                nn.init.zeros_(self.tf_encoder.weight)
            elif self.tf_encoder_local:
                # phi acts on one RGBA delta sample (4 channels) -> rank, plus
                # the log-ratio opacity coordinate when tf_log_ratio_encoder is
                # set, so the residual shares the lookup's coordinates.
                self.tf_encoder = nn.Sequential(
                    nn.Linear(5 if self.tf_log_ratio_encoder else 4,
                              self.tf_hidden, bias=False),
                    nn.ReLU(inplace=False),
                    nn.Linear(self.tf_hidden, self.tf_local_rank, bias=False),
                ).cuda()
                if self.tf_global_context_rank:
                    curve_dim = int(context_descriptors.shape[-1])
                    self.tf_context_encoder = nn.Sequential(
                        nn.Linear(curve_dim, self.tf_hidden, bias=False),
                        nn.ReLU(inplace=False),
                        nn.Linear(self.tf_hidden, self.tf_hidden, bias=False),
                    ).cuda()
                    self.tf_context_encoder_psi = nn.Sequential(
                        nn.ReLU(inplace=False),
                        nn.Linear(self.tf_hidden,
                                  self.tf_global_context_rank, bias=False),
                    ).cuda()
            elif self.tf_encoder_pooled:
                self.tf_encoder = nn.Sequential(
                    nn.Linear(input_dim, self.tf_hidden, bias=False),
                    nn.ReLU(inplace=False),
                    nn.Linear(self.tf_hidden, self.tf_hidden, bias=False),
                ).cuda()
                self.tf_encoder_psi = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Linear(self.tf_hidden, self.tf_rank, bias=False),
                ).cuda()
            else:
                self.tf_encoder = nn.Sequential(
                    nn.Linear(input_dim, self.tf_hidden, bias=False),
                    nn.ReLU(inplace=False),
                    nn.Linear(self.tf_hidden, self.tf_rank, bias=False),
                ).cuda()
        elif (self.tf_encoder is not None and self.tf_encoder_type == "functional"
              and self.tf_encoder[0].in_features != input_dim):
            raise ValueError("TF descriptor dimension changed after encoder initialization")

    def load_lookup_descriptors(self, path):
        """Attach per-Gaussian volume/mask descriptors sampled at init
        (factorsplat_lookup_descriptors.py). Row order must match the loaded
        init PLY. joint mode loads the packed empirical joint p_i(l,h); the
        separable mode loads dense p_i(l), q_i(h) only."""
        if not (self.tf_use_lookup or self.tf_encoder_local):
            return
        if self._tf_bank_lookup is None:
            raise RuntimeError("set_tf_bank must run before load_lookup_descriptors")
        payload = dict(np.load(path))
        label_ids = np.asarray(payload["label_ids"]).astype(int).tolist()
        if self._tf_label_ids is not None and label_ids != self._tf_label_ids:
            raise ValueError("lookup descriptor label order does not match tf_bank")
        count = self.get_xyz.shape[0]
        bins = self._tf_bank_lookup.shape[2]
        native = int(payload["hu"].shape[0]) if "hu" in payload else 256

        if self.tf_lookup_mode == "separable":
            probs = np.asarray(payload["label_probs"], dtype=np.float32)
            hist = np.asarray(payload["intensity_hist"], dtype=np.float32)
            if probs.shape[0] != count or hist.shape[0] != count:
                raise ValueError(f"lookup descriptors cover {probs.shape[0]} "
                                 f"Gaussians but the model has {count}")
            if probs.shape[1] != self._tf_bank_lookup.shape[1]:
                raise ValueError("descriptor label axis does not match tf_bank")
            if hist.shape[1] % bins == 0:
                hist = hist.reshape(hist.shape[0], bins,
                                    hist.shape[1] // bins).sum(axis=2)
            elif hist.shape[1] != bins:
                raise ValueError(f"cannot rebin {hist.shape[1]} bins to {bins}")
            self._tf_lookup_p = torch.tensor(probs, device="cuda")
            self._tf_lookup_q = torch.tensor(hist, device="cuda")
            covered = float((probs.sum(axis=1) > 0).mean())
            samples = "separable p*q"
        else:
            if "sample_ids" in payload:            # packed npz (native grid)
                ids = np.asarray(payload["sample_ids"], dtype=np.int64)
                counts = np.asarray(payload["sample_counts"], dtype=np.int64)
                col, hu_bin = ids // native, ids % native
            elif "sample_label_cols" in payload:   # legacy unpacked npz
                cols = np.asarray(payload["sample_label_cols"], dtype=np.int64)
                hu_bins = np.asarray(payload["sample_bins"], dtype=np.int64)
                valid = cols >= 0
                order = np.argsort(~valid, axis=1, kind="stable")
                col = np.take_along_axis(np.where(valid, cols, 0), order, axis=1)
                hu_bin = np.take_along_axis(hu_bins, order, axis=1)
                counts = valid.sum(axis=1)
            else:
                raise ValueError("npz has no joint sample arrays; regenerate "
                                 "descriptors or use --tf_lookup_mode separable")
            if col.shape[0] != count:
                raise ValueError(f"lookup descriptors cover {col.shape[0]} "
                                 f"Gaussians but the model has {count}")
            packed = col * bins + hu_bin // max(native // bins, 1)
            if packed.max() >= 32768:
                raise ValueError("packed (label, bin) id exceeds int16 range")
            self._tf_lookup_ids = torch.tensor(
                packed.astype(np.int16), device="cuda")
            self._tf_lookup_counts = torch.tensor(
                np.clip(counts, 0, 255).astype(np.uint8), device="cuda")
            if "sample_weights" in payload:
                # Stored uint8 (x255) density weights; renormalize so each
                # primitive's valid samples sum to 1 after dequantization.
                wq = np.asarray(payload["sample_weights"], dtype=np.float32) / 255.0
                if wq.shape != col.shape:
                    raise ValueError("sample_weights shape does not match sample_ids")
                valid = np.arange(col.shape[1])[None, :] < counts[:, None]
                wq = wq * valid
                den = wq.sum(1, keepdims=True)
                wq = np.divide(wq, den, out=np.zeros_like(wq), where=den > 0)
                self._tf_lookup_w = torch.tensor(wq, device="cuda")
            covered = float((counts > 0).mean())
            samples = f"packed joint p(l,h), {col.shape[1]} slots/Gaussian"
        print(f"Loaded lookup descriptors for {count} Gaussians "
              f"({covered:.1%} with foreground support, {samples}) from {path}")

    def init_veg_scalar(self, path):
        """Initialize a fixed region and bounded intensity for adapted VEG.

        The region is the density-weighted majority label in the primitive's
        initialization window.  Only the intensity coordinate inside that
        region is learnable.  This preserves categorical material identity and
        forbids interpolation across adjacent packed region curves.
        """
        if not self.tf_veg_packed:
            return
        payload = dict(np.load(path))
        bins = self._tf_bank_lookup.shape[2]
        native = int(payload["hu"].shape[0]) if "hu" in payload else 256
        ids = np.asarray(payload["sample_ids"], dtype=np.int64)
        counts = np.asarray(payload["sample_counts"], dtype=np.int64)
        if ids.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("veg init descriptors do not match Gaussian count")
        col, hu_bin = ids // native, ids % native
        K = ids.shape[1]
        valid = np.arange(K)[None, :] < counts[:, None]
        if "sample_weights" in payload:
            weights = np.asarray(payload["sample_weights"], dtype=np.float32) / 255.0
            if weights.shape != ids.shape:
                raise ValueError("sample_weights shape does not match sample_ids")
            weights *= valid
            den = weights.sum(1, keepdims=True)
            weights = np.divide(weights, den, out=np.zeros_like(weights),
                                where=den > 0)
        else:
            weights = valid.astype(np.float32) / np.maximum(counts, 1)[:, None]

        label_count = self._tf_bank_lookup.shape[1]
        if np.any(col[valid] < 0) or np.any(col[valid] >= label_count):
            raise ValueError("VEG descriptor label column lies outside TF bank")
        mass = np.zeros((ids.shape[0], label_count), dtype=np.float32)
        rows = np.arange(ids.shape[0])
        for k in range(K):
            np.add.at(mass, (rows, np.clip(col[:, k], 0, label_count - 1)),
                      weights[:, k])
        label = mass.argmax(axis=1).astype(np.int64)
        has_support = counts > 0

        on_label = valid & (col == label[:, None])
        label_weights = weights * on_label
        label_den = label_weights.sum(1)
        native_position = (hu_bin.astype(np.float32) * (bins - 1)
                           / max(native - 1, 1))
        local_bin = np.divide(
            (native_position * label_weights).sum(1), label_den,
            out=np.zeros(ids.shape[0], dtype=np.float32), where=label_den > 0)
        fraction = np.clip(local_bin / max(bins - 1, 1), 1e-3, 1.0 - 1e-3)
        raw = np.log(fraction / (1.0 - fraction)).astype(np.float32)[:, None]
        self._tf_veg_v = nn.Parameter(
            torch.tensor(raw, device="cuda").requires_grad_(True))
        self._tf_veg_label = torch.tensor(label, dtype=torch.int16, device="cuda")
        self._tf_veg_has_support = torch.tensor(
            has_support, dtype=torch.bool, device="cuda")
        print(f"Initialized region-constrained VEG coordinates for {len(raw)} "
              f"Gaussians from {path}")

    def _veg_rgba(self, tf_index):
        """Interpolate within each Gaussian's fixed region curve. Returns [N,4]."""
        if self._tf_veg_v is None or self._tf_veg_label is None:
            raise RuntimeError("region-constrained VEG coordinates are uninitialized")
        lut = self._tf_bank_lookup[tf_index]                         # [L,B,4]
        bins = lut.shape[1]
        if bins < 2:
            raise ValueError("adapted VEG requires at least two intensity bins")
        local = torch.sigmoid(self._tf_veg_v.squeeze(1)) * (bins - 1)
        b0 = local.floor().long().clamp(0, bins - 2)
        w = (local - b0.float()).unsqueeze(1)
        label = self._tf_veg_label.long()
        return lut[label, b0] * (1 - w) + lut[label, b0 + 1] * w

    def _veg_visibility(self, tf_index):
        """Exact authored visibility for the fixed adapted-VEG region label."""
        n = self.get_xyz.shape[0]
        if self._tf_label_visible is None or self._tf_veg_label is None:
            return torch.ones(n, 1, device=self.get_xyz.device)
        removed = ((~self._tf_label_visible[tf_index])
                   & self._tf_label_visible[self._tf_base_index])
        suppress = removed[self._tf_veg_label.long()]
        if self._tf_veg_has_support is not None:
            suppress = suppress & self._tf_veg_has_support
        return (~suppress).float().unsqueeze(1)

    @torch.no_grad()
    def bake_appearance(self, tf_index):
        """Freeze the conditioned appearance of one preset into the base
        tensors and disable the conditional branches.

        The TF code z_T and the lookup delta depend on the PRESET only, so at a
        fixed transfer function the conditioned DC color and opacity logit are
        constants. Writing them into (_features_dc, _features_rest, _opacity)
        yields a checkpoint that is structurally an ordinary dGS model and
        renders bit-identically, moving the conditioning cost from once per
        FRAME to once per preset SWITCH. Call again (after restoring) to switch
        presets; `unbake()` puts the conditional branches back.
        """
        if self._tf_baked_state is None:
            self._tf_baked_state = {
                "features_dc": self._features_dc.detach().clone(),
                "features_rest": self._features_rest.detach().clone(),
                "opacity": self._opacity.detach().clone(),
                "condition_color": self.tf_condition_color,
                "condition_opacity": self.tf_condition_opacity,
                "use_lookup": self.tf_use_lookup,
                "veg_packed": self.tf_veg_packed,
            }
        else:
            self._restore_pre_bake()
        camera = _TFOnlyCamera(int(tf_index))
        shs, opacity = self.conditioned_appearance(camera, 1.0)
        clamped = opacity.clamp(1e-6, 1.0 - 1e-6)
        self._features_dc = nn.Parameter(shs[:, :1, :].contiguous())
        self._features_rest = nn.Parameter(shs[:, 1:, :].contiguous())
        self._opacity = nn.Parameter(torch.log(clamped / (1.0 - clamped)))
        self.tf_condition_color = False
        self.tf_condition_opacity = False
        self.tf_use_lookup = False
        self.tf_veg_packed = False
        self._tf_baked_index = int(tf_index)

    def _restore_pre_bake(self):
        state = self._tf_baked_state
        self._features_dc = nn.Parameter(state["features_dc"].clone())
        self._features_rest = nn.Parameter(state["features_rest"].clone())
        self._opacity = nn.Parameter(state["opacity"].clone())
        self.tf_condition_color = state["condition_color"]
        self.tf_condition_opacity = state["condition_opacity"]
        self.tf_use_lookup = state["use_lookup"]
        self.tf_veg_packed = state["veg_packed"]

    def unbake(self):
        """Restore the pre-bake tensors and re-enable conditioning."""
        if self._tf_baked_state is None:
            return
        self._restore_pre_bake()
        self._tf_baked_state = None
        self._tf_baked_index = None

    def load_refresh_grid(self, path):
        """Attach the label/HU id volume + box transform for train-time refresh.

        Enables re-sampling each primitive's 3^3 window at its CURRENT position
        after densification, so the samples describe where the primitive
        actually is. Kept as a plain buffer (no grad): the refresh runs under
        no_grad and its output is a constant, so the transfer function still
        cannot move a Gaussian.
        """
        if not os.path.isfile(path):
            print(f"[refresh] no grid at {path}; descriptors stay fixed at init")
            return False
        with np.load(path) as g:
            self._refresh_id_vol = torch.tensor(
                np.asarray(g["id_vol"], dtype=np.int32), device="cuda")
            self._refresh_origin = torch.tensor(
                np.asarray(g["origin"], dtype=np.float32), device="cuda")
            self._refresh_spacing = torch.tensor(
                np.asarray(g["spacing"], dtype=np.float32), device="cuda")
            self._refresh_size = torch.tensor(
                np.asarray(g["size"], dtype=np.int64), device="cuda")
            self._refresh_center = torch.tensor(
                np.asarray(g["volume_center"], dtype=np.float32), device="cuda")
            self._refresh_hu_samples = int(g["hu_samples"])
            self._refresh_centered = bool(g["centered_init"])
        z, y, x = self._refresh_id_vol.shape
        print(f"[refresh] grid {x}x{y}x{z} loaded "
              f"({self._refresh_id_vol.numel()*4/2**20:.0f} MiB); "
              f"centered_init={self._refresh_centered}")
        return True

    @torch.no_grad()
    def refresh_descriptors(self):
        """Re-sample the 3^3 window at current positions: reassign the (label,
        HU-bin) ids and recompute density weights from the CURRENT covariance.

        Detached by construction -- the ids and weights are constants, so no
        gradient path is opened from appearance into geometry.
        """
        if getattr(self, "_refresh_id_vol", None) is None:
            return False
        xyz = self.get_xyz.detach()
        if self._refresh_centered:
            xyz = xyz + self._refresh_center[None, :]
        idx = torch.round((xyz - self._refresh_origin[None, :])
                          / self._refresh_spacing[None, :]).long()
        size = self._refresh_size
        idx = idx.clamp(torch.zeros_like(size)[None, :], (size - 1)[None, :])
        rng = torch.arange(-1, 2, device=xyz.device)
        offs = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"),
                           dim=-1).reshape(-1, 3)                    # [27,3]
        nb = (idx[:, None, :] + offs[None, :, :]).clamp(
            torch.zeros_like(size)[None, None, :], (size - 1)[None, None, :])
        z, y, x = self._refresh_id_vol.shape
        flat = (nb[..., 2] * y + nb[..., 1]) * x + nb[..., 0]
        ids = self._refresh_id_vol.reshape(-1)[flat]                 # [N,27] or -1
        valid = ids >= 0

        # Experimental material-preserving refresh.  Lock each primitive to
        # the region carrying most of its pre-refresh descriptor mass, then
        # refresh HU samples only from voxels of that region.  This lets the
        # intensity support and covariance weights track geometry without a
        # center crossing an organ boundary silently changing material identity.
        # If the current window contains no voxel of the locked region, retain
        # the previous descriptor rather than borrowing another anatomy.
        old_ids = self._tf_lookup_ids
        old_counts = self._tf_lookup_counts
        old_weights = self._sample_weights() if old_ids is not None else None
        locked_has_match = None
        if self.tf_refresh_label_locked and old_ids is not None:
            bins = self._tf_bank_lookup.shape[2]
            old_valid = (torch.arange(old_ids.shape[1], device=ids.device)[None, :]
                         < old_counts.long()[:, None])
            old_labels = old_ids.long() // bins
            label_mass = torch.zeros(
                old_ids.shape[0], self._tf_bank_lookup.shape[1],
                device=ids.device, dtype=torch.float32)
            label_mass.scatter_add_(
                1, old_labels.clamp(0, label_mass.shape[1] - 1),
                old_weights * old_valid.float())
            locked_label = label_mass.argmax(dim=1)
            refreshed_label = torch.where(valid, ids, torch.zeros_like(ids)) \
                // self._refresh_hu_samples
            valid = (valid & (refreshed_label == locked_label[:, None])
                     & (old_counts.long()[:, None] > 0))
            locked_has_match = valid.any(dim=1)

        # density weights from the CURRENT covariance, in the primitive frame
        rel = (nb - idx[:, None, :]).float() * self._refresh_spacing[None, None, :]
        sigma = self.get_scaling.detach().clamp_min(1e-6)             # [N,3]
        q = torch.nn.functional.normalize(self.get_rotation.detach(), dim=1)
        w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y0**2 + z0**2), 2*(x0*y0 - w0*z0),     2*(x0*z0 + w0*y0),
            2*(x0*y0 + w0*z0),     1 - 2*(x0**2 + z0**2), 2*(y0*z0 - w0*x0),
            2*(x0*z0 - w0*y0),     2*(y0*z0 + w0*x0),     1 - 2*(x0**2 + y0**2),
        ], dim=1).reshape(-1, 3, 3)
        local = torch.einsum("nij,nkj->nki", R.transpose(1, 2), rel)
        maha = ((local / sigma[:, None, :]) ** 2).sum(-1)
        wts = torch.exp(-0.5 * maha) * valid.float()
        # a primitive whose density underflows everywhere falls back to uniform
        flat_w = (wts.sum(1) <= 1e-12) & (valid.any(1))
        wts[flat_w] = valid[flat_w].float()
        wts = wts / wts.sum(1, keepdim=True).clamp_min(1e-12)

        # pack valid-first so counts remain the contract the gather expects
        order = torch.argsort((~valid).to(torch.uint8), dim=1, stable=True)
        ids_s = torch.gather(torch.where(valid, ids, torch.zeros_like(ids)), 1, order)
        wts_s = torch.gather(wts, 1, order)
        counts = valid.sum(1)
        bins = self._tf_bank_lookup.shape[2]
        S = self._refresh_hu_samples
        col, hu_bin = ids_s // S, ids_s % S
        packed = col * bins + hu_bin // max(S // bins, 1)
        refreshed_ids = packed.to(torch.int16)
        refreshed_counts = counts.clamp(0, 255).to(torch.uint8)
        if locked_has_match is not None:
            keep_old = ~locked_has_match
            refreshed_ids[keep_old] = old_ids[keep_old]
            refreshed_counts[keep_old] = old_counts[keep_old]
            wts_s[keep_old] = old_weights[keep_old]
        self._tf_lookup_ids = refreshed_ids
        self._tf_lookup_counts = refreshed_counts
        self._tf_lookup_w = wts_s
        return True

    def _sample_weights(self):
        """Per-sample weights [N, K] used by both local branches: the stored
        density weights when present, else uniform 1/K_i over valid samples."""
        ids = self._tf_lookup_ids
        counts = self._tf_lookup_counts.long()
        if self._tf_lookup_w is not None:
            return self._tf_lookup_w
        mask = (torch.arange(ids.shape[1], device=ids.device)[None, :]
                < counts[:, None])
        return mask.float() / counts.clamp(min=1)[:, None].float()

    def _gather_operator(self):
        """Cached sparse [N, L*B] matrix S with S[i, u] = sum_k w_ik [u_ik = u].

        Both local branches contract the SAME (ids, weights) against different
        per-entry tables, so each is exactly S @ table.  Writing it as one
        sparse product avoids materializing the [N, K, r] intermediate that the
        gather form builds (231 MiB forward and again backward at N=280k, K=27,
        r=8, to produce 8.5 MiB) and measured 15.45 vs 25.76 ms per fwd+bwd.

        ids/counts/weights change only at load, descriptor refresh, pruning and
        densification, so the operator is rebuilt on demand.  Invalidation keys
        on a fingerprint of the descriptor tensors rather than on hooks at each
        mutation site, so a missed site self-heals instead of silently training
        against stale support.
        """
        ids = self._tf_lookup_ids
        weights = self._sample_weights()
        bins = self._tf_bank_lookup.shape[2]
        n_entries = self._tf_bank_lookup.shape[1] * bins
        sig = (ids.data_ptr(), ids.shape, weights.data_ptr(), weights.shape,
               int(self._tf_lookup_counts.sum().item()), n_entries)
        if self._tf_gather_op is not None and self._tf_gather_sig == sig:
            return self._tf_gather_op
        n = ids.shape[0]
        k = ids.shape[1]
        rows = torch.arange(n, device=ids.device).repeat_interleave(k)
        cols = ids.long().reshape(-1)
        vals = weights.reshape(-1)
        op = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals,
                                     (n, n_entries)).coalesce()
        self._tf_gather_op, self._tf_gather_sig = op, sig
        return op

    def _contract_local(self, tables):
        """Apply the cached operator to a list of [L*B, c] tables at once.

        Concatenating the tables costs one sparse product instead of one per
        branch; the caller splits the [N, sum(c)] result.
        """
        op = self._gather_operator()
        wide = torch.cat(tables, dim=1) if len(tables) > 1 else tables[0]
        return torch.sparse.mm(op, wide)

    def _local_tf_code(self, tf_index, opacity_only=False):
        """Per-Gaussian TF code from the primitive's own (label, bin) samples.

        phi is evaluated ONCE over the (L*B, 4) delta table for this preset, then
        the packed descriptors gather and average the resulting codes, so the
        cost is a gather -- not an MLP per sample. Returns [N, r]."""
        bank = self._tf_bank_lookup
        if bank is None or self._tf_lookup_ids is None:
            raise RuntimeError("local TF encoder requires packed descriptors")
        delta_r = (bank[tf_index] - bank[self._tf_base_index]).reshape(-1, 4)
        if opacity_only:
            # A color-only authored edit must not alter extinction. Keeping the
            # alpha coordinate and zeroing RGB before the bias-free encoder
            # gives that invariant exactly without another encoder or factors.
            delta_r = torch.cat((torch.zeros_like(delta_r[:, :3]),
                                 delta_r[:, 3:4]), dim=1)
        if self.tf_log_ratio_encoder:
            la = self._tf_bank_log_alpha
            if la is None:
                la = torch.log(bank[..., 3] + self.TF_ALPHA_EPS)
            lr = (la[tf_index] - la[self._tf_base_index]).reshape(-1, 1)
            delta_r = torch.cat([delta_r, lr.clamp(-6.0, 6.0)], dim=1)  # [L*B,5]
        table = self.tf_encoder(delta_r)                             # [L*B, r]
        return self._contract_local([table])                         # [N, r]

    def _alpha_curve_changed(self, tf_index):
        """Return whether the authored alpha table differs from the base TF."""
        bank = self._tf_bank_lookup
        if bank is None:
            raise RuntimeError("alpha-identity gating requires a loaded TF bank")
        return bool(torch.any(
            bank[tf_index, ..., 3] != bank[self._tf_base_index, ..., 3]))

    def _bounded_opacity_offset(self, opacity_delta):
        """Convert the factor response to a bounded opacity-logit offset."""
        offset = self.tf_opacity_scale * opacity_delta
        if self.tf_opacity_residual_clip > 0:
            offset = offset.clamp(-self.tf_opacity_residual_clip,
                                  self.tf_opacity_residual_clip)
        return offset

    def _opacity_factor_offset(self, tf_index):
        """Learned opacity-logit correction for one TF, before envelopes."""
        if self.tf_encoder_local:
            code = self._local_context_code(
                tf_index, opacity_only=self.tf_opacity_alpha_only)
            delta = torch.einsum("nr,nr->n", self._tf_opacity_factors, code)
        else:
            code = self._code_for_index(tf_index)
            delta = torch.einsum("nr,r->n", self._tf_opacity_factors, code)
        return self._bounded_opacity_offset(delta)

    @torch.no_grad()
    def _opacity_training_envelope(self):
        """Per-Gaussian min/max learned correction over training TFs.

        The cache is intended for a frozen checkpoint at inference. It adds no
        fitted parameter and leaves every training-preset response unchanged.
        """
        cached = self._tf_opacity_envelope
        if cached is not None and cached[0].shape[0] == self.get_xyz.shape[0]:
            return cached
        if not self._tf_train_rows:
            raise RuntimeError("opacity training envelope needs TF split metadata")
        lower = upper = None
        for row in self._tf_train_rows:
            if (self.tf_opacity_alpha_identity_gate
                    and not self._alpha_curve_changed(row)):
                offset = torch.zeros(
                    self.get_xyz.shape[0], device=self.get_xyz.device)
            else:
                offset = self._opacity_factor_offset(row)
            lower = offset if lower is None else torch.minimum(lower, offset)
            upper = offset if upper is None else torch.maximum(upper, offset)
        self._tf_opacity_envelope = (lower, upper)
        return self._tf_opacity_envelope

    def _authored_visibility(self, tf_index):
        """[N,1] float gate: 0 where the preset REMOVES the primitive's material
        (authored alpha zero at every descriptor sample under `tf_index`, but
        nonzero somewhere under the base preset), else 1.  This is exact renderer semantics -- material the preset assigns
        zero opacity contributes nothing -- and unlike the additive logit offset
        it cannot under-shoot on full visibility removal (`hide` presets), where
        a saturated sigmoid leaves anatomy ghost-visible.  Primitives without
        descriptor support (count 0) stay visible, matching the d_i = 0
        convention of the lookup."""
        if self._tf_label_visible is not None:
            # Label-level rule: a label the preset toggles IsVisible=false on
            # (but which is visible at base) is masked WHOLESALE, exactly matching
            # the renderer's authored semantics. A primitive is gated only when
            # every valid descriptor sample lies on a removed label, so
            # boundary primitives spanning a surviving label keep contributing.
            removed_lab = (~self._tf_label_visible[tf_index]) & \
                self._tf_label_visible[self._tf_base_index]              # [L]
            if not bool(removed_lab.any()):
                n = self.get_xyz.shape[0]
                return torch.ones(n, 1, device=self.get_xyz.device)
            bins = self._tf_bank_lookup.shape[2]
            if self._tf_lookup_ids is not None:
                ids = self._tf_lookup_ids.long()                          # [N,K]
                counts = self._tf_lookup_counts.long()                    # [N]
                labs = ids // bins                                        # [N,K]
                valid = (torch.arange(ids.shape[1], device=ids.device)[None, :]
                         < counts[:, None])
                tau = self.tf_gate_removed_mass
                if self.tf_soft_visibility_gate:
                    if self._tf_lookup_w is not None:
                        w = self._tf_lookup_w * valid.float()
                    else:
                        w = valid.float() / counts.clamp(min=1)[:, None]
                    removed_mass = (removed_lab[labs].float() * w).sum(dim=1)
                    surviving = (1.0 - removed_mass).clamp(0.0, 1.0)
                    return torch.where(counts > 0, surviving,
                                       torch.ones_like(surviving)).unsqueeze(1)
                if tau >= 1.0:  # keep-if-any-survives (exact original rule)
                    on_kept = (~removed_lab[labs]) & valid
                    keep = on_kept.any(dim=1) | (counts == 0)
                    return keep.float().unsqueeze(1)
                # Mass-threshold rule: descriptor weights (density-based after
                # refresh, uniform otherwise) say how much of the primitive's
                # material lies on removed labels.
                if self._tf_lookup_w is not None:
                    w = self._tf_lookup_w * valid.float()
                else:
                    w = valid.float() / counts.clamp(min=1)[:, None]
                removed_mass = (removed_lab[labs].float() * w).sum(dim=1)
                keep = (removed_mass < tau) | (counts == 0)
                return keep.float().unsqueeze(1)
            kept_mass = (self._tf_lookup_p * (~removed_lab).float()[None, :]).sum(1)
            no_support = self._tf_lookup_p.sum(1) == 0
            if self.tf_soft_visibility_gate:
                total_mass = self._tf_lookup_p.sum(1).clamp(min=1e-12)
                surviving = (kept_mass / total_mass).clamp(0.0, 1.0)
                return torch.where(no_support, torch.ones_like(surviving),
                                   surviving).unsqueeze(1)
            return ((kept_mass > 0) | no_support).float().unsqueeze(1)

        alpha = self._tf_bank_lookup[tf_index][..., 3]                   # [L,B]
        base = self._tf_bank_lookup[self._tf_base_index][..., 3]         # [L,B]
        # Gate only material the preset REMOVED (visible at base, zero under T).
        # Gating every authored-invisible primitive also killed always-invisible
        # material that training legitimately uses as scaffolding (large fitted
        # footprints exceed the 3^3 descriptor window), costing ~0.8 dB PSNR on
        # near-distribution splits; the removed-only rule leaves the base render
        # bit-identical and still zeroes hidden anatomy exactly.
        if self._tf_lookup_ids is not None:
            ids = self._tf_lookup_ids.long()                             # [N,K]
            counts = self._tf_lookup_counts.long()                       # [N]
            valid = (torch.arange(ids.shape[1], device=ids.device)[None, :]
                     < counts[:, None]).float()
            a_now = (alpha.reshape(-1)[ids] * valid).amax(dim=1)         # [N]
            a_base = (base.reshape(-1)[ids] * valid).amax(dim=1)         # [N]
            removed = (a_now == 0) & (a_base > 0) & (counts > 0)
            return (~removed).float().unsqueeze(1)
        p_pos = (self._tf_lookup_p > 0).float()
        q_pos = (self._tf_lookup_q > 0).float()
        now = torch.einsum("nl,lb,nb->n", p_pos, (alpha > 0).float(), q_pos)
        was = torch.einsum("nl,lb,nb->n", p_pos, (base > 0).float(), q_pos)
        removed = (now == 0) & (was > 0)
        return (~removed).float().unsqueeze(1)


    def _lookup_delta(self, tf_index):
        """Eq. 6: locally relevant RGBA change of the preset relative to the
        base preset. joint mode averages R_T - R_T0 over the window's packed
        (label, bin) samples with reconstructed uniform weights; separable
        mode contracts the p(l) q(h) product. Returns [N, 4]."""
        bank = self._tf_bank_lookup
        delta_r = bank[tf_index] - bank[self._tf_base_index]            # [L,B,4]
        if self._tf_lookup_ids is not None:
            return self._contract_local([delta_r.reshape(-1, 4)])       # [N,4]
        per_label = torch.einsum("nb,lbc->nlc", self._tf_lookup_q, delta_r)
        return torch.einsum("nl,nlc->nc", self._tf_lookup_p, per_label)

    def _log_ratio(self, tf_index, eps=None, clamp=6.0):
        """[N,1] density-weighted log((a_T+eps)/(a_T0+eps)) over the packed
        window samples (geometric-mean ratio), clamped to +-clamp.

        Reads the log-alpha table cached at bank load; the per-call logs it
        replaces cost two [L,B] transcendental passes per training iteration.
        """
        eps = self.TF_ALPHA_EPS if eps is None else eps
        if self._tf_bank_log_alpha is not None:
            la = self._tf_bank_log_alpha
        else:                                    # legacy checkpoint path
            la = torch.log(self._tf_bank_lookup[..., 3] + eps)
        lr = la[tf_index] - la[self._tf_base_index]                     # [L,B]
        if self._tf_lookup_ids is not None:
            out = self._contract_local([lr.reshape(-1, 1)])             # [N,1]
        else:
            per_label = torch.einsum("nb,lb->nl", self._tf_lookup_q, lr)
            out = torch.einsum("nl,nl->n", self._tf_lookup_p, per_label).unsqueeze(1)
        return out.clamp(-clamp, clamp)

    def _initialize_tf_factors(self, count):
        # Only the ACTIVE branches carry parameters: lookup-only runs allocate
        # no residual factors; color/opacity ablations allocate one tensor.
        device = self._xyz.device
        self._tf_color_factors = nn.Parameter(
            (1e-3 * torch.randn(count, self.tf_color_coeffs, 3, self.tf_rank,
                                device=device)).requires_grad_(True)
        ) if self.tf_condition_color else None
        self._tf_opacity_factors = nn.Parameter(
            (1e-3 * torch.randn(count, self.tf_rank, device=device)).requires_grad_(True)
        ) if self.tf_condition_opacity else None

    def create_from_pcd(self, pcd, spatial_lr_scale, mcmc_cap_max=None,
                        densification_strategy="standard"):
        super().create_from_pcd(pcd, spatial_lr_scale, mcmc_cap_max,
                                densification_strategy)
        self._initialize_tf_factors(self.get_xyz.shape[0])

    def training_setup(self, training_args):
        if self._tf_descriptors is None:
            raise RuntimeError("FactorSplat requires tf_bank.npz in the dataset root")
        if self.tf_factors_active and self.tf_encoder is None:
            raise RuntimeError("residual branch active but TF encoder was never built")
        if (self.tf_veg_packed
                and (self._tf_veg_v is None or self._tf_veg_label is None)):
            raise RuntimeError("tf_veg_packed requires init_veg_scalar or a sidecar")
        if getattr(training_args, "densification_strategy", "standard") != "standard":
            raise ValueError("FactorSplat currently supports standard densification only")
        count = self.get_xyz.shape[0]
        expected = {
            "color": (count, self.tf_color_coeffs, 3, self.tf_rank),
            "opacity": (count, self.tf_rank),
        }
        bad = ((self.tf_condition_color and
                (not isinstance(self._tf_color_factors, nn.Parameter)
                 or tuple(self._tf_color_factors.shape) != expected["color"]))
               or (self.tf_condition_opacity and
                   (not isinstance(self._tf_opacity_factors, nn.Parameter)
                    or tuple(self._tf_opacity_factors.shape) != expected["opacity"])))
        if bad:
            self._initialize_tf_factors(count)
        super().training_setup(training_args)
        if self._tf_color_factors is not None:
            self.optimizer.add_param_group({
                "params": [self._tf_color_factors],
                "lr": training_args.tf_factor_lr,
                "name": "tf_color_factors",
                "per_gaussian": True,
            })
        if self._tf_opacity_factors is not None:
            self.optimizer.add_param_group({
                "params": [self._tf_opacity_factors],
                "lr": training_args.tf_factor_lr,
                "name": "tf_opacity_factors",
                "per_gaussian": True,
            })
        if self.tf_encoder is not None:
            enc_params = list(self.tf_encoder.parameters())
            if self.tf_encoder_psi is not None:
                enc_params += list(self.tf_encoder_psi.parameters())
            if self.tf_context_encoder is not None:
                enc_params += list(self.tf_context_encoder.parameters())
                enc_params += list(self.tf_context_encoder_psi.parameters())
            self.optimizer.add_param_group({
                "params": enc_params,
                "lr": training_args.tf_encoder_lr,
                "name": "tf_encoder",
                "per_gaussian": False,
            })
        if self.tf_veg_packed and self._tf_veg_v is not None:
            self.optimizer.add_param_group({
                "params": [self._tf_veg_v],
                "lr": getattr(training_args, "tf_veg_v_lr", 0.01),
                "name": "tf_veg_v",
                "per_gaussian": True,
            })
        if self.tf_use_lookup and self._tf_lookup_gain is not None:
            self.optimizer.add_param_group({
                "params": [self._tf_lookup_gain],
                "lr": training_args.tf_encoder_lr,
                "name": "tf_lookup_gain",
                "per_gaussian": False,
            })

    def _code_for_index(self, index):
        if self.tf_encoder_type == "embedding":
            if (self.tf_embedding_fallback == "nearest"
                    and index not in self._tf_train_rows
                    and self._tf_nearest_train is not None):
                index = int(self._tf_nearest_train[index])
            return self.tf_encoder(
                torch.tensor(index, device=self.tf_encoder.weight.device)
            )
        row = self._tf_descriptors[index]
        if self.tf_encoder_pooled:
            return self.tf_encoder_psi(self.tf_encoder(row).mean(dim=0))
        return self.tf_encoder(row)

    def _context_code_for_index(self, index):
        """Permutation-invariant scene-wide summary of all authored curves."""
        if not self.tf_global_context_rank:
            return None
        curves = self._tf_context_descriptors[index]                 # [L,4S]
        return self.tf_context_encoder_psi(
            self.tf_context_encoder(curves).mean(dim=0))             # [r_g]

    def _local_context_code(self, index, opacity_only=False):
        """Concatenate local and global codes without increasing tf_rank."""
        local = self._local_tf_code(index, opacity_only=opacity_only)
        if not self.tf_global_context_rank:
            return local
        context = self._context_code_for_index(index)
        context = context.unsqueeze(0).expand(local.shape[0], -1)
        return torch.cat((local, context), dim=1)                     # [N,r]

    def get_pruning_opacity(self):
        """TF-aware pruning opacity: max over TRAINING presets of the
        conditioned opacity (view gate excluded; it is TF-independent).
        A primitive that some training preset reveals must not be deleted
        because the shared/base logit alone falls below the threshold --
        that would permanently remove anatomy other presets need."""
        if self.tf_veg_packed and self._tf_veg_v is not None:
            if not self.tf_aware_prune or not self._tf_train_rows:
                alpha = self._veg_rgba(self._tf_base_index)[:, 3:4]
                return alpha * self._veg_visibility(self._tf_base_index)
            with torch.no_grad():
                best = None
                for row in self._tf_train_rows:
                    alpha = (self._veg_rgba(row)[:, 3:4]
                             * self._veg_visibility(row))
                    best = alpha if best is None else torch.maximum(best, alpha)
                return best
        if (not self.tf_aware_prune or self._tf_descriptors is None
                or not self._tf_train_rows):
            return self.get_opacity
        with torch.no_grad():
            best_logit = None
            for row in self._tf_train_rows:
                logit = self._opacity
                if self.tf_use_lookup and self.tf_lookup_ready:
                    if self.tf_opacity_log_ratio:
                        logit = logit + self._tf_lookup_gain[3] * self._log_ratio(row)
                    else:
                        delta = self._lookup_delta(row)
                        logit = logit + self.tf_lookup_opacity_scale * self._tf_lookup_gain[3] * delta[:, 3:4]
                if self.tf_condition_opacity:
                    if (not self.tf_opacity_alpha_identity_gate
                            or self._alpha_curve_changed(row)):
                        if self.tf_encoder_local:
                            offset = torch.einsum(
                                "nr,nr->n", self._tf_opacity_factors,
                                self._local_context_code(
                                    row,
                                    opacity_only=self.tf_opacity_alpha_only))
                        else:
                            code = self._code_for_index(row)
                            offset = torch.einsum(
                                "nr,r->n", self._tf_opacity_factors, code)
                        logit = logit + self._bounded_opacity_offset(
                            offset)[:, None]
                best_logit = logit if best_logit is None                     else torch.maximum(best_logit, logit)
            return torch.sigmoid(best_logit)

    def densify_and_prune(self, max_grad, min_opacity, extent,
                          max_screen_size, iteration):
        super().densify_and_prune(
            max_grad, min_opacity, extent, max_screen_size, iteration)
        cap = self.tf_veg_max_gaussians if self.tf_veg_packed else 0
        count = self.get_xyz.shape[0]
        if cap > 0 and count > cap:
            # Use the same max-over-training-TFs opacity that drives semantic
            # pruning, retaining the most visible primitives under the budget.
            score = self.get_pruning_opacity().squeeze(1)
            keep = torch.topk(score, cap, sorted=False).indices
            prune = torch.ones(count, dtype=torch.bool, device=score.device)
            prune[keep] = False
            self.prune_points(prune)
            print(f"[ITER {iteration}] capped adapted VEG at {cap} Gaussians")

    def conditioned_appearance(self, viewpoint_camera, opacity_scale):
        index = getattr(viewpoint_camera, "tf_index", None)
        if index is None:
            return super().conditioned_appearance(viewpoint_camera, opacity_scale)
        index = int(index)
        if not 0 <= index < self._tf_descriptors.shape[0]:
            raise IndexError(f"tf_index={index} outside bank of size "
                             f"{self._tf_descriptors.shape[0]}")
        if self.tf_veg_packed and self._tf_veg_v is not None:
            rgba = self._veg_rgba(index)
            C0 = 0.28209479177387814
            dc = (rgba[:, :3] - 0.5) / C0
            shs = torch.cat((dc[:, None, :], self._features_rest), dim=1)
            opacity = rgba[:, 3:4].clamp(1e-5, 1 - 1e-5)
            if self.tf_exact_visibility_gate:
                opacity = opacity * self._veg_visibility(index)
            return shs, opacity * opacity_scale

        code = (self._code_for_index(index)
                if (self.tf_factors_active and not self.tf_encoder_local) else None)

        dc = self._features_dc[:, 0, :]
        logit = self._opacity
        if self.tf_use_lookup and self.tf_lookup_ready:
            delta = self._lookup_delta(index)                            # [N,4]
            # RGB deltas live in [0,1] LUT units; SH DC coeffs relate to RGB via
            # rgb = 0.5 + C0*dc, so the conversion to DC space divides by C0.
            C0 = 0.28209479177387814
            dc = dc + (self.tf_lookup_color_scale / C0) * \
                self._tf_lookup_gain[:3] * delta[:, :3]
            if self.tf_opacity_log_ratio:
                # gain init 1 => exact multiplicative prior at the start
                logit = logit + self._tf_lookup_gain[3] * self._log_ratio(index)
            else:
                logit = logit + self.tf_lookup_opacity_scale * \
                    self._tf_lookup_gain[3] * delta[:, 3:4]

        rest = self._features_rest
        if self.tf_encoder_local:
            zi = self._local_context_code(index)                      # [N, r]
        if self.tf_condition_color:
            color_delta = self.tf_color_scale * torch.tanh(
                torch.einsum("nkcr,nr->nkc", self._tf_color_factors, zi)
                if self.tf_encoder_local else
                torch.einsum("nkcr,r->nkc", self._tf_color_factors, code))
            dc = dc + color_delta[:, 0, :]
            extra = self.tf_color_coeffs - 1
            if extra > 0:
                rest = torch.cat((rest[:, :extra, :] + color_delta[:, 1:, :],
                                  rest[:, extra:, :]), dim=1)
        shs = torch.cat((dc[:, None, :], rest), dim=1)

        if self.tf_condition_opacity:
            if (not self.tf_opacity_alpha_identity_gate
                    or self._alpha_curve_changed(index)):
                if self.tf_encoder_local:
                    opacity_code = (
                        self._local_context_code(index, opacity_only=True)
                        if self.tf_opacity_alpha_only else zi)
                    opacity_delta = torch.einsum(
                        "nr,nr->n", self._tf_opacity_factors, opacity_code)
                else:
                    opacity_delta = torch.einsum(
                        "nr,r->n", self._tf_opacity_factors, code)
                opacity_offset = self._bounded_opacity_offset(opacity_delta)
                if self.tf_opacity_train_envelope:
                    lower, upper = self._opacity_training_envelope()
                    opacity_offset = torch.maximum(
                        lower, torch.minimum(upper, opacity_offset))
                logit = logit + opacity_offset[:, None]
        opacity = torch.sigmoid(logit)
        if self.tf_exact_visibility_gate and self.tf_lookup_ready \
                and self._tf_bank_lookup is not None:
            opacity = opacity * self._authored_visibility(index)
        return shs, opacity * opacity_scale

    def reset_opacity(self):
        # Adapted VEG: opacity is read from the LUT at (label_i, v_i); the shared logit
        # is unused and receives no gradients, so it has no optimizer state to
        # rewrite (and resetting it would do nothing anyway).
        if self.tf_veg_packed:
            return
        super().reset_opacity()

    def _prune_optimizer(self, mask):
        tensors = super()._prune_optimizer(mask)
        if "tf_color_factors" in tensors:
            self._tf_color_factors = tensors["tf_color_factors"]
        if "tf_opacity_factors" in tensors:
            self._tf_opacity_factors = tensors["tf_opacity_factors"]
        if "tf_veg_v" in tensors:
            self._tf_veg_v = tensors["tf_veg_v"]
        for name in ("_tf_veg_label", "_tf_veg_has_support"):
            tensor = getattr(self, name)
            if tensor is not None and tensor.shape[0] == mask.shape[0]:
                setattr(self, name, tensor[mask])
        for name in ("_tf_lookup_p", "_tf_lookup_q",
                     "_tf_lookup_ids", "_tf_lookup_counts", "_tf_lookup_w"):
            tensor = getattr(self, name)
            if tensor is not None and tensor.shape[0] == mask.shape[0]:
                setattr(self, name, tensor[mask])
        return tensors

    def _append_factor(self, name, attribute, extension):
        group = next(g for g in self.optimizer.param_groups if g["name"] == name)
        old_parameter = group["params"][0]
        state = self.optimizer.state.get(old_parameter)
        if state is not None:
            state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(extension)), dim=0)
            state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
            del self.optimizer.state[old_parameter]
        parameter = nn.Parameter(torch.cat((old_parameter, extension), dim=0).requires_grad_(True))
        group["params"][0] = parameter
        if state is not None:
            self.optimizer.state[parameter] = state
        setattr(self, attribute, parameter)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation,
                              new_mean_view, new_mean_time, new_L_22_inv,
                              new_v_12_direction=None, new_lambda_view=None,
                              new_lambda_time=None):
        added = int(new_xyz.shape[0])
        super().densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities,
            new_scaling, new_rotation, new_mean_view, new_mean_time, new_L_22_inv,
            new_v_12_direction, new_lambda_view, new_lambda_time,
        )
        if added:
            # Children inherit BOTH the fixed lookup descriptors and the learned
            # TF factors from the nearest pre-existing Gaussian (clones sit
            # exactly on their parent; split children land within the parent
            # footprint). Zero-initializing children's factors would make every
            # densified Gaussian TF-unresponsive until gradients rebuild it --
            # an artificial cold-start handicap for the residual branch.
            old = self.get_xyz[: self.get_xyz.shape[0] - added].detach()
            nearest = []
            for chunk in new_xyz.detach().split(1024):
                nearest.append(torch.cdist(chunk, old).argmin(dim=1))
            nearest = torch.cat(nearest)
            if self._tf_color_factors is not None:
                self._append_factor(
                    "tf_color_factors", "_tf_color_factors",
                    self._tf_color_factors.detach()[nearest].clone(),
                )
            if self._tf_opacity_factors is not None:
                self._append_factor(
                    "tf_opacity_factors", "_tf_opacity_factors",
                    self._tf_opacity_factors.detach()[nearest].clone(),
                )
            if self._tf_veg_v is not None:
                self._append_factor(
                    "tf_veg_v", "_tf_veg_v",
                    self._tf_veg_v.detach()[nearest].clone(),
                )
                self._tf_veg_label = torch.cat(
                    (self._tf_veg_label, self._tf_veg_label[nearest]), dim=0)
                self._tf_veg_has_support = torch.cat(
                    (self._tf_veg_has_support,
                     self._tf_veg_has_support[nearest]), dim=0)
            for name in ("_tf_lookup_p", "_tf_lookup_q",
                         "_tf_lookup_ids", "_tf_lookup_counts", "_tf_lookup_w"):
                tensor = getattr(self, name)
                if tensor is not None:
                    setattr(self, name,
                            torch.cat((tensor, tensor[nearest]), dim=0))

    @staticmethod
    def _sidecar(path):
        return path + ".factorsplat.pt"

    def save_ply(self, path):
        super().save_ply(path)
        payload = {
            "version": 3,
            "tf_rank": self.tf_rank,
            "tf_hidden": self.tf_hidden,
            "tf_samples": self.tf_samples,
            "tf_ids": self._tf_ids,
            "tf_encoder_type": self.tf_encoder_type,
            "tf_condition_color": self.tf_condition_color,
            "tf_condition_opacity": self.tf_condition_opacity,
            "tf_color_sh_degree": self.tf_color_sh_degree,
            "tf_use_lookup": self.tf_use_lookup,
            "tf_encoder_local": self.tf_encoder_local,
            "tf_global_context_rank": self.tf_global_context_rank,
            "tf_opacity_alpha_only": self.tf_opacity_alpha_only,
            "tf_opacity_alpha_identity_gate": self.tf_opacity_alpha_identity_gate,
            "tf_opacity_train_envelope": self.tf_opacity_train_envelope,
            "tf_opacity_residual_clip": self.tf_opacity_residual_clip,
            "tf_refresh_label_locked": self.tf_refresh_label_locked,
            "tf_soft_visibility_gate": self.tf_soft_visibility_gate,
            "tf_veg_packed": self.tf_veg_packed,
            "tf_veg_max_gaussians": self.tf_veg_max_gaussians,
            "tf_lookup_mode": self.tf_lookup_mode if self.tf_use_lookup else None,
        }
        # Only ACTIVE branches are stored: no residual factors or encoder for
        # lookup-only runs; one factor tensor for color/opacity ablations.
        if self._tf_color_factors is not None:
            payload["color_factors"] = self._tf_color_factors.detach().cpu()
        if self._tf_opacity_factors is not None:
            payload["opacity_factors"] = self._tf_opacity_factors.detach().cpu()
        if self.tf_encoder is not None:
            payload["encoder"] = {key: value.detach().cpu()
                                  for key, value in self.tf_encoder.state_dict().items()}
        if self.tf_context_encoder is not None:
            payload["context_encoder"] = {
                key: value.detach().cpu()
                for key, value in self.tf_context_encoder.state_dict().items()}
            payload["context_encoder_psi"] = {
                key: value.detach().cpu()
                for key, value in self.tf_context_encoder_psi.state_dict().items()}
        if self.tf_veg_packed and self._tf_veg_v is not None:
            payload["veg_coordinate_mode"] = "fixed_label_logit_v1"
            payload["veg_v"] = self._tf_veg_v.detach().cpu()
            payload["veg_label"] = self._tf_veg_label.detach().cpu()
            payload["veg_has_support"] = self._tf_veg_has_support.detach().cpu()
        if self.tf_use_lookup and self.tf_lookup_ready:
            payload["lookup_gain"] = self._tf_lookup_gain.detach().cpu()
        if (self.tf_use_lookup or self.tf_encoder_local) and self.tf_lookup_ready:
            if self._tf_lookup_ids is not None:
                payload["lookup_ids"] = self._tf_lookup_ids.detach().cpu()
                payload["lookup_counts"] = self._tf_lookup_counts.detach().cpu()
                if self._tf_lookup_w is not None:
                    payload["lookup_w"] = self._tf_lookup_w.detach().cpu()
            else:
                payload["lookup_p"] = self._tf_lookup_p.detach().cpu()
                payload["lookup_q"] = self._tf_lookup_q.detach().cpu()
        torch.save(payload, self._sidecar(path))

    def load_ply(self, path):
        super().load_ply(path)
        sidecar = self._sidecar(path)
        if not os.path.isfile(sidecar):
            self._initialize_tf_factors(self.get_xyz.shape[0])
            return
        payload = torch.load(sidecar, map_location="cuda")
        version = int(payload.get("version", 1))
        if int(payload["tf_rank"]) != self.tf_rank:
            raise ValueError("FactorSplat sidecar rank does not match --tf_rank")
        if int(payload.get("tf_global_context_rank", 0)) != self.tf_global_context_rank:
            raise ValueError("sidecar global-context rank does not match configuration")
        saved_type = payload.get("tf_encoder_type", "functional")
        if saved_type != self.tf_encoder_type:
            raise ValueError(f"sidecar encoder type {saved_type} != --tf_encoder_type {self.tf_encoder_type}")
        count = self.get_xyz.shape[0]

        def _factor(key):
            if key not in payload:
                raise ValueError(f"active branch needs '{key}' but the sidecar has none")
            tensor = payload[key].to(device="cuda", dtype=torch.float32)
            if tensor.shape[0] != count:
                raise ValueError("FactorSplat sidecar and PLY have different Gaussian counts")
            if key == "color_factors":
                if tensor.dim() == 3:      # legacy DC-only [N, 3, r]
                    tensor = tensor[:, None, :, :]
                if tensor.shape[1] != self.tf_color_coeffs:
                    raise ValueError(
                        f"sidecar conditions {tensor.shape[1]} SH color coeffs but "
                        f"--tf_color_sh_degree {self.tf_color_sh_degree} expects "
                        f"{self.tf_color_coeffs}")
            return nn.Parameter(tensor.requires_grad_(True))

        # v1 always stored both factors + encoder; v2 stores active ones only.
        self._tf_color_factors = _factor("color_factors") if self.tf_condition_color else None
        self._tf_opacity_factors = _factor("opacity_factors") if self.tf_condition_opacity else None
        if self.tf_factors_active:
            self.tf_encoder.load_state_dict(payload["encoder"])
            if self.tf_context_encoder is not None:
                self.tf_context_encoder.load_state_dict(
                    payload["context_encoder"])
                self.tf_context_encoder_psi.load_state_dict(
                    payload["context_encoder_psi"])
        if self.tf_veg_packed:
            mode = payload.get("veg_coordinate_mode")
            if mode != "fixed_label_logit_v1":
                raise ValueError(
                    "incompatible adapted-VEG sidecar: expected "
                    "veg_coordinate_mode='fixed_label_logit_v1'; legacy "
                    "sigmoid/direct-index checkpoints must be retrained")
            required = ("veg_v", "veg_label", "veg_has_support")
            if any(key not in payload for key in required):
                raise ValueError("region-constrained VEG sidecar is incomplete")
            veg_v = payload["veg_v"].to(device="cuda", dtype=torch.float32)
            veg_label = payload["veg_label"].to(
                device="cuda", dtype=torch.int16)
            veg_has_support = payload["veg_has_support"].to(
                device="cuda", dtype=torch.bool)
            if any(tensor.shape[0] != count for tensor in
                   (veg_v, veg_label, veg_has_support)):
                raise ValueError("adapted-VEG sidecar and PLY counts differ")
            self._tf_veg_v = nn.Parameter(veg_v.requires_grad_(True))
            self._tf_veg_label = veg_label
            self._tf_veg_has_support = veg_has_support
        if self.tf_use_lookup:
            self._tf_lookup_gain = nn.Parameter(
                payload["lookup_gain"].to("cuda").requires_grad_(True))
        if self.tf_use_lookup or self.tf_encoder_local:
            if version >= 2:
                if self.tf_lookup_mode == "joint":
                    if "lookup_ids" not in payload:
                        raise ValueError("joint mode but v2 sidecar has no packed lookup")
                    self._tf_lookup_ids = payload["lookup_ids"].to("cuda")
                    self._tf_lookup_counts = payload["lookup_counts"].to("cuda")
                    if "lookup_w" in payload:
                        self._tf_lookup_w = payload["lookup_w"].to("cuda")
                else:
                    self._tf_lookup_p = payload["lookup_p"].to("cuda")
                    self._tf_lookup_q = payload["lookup_q"].to("cuda")
            else:
                # v1 -> convert in memory; saving again writes v2.
                if self.tf_lookup_mode == "joint" and "lookup_l" in payload:
                    l = payload["lookup_l"].long()
                    b = payload["lookup_b"].long()
                    valid = payload["lookup_w"] > 0
                    order = torch.argsort((~valid).to(torch.uint8), dim=1, stable=True)
                    l = torch.gather(l, 1, order)
                    b = torch.gather(b, 1, order)
                    valid = torch.gather(valid, 1, order)
                    self._tf_lookup_ids = (l * self.tf_lookup_bins + b).to(torch.int16).cuda()
                    self._tf_lookup_counts = valid.sum(dim=1).to(torch.uint8).cuda()
                elif "lookup_p" in payload:
                    self._tf_lookup_p = payload["lookup_p"].to("cuda")
                    self._tf_lookup_q = payload["lookup_q"].to("cuda")
                else:
                    raise ValueError("active local branch needs packed descriptors, "
                                     "but the sidecar has none")

    def capture(self):
        state = {"base": super().capture(), "version": 3}
        if self._tf_color_factors is not None:
            state["color_factors"] = self._tf_color_factors
        if self._tf_opacity_factors is not None:
            state["opacity_factors"] = self._tf_opacity_factors
        if self.tf_encoder is not None:
            state["encoder"] = self.tf_encoder.state_dict()
        if self.tf_context_encoder is not None:
            state["context_encoder"] = self.tf_context_encoder.state_dict()
            state["context_encoder_psi"] = self.tf_context_encoder_psi.state_dict()
        if self.tf_veg_packed and self._tf_veg_v is not None:
            state["veg_coordinate_mode"] = "fixed_label_logit_v1"
            state["veg_v"] = self._tf_veg_v
            state["veg_label"] = self._tf_veg_label
            state["veg_has_support"] = self._tf_veg_has_support
        if self.tf_use_lookup and self.tf_lookup_ready:
            state["lookup_gain"] = self._tf_lookup_gain
        if (self.tf_use_lookup or self.tf_encoder_local) and self.tf_lookup_ready:
            for key, name in (("lookup_ids", "_tf_lookup_ids"),
                              ("lookup_counts", "_tf_lookup_counts"),
                              ("lookup_w", "_tf_lookup_w"),
                              ("lookup_p", "_tf_lookup_p"),
                              ("lookup_q", "_tf_lookup_q")):
                tensor = getattr(self, name)
                if tensor is not None:
                    state[key] = tensor
        return state

    def restore(self, model_args, training_args):
        if self.tf_condition_color:
            self._tf_color_factors = nn.Parameter(
                model_args["color_factors"].to("cuda").requires_grad_(True))
        if self.tf_condition_opacity:
            self._tf_opacity_factors = nn.Parameter(
                model_args["opacity_factors"].to("cuda").requires_grad_(True))
        if self.tf_factors_active:
            self.tf_encoder.load_state_dict(model_args["encoder"])
            if self.tf_context_encoder is not None:
                self.tf_context_encoder.load_state_dict(
                    model_args["context_encoder"])
                self.tf_context_encoder_psi.load_state_dict(
                    model_args["context_encoder_psi"])
        if self.tf_veg_packed:
            if model_args.get("veg_coordinate_mode") != "fixed_label_logit_v1":
                raise ValueError(
                    "incompatible adapted-VEG checkpoint coordinate mode")
            self._tf_veg_v = nn.Parameter(
                model_args["veg_v"].to("cuda").requires_grad_(True))
            self._tf_veg_label = model_args["veg_label"].to(
                device="cuda", dtype=torch.int16)
            self._tf_veg_has_support = model_args["veg_has_support"].to(
                device="cuda", dtype=torch.bool)
        if self.tf_use_lookup and "lookup_gain" in model_args:
            self._tf_lookup_gain = nn.Parameter(
                model_args["lookup_gain"].to("cuda").requires_grad_(True))
        if self.tf_use_lookup or self.tf_encoder_local:
            for key, name in (("lookup_ids", "_tf_lookup_ids"),
                              ("lookup_counts", "_tf_lookup_counts"),
                              ("lookup_p", "_tf_lookup_p"),
                              ("lookup_q", "_tf_lookup_q")):
                if key in model_args:
                    setattr(self, name, model_args[key].to("cuda"))
        super().restore(model_args["base"], training_args)
