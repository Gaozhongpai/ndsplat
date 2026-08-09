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


class GaussianModel(DGSModel):
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
                 tf_lookup_mode: str = "joint",
                 tf_lookup_bins: int = 64,
                 tf_lookup_color_scale: float = 1.0,
                 tf_lookup_opacity_scale: float = 4.0):
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
        # Adapted VEG reference: one learnable scalar u_i per Gaussian; DC
        # color and opacity are read from the packed 1D LUT T'(u)=bank[t]
        # flattened over (label, bin). Excludes factors/encoder/lookup.
        self.tf_veg_packed = bool(tf_veg_packed)
        if self.tf_veg_packed and (tf_condition_color or tf_condition_opacity
                                   or tf_use_lookup):
            raise ValueError("tf_veg_packed excludes the FactorSplat branches")
        self._tf_veg_u = None            # [N, 1] raw scalar (sigmoid -> LUT pos)
        if tf_lookup_mode not in ("joint", "separable"):
            raise ValueError(f"tf_lookup_mode must be joint|separable, got {tf_lookup_mode}")
        self.tf_lookup_mode = tf_lookup_mode
        self.tf_lookup_bins = int(tf_lookup_bins)
        self.tf_lookup_color_scale = float(tf_lookup_color_scale)
        self.tf_lookup_opacity_scale = float(tf_lookup_opacity_scale)
        self._tf_bank_lookup = None      # [T, L, B, 4] raw RGBA, downsampled bins
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
        if self.tf_use_lookup or self.tf_veg_packed:
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
            self._tf_label_ids = np.asarray(bank["label_ids"]).astype(int).tolist() \
                if "label_ids" in bank else None
            if self._tf_lookup_gain is None:
                self._tf_lookup_gain = nn.Parameter(
                    torch.ones(4, device="cuda").requires_grad_(True))
        # RGB is irrelevant where the TF is transparent. Premultiplication keeps
        # hidden-label and zero-alpha colors from becoming spurious conditions.
        rgba = rgba.copy()
        rgba[..., :3] *= rgba[..., 3:4]
        sample_count = min(self.tf_samples, rgba.shape[2])
        sample_indices = np.linspace(0, rgba.shape[2] - 1, sample_count).round().astype(int)
        descriptors = rgba[:, :, sample_indices, :].reshape(rgba.shape[0], -1)
        tf_ids = np.asarray(bank["tf_ids"]).astype(str).tolist()
        base_index = next((i for i, value in enumerate(tf_ids)
                           if value == "train_00_base"), 0)
        self._tf_base_index = base_index
        # Centering gives the authored base TF an exact zero code. Bias-free
        # layers consequently preserve the input dGS checkpoint at T_base.
        descriptors = descriptors - descriptors[base_index:base_index + 1]
        self._tf_descriptors = torch.tensor(descriptors, dtype=torch.float32, device="cuda")
        self._tf_ids = tf_ids
        input_dim = int(descriptors.shape[1])
        # Seen-only baseline bookkeeping: rows whose id marks a training preset,
        # and each row's nearest training row in descriptor space (used when the
        # embedding encoder must answer for an unseen preset).
        self._tf_train_rows = [i for i, t in enumerate(tf_ids) if t.startswith("train")]
        if self._tf_train_rows:
            train = self._tf_descriptors[self._tf_train_rows]           # [Ttr, D]
            dist = torch.cdist(self._tf_descriptors, train)             # [T, Ttr]
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
        if not self.tf_use_lookup:
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
            covered = float((counts > 0).mean())
            samples = f"packed joint p(l,h), {col.shape[1]} slots/Gaussian"
        print(f"Loaded lookup descriptors for {count} Gaussians "
              f"({covered:.1%} with foreground support, {samples}) from {path}")

    def init_veg_scalar(self, path):
        """Initialize u_i from the packed volume descriptor (majority id of
        the window samples), mapped through the inverse sigmoid."""
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
        packed = col * bins + hu_bin // max(native // bins, 1)      # [N,K]
        K = packed.shape[1]
        valid = np.arange(K)[None, :] < counts[:, None]
        mean_u = (packed * valid).sum(1) / np.maximum(counts, 1)
        span = self._tf_bank_lookup.shape[1] * bins - 1
        frac = np.clip(mean_u / span, 1e-4, 1 - 1e-4)
        raw = np.log(frac / (1 - frac)).astype(np.float32)[:, None]
        self._tf_veg_u = nn.Parameter(
            torch.tensor(raw, device="cuda").requires_grad_(True))
        print(f"Initialized VEG packed scalars for {len(raw)} Gaussians from {path}")

    def _veg_rgba(self, tf_index):
        """Linear interpolation of the packed 1D LUT at u_i. Returns [N,4]."""
        lut = self._tf_bank_lookup[tf_index].reshape(-1, 4)          # [L*B, 4]
        span = lut.shape[0] - 1
        u = torch.sigmoid(self._tf_veg_u).squeeze(1) * span          # [N]
        i0 = u.floor().long().clamp(0, span - 1)
        w = (u - i0.float()).unsqueeze(1)
        return lut[i0] * (1 - w) + lut[i0 + 1] * w

    def _lookup_delta(self, tf_index):
        """Eq. 6: locally relevant RGBA change of the preset relative to the
        base preset. joint mode averages R_T - R_T0 over the window's packed
        (label, bin) samples with reconstructed uniform weights; separable
        mode contracts the p(l) q(h) product. Returns [N, 4]."""
        bank = self._tf_bank_lookup
        delta_r = bank[tf_index] - bank[self._tf_base_index]            # [L,B,4]
        if self._tf_lookup_ids is not None:
            ids = self._tf_lookup_ids.long()                            # [N,K]
            counts = self._tf_lookup_counts.long()                      # [N]
            vals = delta_r.reshape(-1, 4)[ids]                          # [N,K,4]
            mask = (torch.arange(ids.shape[1], device=ids.device)[None, :]
                    < counts[:, None])
            weights = mask.float() / counts.clamp(min=1)[:, None].float()
            return (vals * weights.unsqueeze(-1)).sum(dim=1)
        per_label = torch.einsum("nb,lbc->nlc", self._tf_lookup_q, delta_r)
        return torch.einsum("nl,nlc->nc", self._tf_lookup_p, per_label)

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
        if self.tf_veg_packed and self._tf_veg_u is None:
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
            self.optimizer.add_param_group({
                "params": list(self.tf_encoder.parameters()),
                "lr": training_args.tf_encoder_lr,
                "name": "tf_encoder",
                "per_gaussian": False,
            })
        if self.tf_veg_packed and self._tf_veg_u is not None:
            self.optimizer.add_param_group({
                "params": [self._tf_veg_u],
                "lr": training_args.tf_factor_lr,
                "name": "tf_veg_u",
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
        return self.tf_encoder(self._tf_descriptors[index])

    def get_pruning_opacity(self):
        """TF-aware pruning opacity: max over TRAINING presets of the
        conditioned opacity (view gate excluded; it is TF-independent).
        A primitive that some training preset reveals must not be deleted
        because the shared/base logit alone falls below the threshold --
        that would permanently remove anatomy other presets need."""
        if self.tf_veg_packed and self._tf_veg_u is not None:
            if not self.tf_aware_prune or not self._tf_train_rows:
                return self._veg_rgba(self._tf_base_index)[:, 3:4]
            with torch.no_grad():
                best = None
                for row in self._tf_train_rows:
                    alpha = self._veg_rgba(row)[:, 3:4]
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
                    delta = self._lookup_delta(row)
                    logit = logit + self.tf_lookup_opacity_scale *                         self._tf_lookup_gain[3] * delta[:, 3:4]
                if self.tf_condition_opacity:
                    code = self._code_for_index(row)
                    offset = torch.einsum("nr,r->n", self._tf_opacity_factors, code)
                    logit = logit + self.tf_opacity_scale * offset[:, None]
                best_logit = logit if best_logit is None                     else torch.maximum(best_logit, logit)
            return torch.sigmoid(best_logit)

    def conditioned_appearance(self, viewpoint_camera, opacity_scale):
        index = getattr(viewpoint_camera, "tf_index", None)
        if index is None:
            return super().conditioned_appearance(viewpoint_camera, opacity_scale)
        index = int(index)
        if not 0 <= index < self._tf_descriptors.shape[0]:
            raise IndexError(f"tf_index={index} outside bank of size "
                             f"{self._tf_descriptors.shape[0]}")
        if self.tf_veg_packed and self._tf_veg_u is not None:
            rgba = self._veg_rgba(index)
            C0 = 0.28209479177387814
            dc = (rgba[:, :3] - 0.5) / C0
            shs = torch.cat((dc[:, None, :], self._features_rest), dim=1)
            opacity = rgba[:, 3:4].clamp(1e-5, 1 - 1e-5)
            return shs, opacity * opacity_scale

        code = self._code_for_index(index) if self.tf_factors_active else None

        dc = self._features_dc[:, 0, :]
        logit = self._opacity
        if self.tf_use_lookup and self.tf_lookup_ready:
            delta = self._lookup_delta(index)                            # [N,4]
            # RGB deltas live in [0,1] LUT units; SH DC coeffs relate to RGB via
            # rgb = 0.5 + C0*dc, so the conversion to DC space divides by C0.
            C0 = 0.28209479177387814
            dc = dc + (self.tf_lookup_color_scale / C0) * \
                self._tf_lookup_gain[:3] * delta[:, :3]
            logit = logit + self.tf_lookup_opacity_scale * \
                self._tf_lookup_gain[3] * delta[:, 3:4]

        rest = self._features_rest
        if self.tf_condition_color:
            color_delta = self.tf_color_scale * torch.tanh(
                torch.einsum("nkcr,r->nkc", self._tf_color_factors, code))
            dc = dc + color_delta[:, 0, :]
            extra = self.tf_color_coeffs - 1
            if extra > 0:
                rest = torch.cat((rest[:, :extra, :] + color_delta[:, 1:, :],
                                  rest[:, extra:, :]), dim=1)
        shs = torch.cat((dc[:, None, :], rest), dim=1)

        if self.tf_condition_opacity:
            opacity_delta = torch.einsum("nr,r->n", self._tf_opacity_factors, code)
            logit = logit + self.tf_opacity_scale * opacity_delta[:, None]
        opacity = torch.sigmoid(logit)
        return shs, opacity * opacity_scale

    def _prune_optimizer(self, mask):
        tensors = super()._prune_optimizer(mask)
        if "tf_color_factors" in tensors:
            self._tf_color_factors = tensors["tf_color_factors"]
        if "tf_opacity_factors" in tensors:
            self._tf_opacity_factors = tensors["tf_opacity_factors"]
        if "tf_veg_u" in tensors:
            self._tf_veg_u = tensors["tf_veg_u"]
        for name in ("_tf_lookup_p", "_tf_lookup_q",
                     "_tf_lookup_ids", "_tf_lookup_counts"):
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
            if self._tf_veg_u is not None:
                self._append_factor(
                    "tf_veg_u", "_tf_veg_u",
                    self._tf_veg_u.detach()[nearest].clone(),
                )
            for name in ("_tf_lookup_p", "_tf_lookup_q",
                         "_tf_lookup_ids", "_tf_lookup_counts"):
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
            "version": 2,
            "tf_rank": self.tf_rank,
            "tf_hidden": self.tf_hidden,
            "tf_samples": self.tf_samples,
            "tf_ids": self._tf_ids,
            "tf_encoder_type": self.tf_encoder_type,
            "tf_condition_color": self.tf_condition_color,
            "tf_condition_opacity": self.tf_condition_opacity,
            "tf_color_sh_degree": self.tf_color_sh_degree,
            "tf_use_lookup": self.tf_use_lookup,
            "tf_veg_packed": self.tf_veg_packed,
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
        if self.tf_veg_packed and self._tf_veg_u is not None:
            payload["veg_u"] = self._tf_veg_u.detach().cpu()
        if self.tf_use_lookup and self.tf_lookup_ready:
            payload["lookup_gain"] = self._tf_lookup_gain.detach().cpu()
            if self._tf_lookup_ids is not None:
                payload["lookup_ids"] = self._tf_lookup_ids.detach().cpu()
                payload["lookup_counts"] = self._tf_lookup_counts.detach().cpu()
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
        if self.tf_veg_packed:
            if "veg_u" not in payload:
                raise ValueError("tf_veg_packed set but sidecar has no veg_u")
            self._tf_veg_u = nn.Parameter(
                payload["veg_u"].to("cuda").requires_grad_(True))
        if self.tf_use_lookup:
            self._tf_lookup_gain = nn.Parameter(
                payload["lookup_gain"].to("cuda").requires_grad_(True))
            if version >= 2:
                if self.tf_lookup_mode == "joint":
                    if "lookup_ids" not in payload:
                        raise ValueError("joint mode but v2 sidecar has no packed lookup")
                    self._tf_lookup_ids = payload["lookup_ids"].to("cuda")
                    self._tf_lookup_counts = payload["lookup_counts"].to("cuda")
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
                    raise ValueError("--tf_use_lookup set but sidecar has no lookup descriptors")

    def capture(self):
        state = {"base": super().capture(), "version": 2}
        if self._tf_color_factors is not None:
            state["color_factors"] = self._tf_color_factors
        if self._tf_opacity_factors is not None:
            state["opacity_factors"] = self._tf_opacity_factors
        if self.tf_encoder is not None:
            state["encoder"] = self.tf_encoder.state_dict()
        if self.tf_use_lookup and self.tf_lookup_ready:
            state["lookup_gain"] = self._tf_lookup_gain
            for key, name in (("lookup_ids", "_tf_lookup_ids"),
                              ("lookup_counts", "_tf_lookup_counts"),
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
        if self.tf_use_lookup and "lookup_gain" in model_args:
            self._tf_lookup_gain = nn.Parameter(
                model_args["lookup_gain"].to("cuda").requires_grad_(True))
            for key, name in (("lookup_ids", "_tf_lookup_ids"),
                              ("lookup_counts", "_tf_lookup_counts"),
                              ("lookup_p", "_tf_lookup_p"),
                              ("lookup_q", "_tf_lookup_q")):
                if key in model_args:
                    setattr(self, name, model_args[key].to("cuda"))
        super().restore(model_args["base"], training_args)
