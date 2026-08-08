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
                 tf_encoder_type: str = "functional",
                 tf_embedding_fallback: str = "nearest",
                 tf_aware_prune: bool = True,
                 tf_use_lookup: bool = False,
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
        self.tf_lookup_bins = int(tf_lookup_bins)
        self.tf_lookup_color_scale = float(tf_lookup_color_scale)
        self.tf_lookup_opacity_scale = float(tf_lookup_opacity_scale)
        self._tf_bank_lookup = None      # [T, L, B, 4] raw RGBA, downsampled bins
        self._tf_base_index = 0
        self._tf_label_ids = None
        self._tf_lookup_p = None         # [N, L] per-Gaussian label distribution
        self._tf_lookup_q = None         # [N, B] per-Gaussian intensity histogram
        # Joint descriptor p_i(l,h): the window's voxel samples (label col, HU
        # bin, weight). Exact empirical joint -- preferred over the separable
        # p*q approximation when present.
        self._tf_lookup_l = None         # [N, K] int16 label cols (0 where invalid)
        self._tf_lookup_b = None         # [N, K] int16 HU bins at bank resolution
        self._tf_lookup_w = None         # [N, K] float weights (0 where invalid)
        self._tf_lookup_gain = None      # learned global RGBA gain (4,)

        self.tf_encoder = None
        self._tf_descriptors = None
        self._tf_ids = []
        self._tf_color_factors = torch.empty(0)
        self._tf_opacity_factors = torch.empty(0)

    def set_tf_bank(self, bank):
        rgba = np.asarray(bank["rgba"], dtype=np.float32)
        if rgba.ndim != 4 or rgba.shape[-1] != 4:
            raise ValueError(f"expected TF bank rgba [T,L,K,4], got {rgba.shape}")
        if self.tf_use_lookup:
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
        if self.tf_encoder is None:
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
        elif self.tf_encoder_type == "functional" and self.tf_encoder[0].in_features != input_dim:
            raise ValueError("TF descriptor dimension changed after encoder initialization")

    def load_lookup_descriptors(self, path):
        """Attach per-Gaussian (label distribution, intensity histogram) sampled
        from the volume+mask at init (factorsplat_lookup_descriptors.py). Must be
        called after the init PLY is loaded; row order must match."""
        if not self.tf_use_lookup:
            return
        if self._tf_bank_lookup is None:
            raise RuntimeError("set_tf_bank must run before load_lookup_descriptors")
        with np.load(path) as payload:
            probs = np.asarray(payload["label_probs"], dtype=np.float32)
            hist = np.asarray(payload["intensity_hist"], dtype=np.float32)
            label_ids = np.asarray(payload["label_ids"]).astype(int).tolist()
            sample_cols = (np.asarray(payload["sample_label_cols"], dtype=np.int64)
                           if "sample_label_cols" in payload else None)
            sample_bins = (np.asarray(payload["sample_bins"], dtype=np.int64)
                           if "sample_bins" in payload else None)
        if self._tf_label_ids is not None and label_ids != self._tf_label_ids:
            raise ValueError("lookup descriptor label order does not match tf_bank")
        count = self.get_xyz.shape[0]
        if probs.shape[0] != count or hist.shape[0] != count:
            raise ValueError(f"lookup descriptors cover {probs.shape[0]} Gaussians "
                             f"but the model has {count}")
        if probs.shape[1] != self._tf_bank_lookup.shape[1]:
            raise ValueError("lookup descriptor label axis does not match tf_bank")
        bins = self._tf_bank_lookup.shape[2]
        native = int(hist.shape[1])          # npz HU-grid resolution (pre-rebin)
        if hist.shape[1] % bins == 0:
            # Histograms are distributions: rebin by summation.
            hist = hist.reshape(hist.shape[0], bins, hist.shape[1] // bins).sum(axis=2)
        elif hist.shape[1] != bins:
            raise ValueError(f"cannot rebin {hist.shape[1]}-bin histograms to {bins}")
        self._tf_lookup_p = torch.tensor(probs, device="cuda")
        self._tf_lookup_q = torch.tensor(hist, device="cuda")
        mode = "separable p*q"
        if sample_cols is not None and sample_bins is not None:
            if sample_cols.shape[0] != count:
                raise ValueError("joint sample arrays do not match Gaussian count")
            valid = sample_cols >= 0
            weights = valid.astype(np.float32)
            denom = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, denom, out=np.zeros_like(weights),
                                where=denom > 0)
            rebin = max(native // bins, 1)
            self._tf_lookup_l = torch.tensor(
                np.where(valid, sample_cols, 0).astype(np.int16), device="cuda")
            self._tf_lookup_b = torch.tensor(
                np.clip(sample_bins // rebin, 0, bins - 1).astype(np.int16),
                device="cuda")
            self._tf_lookup_w = torch.tensor(weights, device="cuda")
            mode = f"joint p(l,h) ({sample_cols.shape[1]} samples/Gaussian)"
        covered = float((probs.sum(axis=1) > 0).mean())
        print(f"Loaded lookup descriptors for {count} Gaussians "
              f"({covered:.1%} with foreground support, {mode}) from {path}")

    def _lookup_delta(self, tf_index):
        """Eq. 6: locally relevant RGBA change of the preset relative to the
        base preset. Uses the exact empirical joint p_i(l,h) (window voxel
        samples) when available, else the separable p_i(l) q_i(h) product.
        Returns [N, 4]."""
        bank = self._tf_bank_lookup
        delta_r = bank[tf_index] - bank[self._tf_base_index]            # [L,B,4]
        if self._tf_lookup_l is not None:
            vals = delta_r[self._tf_lookup_l.long(),
                           self._tf_lookup_b.long()]                    # [N,K,4]
            return (vals * self._tf_lookup_w.unsqueeze(-1)).sum(dim=1)
        per_label = torch.einsum("nb,lbc->nlc", self._tf_lookup_q, delta_r)
        return torch.einsum("nl,nlc->nc", self._tf_lookup_p, per_label)

    def _initialize_tf_factors(self, count):
        device = self._xyz.device
        self._tf_color_factors = nn.Parameter(
            (1e-3 * torch.randn(count, 3, self.tf_rank, device=device)).requires_grad_(True)
        )
        self._tf_opacity_factors = nn.Parameter(
            (1e-3 * torch.randn(count, self.tf_rank, device=device)).requires_grad_(True)
        )

    def create_from_pcd(self, pcd, spatial_lr_scale, mcmc_cap_max=None,
                        densification_strategy="standard"):
        super().create_from_pcd(pcd, spatial_lr_scale, mcmc_cap_max,
                                densification_strategy)
        self._initialize_tf_factors(self.get_xyz.shape[0])

    def training_setup(self, training_args):
        if self.tf_encoder is None or self._tf_descriptors is None:
            raise RuntimeError("FactorSplat requires tf_bank.npz in the dataset root")
        if getattr(training_args, "densification_strategy", "standard") != "standard":
            raise ValueError("FactorSplat currently supports standard densification only")
        if self._tf_color_factors.shape[0] != self.get_xyz.shape[0]:
            self._initialize_tf_factors(self.get_xyz.shape[0])
        super().training_setup(training_args)
        self.optimizer.add_param_group({
            "params": [self._tf_color_factors],
            "lr": training_args.tf_factor_lr,
            "name": "tf_color_factors",
            "per_gaussian": True,
        })
        self.optimizer.add_param_group({
            "params": [self._tf_opacity_factors],
            "lr": training_args.tf_factor_lr,
            "name": "tf_opacity_factors",
            "per_gaussian": True,
        })
        self.optimizer.add_param_group({
            "params": list(self.tf_encoder.parameters()),
            "lr": training_args.tf_encoder_lr,
            "name": "tf_encoder",
            "per_gaussian": False,
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

    def _tf_code(self, viewpoint_camera):
        index = getattr(viewpoint_camera, "tf_index", None)
        if index is None:
            return None
        index = int(index)
        if not 0 <= index < self._tf_descriptors.shape[0]:
            raise IndexError(f"tf_index={index} outside bank of size "
                             f"{self._tf_descriptors.shape[0]}")
        return self._code_for_index(index)

    def get_pruning_opacity(self):
        """TF-aware pruning opacity: max over TRAINING presets of the
        conditioned opacity (view gate excluded; it is TF-independent).
        A primitive that some training preset reveals must not be deleted
        because the shared/base logit alone falls below the threshold --
        that would permanently remove anatomy other presets need."""
        if (not self.tf_aware_prune or self._tf_descriptors is None
                or not self._tf_train_rows):
            return self.get_opacity
        with torch.no_grad():
            best_logit = None
            for row in self._tf_train_rows:
                logit = self._opacity
                if self.tf_use_lookup and self._tf_lookup_p is not None:
                    delta = self._lookup_delta(row)
                    logit = logit + self.tf_lookup_opacity_scale *                         self._tf_lookup_gain[3] * delta[:, 3:4]
                if self.tf_condition_opacity:
                    code = self._code_for_index(row)
                    offset = torch.einsum("nr,r->n", self._tf_opacity_factors, code)
                    logit = logit + self.tf_opacity_scale * offset[:, None]
                best_logit = logit if best_logit is None                     else torch.maximum(best_logit, logit)
            return torch.sigmoid(best_logit)

    def conditioned_appearance(self, viewpoint_camera, opacity_scale):
        code = self._tf_code(viewpoint_camera)
        if code is None:
            return super().conditioned_appearance(viewpoint_camera, opacity_scale)

        dc = self._features_dc[:, 0, :]
        logit = self._opacity
        if self.tf_use_lookup and self._tf_lookup_p is not None:
            delta = self._lookup_delta(int(viewpoint_camera.tf_index))   # [N,4]
            # RGB deltas live in [0,1] LUT units; SH DC coeffs relate to RGB via
            # rgb = 0.5 + C0*dc, so the conversion to DC space divides by C0.
            C0 = 0.28209479177387814
            dc = dc + (self.tf_lookup_color_scale / C0) * \
                self._tf_lookup_gain[:3] * delta[:, :3]
            logit = logit + self.tf_lookup_opacity_scale * \
                self._tf_lookup_gain[3] * delta[:, 3:4]

        if self.tf_condition_color:
            color_delta = torch.einsum("ncr,r->nc", self._tf_color_factors, code)
            dc = dc + self.tf_color_scale * torch.tanh(color_delta)
        shs = torch.cat((dc[:, None, :], self._features_rest), dim=1)

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
        if self._tf_lookup_p is not None and mask.shape[0] == self._tf_lookup_p.shape[0]:
            for name in ("_tf_lookup_p", "_tf_lookup_q", "_tf_lookup_l",
                         "_tf_lookup_b", "_tf_lookup_w"):
                tensor = getattr(self, name)
                if tensor is not None:
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
            self._append_factor(
                "tf_color_factors", "_tf_color_factors",
                torch.zeros(added, 3, self.tf_rank, device=self._xyz.device),
            )
            self._append_factor(
                "tf_opacity_factors", "_tf_opacity_factors",
                torch.zeros(added, self.tf_rank, device=self._xyz.device),
            )
            if self._tf_lookup_p is not None:
                # Descriptors are fixed volume samples, so children inherit them
                # from the nearest pre-existing Gaussian (clones sit exactly on
                # their parent; split children land within the parent footprint).
                old = self.get_xyz[: self.get_xyz.shape[0] - added].detach()
                nearest = []
                for chunk in new_xyz.detach().split(1024):
                    nearest.append(torch.cdist(chunk, old).argmin(dim=1))
                nearest = torch.cat(nearest)
                for name in ("_tf_lookup_p", "_tf_lookup_q", "_tf_lookup_l",
                             "_tf_lookup_b", "_tf_lookup_w"):
                    tensor = getattr(self, name)
                    if tensor is not None:
                        setattr(self, name,
                                torch.cat((tensor, tensor[nearest]), dim=0))

    @staticmethod
    def _sidecar(path):
        return path + ".factorsplat.pt"

    def save_ply(self, path):
        super().save_ply(path)
        encoder_state = {key: value.detach().cpu()
                         for key, value in self.tf_encoder.state_dict().items()}
        payload = {
            "version": 1,
            "tf_rank": self.tf_rank,
            "tf_hidden": self.tf_hidden,
            "tf_samples": self.tf_samples,
            "tf_ids": self._tf_ids,
            "tf_encoder_type": self.tf_encoder_type,
            "color_factors": self._tf_color_factors.detach().cpu(),
            "opacity_factors": self._tf_opacity_factors.detach().cpu(),
            "encoder": encoder_state,
        }
        if self.tf_use_lookup and self._tf_lookup_p is not None:
            payload["lookup_p"] = self._tf_lookup_p.detach().cpu()
            payload["lookup_q"] = self._tf_lookup_q.detach().cpu()
            payload["lookup_gain"] = self._tf_lookup_gain.detach().cpu()
            for key, name in (("lookup_l", "_tf_lookup_l"),
                              ("lookup_b", "_tf_lookup_b"),
                              ("lookup_w", "_tf_lookup_w")):
                tensor = getattr(self, name)
                if tensor is not None:
                    payload[key] = tensor.detach().cpu()
        torch.save(payload, self._sidecar(path))

    def load_ply(self, path):
        super().load_ply(path)
        sidecar = self._sidecar(path)
        if not os.path.isfile(sidecar):
            self._initialize_tf_factors(self.get_xyz.shape[0])
            return
        payload = torch.load(sidecar, map_location="cuda")
        if int(payload["tf_rank"]) != self.tf_rank:
            raise ValueError("FactorSplat sidecar rank does not match --tf_rank")
        saved_type = payload.get("tf_encoder_type", "functional")
        if saved_type != self.tf_encoder_type:
            raise ValueError(f"sidecar encoder type {saved_type} != --tf_encoder_type {self.tf_encoder_type}")
        color = payload["color_factors"].to(device="cuda", dtype=torch.float32)
        opacity = payload["opacity_factors"].to(device="cuda", dtype=torch.float32)
        if color.shape[0] != self.get_xyz.shape[0] or opacity.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("FactorSplat sidecar and PLY have different Gaussian counts")
        self._tf_color_factors = nn.Parameter(color.requires_grad_(True))
        self._tf_opacity_factors = nn.Parameter(opacity.requires_grad_(True))
        self.tf_encoder.load_state_dict(payload["encoder"])
        if self.tf_use_lookup:
            if "lookup_p" not in payload:
                raise ValueError("--tf_use_lookup set but sidecar has no lookup descriptors")
            self._tf_lookup_p = payload["lookup_p"].to("cuda")
            self._tf_lookup_q = payload["lookup_q"].to("cuda")
            self._tf_lookup_gain = nn.Parameter(
                payload["lookup_gain"].to("cuda").requires_grad_(True))
            for key, name in (("lookup_l", "_tf_lookup_l"),
                              ("lookup_b", "_tf_lookup_b"),
                              ("lookup_w", "_tf_lookup_w")):
                if key in payload:
                    setattr(self, name, payload[key].to("cuda"))

    def capture(self):
        state = {
            "base": super().capture(),
            "color_factors": self._tf_color_factors,
            "opacity_factors": self._tf_opacity_factors,
            "encoder": self.tf_encoder.state_dict(),
        }
        if self.tf_use_lookup and self._tf_lookup_p is not None:
            state["lookup_p"] = self._tf_lookup_p
            state["lookup_q"] = self._tf_lookup_q
            state["lookup_gain"] = self._tf_lookup_gain
            for key, name in (("lookup_l", "_tf_lookup_l"),
                              ("lookup_b", "_tf_lookup_b"),
                              ("lookup_w", "_tf_lookup_w")):
                tensor = getattr(self, name)
                if tensor is not None:
                    state[key] = tensor
        return state

    def restore(self, model_args, training_args):
        self._tf_color_factors = nn.Parameter(
            model_args["color_factors"].to("cuda").requires_grad_(True)
        )
        self._tf_opacity_factors = nn.Parameter(
            model_args["opacity_factors"].to("cuda").requires_grad_(True)
        )
        self.tf_encoder.load_state_dict(model_args["encoder"])
        if self.tf_use_lookup and "lookup_p" in model_args:
            self._tf_lookup_p = model_args["lookup_p"].to("cuda")
            self._tf_lookup_q = model_args["lookup_q"].to("cuda")
            self._tf_lookup_gain = nn.Parameter(
                model_args["lookup_gain"].to("cuda").requires_grad_(True))
            for key, name in (("lookup_l", "_tf_lookup_l"),
                              ("lookup_b", "_tf_lookup_b"),
                              ("lookup_w", "_tf_lookup_w")):
                if key in model_args:
                    setattr(self, name, model_args[key].to("cuda"))
        super().restore(model_args["base"], training_args)
