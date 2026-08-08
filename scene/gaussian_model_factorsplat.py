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
                 tf_condition_opacity: bool = True):
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

        self.tf_encoder = None
        self._tf_descriptors = None
        self._tf_ids = []
        self._tf_color_factors = torch.empty(0)
        self._tf_opacity_factors = torch.empty(0)

    def set_tf_bank(self, bank):
        rgba = np.asarray(bank["rgba"], dtype=np.float32)
        if rgba.ndim != 4 or rgba.shape[-1] != 4:
            raise ValueError(f"expected TF bank rgba [T,L,K,4], got {rgba.shape}")
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
        # Centering gives the authored base TF an exact zero code. Bias-free
        # layers consequently preserve the input dGS checkpoint at T_base.
        descriptors = descriptors - descriptors[base_index:base_index + 1]
        self._tf_descriptors = torch.tensor(descriptors, dtype=torch.float32, device="cuda")
        self._tf_ids = tf_ids
        input_dim = int(descriptors.shape[1])
        if self.tf_encoder is None:
            self.tf_encoder = nn.Sequential(
                nn.Linear(input_dim, self.tf_hidden, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(self.tf_hidden, self.tf_rank, bias=False),
            ).cuda()
        elif self.tf_encoder[0].in_features != input_dim:
            raise ValueError("TF descriptor dimension changed after encoder initialization")

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

    def _tf_code(self, viewpoint_camera):
        index = getattr(viewpoint_camera, "tf_index", None)
        if index is None:
            return None
        index = int(index)
        if not 0 <= index < self._tf_descriptors.shape[0]:
            raise IndexError(f"tf_index={index} outside bank of size "
                             f"{self._tf_descriptors.shape[0]}")
        return self.tf_encoder(self._tf_descriptors[index])

    def conditioned_appearance(self, viewpoint_camera, opacity_scale):
        code = self._tf_code(viewpoint_camera)
        if code is None:
            return super().conditioned_appearance(viewpoint_camera, opacity_scale)

        if self.tf_condition_color:
            color_delta = torch.einsum("ncr,r->nc", self._tf_color_factors, code)
            dc = self._features_dc[:, 0, :] + self.tf_color_scale * torch.tanh(color_delta)
            shs = torch.cat((dc[:, None, :], self._features_rest), dim=1)
        else:
            shs = self.get_features

        if self.tf_condition_opacity:
            opacity_delta = torch.einsum("nr,r->n", self._tf_opacity_factors, code)
            opacity = torch.sigmoid(
                self._opacity + self.tf_opacity_scale * opacity_delta[:, None]
            )
        else:
            opacity = self.get_opacity
        return shs, opacity * opacity_scale

    def _prune_optimizer(self, mask):
        tensors = super()._prune_optimizer(mask)
        if "tf_color_factors" in tensors:
            self._tf_color_factors = tensors["tf_color_factors"]
        if "tf_opacity_factors" in tensors:
            self._tf_opacity_factors = tensors["tf_opacity_factors"]
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

    @staticmethod
    def _sidecar(path):
        return path + ".factorsplat.pt"

    def save_ply(self, path):
        super().save_ply(path)
        encoder_state = {key: value.detach().cpu()
                         for key, value in self.tf_encoder.state_dict().items()}
        torch.save({
            "version": 1,
            "tf_rank": self.tf_rank,
            "tf_hidden": self.tf_hidden,
            "tf_samples": self.tf_samples,
            "tf_ids": self._tf_ids,
            "color_factors": self._tf_color_factors.detach().cpu(),
            "opacity_factors": self._tf_opacity_factors.detach().cpu(),
            "encoder": encoder_state,
        }, self._sidecar(path))

    def load_ply(self, path):
        super().load_ply(path)
        sidecar = self._sidecar(path)
        if not os.path.isfile(sidecar):
            self._initialize_tf_factors(self.get_xyz.shape[0])
            return
        payload = torch.load(sidecar, map_location="cuda")
        if int(payload["tf_rank"]) != self.tf_rank:
            raise ValueError("FactorSplat sidecar rank does not match --tf_rank")
        color = payload["color_factors"].to(device="cuda", dtype=torch.float32)
        opacity = payload["opacity_factors"].to(device="cuda", dtype=torch.float32)
        if color.shape[0] != self.get_xyz.shape[0] or opacity.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("FactorSplat sidecar and PLY have different Gaussian counts")
        self._tf_color_factors = nn.Parameter(color.requires_grad_(True))
        self._tf_opacity_factors = nn.Parameter(opacity.requires_grad_(True))
        self.tf_encoder.load_state_dict(payload["encoder"])

    def capture(self):
        return {
            "base": super().capture(),
            "color_factors": self._tf_color_factors,
            "opacity_factors": self._tf_opacity_factors,
            "encoder": self.tf_encoder.state_dict(),
        }

    def restore(self, model_args, training_args):
        self._tf_color_factors = nn.Parameter(
            model_args["color_factors"].to("cuda").requires_grad_(True)
        )
        self._tf_opacity_factors = nn.Parameter(
            model_args["opacity_factors"].to("cuda").requires_grad_(True)
        )
        self.tf_encoder.load_state_dict(model_args["encoder"])
        super().restore(model_args["base"], training_args)
