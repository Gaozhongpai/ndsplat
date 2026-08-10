#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got {v}')

class GroupParams:
    pass

class ParamGroup:
    # Parameters that should accept explicit True/False values from command line
    EXPLICIT_BOOL_PARAMS = {
        'use_view_dependent_pos',
        'use_rot_scale_l_triangle',
        'use_opacity_pos_decouple',
        'tf_condition_color',
        'tf_condition_opacity',
        'tf_use_lookup',
        'tf_aware_prune',
        'tf_veg_packed',
        'tf_encoder_pooled',
        'tf_encoder_local',
        'tf_refresh_descriptors',
        'tf_uniform_sample_weights',
    }

    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    # Use explicit bool type for configurable parameters, store_true for others
                    if key in self.EXPLICIT_BOOL_PARAMS:
                        group.add_argument("--" + key, default=value, type=str2bool, nargs='?', const=True)
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.seed = 0  # RNG seed for random/numpy/torch. Default 0 reproduces all previously published runs.
        self.mode = "dgs"  # Options: "3dgs", "ndgs", "ubs", "dgs", "dbs", "dbs-sh"
        self.input_dim = 6  # Gaussian dimension: 6 for 6DGS/UBS, 7 for 7DGS (with time)
        self.use_rot_scale_l_triangle = False  # If True: use rotation-scale-l_triangle (UBS-style), If False: use diagonal-l_triangle (NDGS-style)
        self.learnable_lambda_opc = False  # If True: make lambda_opc a learnable parameter per Gaussian
        self.use_jpeg_compression = False  # If True: use JPEG compression for images to save GPU memory (slower but memory-efficient)
        # DGS view-dependent flags (only used when mode="dgs")
        self.use_view_dependent_pos = True  # Enable view-dependent position shift
        self.use_opacity_pos_decouple = False  # If True: decouple position and opacity by setting lambda_view=lambda_time=0 (not learnable)
        self.direct_unrestricted = False  # Ablation (NeurIPS'26 rebuttal): replace dGS's constrained factorization
                                          # M = V_pq diag(Lambda) V_qq with a FREE matrix M in R^{3xC}, so that
                                          # (Sigma_cond, M, V_qq) is the unrestricted conditional tuple -- a bijective
                                          # reparameterization of the joint Gaussian covariance (Appendix D).
                                          # N-DGS vs. this isolates the coordinate change; this vs. dGS isolates the
                                          # spatial normalization, bounded Lambda, and structured regression prior.
        self.l_22_inv_init_scale = 1.0  # Initialization scale for L_22_inv diagonal (1.0 for standard, 2.0 for PBR scenes)
        self.lambda_init = -1.2  # Initial value for lambda_view and lambda_time parameters
        self.beta_init_view = -3.0  # dBS-only: raw init for view beta dims (7D); activated beta = 4*exp(beta_init_view)
        self.lambda_opc = 0.35  # Default lambda_opc for opacity scaling (0.35 standard, 0.01 for dnerf, 0.2 for PBR)
        self.use_gsplat = False  # If True: use gsplat rasterizer instead of TCGS for UBS/DGS modes
        self.mip3dgs = False  # Mip-Splatting: 3D smoothing filter + 2D antialiasing (Gaussian-kernel TCGS modes: dgs/ndgs)
        self.clip_operator = "analytic"  # XClipGS clip operator: "analytic" (Ours, exact half-space), "moment" (MM, moment-matched truncation), "hardcull" (HC, per-primitive keep/drop). dgs mode.
        # ClipGS baseline reimpl (--mode clipgs): 3DGS + STE hard-cull + deform MLP.
        self.clipgs_deform_scale = False  # let the deform MLP also predict log-scale offsets (else position-only)
        self.clipgs_deform_lr = 1e-4      # learning rate for the ClipGS deformation MLP
        # FactorSplat: functional transfer-function conditioning on dGS appearance.
        self.tf_rank = 8
        self.tf_hidden = 64
        self.tf_samples = 32
        self.tf_encoder_type = "functional"
        self.tf_embedding_fallback = "nearest"
        self.tf_color_scale = 0.25
        self.tf_opacity_scale = 4.0
        self.tf_condition_color = True
        self.tf_condition_opacity = True
        # SH degree of the conditioned color residual: 1 = DC + first-order
        # band (the standard model at --sh_degree 1); 0 = DC-only ablation.
        self.tf_color_sh_degree = 1
        # Local-lookup branch (Eq. 6): per-Gaussian label/intensity descriptors
        # predict the zeroth-order RGBA edit directly from the preset.
        # Prune on max-over-training-presets opacity so no preset-revealed
        # anatomy is deleted for having a low shared/base opacity.
        self.tf_aware_prune = True
        self.tf_use_lookup = False
        # Adapted VEG reference: per-Gaussian scalar + packed 1D LUT readout.
        self.tf_veg_packed = False
        # Label-order-invariant pooled TF encoder (per-curve phi + mean pool + psi)
        self.tf_encoder_pooled = False
        # Local functional residual: z_{i,T} = mean_k phi(dR_T at the primitive's
        # own (region, bin) samples). Uses the packed descriptor independently
        # of whether the physical lookup branch is enabled.
        self.tf_encoder_local = False
        # Re-sample descriptors (ids + density weights) after each densification
        # step, from the current positions/covariances. Needs
        # points3d_refresh_grid.npz in the dataset root.
        self.tf_refresh_descriptors = False
        # Ablation: force uniform 1/K_i sample weights (ignore stored/refresh
        # density weights) so the density-weighting component can be removed
        # without regenerating datasets.
        self.tf_uniform_sample_weights = False
        self.tf_lookup_mode = "joint"  # joint = packed empirical p(l,h); separable = p(l)q(h) ablation
        self.tf_lookup_bins = 64
        self.tf_lookup_color_scale = 1.0
        self.tf_lookup_opacity_scale = 4.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.mv = 1
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Training iterations
        self.iterations = 30_000

        # Position learning rates (3DGS-style with scheduling)
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        # 3DGS learning rates
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.m_lr = 0.0   # LR for the free regression operator M (--direct_unrestricted only).
                          # 0.0 => reuse rotation_lr (original behavior). Set explicitly to avoid
                          # the scale mismatch: dGS's rotation_lr acts on a unit-normalized
                          # direction later multiplied by s-bar, so a raw M wants ~s-bar times
                          # that value (s-bar median 3e-3..6e-3 => m_lr ~3e-6..6e-6).

        # UBS-specific learning rates
        self.mean_lr = 0.001
        self.beta_lr = 0.001
        self.rgb_lr = 0.001
        self.tf_factor_lr = 0.0025
        self.tf_encoder_lr = 0.001
        self.tf_veg_u_lr = 0.5  # adapted-VEG scalar step, in LUT-index units/iter
        self.scale_lr = 0.005
        self.l_triangle_lr = 0.001

        # Densification parameters
        self.percent_dense = 0.01
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.opacity_reset_interval = 3000

        # Densification strategy: "standard" or "mcmc"
        self.densification_strategy = "standard"

        # Mip-Splatting: refresh cadence of the 3D smoothing filter (iterations).
        # The filter depends on Gaussian positions, so it is also refreshed
        # whenever densification changes the primitive count.
        self.mip_filter_interval = 100

        # MCMC-specific parameters (only used when densification_strategy="mcmc")
        self.mcmc_cap_max = 300_000  # Maximum number of Gaussians
        self.mcmc_refine_interval = 100  # Interval for MCMC refinement
        self.mcmc_densify_until_iter = 25_000  # MCMC densifies longer than standard (25k vs 15k)
        self.noise_lr = 1.0  # Noise learning rate for MCMC spatial perturbation
        self.opacity_reg = 0.01  # Opacity regularization weight for MCMC
        self.scale_reg = 0.01  # Scale regularization weight for MCMC

        # Loss parameters
        self.lambda_dssim = 0.2
        self.random_background = False

        super().__init__(parser, "Optimization Parameters")

class ViewerParams(ParamGroup):
    def __init__(self, parser):
        self.port = 8080
        self.disable_viewer = False
        super().__init__(parser, "Viewer Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
