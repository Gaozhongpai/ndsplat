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
        'use_view_dependent_rotation',
        'use_rot_scale_l_triangle',
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
        self.mode = "dgs"  # Options: "ddgs", "3dgs", "ubs", "ndgs", "ndgs-2sh", "ndgs-color", "dgs", "dgs-color"
        self.input_dim = 6  # Gaussian dimension: 6 for 6DGS/UBS, 7 for 7DGS (with time)
        self.use_rot_scale_l_triangle = False  # If True: use rotation-scale-l_triangle (UBS-style), If False: use diagonal-l_triangle (NDGS-style)
        self.learnable_lambda_opc = False  # If True: make lambda_opc a learnable parameter per Gaussian
        self.use_jpeg_compression = False  # If True: use JPEG compression for images to save GPU memory (slower but memory-efficient)
        # DGS view-dependent flags (only used when mode="dgs")
        self.use_view_dependent_pos = True  # Enable view-dependent position shift
        self.use_view_dependent_rotation = True  # Enable time-dependent rotation (only when input_dim=7)
        self.l_22_inv_init_scale = 1.0  # Initialization scale for L_22_inv diagonal (1.0 for standard, 2.0 for PBR scenes)
        self.lambda_opc = 0.35  # Default lambda_opc for opacity scaling (0.35 standard, 0.01 for dnerf, 0.2 for PBR)
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

        # UBS-specific learning rates
        self.mean_lr = 0.0001
        self.beta_lr = 0.0001
        self.rgb_lr = 0.0025
        self.scale_lr = 0.005
        self.l_triangle_lr = 0.001

        # Densification parameters
        self.percent_dense = 0.01
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.opacity_reset_interval = 3000

        # Densification strategy: "standard", "mcmc", or "fastgs"
        self.densification_strategy = "standard"  # Options: "standard" (gradient-based), "mcmc" (MCMC sampling), "fastgs" (multi-view consistent)

        # MCMC-specific parameters (only used when densification_strategy="mcmc")
        self.mcmc_cap_max = 300_000  # Maximum number of Gaussians
        self.mcmc_refine_interval = 100  # Interval for MCMC refinement
        self.mcmc_densify_until_iter = 25_000  # MCMC densifies longer than standard (25k vs 15k)
        self.mcmc_add_rate = 0.25  # Rate of adding new Gaussians (fraction per refinement)
        self.mcmc_remove_rate = 0.1  # Rate of removing Gaussians (fraction per refinement)
        self.noise_lr = 1.0  # Noise learning rate for MCMC spatial perturbation (matching UBS)
        self.opacity_reg = 0.01  # Opacity regularization weight for MCMC
        self.scale_reg = 0.01  # Scale regularization weight for MCMC

        # NDGS position shift regularization (constrains position shift to stay within spatial scale)
        self.shift_reg = 0.0  # Weight for position shift regularization (0 = disabled, try 0.01-0.1)
        self.max_shift_ratio = 2.0  # Maximum position shift ratio relative to spatial scale

        # FastGS-specific parameters (only used when densification_strategy="fastgs")
        # Adapted from FastGS (arXiv:2511.04283) for multi-view consistent densification
        self.fastgs_loss_thresh = 0.3  # Threshold for high-error pixel detection
        self.fastgs_grad_thresh = 0.0002  # Gradient threshold for cloning (XY screen-space gradients)
        self.fastgs_grad_abs_thresh = 0.0004  # Gradient threshold for splitting (Z depth gradients) - FastGS big default
        self.fastgs_densify_score_thresh = 2  # Minimum importance score for densification
        self.fastgs_prune_budget_ratio = 0.2  # Fraction of prunable Gaussians to actually prune
        self.fastgs_final_prune_interval = 3000  # Interval for final pruning after 15k iterations
        self.fastgs_final_prune_start = 15_000  # Start iteration for final pruning
        self.fastgs_final_prune_end = 30_000  # End iteration for final pruning
        self.fastgs_num_sample_cams = 10  # Number of cameras to sample for multi-view scoring

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
