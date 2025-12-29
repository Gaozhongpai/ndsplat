#!/bin/bash

# Run DGS MCMC benchmarks on selected datasets
#
# Modes (all with MCMC densification):
# | Mode         | Output Dir                     | Description                            |
# |--------------|--------------------------------|----------------------------------------|
# | 3dgs         | output/mcmc/3dgs/...           | Baseline 3D Gaussian Splatting         |
# | opacity_only | output/mcmc/opacity_only/...   | Opacity conditioning only (no position)|
# | opacity_pos  | output/mcmc/opacity_pos/...    | Opacity + Position conditioning        |
# | ndgs         | output/mcmc/ndgs/...           | N-DGS with full Cholesky precision     |
#
# MCMC Parameters:
# - densification_strategy: mcmc
# - mcmc_cap_max: Scene-specific (300k for NeRF Synthetic, variable for 360_v2)
# - noise_lr: 1.0
# - opacity_reg: 0.01
# - scale_reg: 0.01

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Running DGS MCMC ablation benchmarks"
echo "=============================================="

# ============================================
# Static scene benchmarks with MCMC
# ============================================
echo ""
echo "[1/2] Running NeRF Synthetic benchmark (MCMC)..."
echo ""
bash "$SCRIPT_DIR/dgs_nerf_synthetic_mcmc.sh"

echo ""
echo "[2/2] Running Mip-NeRF 360 v2 benchmark (MCMC)..."
echo ""
bash "$SCRIPT_DIR/dgs_360_v2_mcmc.sh"

echo ""
echo "=============================================="
echo "All DGS MCMC ablation benchmarks completed!"
echo "=============================================="
