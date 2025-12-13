#!/bin/bash

# Run DGS benchmarks on all datasets
#
# Modes:
# | Mode         | Output Dir                | Description                            |
# |--------------|---------------------------|----------------------------------------|
# | opacity_only | output/opacity_only/...   | Opacity conditioning only (no position)|
# | opacity_pos  | output/opacity_pos/...    | Opacity + Position conditioning        |
# | ndgs         | output/ndgs/...           | N-DGS with full Cholesky precision     |
#
# Note: Rotation conditioning is only available for dynamic scenes (C=4 with time)
# Note: Scale is NOT view-dependent (use get_scaling directly)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Running DGS ablation benchmarks"
echo "=============================================="

echo ""
echo "[1/2] Running NeRF Synthetic benchmark..."
echo ""
bash "$SCRIPT_DIR/dgs_nerf_synthetic.sh"

echo ""
echo "[2/2] Running Tanks & Temples PBR benchmark..."
echo ""
bash "$SCRIPT_DIR/dgs_6dgs_pbr.sh"

echo ""
echo "=============================================="
echo "All DGS ablation benchmarks completed!"
echo "=============================================="
