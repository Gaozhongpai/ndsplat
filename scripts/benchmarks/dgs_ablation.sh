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

# ============================================
# Static scene benchmarks (input_dim=6)
# ============================================
echo ""
echo "[1/5] Running Medical PBR benchmark (static)..."
echo ""
bash "$SCRIPT_DIR/dgs_medical_pbr.sh"

echo ""
echo "[2/5] Running NeRF Synthetic benchmark (static)..."
echo ""
bash "$SCRIPT_DIR/dgs_nerf_synthetic.sh"

echo ""
echo "[3/5] Running Tanks & Temples PBR benchmark (static)..."
echo ""
bash "$SCRIPT_DIR/dgs_6dgs_pbr.sh"

# ============================================
# Dynamic scene benchmarks (input_dim=7, mv=4)
# ============================================
echo ""
echo "[4/5] Running D-NeRF benchmark (dynamic)..."
echo ""
bash "$SCRIPT_DIR/dgs_dnerf.sh"

echo ""
echo "[5/5] Running 7DGS PBR benchmark (dynamic)..."
echo ""
bash "$SCRIPT_DIR/dgs_7dgs_pbr.sh"

echo ""
echo "=============================================="
echo "All DGS ablation benchmarks completed!"
echo "=============================================="
