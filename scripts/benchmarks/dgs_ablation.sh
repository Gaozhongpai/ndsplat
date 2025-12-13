#!/bin/bash

# Run DGS benchmarks on all datasets
#
# Modes:
# | Mode     | Output Dir              | Description                                    |
# |----------|-------------------------|------------------------------------------------|
# | dgs      | output/dgs/...          | DGS with bounded v_12 parameterization         |
# | ndgs     | output/ndgs/...         | N-DGS with full Cholesky precision             |
# | dgs-full | output/dgs-full_*/...   | DGS-full with configurable view-dependent flags|
#
# DGS-full Configurations:
# | Config       | Output Dir                        | pos | scale | rot |
# |--------------|-----------------------------------|-----|-------|-----|
# | no_view_dep  | output/dgs-full_no_view_dep/...   |  -  |   -   |  -  |
# | pos_only     | output/dgs-full_pos_only/...      |  +  |   -   |  -  |
# | pos_rot      | output/dgs-full_pos_rot/...       |  +  |   -   |  +  |
# | rot_only     | output/dgs-full_rot_only/...      |  -  |   -   |  +  |

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
