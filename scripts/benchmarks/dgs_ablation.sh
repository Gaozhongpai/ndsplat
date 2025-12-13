#!/bin/bash

# Run DGS view-dependent ablation benchmarks on all datasets
#
# Configurations:
# | Config       | Output Dir                   | pos | scale | rot |
# |--------------|------------------------------|-----|-------|-----|
# | no_view_dep  | output/dgs_no_view_dep/...   |  -  |   -   |  -  |
# | pos_only     | output/dgs_pos_only/...      |  +  |   -   |  -  |
# | pos_scale    | output/dgs_pos_scale/...     |  +  |   +   |  -  |
# | full         | output/dgs_full/...          |  +  |   +   |  +  |
# | scale_rot    | output/dgs_scale_rot/...     |  -  |   +   |  +  |

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Running DGS ablation benchmarks"
echo "=============================================="

echo ""
echo "[1/2] Running Tanks & Temples PBR benchmark..."
echo ""
bash "$SCRIPT_DIR/dgs_6dgs_pbr.sh"

echo ""
echo "[2/2] Running NeRF Synthetic benchmark..."
echo ""
bash "$SCRIPT_DIR/dgs_nerf_synthetic.sh"

echo ""
echo "=============================================="
echo "All DGS ablation benchmarks completed!"
echo "=============================================="
