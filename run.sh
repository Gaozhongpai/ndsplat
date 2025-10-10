#!/bin/bash

# Master script for running 6DGS experiments
# Usage: ./run.sh <category> <script>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
    cat << USAGE
6DGS Experiment Runner

Usage: ./run.sh <category> <script>

Categories:
  benchmark    - Run standard benchmark evaluations
  ablation     - Run ablation studies
  test         - Run test/debug experiments

Benchmarks:
  ./run.sh benchmark mipnerf360        - Mip-NeRF 360 scenes (room, stump, treehill)
  ./run.sh benchmark nerf_synthetic    - All NeRF synthetic scenes
  ./run.sh benchmark tanks_temples     - Tanks & Temples volumetric scenes
  ./run.sh benchmark shiny_blender     - Shiny Blender scenes (8 scenes)

Ablations:
  ./run.sh ablation nerf_synthetic     - NeRF synthetic ablation (lego)
  ./run.sh ablation deepdrr            - DeepDRR clinic datasets
  ./run.sh ablation deepdrr_entangled  - DeepDRR entangled ablation

Tests:
  ./run.sh test ct_data                - CT data (chest, foot, abdomen, jaw)
  ./run.sh test deepdrr                - DeepDRR test
  ./run.sh test naf_ct                 - NAF CT test data
  ./run.sh test dicom                  - DICOM CT data
  ./run.sh test ljubljana_scaled       - Ljubljana scaled datasets
  ./run.sh test dgs_tanks_temples      - DGS Tanks & Temples
  ./run.sh test dgs_nerf               - DGS NeRF synthetic

Examples:
  ./run.sh benchmark mipnerf360
  ./run.sh ablation nerf_synthetic
  ./run.sh test ct_data

List available scripts:
  ./run.sh list

USAGE
}

list_scripts() {
    echo "Available scripts:"
    echo ""
    echo "Benchmarks:"
    ls -1 "${SCRIPT_DIR}/scripts/benchmarks/" | sed 's/\.sh$//' | sed 's/^/  - /'
    echo ""
    echo "Ablations:"
    ls -1 "${SCRIPT_DIR}/scripts/ablations/" | sed 's/\.sh$//' | sed 's/^/  - /'
    echo ""
    echo "Tests:"
    ls -1 "${SCRIPT_DIR}/scripts/tests/" | sed 's/\.sh$//' | sed 's/^/  - /'
}

if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

if [ "$1" == "list" ]; then
    list_scripts
    exit 0
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo ""
    show_usage
    exit 1
fi

CATEGORY=$1
SCRIPT_NAME=$2

case "$CATEGORY" in
    benchmark)
        SCRIPT_PATH="${SCRIPT_DIR}/scripts/benchmarks/${SCRIPT_NAME}.sh"
        ;;
    ablation)
        SCRIPT_PATH="${SCRIPT_DIR}/scripts/ablations/${SCRIPT_NAME}.sh"
        ;;
    test)
        SCRIPT_PATH="${SCRIPT_DIR}/scripts/tests/${SCRIPT_NAME}.sh"
        ;;
    *)
        echo "Error: Unknown category '$CATEGORY'"
        echo "Valid categories: benchmark, ablation, test"
        echo ""
        show_usage
        exit 1
        ;;
esac

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    echo ""
    echo "Run './run.sh list' to see available scripts"
    exit 1
fi

echo "Running: $CATEGORY/$SCRIPT_NAME"
echo "=========================================="
echo ""

bash "$SCRIPT_PATH"
