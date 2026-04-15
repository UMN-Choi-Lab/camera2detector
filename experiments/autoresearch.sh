#!/bin/bash
set -euo pipefail

# Quick pre-check: syntax
python3 -c "import backend.services.trajectory_cluster" 2>&1 || { echo "METRIC roi_quality=0"; exit 1; }

# Run evaluation
python3 experiments/evaluate_roi.py
