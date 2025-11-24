#!/bin/bash
# Script to run validation with conda environment

# Activate conda environment
source ~/.conda/envs/alphabuilder/bin/activate

# Run validation
python alphabuilder/src/core/validate_physics.py
