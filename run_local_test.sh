#!/bin/bash
# Simple script to run Perceiver IO testing locally on CPU

# Set up environment
echo "Setting up local testing environment..."

# Ensure we have the necessary directories
mkdir -p results_local
mkdir -p logs

# Enable tensor cores properly
echo "Enabling tensor cores for better performance..."
python -c "import torch; torch.set_float32_matmul_precision('medium')"

# Run the local test script
echo "Starting local CPU testing..."
python test_perceiver_local.py

# Show results
echo "Check the results_local directory for testing outputs."