#!/bin/bash

# Step 1: Navigate to the mavrick directory
cd mavrick || { echo "Directory 'mavrick' not found"; exit 1; }

# Step 2: Export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# Step 3: Install the package in editable mode
pip install -e . 