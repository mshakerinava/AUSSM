#!/bin/bash

# Load necessary modules
module load libffi
module load python/3.10
module load cuda/12.6.0

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment if it doesn't exist
[ -d ".venv" ] || uv venv

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
