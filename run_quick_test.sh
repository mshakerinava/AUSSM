#!/bin/bash
# Quick test script to run a small experiment

# Example: Compare different layer configurations
python3 run_experiments.py \
  --mode slurm \
  --total_layers 3 \
  --min_aussm 0 \
  --max_aussm 3 \
  --d_models 512 \
  --d_states 32 \
  --learning_rates 1e-4 \
  --batch_sizes 8 \
  --num_epochs 1 \
  --seeds 42 \
  --wandb_entity "khavarib" \
  --wandb_project "aussm-language-modeling" \
  --wandb_group "quick_test" \
  --max_parallel 1

