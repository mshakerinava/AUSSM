# AUSSM Language Modeling Experiments

This directory contains scripts for running language modeling experiments with AUSSM and Mamba models, comparing different layer configurations and hyperparameters.

## Files

- `language_modeling_aussm.py`: Main training script with wandb integration
- `run_experiments.py`: Experiment runner for hyperparameter sweeps
- `example_config.json`: Example configuration file

## Quick Start

### 1. Single Experiment

Run a single experiment directly:

```bash
python3 language_modeling_aussm.py \
  --layers "m|m|a" \
  --d_model 512 \
  --d_state 16 \
  --learning_rate 1e-4 \
  --batch_size 8 \
  --num_epochs 3 \
  --wandb_project "aussm-lm"
```

### 2. Hyperparameter Sweep (Local)

Run multiple experiments locally:

```bash
python3 run_experiments.py \
  --mode local \
  --total_layers 3 \
  --d_states 16 32 \
  --learning_rates 1e-4 5e-4 \
  --seeds 42 43 44 \
  --wandb_project "aussm-lm" \
  --wandb_group "layer_sweep" \
  --max_parallel 2
```

### 3. Hyperparameter Sweep (SLURM)

Create and submit a SLURM array job:

```bash
python3 run_experiments.py \
  --mode slurm \
  --total_layers 3 \
  --d_states 16 32 \
  --learning_rates 1e-4 5e-4 \
  --seeds 42 43 44 \
  --wandb_project "aussm-lm" \
  --wandb_group "layer_sweep" \
  --slurm_partition long \
  --slurm_time "24:00:00" \
  --output_dir experiments

# Then submit the generated script
sbatch experiments/run_array.slurm
```

### 4. Using a Config File

Create a JSON config file (see `example_config.json`) and run:

```bash
python3 run_experiments.py --config_file example_config.json
```

## Layer Configurations

Layer configurations are specified as strings with `|` separators:
- `m` = Mamba layer
- `a` = AUSSM layer

Examples:
- `"m|m|m"` = 3 Mamba layers
- `"a|a|a"` = 3 AUSSM layers
- `"m|m|a"` = 2 Mamba + 1 AUSSM
- `"a|m|a"` = AUSSM-Mamba-AUSSM

The experiment runner can automatically generate all combinations for a given total number of layers.

## Wandb Integration

All experiments are automatically logged to wandb with:
- Loss and perplexity (train and validation)
- Model configuration (layers, dimensions, etc.)
- Training hyperparameters
- Number of AUSSM vs Mamba layers for easy filtering

### Comparing Models in Wandb

In wandb, you can:
1. Filter by `num_aussm_layers` and `num_mamba_layers` to compare different layer compositions
2. Group by `wandb_group` to organize related experiments
3. Plot validation perplexity vs number of AUSSM layers
4. Compare different hyperparameter settings

Example wandb query:
- Filter: `num_aussm_layers == 1 AND num_mamba_layers == 2`
- Plot: `val/perplexity` vs `epoch`
- Compare: Different `learning_rate` values

## Command Line Arguments

### `language_modeling_aussm.py`

**Model arguments:**
- `--layers`: Layer configuration (e.g., "m|m|a")
- `--d_model`: Model dimension (default: 512)
- `--d_state`: SSM state dimension (default: 16)
- `--mamba_expand`: Mamba expansion factor (default: 2)

**Training arguments:**
- `--batch_size`: Batch size (default: 8)
- `--seq_length`: Sequence length (default: 512)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--num_epochs`: Number of epochs (default: 3)
- `--seed`: Random seed (default: 42)

**Wandb arguments:**
- `--wandb_project`: Project name
- `--wandb_entity`: Entity/team name (optional)
- `--wandb_group`: Group name for organizing runs (optional)
- `--wandb_run_name`: Custom run name (optional)
- `--no_wandb`: Disable wandb logging

### `run_experiments.py`

**Execution mode:**
- `--mode`: `local`, `slurm`, or `dry-run`

**Layer configurations:**
- `--layer_configs`: Explicit list of layer configs (e.g., "m|m|a" "a|a|m")
- `--total_layers`: Total number of layers (used if layer_configs not provided)
- `--min_aussm`: Minimum AUSSM layers
- `--max_aussm`: Maximum AUSSM layers

**Hyperparameter sweeps:**
- `--d_models`: List of model dimensions
- `--d_states`: List of SSM state dimensions
- `--mamba_expands`: List of Mamba expansion factors
- `--batch_sizes`: List of batch sizes
- `--learning_rates`: List of learning rates
- `--weight_decays`: List of weight decay values
- `--seq_lengths`: List of sequence lengths
- `--seeds`: List of random seeds

**Other:**
- `--config_file`: JSON config file (overrides command line args)
- `--output_dir`: Output directory for logs (default: "experiments")
- `--max_parallel`: Max parallel jobs for local mode (default: 4)

## Example Workflows

### Compare AUSSM vs Mamba layers

```bash
# Run experiments with different layer compositions
python3 run_experiments.py \
  --mode local \
  --total_layers 3 \
  --min_aussm 0 \
  --max_aussm 3 \
  --d_states 16 \
  --learning_rates 1e-4 \
  --seeds 42 43 44 \
  --wandb_project "aussm-lm" \
  --wandb_group "layer_comparison"
```

This will generate and run all combinations:
- `m|m|m` (0 AUSSM, 3 Mamba)
- `m|m|a`, `m|a|m`, `a|m|m` (1 AUSSM, 2 Mamba)
- `m|a|a`, `a|m|a`, `a|a|m` (2 AUSSM, 1 Mamba)
- `a|a|a` (3 AUSSM, 0 Mamba)

### Hyperparameter tuning

```bash
python3 run_experiments.py \
  --mode local \
  --layer_configs "m|m|a" \
  --d_models 256 512 768 \
  --d_states 16 32 64 \
  --learning_rates 1e-4 5e-4 1e-3 \
  --seeds 42 \
  --wandb_project "aussm-lm" \
  --wandb_group "hyperparameter_tuning"
```

## Tips

1. **Start small**: Test with a few experiments first using `--mode dry-run` to see what will be run
2. **Use wandb groups**: Organize related experiments with `--wandb_group`
3. **Monitor resources**: For local mode, adjust `--max_parallel` based on your GPU memory
4. **SLURM arrays**: For large sweeps, use SLURM mode to parallelize across cluster nodes
5. **Config files**: Save your experiment configs as JSON files for reproducibility

