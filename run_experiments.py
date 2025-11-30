#!/usr/bin/env python3
"""
Experiment runner for AUSSM/Mamba language modeling hyperparameter sweeps.

This script allows you to:
- Sweep over different layer configurations (AUSSM vs Mamba layers)
- Sweep over model and training hyperparameters
- Run experiments locally or submit to SLURM
- Track all experiments in wandb with proper grouping
"""

import argparse
import subprocess
import itertools
import json
from pathlib import Path
from typing import List, Dict, Any
import os


def generate_layer_configs(
    total_layers: int,
    min_aussm: int = 0,
    max_aussm: int = None
) -> List[str]:
    """
    Generate all possible layer configurations with given total layers.
    
    Args:
        total_layers: Total number of layers
        min_aussm: Minimum number of AUSSM layers
        max_aussm: Maximum number of AUSSM layers (default: total_layers)
    
    Returns:
        List of layer configuration strings (e.g., ["m|m|m", "m|m|a", "m|a|m", ...])
    """
    if max_aussm is None:
        max_aussm = total_layers
    
    configs = []
    for num_aussm in range(min_aussm, max_aussm + 1):
        num_mamba = total_layers - num_aussm
        # Generate all permutations of a's and m's
        layer_list = ['a'] * num_aussm + ['m'] * num_mamba
        # Use itertools to get unique combinations (since order matters for layers)
        # We'll generate all unique permutations
        seen = set()
        for perm in itertools.permutations(layer_list):
            if perm not in seen:
                seen.add(perm)
                configs.append('|'.join(perm))
    
    return configs


def create_slurm_script(
    script_path: Path,
    python_script: str,
    job_name: str,
    output_dir: Path,
    array_size: int,
    **slurm_args
):
    """Create a SLURM array job script."""
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm-%A_%a.out
#SBATCH --error={output_dir}/slurm-%A_%a.err
#SBATCH --time={slurm_args.get('time', '24:00:00')}
#SBATCH --partition={slurm_args.get('partition', 'long')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_args.get('cpus_per_task', 8)}
#SBATCH --mem={slurm_args.get('mem', '16G')}
#SBATCH --gres=gpu:{slurm_args.get('gpus', 1)}
#SBATCH --array=0-{array_size-1}

set -euo pipefail

# Create log directory
mkdir -p {output_dir}

# Load environment if needed
{slurm_args.get('env_setup', '')}

# Change to script directory
cd {Path(python_script).parent.absolute()}

# Run the experiment
python3 {Path(python_script).name} --array_task_id $SLURM_ARRAY_TASK_ID --experiment_config {output_dir.absolute()}/experiment_config.json
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description='Run AUSSM language modeling experiments')
    
    # Experiment configuration
    parser.add_argument('--mode', type=str, choices=['local', 'slurm', 'dry-run'],
                       default='local',
                       help='Execution mode: local (run sequentially), slurm (submit array job), or dry-run (just print)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='JSON file with experiment configuration (overrides command line args)')
    
    # Layer configurations
    parser.add_argument('--layer_configs', type=str, nargs='+', default=None,
                       help='Explicit layer configurations (e.g., "m|m|a" "a|a|m")')
    parser.add_argument('--total_layers', type=int, default=3,
                       help='Total number of layers (used if layer_configs not provided)')
    parser.add_argument('--min_aussm', type=int, default=0,
                       help='Minimum number of AUSSM layers')
    parser.add_argument('--max_aussm', type=int, default=None,
                       help='Maximum number of AUSSM layers')
    
    # Model hyperparameters
    parser.add_argument('--d_models', type=int, nargs='+', default=[512],
                       help='Model dimensions to sweep')
    parser.add_argument('--d_states', type=int, nargs='+', default=[16],
                       help='SSM state dimensions to sweep')
    parser.add_argument('--mamba_expands', type=int, nargs='+', default=[2],
                       help='Mamba expansion factors to sweep')
    
    # Training hyperparameters
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8],
                       help='Batch sizes to sweep')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[1e-4],
                       help='Learning rates to sweep')
    parser.add_argument('--weight_decays', type=float, nargs='+', default=[0.01],
                       help='Weight decays to sweep')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[512],
                       help='Sequence lengths to sweep')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds to run')
    
    # Wandb settings
    parser.add_argument('--wandb_project', type=str, default='aussm-language-modeling',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='khavarib',
                       help='Wandb entity/team name')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name for organizing runs')
    
    # SLURM settings
    parser.add_argument('--slurm_partition', type=str, default='long',
                       help='SLURM partition')
    parser.add_argument('--slurm_time', type=str, default='24:00:00',
                       help='SLURM time limit')
    parser.add_argument('--slurm_mem', type=str, default='16G',
                       help='SLURM memory')
    parser.add_argument('--slurm_gpus', type=int, default=1,
                       help='Number of GPUs per job')
    parser.add_argument('--slurm_cpus', type=int, default=4,
                       help='Number of CPUs per job')
    parser.add_argument('--slurm_env_setup', type=str, default='',
                       help='Commands to set up environment (e.g., "source ~/.profile")')
    
    # Other settings
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for logs and configs')
    parser.add_argument('--python_script', type=str, default='language_modeling_aussm.py',
                       help='Python training script to run')
    parser.add_argument('--max_parallel', type=int, default=4,
                       help='Maximum parallel jobs for local mode')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Generate layer configurations
    if args.layer_configs:
        layer_configs = args.layer_configs
    else:
        layer_configs = generate_layer_configs(
            args.total_layers,
            args.min_aussm,
            args.max_aussm
        )
    
    print(f"Generated {len(layer_configs)} layer configurations:")
    for config in layer_configs:
        print(f"  - {config}")
    
    # Generate all experiment combinations
    experiments = []
    for layer_config in layer_configs:
        for d_model in args.d_models:
            for d_state in args.d_states:
                for mamba_expand in args.mamba_expands:
                    for batch_size in args.batch_sizes:
                        for lr in args.learning_rates:
                            for wd in args.weight_decays:
                                for seq_len in args.seq_lengths:
                                    for seed in args.seeds:
                                        experiments.append({
                                            'layers': layer_config,
                                            'd_model': d_model,
                                            'd_state': d_state,
                                            'mamba_expand': mamba_expand,
                                            'batch_size': batch_size,
                                            'learning_rate': lr,
                                            'weight_decay': wd,
                                            'seq_length': seq_len,
                                            'seed': seed,
                                            'num_epochs': args.num_epochs,
                                            'wandb_project': args.wandb_project,
                                            'wandb_entity': args.wandb_entity,
                                            'wandb_group': args.wandb_group,
                                        })
    
    print(f"\nTotal experiments: {len(experiments)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config_file = output_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'experiments': experiments,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"Saved experiment configuration to {config_file}")
    
    if args.mode == 'dry-run':
        print("\nDry-run mode: Would run the following experiments:")
        for i, exp in enumerate(experiments[:5]):  # Show first 5
            print(f"\nExperiment {i}:")
            for key, value in exp.items():
                print(f"  {key}: {value}")
        if len(experiments) > 5:
            print(f"\n... and {len(experiments) - 5} more experiments")
        return
    
    # Run experiments
    if args.mode == 'local':
        print("\nRunning experiments locally...")
        import concurrent.futures
        
        def run_experiment(exp_idx, exp_config):
            """Run a single experiment."""
            cmd = [
                'python3', args.python_script,
                '--layers', exp_config['layers'],
                '--d_model', str(exp_config['d_model']),
                '--d_state', str(exp_config['d_state']),
                '--mamba_expand', str(exp_config['mamba_expand']),
                '--batch_size', str(exp_config['batch_size']),
                '--learning_rate', str(exp_config['learning_rate']),
                '--weight_decay', str(exp_config['weight_decay']),
                '--seq_length', str(exp_config['seq_length']),
                '--num_epochs', str(exp_config['num_epochs']),
                '--seed', str(exp_config['seed']),
                '--wandb_project', exp_config['wandb_project'],
            ]
            if exp_config['wandb_entity']:
                cmd.extend(['--wandb_entity', exp_config['wandb_entity']])
            if exp_config['wandb_group']:
                cmd.extend(['--wandb_group', exp_config['wandb_group']])
            
            run_name = f"exp{exp_idx}_layers{exp_config['layers']}_d{exp_config['d_model']}_s{exp_config['d_state']}_lr{exp_config['learning_rate']}"
            cmd.extend(['--wandb_run_name', run_name])
            
            print(f"Running experiment {exp_idx + 1}/{len(experiments)}: {run_name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error in experiment {exp_idx}: {result.stderr}")
            return result
        
        # Run with limited parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = [executor.submit(run_experiment, i, exp) for i, exp in enumerate(experiments)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Wait for completion and check for errors
        
        print("\nAll experiments completed!")
    
    elif args.mode == 'slurm':
        print("\nCreating SLURM array job...")
        
        # Create a wrapper script that reads from the config file
        wrapper_script = output_dir / 'run_single_experiment.py'
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        python_script_path = Path(args.python_script)
        if python_script_path.is_absolute():
            python_script_abs = python_script_path
        else:
            python_script_abs = script_dir / python_script_path
        python_script_abs = python_script_abs.absolute()
        
        wrapper_content = f"""#!/usr/bin/env python3
import json
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    task_id = int(sys.argv[sys.argv.index('--array_task_id') + 1])
    config_file = sys.argv[sys.argv.index('--experiment_config') + 1]
    
    with open(config_file) as f:
        config = json.load(f)
    
    exp = config['experiments'][task_id]
    # Use absolute path to training script
    script_path = Path('{python_script_abs}')
    
    cmd = [
        'python3', str(script_path),
        '--layers', exp['layers'],
        '--d_model', str(exp['d_model']),
        '--d_state', str(exp['d_state']),
        '--mamba_expand', str(exp['mamba_expand']),
        '--batch_size', str(exp['batch_size']),
        '--learning_rate', str(exp['learning_rate']),
        '--weight_decay', str(exp['weight_decay']),
        '--seq_length', str(exp['seq_length']),
        '--num_epochs', str(exp['num_epochs']),
        '--seed', str(exp['seed']),
        '--wandb_project', exp['wandb_project'],
    ]
    if exp.get('wandb_entity'):
        cmd.extend(['--wandb_entity', exp['wandb_entity']])
    if exp.get('wandb_group'):
        cmd.extend(['--wandb_group', exp['wandb_group']])
    
    run_name = f"exp{{task_id}}_layers{{exp['layers']}}_d{{exp['d_model']}}_s{{exp['d_state']}}_lr{{exp['learning_rate']}}"
    cmd.extend(['--wandb_run_name', run_name])
    
    subprocess.run(cmd)
"""
        wrapper_script.write_text(wrapper_content)
        wrapper_script.chmod(0o755)
        
        # Create SLURM script
        slurm_script = output_dir / 'run_array.slurm'
        # Get absolute paths for SLURM script
        wrapper_script_abs = wrapper_script.absolute()
        output_dir_abs = output_dir.absolute()
        create_slurm_script(
            slurm_script,
            str(wrapper_script_abs),
            'aussm_lm',
            output_dir_abs,
            len(experiments),
            partition=args.slurm_partition,
            time=args.slurm_time,
            mem=args.slurm_mem,
            gpus=args.slurm_gpus,
            cpus_per_task=args.slurm_cpus,
            env_setup=args.slurm_env_setup
        )
        
        print(f"Created SLURM script: {slurm_script}")
        print(f"Submit with: sbatch {slurm_script}")


if __name__ == "__main__":
    main()

