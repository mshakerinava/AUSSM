import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import sys
import os
import argparse
import math
import wandb
from pathlib import Path

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'extension-cpp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-modules'))
from wavesAI.model.aussm import SSMSeq2Seq


def count_layers(layer_string):
    """Count AUSSM and Mamba layers in the layer configuration string."""
    layers = layer_string.replace(" ", "").split("|")
    num_aussm = sum(1 for l in layers if l == "a")
    num_mamba = sum(1 for l in layers if l == "m")
    return num_aussm, num_mamba, len(layers)


def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)


def get_weight_norm(model):
    """Calculate the L2 norm of all model parameters."""
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def get_grad_norm(model):
    """Calculate the L2 norm of all model gradients."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def tokenize_function(examples, tokenizer, seq_length):
    """Tokenize examples for language modeling."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=seq_length + 1,  # +1 for shift
        padding="max_length",
        return_tensors="pt"
    )


class WikiTextDataset(Dataset):
    """Dataset class for WikiText language modeling."""
    def __init__(self, tokenized_data):
        self.data = tokenized_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        return input_ids[:-1], input_ids[1:]


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, vocab_size, global_step, max_grad_norm=1.0, log_interval=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits = model(inputs)  # (batch, seq_len, vocab_size)
        
        # Reshape for loss: (batch * seq_len, vocab_size) and (batch * seq_len,)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = get_grad_norm(model)
        
        # Gradient clipping
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm_clipped = get_grad_norm(model)
        else:
            grad_norm_clipped = grad_norm
        
        optimizer.step()
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate weight norm and current learning rate
        weight_norm = get_weight_norm(model)
        current_lr = optimizer.param_groups[0]['lr']
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            perplexity = calculate_perplexity(loss.item())
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}, LR: {current_lr:.2e}")
            if wandb.run is not None:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_perplexity": perplexity,
                    "train/batch": batch_idx,
                    "train/weight_norm": weight_norm,
                    "train/grad_norm": grad_norm,
                    "train/grad_norm_clipped": grad_norm_clipped,
                    "train/learning_rate": current_lr,
                    "global_step": global_step
                })
        
        global_step += 1
    
    avg_loss = total_loss / num_batches
    avg_perplexity = calculate_perplexity(avg_loss)
    return avg_loss, avg_perplexity, global_step


def validate(model, val_loader, criterion, device, vocab_size):
    """Validate the model."""
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    avg_val_perplexity = calculate_perplexity(avg_val_loss)
    return avg_val_loss, avg_val_perplexity


def main():
    parser = argparse.ArgumentParser(description='Train AUSSM/Mamba language model on WikiText-2')
    
    # Model arguments
    parser.add_argument('--layers', type=str, default='m|m|a',
                       help='Layer configuration string (e.g., "m|m|a" for Mamba-Mamba-AUSSM)')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--d_state', type=int, default=16,
                       help='SSM state dimension')
    parser.add_argument('--mamba_expand', type=int, default=2,
                       help='Mamba expansion factor')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping (0.0 to disable)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval in batches')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='aussm-language-modeling',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='khavarib',
                       help='Wandb entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (auto-generated if not provided)')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name for organizing runs')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext', 'ptb', 'tinystories', 'simplebooks'],
                       help='Dataset to use: wikitext, ptb (Penn Treebank), tinystories, or simplebooks')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detected if not provided)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Count layers
    num_aussm, num_mamba, total_layers = count_layers(args.layers)
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"layers_{args.layers}_d{args.d_model}_s{args.d_state}_lr{args.learning_rate}"
        
        wandb_kwargs = {
            'project': args.wandb_project,
            'name': run_name,
            'config': {
                "dataset": args.dataset,
                "layers": args.layers,
                "num_aussm_layers": num_aussm,
                "num_mamba_layers": num_mamba,
                "total_layers": total_layers,
                "d_model": args.d_model,
                "d_state": args.d_state,
                "mamba_expand": args.mamba_expand,
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "warmup_steps": args.warmup_steps,
                "max_grad_norm": args.max_grad_norm,
                "seed": args.seed,
            }
        }
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        if args.wandb_group:
            wandb_kwargs['group'] = args.wandb_group
        
        wandb.init(**wandb_kwargs)
    
    # Load dataset based on choice
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    elif args.dataset == 'ptb':
        dataset = load_dataset("ptb_text_only")
        # PTB uses 'sentence' column instead of 'text'
        if 'sentence' in dataset['train'].column_names:
            dataset = dataset.rename_column('sentence', 'text')
    elif args.dataset == 'tinystories':
        # TinyStories dataset - very small and simple
        dataset = load_dataset("roneneldan/TinyStories")
        # TinyStories uses 'text' column
    elif args.dataset == 'simplebooks':
        # SimpleBooks dataset - simplified books
        dataset = load_dataset("simplebooks")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenize_fn = lambda examples: tokenize_function(examples, tokenizer, args.seq_length)
    tokenized_train = dataset["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_val = dataset["validation"].map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    # Create data loaders
    train_dataset = WikiTextDataset(tokenized_train)
    val_dataset = WikiTextDataset(tokenized_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print(f"Initializing model with layers: {args.layers}")
    print(f"  - AUSSM layers: {num_aussm}")
    print(f"  - Mamba layers: {num_mamba}")
    print(f"  - Total layers: {total_layers}")
    
    model = SSMSeq2Seq(
        d_model=args.d_model,
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        layers=args.layers,
        d_state=args.d_state,
        mamba_expand=args.mamba_expand,
        verbose=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if wandb.run is not None:
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params
        })
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Calculate total training steps for scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    
    # Learning rate scheduler: warmup + cosine decay
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_perplexity, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, vocab_size, global_step, args.max_grad_norm, args.log_interval
        )
        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
        
        # Validate
        val_loss, val_perplexity = validate(model, val_loader, criterion, device, vocab_size)
        print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
        
        # Calculate metrics for logging
        weight_norm = get_weight_norm(model)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/perplexity": train_perplexity,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "train/weight_norm": weight_norm,
                "train/learning_rate": current_lr,
                "global_step": global_step
            })
    
    print("\nTraining completed!")
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
