#!/usr/bin/env python3
"""
Training script for autoregressive SMILES GPT decoder.

This trains a pure generative model using reconstruction loss to learn
molecular SMILES generation. Future versions will add protein conditioning.
"""
import argparse
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
import numpy as np

from model.config import ModelConfig
from model.decoder import SMILESGPTDecoder
from molecule_utils.tokenizer import SMILESTokenizer
from molecule_utils.dataset import SMILESDataset, collate_fn, count_lines

# region: Logging and Directory Setup
# ==============================================================================

def setup_run_directory(base_output_dir: str) -> Path:
    """Create a timestamped run directory and setup logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = run_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    print(f"Created run directory: {run_dir}")
    logging.info(f"Started new training run in {run_dir}")
    return run_dir

def save_run_config(args: argparse.Namespace, run_dir: Path, tokenizer: SMILESTokenizer):
    """Save the configuration for this training run."""
    config_dict = vars(args).copy()
    config_dict['vocab_size'] = tokenizer.vocab_size
    config_dict['run_directory'] = str(run_dir)
    config_dict['timestamp'] = datetime.now().isoformat()
    
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info("Saved run configuration")

def save_training_metrics(run_dir: Path, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
    """Save training metrics to a JSON file."""
    metrics_file = run_dir / "metrics.jsonl"
    
    metrics_entry = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'train_loss': train_metrics['loss'],
        'train_reconstruction_loss': train_metrics['reconstruction_loss'],
        'train_perplexity': train_metrics.get('perplexity', 0),
        'train_accuracy': train_metrics.get('accuracy', 0),
        'train_entropy': train_metrics.get('entropy', 0),
        'val_loss': val_metrics['loss'],
        'val_reconstruction_loss': val_metrics['reconstruction_loss'],
        'val_perplexity': val_metrics.get('perplexity', 0),
        'val_accuracy': val_metrics.get('accuracy', 0),
        'val_entropy': val_metrics.get('entropy', 0),
        'learning_rate': lr
    }
    
    # Append to JSONL file (one JSON object per line)
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics_entry) + '\n')

# endregion

# region: Core Training Functions
# ==============================================================================

def analyze_sequence_lengths(data_path: str, tokenizer: SMILESTokenizer, n_samples: int = 10000):
    """Analyze actual sequence lengths in the dataset."""
    lengths = []
    for chunk in pd.read_csv(data_path, usecols=[0], header=None, chunksize=1000):
        # Respect n_samples cap
        if len(lengths) >= n_samples:
            break
        take = max(0, n_samples - len(lengths))
        for smiles in chunk[0][:take]:
            if isinstance(smiles, str):
                tokens = tokenizer.encode(smiles, add_special_tokens=True)
                lengths.append(len(tokens))
    if not lengths:
        print("No sequences found for length analysis.")
        return
    print(f"\nSequence Length Analysis (n={len(lengths)}):")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"  Std: {np.std(lengths):.1f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.1f}")

def _autoregressive_collate_fn(batch: List[torch.Tensor], tokenizer: SMILESTokenizer, max_seq_len: int = 256) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate function with optional cap."""
    # Find the maximum length in this specific batch
    max_len = min(max(len(t) for t in batch), max_seq_len)
    
    input_ids = []
    for t in batch:
        if len(t) > max_len:
            padded = t[:max_len]
        else:
            padded = torch.cat([
                t,
                torch.full((max_len - len(t),), tokenizer.pad_token_id, dtype=torch.long)
            ])
        input_ids.append(padded)
    
    input_ids = torch.stack(input_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def _train_epoch(model: SMILESGPTDecoder, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    """Train one epoch with autoregressive reconstruction loss."""
    model.train()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_entropy = 0.0
    num_batches = 0
    
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for i, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with autocast(device.type, enabled=args.use_amp):
            # Compute autoregressive loss
            loss_dict = model.compute_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                protein_embeddings=None  # TODO: Add protein conditioning
            )
            
            loss = loss_dict['loss'] / args.grad_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % args.grad_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss_dict['loss'].item()
        total_perplexity += loss_dict['perplexity'].item()
        total_accuracy += loss_dict['accuracy'].item()
        total_entropy += loss_dict['entropy'].item()
        num_batches += 1
        
        # Update progress bar with more detailed stats
        progress_bar.set_postfix({
            'loss': f"{loss_dict['loss'].item():.3f}",
            'ppl': f"{loss_dict['perplexity'].item():.2f}",
            'acc': f"{loss_dict['accuracy'].item():.3f}",
            'ent': f"{loss_dict['entropy'].item():.2f}",
            'seq_len': f"{loss_dict['average_sequence_length'].item():.1f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'accuracy': total_accuracy / num_batches,
        'entropy': total_entropy / num_batches
    }

def _validate(model: SMILESGPTDecoder, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    """Validate model with reconstruction loss."""
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_entropy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast(device.type, enabled=args.use_amp):
                loss_dict = model.compute_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    protein_embeddings=None
                )
            
            total_loss += loss_dict['loss'].item()
            total_perplexity += loss_dict['perplexity'].item()
            total_accuracy += loss_dict['accuracy'].item()
            total_entropy += loss_dict['entropy'].item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'accuracy': total_accuracy / num_batches,
        'entropy': total_entropy / num_batches
    }

def generate_molecules(model: SMILESGPTDecoder, tokenizer: SMILESTokenizer, device: torch.device, args) -> List[str]:
    """Generate molecules using the model's built-in generation method."""
    model.eval()
    
    # Generate multiple sequences
    generated_ids = model.generate(
        prompt_ids=None,  # Start from BOS
        protein_embeddings=None,  # TODO: Add protein conditioning
        max_length=args.max_seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_generated
    )
    
    # Decode to SMILES strings
    generated_molecules = []
    for seq_ids in generated_ids:
        smiles_str = tokenizer.decode(seq_ids.tolist(), skip_special_tokens=True)
        generated_molecules.append(smiles_str)
    
    return generated_molecules

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, run_dir):
    """Save a complete training checkpoint with epoch-specific naming."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    
    # Save epoch-specific checkpoint
    epoch_checkpoint_path = run_dir / f"model_ep{epoch:03d}.pt"
    torch.save(checkpoint, epoch_checkpoint_path)
    
    # Also save as latest checkpoint for easy resuming
    latest_checkpoint_path = run_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_checkpoint_path)
    
    logging.info(f"Saved checkpoint for epoch {epoch} to {epoch_checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load a training checkpoint and return the epoch and best validation loss."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a full checkpoint or just model weights
    if 'model_state_dict' in checkpoint:
        # Full checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        logging.info(f"Resuming from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")
        return start_epoch, best_val_loss
    else:
        # Just model weights (like best_model.pt)
        model.load_state_dict(checkpoint)
        logging.info("Loaded model weights only. Starting from epoch 0 with fresh optimizer/scheduler.")
        return 0, float('inf')

def save_generated_molecules(molecules: List[str], run_dir: Path, epoch: int):
    """Save generated molecules to file."""
    molecules_file = run_dir / f"generated_molecules_ep{epoch:03d}.txt"
    with open(molecules_file, 'w') as f:
        for i, mol in enumerate(molecules):
            f.write(f"{i+1}: {mol}\n")
    logging.info(f"Saved {len(molecules)} generated molecules to {molecules_file}")

# endregion

def analyze_dataset(dataset, name="Dataset"):
    """Analyze dataset diversity"""
    samples = []
    for i, item in enumerate(dataset):
        if i >= 1000:  # Sample first 1000
            break
        samples.append(item)
    
    # Check uniqueness
    unique_samples = len(set([tuple(s.tolist()) for s in samples]))
    print(f"\n{name} Analysis:")
    print(f"  Unique sequences in first 1000: {unique_samples}/1000")
    print(f"  Average length: {sum(len(s) for s in samples)/len(samples):.1f}")
    
    # Check diversity (pairwise similarity)
    if len(samples) > 100:
        sample_subset = samples[:100]
        similarities = []
        for i in range(len(sample_subset)):
            for j in range(i+1, len(sample_subset)):
                seq1 = set(sample_subset[i].tolist())
                seq2 = set(sample_subset[j].tolist())
                jaccard = len(seq1.intersection(seq2)) / len(seq1.union(seq2))
                similarities.append(jaccard)
        print(f"  Average Jaccard similarity: {sum(similarities)/len(similarities):.3f}")

def train(args, tokenizer, device):
    """Main training loop for autoregressive SMILES generation."""
    # Setup run directory and logging
    run_dir = setup_run_directory(args.output_dir)
    save_run_config(args, run_dir, tokenizer)
    
    logging.info("Starting Autoregressive SMILES Training")
    
    # Config
    if args.model_size == "small": config = ModelConfig.small_config()
    elif args.model_size == "standard": config = ModelConfig.standard_config()
    else: config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size

    # Datasets and loaders
    total_lines = args.dataset_subset_size or count_lines(args.data_path)
    
    # Ensure validation set is carved out correctly
    val_size = min(args.val_set_size, int(total_lines * 0.2))
    train_size = total_lines - val_size
    train_split_ratio = train_size / total_lines if total_lines > 0 else 0.0
    logging.info(f"Dataset size: {total_lines} (Train: {train_size}, Val: {val_size})")

    train_dataset = SMILESDataset(args.data_path, tokenizer, config.max_seq_len, total_lines=total_lines, split='train', split_ratio=train_split_ratio)
    val_dataset = SMILESDataset(args.data_path, tokenizer, config.max_seq_len, total_lines=total_lines, split='val', split_ratio=train_split_ratio)
    
    # Debug dataset diversity
    analyze_dataset(train_dataset, "Train")
    analyze_dataset(val_dataset, "Validation")
    
    # Use simple collate function for autoregressive training
    from functools import partial
    collate_fn = partial(_autoregressive_collate_fn, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    # Note: IterableDataset handles shuffling internally, so we don't use shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    # Model, optimizer, scaler
    model = SMILESGPTDecoder(config).to(device)
    
    # Set tokenizer for special tokens
    model.set_tokenizer(tokenizer)
    
    # Configure label smoothing
    model.label_smoothing = args.label_smoothing
    
    logging.info(f"Initialized model with label smoothing={args.label_smoothing}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    
    # Add ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Minimize loss
        factor=0.8,          # Reduce LR by 20% when stuck
        patience=3,          # Wait 3 epochs before reducing
        threshold=0.01,      # Minimum change to be considered improvement
        min_lr=1e-6          # Don't go below this
    )
    logging.info("Initialized optimizer with ReduceLROnPlateau scheduler")
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Load checkpoint if resuming training
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler, device
            )
        else:
            logging.warning(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
    elif args.auto_resume:
        # Auto-resume from latest checkpoint in run directory
        checkpoint_path = run_dir / "checkpoint_latest.pt"
        if checkpoint_path.exists():
            start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler, device
            )
        else:
            logging.info("No checkpoint found for auto-resume. Starting from scratch.")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{args.num_epochs}")
        
        train_metrics = _train_epoch(model, train_loader, optimizer, scaler, device, args)
        val_metrics = _validate(model, val_loader, device, args)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics with more detail
        logging.info(f"Epoch {epoch + 1}")
        logging.info(f"  Train - Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}, "
                    f"Acc: {train_metrics['accuracy']:.3f}, Ent: {train_metrics['entropy']:.2f}")
        logging.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}, "
                    f"Acc: {val_metrics['accuracy']:.3f}, Ent: {val_metrics['entropy']:.2f}")
        logging.info(f"  LR: {current_lr:.2e}")
        
        # Save metrics to file
        save_training_metrics(run_dir, epoch + 1, train_metrics, val_metrics, current_lr)
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics['loss'])

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = run_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model with validation loss: {val_metrics['loss']:.4f}")

        # Save checkpoint every epoch (for resuming)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, run_dir)

        if (epoch + 1) % args.generate_interval == 0:
            logging.info("Generating molecules...")
            molecules = generate_molecules(model, tokenizer, device, args)
            save_generated_molecules(molecules, run_dir, epoch + 1)
            
            # Also log a few examples
            for i, mol in enumerate(molecules[:3]):
                logging.info(f"  Generated {i+1}: {mol}")

    logging.info("Training Complete")

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Autoregressive Training for SMILES GPT Decoder")

    # Core arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to training SMILES file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Base directory for run folders")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "standard", "large"], help="Model size")
    
    # Resume training arguments
    resume_group = parser.add_argument_group('Resume training')
    resume_group.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file to resume from")
    resume_group.add_argument("--auto_resume", action="store_true", help="Automatically resume from checkpoint_latest.pt in current run dir")
    
    # Data arguments
    data_group = parser.add_argument_group('Data settings')
    data_group.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json", help="Path to vocabulary file")
    data_group.add_argument("--val_set_size", type=int, default=10000, help="Size of the validation set")
    data_group.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    data_group.add_argument("--dataset_subset_size", type=int, default=None, help="Use a subset of the dataset")

    # Training arguments
    train_group = parser.add_argument_group('Training settings')
    train_group.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    train_group.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    train_group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    train_group.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for reconstruction loss")

    
    # TODO: Protein conditioning arguments (future implementation)
    # protein_group = parser.add_argument_group('Protein conditioning')
    # protein_group.add_argument("--protein_encoder_path", type=str, default=None, help="Path to pretrained protein encoder")
    # protein_group.add_argument("--use_protein_conditioning", action="store_true", help="Enable protein conditioning")

    # Generation arguments
    gen_group = parser.add_argument_group('Generation settings')
    gen_group.add_argument("--generate_interval", type=int, default=5, help="Generate samples every N epochs")
    gen_group.add_argument("--num_generated", type=int, default=5, help="Number of molecules to generate")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_group.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 to disable)")
    gen_group.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    gen_group.add_argument("--use_grammar_constraints", action="store_true", help="Use grammar constraints during generation")
    
    # System arguments
    system_group = parser.add_argument_group('System settings')
    system_group.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    system_group.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    system_group.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")
    print(f"Using device: {device}")
    
    vocab_path = Path(args.vocab_path)
    if not vocab_path.parent.exists(): vocab_path.parent.mkdir(parents=True, exist_ok=True)

    if not vocab_path.exists():
        print(f"Building vocabulary from {args.data_path}...")
        tokenizer = SMILESTokenizer(data_path=args.data_path)
        tokenizer.save_vocabulary(str(vocab_path))
        print(f"Vocabulary saved to {vocab_path}")
    else:
        print(f"Loading vocabulary from {vocab_path}")
        tokenizer = SMILESTokenizer(vocab_path=str(vocab_path))
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Analyze sequence length distribution to choose a reasonable cap
    analyze_sequence_lengths(args.data_path, tokenizer)

    # Run training
    train(args, tokenizer, device)
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main() 