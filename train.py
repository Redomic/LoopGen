#!/usr/bin/env python3
"""
Training script for SELFIES GPT decoder using contrastive learning.
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
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from model.config import ModelConfig
from model.decoder import SELFIESGPTDecoder, ReconstructionLoss, DiversityRegularizer, TrainingStabilizer
from molecule_utils.tokenizer import SELFIESTokenizer
from molecule_utils.dataset import SELFIESDataset, collate_fn, count_lines
from molecule_utils.augmentation import SELFIESAugmenter, create_contrastive_batch

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

def save_run_config(args: argparse.Namespace, run_dir: Path, tokenizer: SELFIESTokenizer):
    """Save the configuration for this training run."""
    config_dict = vars(args).copy()
    config_dict['vocab_size'] = tokenizer.vocab_size
    config_dict['run_directory'] = str(run_dir)
    config_dict['timestamp'] = datetime.now().isoformat()
    
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info("Saved run configuration")

def save_training_metrics(run_dir: Path, epoch: int, train_metrics: Dict, val_loss: float, lr: float):
    """Save training metrics to a JSON file."""
    metrics_file = run_dir / "metrics.jsonl"
    
    metrics_entry = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'train_loss': train_metrics['total_loss'],
        'train_contrastive_loss': train_metrics['contrastive_loss'],
        'train_reconstruction_loss': train_metrics['reconstruction_loss'],
        'train_diversity_loss': train_metrics['diversity_loss'],
        'val_loss': val_loss,
        'learning_rate': lr
    }
    
    # Append to JSONL file (one JSON object per line)
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics_entry) + '\n')

# endregion

# region: Core Training Functions
# ==============================================================================

def _contrastive_collate_fn(batch: List[torch.Tensor], tokenizer: SELFIESTokenizer, augmenter: SELFIESAugmenter, n_augmentations: int) -> Dict[str, torch.Tensor]:
    """Custom collate function for contrastive learning."""
    selfies_strings = [tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in batch]
    augmented_strings, labels = create_contrastive_batch(selfies_strings, augmenter, n_augmentations)
    
    max_len = max(len(tokenizer.encode(s, add_special_tokens=True)) for s in augmented_strings)
    
    input_ids = []
    for s in augmented_strings:
        encoded = tokenizer.encode(s, add_special_tokens=True)
        encoded.extend([tokenizer.pad_token_id] * (max_len - len(encoded)))
        input_ids.append(torch.tensor(encoded[:max_len], dtype=torch.long))
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': (torch.stack(input_ids) != tokenizer.pad_token_id).long(),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

def _train_epoch(model: SELFIESGPTDecoder, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, args: argparse.Namespace):
    """Train one epoch with proper stability monitoring."""
    model.train()
    total_loss, total_contrastive, total_recon, total_div = 0.0, 0.0, 0.0, 0.0
    
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for i, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast(device.type, enabled=args.use_amp):
            # Use updated parameter names
            loss_dict = model.combined_loss(
                input_ids, 
                attention_mask, 
                labels, 
                contrastive_weight=args.alpha, 
                reconstruction_weight=args.beta
            )
            
            loss = loss_dict['loss'] / args.grad_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % args.grad_accumulation_steps == 0:
            # Simple gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update metrics for logging
        total_loss += loss_dict['loss'].item()
        total_contrastive += loss_dict['contrastive_loss'].item()
        total_recon += loss_dict['reconstruction_loss'].item()
        total_div += loss_dict['diversity_loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['loss'].item():.3f}",
            'cont': f"{loss_dict['contrastive_loss'].item():.3f}",
            'recon': f"{loss_dict['reconstruction_loss'].item():.3f}",
            'div': f"{loss_dict['diversity_loss'].item():.3f}",
            'length_bonus': f"{loss_dict.get('length_bonus', 0):.4f}"
        })
    
    return {
        'total_loss': total_loss / len(loader),
        'contrastive_loss': total_contrastive / len(loader),
        'reconstruction_loss': total_recon / len(loader),
        'diversity_loss': total_div / len(loader)
    }

def _validate(model: SELFIESGPTDecoder, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> float:
    """Validate model and return contrastive loss."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast(device.type, enabled=args.use_amp):
                loss_dict = model.combined_loss(
                    input_ids, 
                    attention_mask, 
                    labels, 
                    contrastive_weight=args.alpha, 
                    reconstruction_weight=args.beta
                )
            total_loss += loss_dict['contrastive_loss'].item()
    return total_loss / len(loader)

def generate_molecules(model: nn.Module, tokenizer: SELFIESTokenizer, device: torch.device, args) -> List[str]:
    """Generate molecules with advanced sampling."""
    model.eval()
    generated_molecules = []
    with torch.no_grad():
        for _ in range(args.num_generated):
            current_sequence = [tokenizer.bos_token_id]
            for _ in range(args.max_seq_len):
                input_ids = torch.tensor([current_sequence], device=device)
                model_output = model(input_ids, apply_constraints=args.use_grammar_constraints)
                logits = model_output['logits'][0, -1, :]
                
                logits = logits / args.temperature
                if args.top_k > 0:
                    top_k_vals, _ = torch.topk(logits, args.top_k)
                    logits[logits < top_k_vals[-1]] = -float('Inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()

                if next_token_id == tokenizer.eos_token_id:
                    break
                current_sequence.append(next_token_id)

            generated_molecules.append(tokenizer.decode(current_sequence, skip_special_tokens=True))
    model.train()
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

def train(args, tokenizer, device):
    """Main training loop with anti-collapse mechanisms."""
    # Setup run directory and logging
    run_dir = setup_run_directory(args.output_dir)
    save_run_config(args, run_dir, tokenizer)
    
    logging.info("Starting Robust Contrastive Training")
    
    # Config
    if args.model_size == "small": config = ModelConfig.small_config()
    elif args.model_size == "standard": config = ModelConfig.standard_config()
    else: config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size

    # Datasets and loaders
    augmenter = SELFIESAugmenter(tokenizer)
    total_lines = args.dataset_subset_size or count_lines(args.data_path)
    
    # Ensure validation set is carved out correctly
    val_size = min(args.val_set_size, int(total_lines * 0.2))
    train_size = total_lines - val_size
    train_split_ratio = train_size / total_lines if total_lines > 0 else 0.0
    logging.info(f"Dataset size: {total_lines} (Train: {train_size}, Val: {val_size})")

    train_dataset = SELFIESDataset(args.data_path, tokenizer, config.max_seq_len, total_lines=total_lines, split='train', split_ratio=train_split_ratio)
    val_dataset = SELFIESDataset(args.data_path, tokenizer, config.max_seq_len, total_lines=total_lines, split='val', split_ratio=train_split_ratio)
    
    collate_partial = partial(_contrastive_collate_fn, tokenizer=tokenizer, augmenter=augmenter, n_augmentations=args.n_augmentations)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_partial)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_partial)
    
    # Model, optimizer, scaler
    model = SELFIESGPTDecoder(config).to(device)

    model.temperature = args.contrastive_temperature
    print(f"Set model temperature to: {model.temperature}")
    
    # Initialize clean loss components
    model.reconstruction_loss_fn = ReconstructionLoss(
        pad_token_id=tokenizer.pad_token_id, 
        min_sequence_length=args.min_generation_length
    ).to(device)
    
    logging.info(f"Initialized clean loss components (min_length={args.min_generation_length})")
    
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
        val_loss = _validate(model, val_loader, device, args)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logging.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        
        # Save metrics to file
        save_training_metrics(run_dir, epoch + 1, train_metrics, val_loss, current_lr)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = run_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")

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
    parser = argparse.ArgumentParser(description="Contrastive Training for SELFIES GPT")

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
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    train_group.add_argument("--n_augmentations", type=int, default=10, help="Augmentations per sample")
    train_group.add_argument("--contrastive_temperature", type=float, default=0.1, help="Temperature for contrastive loss")
    train_group.add_argument("--alpha", type=float, default=1.0, help="Weight for contrastive loss")
    train_group.add_argument("--beta", type=float, default=0.5, help="Weight for reconstruction loss")

    
    # Anti-collapse arguments
    collapse_group = parser.add_argument_group('Anti-collapse settings')
    collapse_group.add_argument("--min_generation_length", type=int, default=20, help="Minimum sequence length to prevent collapse")
    collapse_group.add_argument("--length_penalty_weight", type=float, default=5.0, help="Weight for length penalty")
    collapse_group.add_argument("--min_embedding_std", type=float, default=0.1, help="Minimum embedding standard deviation")
    collapse_group.add_argument("--max_similarity", type=float, default=0.8, help="Maximum allowed embedding similarity")

    # Generation arguments
    gen_group = parser.add_argument_group('Generation settings')
    gen_group.add_argument("--generate_interval", type=int, default=5, help="Generate samples every N epochs")
    gen_group.add_argument("--num_generated", type=int, default=5, help="Number of molecules to generate")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_group.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 to disable)")
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
        tokenizer = SELFIESTokenizer(data_path=args.data_path)
        tokenizer.save_vocabulary(str(vocab_path))
        print(f"Vocabulary saved to {vocab_path}")
    else:
        print(f"Loading vocabulary from {vocab_path}")
        tokenizer = SELFIESTokenizer(vocab_path=str(vocab_path))
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Run training
    train(args, tokenizer, device)
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main() 