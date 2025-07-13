#!/usr/bin/env python3
"""
Training script for SELFIES GPT decoder with proper vocabulary building and advanced generation.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from model.config import ModelConfig
from model.decoder import SELFIESGPTDecoder
from molecule_utils.tokenizer import SELFIETokenizer
from molecule_utils.dataset import SELFIESDataset, FixedSizeSELFIESDataset, collate_fn, count_lines


def get_learning_rate(step: int, config: ModelConfig) -> float:
    """Learning rate schedule with warmup and cosine annealing."""
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps
    
    if step < warmup_steps:
        return config.learning_rate * step / warmup_steps
    
    if step > max_steps:
        return config.min_learning_rate
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def estimate_mfu(model: nn.Module, config: ModelConfig, dt: float, fwdbwd_per_iter: int) -> float:
    """Estimate model flops utilization."""
    N = sum(p.numel() for p in model.parameters())
    L, H, Q, T = config.n_layers, config.n_heads, config.d_model // config.n_heads, config.max_seq_len
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0 / dt)
    
    # A100 theoretical peak is 312 TFLOPS with sparsity
    flops_promised = 312e12
    mfu = flops_achieved / flops_promised
    return mfu


def validate_model(model: nn.Module, val_dataloader: DataLoader, device: torch.device, 
                  use_amp: bool = False) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if use_amp:
                with autocast(device.type):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # Shift for causal LM loss
                    shift_logits = outputs[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1), 
                        ignore_index=0  # PAD token
                    )
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1), 
                    ignore_index=0
                )
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def generate_molecules(model: nn.Module, tokenizer: SELFIETokenizer, device: torch.device,
                      num_molecules: int = 5, max_length: int = 128, temperature: float = 1.0,
                      top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1,
                      use_grammar_constraints: bool = True) -> List[str]:
    """Generate molecules with advanced sampling and optional grammar constraints."""
    model.eval()
    generated_molecules = []
    
    with torch.no_grad():
        for i in range(num_molecules):
            # Start with BOS token
            current_sequence = [tokenizer.bos_token_id]
            input_ids = torch.tensor([current_sequence], device=device)
            
            for _ in range(max_length):
                # Get model predictions
                outputs = model(input_ids, apply_constraints=use_grammar_constraints)
                logits = outputs[0, -1, :] if isinstance(outputs, torch.Tensor) else outputs['logits'][0, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(current_sequence):
                        if logits[token_id] < 0:
                            logits[token_id] *= repetition_penalty
                        else:
                            logits[token_id] /= repetition_penalty
                
                # Apply grammar constraints if enabled
                if use_grammar_constraints:
                    valid_tokens = tokenizer.get_valid_next_tokens(current_sequence)
                    
                    mask = torch.full_like(logits, float('-inf'))
                    for token_id in valid_tokens:
                        if token_id < len(logits):
                            mask[token_id] = 0
                    logits = logits + mask
                
                # Temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_actual = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_actual)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                
                # Check for EOS or add token
                if next_token_id == tokenizer.eos_token_id:
                    break
                
                current_sequence.append(next_token_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            
            # Decode the sequence
            molecule = tokenizer.decode(current_sequence, skip_special_tokens=True)
            generated_molecules.append(molecule)
    
    model.train()
    return generated_molecules


def build_or_load_tokenizer(data_path: str, checkpoint_dir: Path) -> SELFIETokenizer:
    """Build tokenizer vocabulary from training data or load from cache in checkpoints dir."""
    vocab_cache_path = checkpoint_dir / "vocab_cache.json"
    
    print("Setting up tokenizer...")
    
    # Check if cached vocabulary exists in checkpoints
    if vocab_cache_path.exists():
        print(f"Loading cached vocabulary from {vocab_cache_path}")
        tokenizer = SELFIETokenizer(vocab_file=str(vocab_cache_path))
    else:
        print(f"No vocabulary cache found. Building vocabulary from training data: {data_path}")
        print("This may take a few minutes for large datasets...")
        
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Build vocabulary from training data
        tokenizer = SELFIETokenizer(training_data_path=data_path)
        
        # Save vocabulary to checkpoints directory
        tokenizer._save_vocab(tokenizer.vocab, str(vocab_cache_path))
        print(f"Vocabulary automatically saved to {vocab_cache_path}")
        
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample vocabulary tokens: {tokenizer.vocab[:20]}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SELFIES GPT decoder")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to training CSV file")
    parser.add_argument("--val_split", type=float, default=0.01,
                       help="Validation split ratio")
    parser.add_argument("--val_set_size", type=int, default=10000,
                       help="Fixed validation set size")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000,
                       help="Size of shuffle buffer for streaming data")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="standard", choices=["standard", "large"],
                       help="Model size configuration")
    parser.add_argument("--max_seq_len", type=int, default=256,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--max_steps", type=int, default=50000,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping value")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--use_amp", action="store_true", 
                       help="Use automatic mixed precision")
    parser.add_argument("--use_compile", action="store_true",
                       help="Use torch.compile for optimization")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Evaluation arguments
    parser.add_argument("--eval_interval", type=int, default=500,
                       help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=100,
                       help="Number of evaluation iterations")
    parser.add_argument("--generate_interval", type=int, default=2000,
                       help="Generation interval")
    parser.add_argument("--num_generated", type=int, default=5,
                       help="Number of molecules to generate")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    parser.add_argument("--use_grammar_constraints", action="store_true",
                       help="Use grammar constraints during generation")
    
    # I/O arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Checkpoint save interval")
    
    # Debug arguments
    parser.add_argument("--validate_split_and_exit", action="store_true",
                       help="Validate data split and exit")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Build or load tokenizer
    checkpoint_dir = Path(args.checkpoint_dir)
    tokenizer = build_or_load_tokenizer(args.data_path, checkpoint_dir)
    
    # Create model configuration
    if args.model_size == "standard":
        config = ModelConfig.standard_config()
    else:
        config = ModelConfig.large_config()
    
    # Update config with CLI args
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.max_seq_len
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.weight_decay = args.weight_decay
    config.max_steps = args.max_steps
    
    print(f"Model configuration: {config}")
    
    # Validate split if requested
    if args.validate_split_and_exit:
        print("Validating data split...")
        total_lines = count_lines(args.data_path)
        train_dataset = SELFIESDataset(
            args.data_path, tokenizer, config.max_seq_len, total_lines,
            split='train', split_ratio=1.0 - (args.val_set_size/total_lines),
            shuffle_buffer_size=args.shuffle_buffer_size
        )
        val_dataset = FixedSizeSELFIESDataset(
            args.data_path, tokenizer, config.max_seq_len,
            num_samples=args.val_set_size, total_lines=total_lines
        )
        
        print(f"Training dataset created successfully")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Show sample data
        train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
        val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
        
        print("\nSample training batch:")
        train_batch = next(iter(train_loader))
        for i, seq in enumerate(train_batch['input_ids'][:2]):
            decoded = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            print(f"  Training sample {i}: {decoded[:100]}...")
        
        print("\nSample validation batch:")
        val_batch = next(iter(val_loader))
        for i, seq in enumerate(val_batch['input_ids'][:2]):
            decoded = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            print(f"  Validation sample {i}: {decoded[:100]}...")
        
        print("\nValidation complete. Exiting.")
        return
    
    # Create datasets
    print("Creating datasets...")
    total_lines = count_lines(args.data_path)
    print(f"Total lines in dataset: {total_lines:,}")
    
    train_dataset = SELFIESDataset(
        args.data_path, tokenizer, config.max_seq_len, total_lines,
        split='train', split_ratio=1.0 - (args.val_set_size/total_lines),
        shuffle_buffer_size=args.shuffle_buffer_size
    )
    
    val_dataset = FixedSizeSELFIESDataset(
        args.data_path, tokenizer, config.max_seq_len,
        num_samples=args.val_set_size, total_lines=total_lines
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    )
    
    print(f"Training dataset ready")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    print("Initializing model...")
    model = SELFIESGPTDecoder(config).to(device)
    
    # Compile model if requested
    if args.use_compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Setup gradient scaler for AMP
    scaler = GradScaler(device.type) if args.use_amp else None
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_step}")
    
    # Training loop
    print(f"Starting training from step {start_step}...")
    model.train()
    
    running_loss = 0.0
    optimizer.zero_grad()
    
    # Progress bar
    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps)
    
    for step, batch in enumerate(train_loader, start=start_step):
        if step >= args.max_steps:
            break
        
        # Update learning rate
        lr = get_learning_rate(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass (no constraints during training to avoid over-masking)
        t0 = time.time()
        
        if args.use_amp:
            with autocast(device.type):
                outputs = model(input_ids, attention_mask=attention_mask, apply_constraints=False)
                # Shift for causal LM loss
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                loss = loss / args.grad_accumulation_steps
                
                # Debug: Check for abnormal loss values
                if loss.item() > 100:
                    print(f"WARNING: Abnormally high loss: {loss.item():.2f}")
                    print(f"Logits shape: {shift_logits.shape}, min/max: {shift_logits.min().item():.2f}/{shift_logits.max().item():.2f}")
                    print(f"Labels shape: {shift_labels.shape}, unique values: {torch.unique(shift_labels).tolist()}")
        else:
            outputs = model(input_ids, attention_mask=attention_mask, apply_constraints=False)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            loss = loss / args.grad_accumulation_steps
            
            # Debug: Check for abnormal loss values
            if loss.item() > 100:
                print(f"WARNING: Abnormally high loss: {loss.item():.2f}")
                print(f"Logits shape: {shift_logits.shape}, min/max: {shift_logits.min().item():.2f}/{shift_logits.max().item():.2f}")
                print(f"Labels shape: {shift_labels.shape}, unique values: {torch.unique(shift_labels).tolist()}")
        
        # Backward pass
        if args.use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        running_loss += loss.item()
        
        # Update weights
        if (step + 1) % args.grad_accumulation_steps == 0:
            if args.use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad()
        
        dt = time.time() - t0
        
        # Update progress bar
        if step % 10 == 0:
            mfu = estimate_mfu(model, config, dt, args.grad_accumulation_steps * args.batch_size)
            pbar.set_description(
                f"loss: {running_loss/(step-start_step+1):.4f}, "
                f"lr: {lr:.2e}, "
                f"mfu: {mfu*100:.2f}%"
            )
        pbar.update(1)
        
        # Validation
        if (step + 1) % args.eval_interval == 0:
            print(f"\nRunning validation at step {step + 1}...")
            val_loss = validate_model(model, val_loader, device, args.use_amp)
            train_loss = running_loss / (step - start_step + 1)
            
            print(f"Step {step + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.__dict__,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'vocab_size': tokenizer.vocab_size
                }
                if scaler:
                    best_checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(best_checkpoint, checkpoint_dir / "best_model.pt")
                print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Generate molecules
        if (step + 1) % args.generate_interval == 0:
            print(f"\nGenerating molecules at step {step + 1}...")
            molecules = generate_molecules(
                model, tokenizer, device,
                num_molecules=args.num_generated,
                max_length=config.max_seq_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_grammar_constraints=args.use_grammar_constraints
            )
            
            print("Generated molecules:")
            for i, mol in enumerate(molecules, 1):
                print(f"  {i}: {mol}")
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            checkpoint = {
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'train_loss': running_loss / (step - start_step + 1),
                'best_val_loss': best_val_loss,
                'vocab_size': tokenizer.vocab_size
            }
            if scaler:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_step_{step + 1}.pt")
            print(f"\nSaved checkpoint at step {step + 1}")
    
    # Final save
    final_checkpoint = {
        'step': args.max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'train_loss': running_loss / (args.max_steps - start_step),
        'best_val_loss': best_val_loss,
        'vocab_size': tokenizer.vocab_size
    }
    if scaler:
        final_checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(final_checkpoint, checkpoint_dir / "final_model.pt")
    print(f"\nTraining completed! Final model saved.")
    
    print("\nRunning final validation...")
    final_val_loss = validate_model(model, val_loader, device, args.use_amp)
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Best validation loss during training: {best_val_loss:.4f}")

    pbar.close()


if __name__ == "__main__":
    main() 