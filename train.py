#!/usr/bin/env python3
"""
Unified training script for SELFIES GPT decoder.
Supports contrastive pre-training and generative fine-tuning.
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
from model.contrastive_model import ContrastiveSELFIESModel
from molecule_utils.tokenizer import SELFIETokenizer
from molecule_utils.dataset import SELFIESDataset, FixedSizeSELFIESDataset, collate_fn, count_lines
from molecule_utils.augmentation import SELFIESAugmenter, create_contrastive_batch

# region: Contrastive Learning Functions
# ==============================================================================

def _contrastive_collate_fn(batch, tokenizer, augmenter, num_augmentations=2):
    """
    Custom collate function for contrastive learning.
    Creates augmented pairs and handles batching.
    """
    selfies_strings = [tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in batch]
    
    augmented_strings, labels = create_contrastive_batch(
        selfies_strings, augmenter, num_augmentations
    )
    
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


def _train_contrastive_epoch(model, loader, optimizer, scaler, device, args):
    """Train one epoch of contrastive learning."""
    model.train()
    totals = {'loss': 0, 'cont': 0, 'recon': 0, 'div': 0}
    
    pbar = tqdm(loader, desc="Contrastive Training")
    optimizer.zero_grad()
    
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast(device.type, enabled=args.use_amp):
            loss_dict = model.combined_loss(
                input_ids, attention_mask, labels,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma
            )
            loss = loss_dict['loss'] / args.grad_accumulation_steps
        
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % args.grad_accumulation_steps == 0:
            if args.use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        totals['loss'] += loss.item() * args.grad_accumulation_steps
        totals['cont'] += loss_dict['contrastive_loss'].item()
        totals['recon'] += loss_dict['reconstruction_loss'].item()
        totals['div'] += loss_dict['diversity_loss'].item()
        
        pbar.set_postfix({k: v / (i + 1) for k, v in totals.items()})
        
    return {k: v / len(loader) for k, v in totals.items()}


def _validate_contrastive(model, loader, device, args):
    """Validate the contrastive model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Contrastive Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast(device.type, enabled=args.use_amp):
                loss = model.forward_contrastive(input_ids, attention_mask, labels)['loss']
            total_loss += loss.item()
    return total_loss / len(loader)

# endregion

# region: Generative Learning Functions
# ==============================================================================

def get_learning_rate(step: int, config: ModelConfig) -> float:
    """Learning rate schedule with warmup and cosine annealing."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_learning_rate
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def validate_model(model: nn.Module, val_dataloader: DataLoader, device: torch.device, use_amp: bool = False) -> float:
    """Run validation and return average loss for the generative model."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with autocast(device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask)
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_dataloader) if val_dataloader else float('inf')


def generate_molecules(model: nn.Module, tokenizer: SELFIETokenizer, device: torch.device, args) -> List[str]:
    """Generate molecules with advanced sampling."""
    model.eval()
    generated_molecules = []
    with torch.no_grad():
        for _ in range(args.num_generated):
            current_sequence = [tokenizer.bos_token_id]
            for _ in range(args.max_seq_len):
                input_ids = torch.tensor([current_sequence], device=device)
                outputs = model(input_ids, apply_constraints=args.use_grammar_constraints)
                logits = outputs[0, -1, :]
                
                # Apply sampling techniques (temperature, top-k, top-p)
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

# endregion

# region: Main Training Phases
# ==============================================================================

def run_contrastive_phase(args, tokenizer, device):
    """Runs the contrastive pre-training phase."""
    print("--- Starting Contrastive Pre-training Phase ---")
    
    # Config
    if args.model_size == "small":
        config = ModelConfig.small_config()
    elif args.model_size == "standard":
        config = ModelConfig.standard_config()
    else: # large
        config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size

    # Datasets and loaders
    augmenter = SELFIESAugmenter(tokenizer)
    
    # Use subset size if provided, otherwise use the full dataset
    total_lines = args.dataset_subset_size or count_lines(args.data_path)
    val_size = min(args.val_set_size, int(total_lines * 0.1)) # Use 10% for val if val_set_size is too large
    train_split_ratio = 1.0 - (val_size / total_lines) if total_lines > 0 else 0.0

    print(f"Dataset size: {total_lines}, Train/Val split: {train_split_ratio:.2f}/{1-train_split_ratio:.2f}")

    train_dataset = SELFIESDataset(
        args.data_path, tokenizer, config.max_seq_len,
        total_lines=total_lines, split='train', split_ratio=train_split_ratio
    )
    val_dataset = SELFIESDataset(
        args.data_path, tokenizer, config.max_seq_len,
        total_lines=total_lines, split='val', split_ratio=train_split_ratio
    )
    
    collate_partial = partial(_contrastive_collate_fn, tokenizer=tokenizer, augmenter=augmenter, num_augmentations=args.num_augmentations)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_partial)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_partial)
    
    # Model, optimizer, scaler
    model = ContrastiveSELFIESModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_val_loss = float('inf')
    output_dir = Path(args.output_dir) / "contrastive"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        train_metrics = _train_contrastive_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss = _validate_contrastive(model, val_loader, device, args)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the base model's state_dict for generative fine-tuning
            torch.save(model.model.state_dict(), output_dir / "best_pretrained_model.pt")
            print(f"Saved best pre-trained model with validation loss: {val_loss:.4f}")

    print("--- Contrastive Pre-training Phase Complete ---")
    return output_dir / "best_pretrained_model.pt"


def run_generative_phase(args, tokenizer, device):
    """Runs the generative fine-tuning phase."""
    print("--- Starting Generative Fine-tuning Phase ---")

    # Config
    if args.model_size == "small":
        config = ModelConfig.small_config()
    elif args.model_size == "standard":
        config = ModelConfig.standard_config()
    else: # large
        config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size
    config.max_steps = args.max_steps

    # Datasets and loaders
    total_lines = args.dataset_subset_size or count_lines(args.data_path)
    val_size = min(args.val_set_size, int(total_lines * 0.1))
    train_split_ratio = 1.0 - (val_size / total_lines) if total_lines > 0 else 0.0
    
    print(f"Dataset size: {total_lines}, Train/Val split: {train_split_ratio:.2f}/{1-train_split_ratio:.2f}")

    train_dataset = SELFIESDataset(
        args.data_path, tokenizer, config.max_seq_len,
        total_lines=total_lines, split='train', split_ratio=train_split_ratio
    )
    val_dataset = SELFIESDataset(
        args.data_path, tokenizer, config.max_seq_len,
        total_lines=total_lines, split='val', split_ratio=train_split_ratio
    )
    
    collate_partial = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_partial)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_partial)

    # Model, optimizer, scaler
    model = SELFIESGPTDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    
    start_step = 0
    if args.load_pretrained:
        print(f"Loading pre-trained model from: {args.load_pretrained}")
        model.load_state_dict(torch.load(args.load_pretrained, map_location=device))

    # Training loop
    output_dir = Path(args.output_dir) / "generative"
    output_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps)
    
    best_val_loss = float('inf')
    train_iter = iter(train_loader)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = get_learning_rate(step, config)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with autocast(device.type, enabled=args.use_amp):
            outputs = model(input_ids, attention_mask=attention_mask)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id)
        
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        if args.use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % args.eval_interval == 0:
            val_loss = validate_model(model, val_loader, device, args.use_amp)
            print(f"\nStep {step + 1}: Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / "best_generative_model.pt")
                print(f"Saved new best model.")
        
        if (step + 1) % args.generate_interval == 0:
            print("\nGenerating molecules...")
            molecules = generate_molecules(model, tokenizer, device, args)
            for i, mol in enumerate(molecules): print(f"  {i+1}: {mol}")

    print("--- Generative Fine-tuning Phase Complete ---")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Unified SELFIES Model Training")

    # Core arguments
    parser.add_argument("--training_mode", type=str, default="end-to-end",
                        choices=["contrastive", "generative", "end-to-end"],
                        help="The training mode to use.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints and results")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small", "standard", "large"],
                        help="Size of the model to train.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to a pretrained model for generative fine-tuning.")
    
    # Data arguments
    data_group = parser.add_argument_group('Data settings')
    data_group.add_argument("--vocab_path", type=str, default="checkpoints/vocab_cache.json", help="Path to vocabulary file (will be created if it doesn't exist)")
    data_group.add_argument("--val_set_size", type=int, default=10000, help="Size of the validation set")
    data_group.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")

    # Training arguments
    train_group = parser.add_argument_group('General training settings')
    train_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    # Contrastive phase arguments
    contrastive_group = parser.add_argument_group('Contrastive pre-training settings')
    contrastive_group.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for contrastive pre-training")
    contrastive_group.add_argument("--num_augmentations", type=int, default=2, help="Number of augmentations per sample for contrastive learning.")
    contrastive_group.add_argument("--alpha", type=float, default=0.5, help="Weight for contrastive loss.")
    contrastive_group.add_argument("--beta", type=float, default=0.1, help="Weight for reconstruction loss in contrastive phase")
    contrastive_group.add_argument("--gamma", type=float, default=0.01, help="Weight for diversity loss in contrastive phase")

    # Generative phase arguments
    generative_group = parser.add_argument_group('Generative fine-tuning settings')
    generative_group.add_argument("--max_steps", type=int, default=50000, help="Maximum training steps for generative phase")
    generative_group.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps for learning rate schedule")
    generative_group.add_argument("--eval_interval", type=int, default=500, help="Interval for validation")
    generative_group.add_argument("--generate_interval", type=int, default=2000, help="Interval for generating sample molecules")
    generative_group.add_argument("--num_generated", type=int, default=5, help="Number of molecules to generate")
    generative_group.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    generative_group.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    generative_group.add_argument("--use_grammar_constraints", action="store_true", help="Use grammar constraints during generation.")
    
    # System arguments
    system_group = parser.add_argument_group('System settings')
    system_group.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    system_group.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    system_group.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    system_group.add_argument("--load_pretrained", type=str, default=None, help="Path to load a pre-trained model for the generative phase")

    # New arguments
    parser.add_argument("--dataset_subset_size", type=int, default=None, help="Use a subset of the dataset for quick testing.")


    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")
    print(f"Using device: {device}")
    
    # Create tokenizer from specified vocab path
    vocab_path = Path(args.vocab_path)
    if not vocab_path.parent.exists():
        print(f"Creating directory for vocabulary: {vocab_path.parent}")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

    if not vocab_path.exists():
        print(f"Vocabulary not found at {vocab_path}. Building from data...")
        tokenizer = SELFIETokenizer(training_data_path=args.data_path)
        tokenizer._save_vocab(tokenizer.vocab, str(vocab_path))
        print(f"Vocabulary built and saved to {vocab_path}")
    else:
        print(f"Loading vocabulary from {vocab_path}")
        tokenizer = SELFIETokenizer(vocab_file=str(vocab_path))
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Run training
    pretrained_model_path = None
    if args.training_mode in ["contrastive", "end-to-end"]:
        pretrained_model_path = run_contrastive_phase(args, tokenizer, device)

        if pretrained_model_path and pretrained_model_path.exists():
            print("\n--- Generating molecules after contrastive phase ---")
            if args.model_size == "small": config = ModelConfig.small_config()
            elif args.model_size == "standard": config = ModelConfig.standard_config()
            else: config = ModelConfig.large_config()
            config.vocab_size = tokenizer.vocab_size

            generative_model = SELFIESGPTDecoder(config).to(device)
            print(f"Loading pretrained weights from {pretrained_model_path}")
            generative_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

            molecules = generate_molecules(generative_model, tokenizer, device, args)
            print("Generated molecules post-contrastive:")
            for i, mol in enumerate(molecules):
                print(f"  {i+1}: {mol}")
            print("--------------------------------------------------")

        # For end-to-end, use the newly trained model
        if args.training_mode == "end-to-end":
            args.load_pretrained = pretrained_model_path
    
    if args.training_mode in ["generative", "end-to-end"]:
        if args.training_mode == "generative" and not args.load_pretrained:
            print("Warning: Running generative training from scratch without a pre-trained model.")
        run_generative_phase(args, tokenizer, device)

    print("\n--- Training Pipeline Complete ---")


if __name__ == "__main__":
    main()

# endregion 