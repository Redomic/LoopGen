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
from model.rl_trainer import MolecularRewardCalculator, PPOTrainer
from model.scheduled_sampling import ScheduledSamplingScheduler
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

def _to_json_serializable(value):
    """Convert any value to JSON-serializable format."""
    if isinstance(value, torch.Tensor):
        return value.item()
    elif isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    else:
        return value

def save_training_metrics(run_dir: Path, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float, phase: Optional[str] = None):
    """Save training metrics to a JSON file."""
    phase_suffix = f"_phase{phase}" if phase else ""
    metrics_file = run_dir / f"metrics{phase_suffix}.jsonl"
    
    metrics_entry = {
        'epoch': epoch,
        'phase': phase,
        'timestamp': datetime.now().isoformat(),
        'train_loss': _to_json_serializable(train_metrics.get('loss', 0)),
        'train_reconstruction_loss': _to_json_serializable(train_metrics.get('reconstruction_loss', 0)),
        'train_perplexity': _to_json_serializable(train_metrics.get('perplexity', 0)),
        'train_accuracy': _to_json_serializable(train_metrics.get('accuracy', 0)),
        'train_entropy': _to_json_serializable(train_metrics.get('entropy', 0)),
        'val_loss': _to_json_serializable(val_metrics.get('loss', 0)),
        'val_reconstruction_loss': _to_json_serializable(val_metrics.get('reconstruction_loss', 0)),
        'val_perplexity': _to_json_serializable(val_metrics.get('perplexity', 0)),
        'val_accuracy': _to_json_serializable(val_metrics.get('accuracy', 0)),
        'val_entropy': _to_json_serializable(val_metrics.get('entropy', 0)),
        'learning_rate': float(lr)
    }
    
    # Add any additional RL metrics if present in train_metrics
    for key in train_metrics.keys():
        if key.startswith('ppo_') or key in ['validity_rate', 'avg_qed', 'avg_sa', 'ppo_weight', 'tf_prob', 'loss_scalar']:
            if key != 'loss_scalar':  # Skip loss_scalar, we already have train_loss
                metrics_entry[f'train_{key}'] = _to_json_serializable(train_metrics[key])
    
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

def protein_conditioned_collate_fn(
    batch: List[Dict[str, torch.Tensor]], 
    smiles_tokenizer: SMILESTokenizer,
    protein_tokenizer,  # ProteinTokenizer
    max_smiles_len: int = 256,
    max_protein_len: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Collate function for protein-ligand pairs with dynamic padding.
    
    Args:
        batch: List of dicts with 'smiles_ids', 'protein_ids', optional 'affinity'
        smiles_tokenizer: SMILES tokenizer for pad token ID
        protein_tokenizer: Protein tokenizer for pad token ID
        max_smiles_len: Maximum SMILES sequence length
        max_protein_len: Maximum protein sequence length
    
    Returns:
        Dictionary with padded tensors
    """
    # Extract sequences
    smiles_seqs = [item['smiles_ids'] for item in batch]
    protein_seqs = [item['protein_ids'] for item in batch]
    
    # Find max lengths in batch
    max_smiles = min(max(len(s) for s in smiles_seqs), max_smiles_len)
    max_protein = min(max(len(p) for p in protein_seqs), max_protein_len)
    
    # Pad SMILES sequences
    padded_smiles = []
    for seq in smiles_seqs:
        if len(seq) > max_smiles:
            padded = seq[:max_smiles]
        else:
            padding = torch.full((max_smiles - len(seq),), smiles_tokenizer.pad_token_id, dtype=torch.long)
            padded = torch.cat([seq, padding])
        padded_smiles.append(padded)
    
    # Pad protein sequences
    padded_proteins = []
    for seq in protein_seqs:
        if len(seq) > max_protein:
            padded = seq[:max_protein]
        else:
            padding = torch.full((max_protein - len(seq),), protein_tokenizer.pad_token_id, dtype=torch.long)
            padded = torch.cat([seq, padding])
        padded_proteins.append(padded)
    
    # Stack into tensors
    input_ids = torch.stack(padded_smiles)
    protein_ids = torch.stack(padded_proteins)
    
    # Create attention masks
    attention_mask = (input_ids != smiles_tokenizer.pad_token_id).long()
    protein_mask = (protein_ids != protein_tokenizer.pad_token_id).long()
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'protein_ids': protein_ids,
        'protein_mask': protein_mask
    }
    
    # Add affinity if present
    if 'affinity' in batch[0]:
        affinities = [item['affinity'] for item in batch if 'affinity' in item]
        if affinities:
            result['affinity'] = torch.stack(affinities)
    
    return result

def _train_epoch(model: SMILESGPTDecoder, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    """Train one epoch with autoregressive reconstruction loss."""
    model.train()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_entropy = 0.0
    num_batches = 0
    
    logging.info(f"Starting training epoch with {len(loader)} batches")
    
    try:
        # Test if we can get an iterator
        logging.info("Creating data loader iterator...")
        loader_iter = iter(loader)
        logging.info("Data loader iterator created successfully")
        
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")
        
        for i, batch in progress_bar:
            if i == 0:
                logging.info(f"Received first batch with keys: {batch.keys()}")
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Get protein data if available
                protein_ids = batch.get('protein_ids')
                protein_mask = batch.get('protein_mask')
                if protein_ids is not None:
                    protein_ids = protein_ids.to(device)
                    protein_mask = protein_mask.to(device)
                
                with autocast(device.type, enabled=args.use_amp):
                    # Compute autoregressive loss
                    loss_dict = model.compute_loss(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        protein_ids=protein_ids,
                        protein_mask=protein_mask
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
            except Exception as batch_error:
                logging.error(f"Error processing batch {i}: {batch_error}")
                logging.error(f"Batch keys: {batch.keys()}")
                if 'input_ids' in batch:
                    logging.error(f"Input shape: {batch['input_ids'].shape}")
                if 'protein_ids' in batch:
                    logging.error(f"Protein shape: {batch['protein_ids'].shape}")
                import traceback
                traceback.print_exc()
                raise
    except Exception as e:
        logging.error(f"Training epoch failed: {e}")
        raise
    
    # Safeguard against division by zero
    if num_batches == 0:
        logging.error("No batches were processed during training epoch!")
        raise RuntimeError("Training failed: No batches processed. Check data loader and model compatibility.")
    
    return {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'accuracy': total_accuracy / num_batches,
        'entropy': total_entropy / num_batches
    }

def _train_epoch_hybrid_rl(
    model: SMILESGPTDecoder, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scaler: GradScaler, 
    device: torch.device, 
    args: argparse.Namespace,
    ppo_trainer,
    scheduled_sampler,
    epoch: int
) -> Dict[str, float]:
    """
    Train one epoch with hybrid teacher forcing + RL approach.
    
    Combines:
    - Standard reconstruction loss (teacher forcing)
    - PPO reinforcement learning (validity rewards)
    - Scheduled sampling (gradual transition)
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_ppo_loss = 0.0
    total_validity = 0.0
    total_qed = 0.0
    total_sa = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_entropy = 0.0
    num_batches = 0
    num_rl_batches = 0
    
    # Running averages for progress bar display
    running_validity = 0.0
    running_qed = 0.0
    running_sa = 0.0
    
    # Get teacher forcing probability for this epoch
    tf_prob = scheduled_sampler.get_probability(epoch)
    
    # Calculate RL weight for this epoch
    if args.use_rl_training and epoch >= args.rl_start_epoch:
        adjusted_epoch = epoch - args.rl_start_epoch
        if args.rl_weight_schedule == 'progressive':
            # Gradually increase RL weight
            ppo_weight = min(args.rl_max_weight, adjusted_epoch / 20.0)
        else:  # fixed
            ppo_weight = args.rl_max_weight
    else:
        ppo_weight = 0.0
    
    logging.info(f"Hybrid RL training: TF prob={tf_prob:.3f}, PPO weight={ppo_weight:.3f}")
    
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Training (RL={ppo_weight:.2f})")
    
    for i, batch in progress_bar:
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get protein data if available
            protein_ids = batch.get('protein_ids')
            protein_mask = batch.get('protein_mask')
            if protein_ids is not None:
                protein_ids = protein_ids.to(device)
                protein_mask = protein_mask.to(device)
            
            with autocast(device.type, enabled=args.use_amp):
                # Compute hybrid loss (reconstruction + PPO)
                # Apply PPO every 5th batch for speed (40-50% faster than every 2nd batch)
                use_ppo_this_batch = (ppo_weight > 0 and i % 5 == 0)
                
                if use_ppo_this_batch:
                    loss_dict = model.compute_loss_with_ppo(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        protein_ids=protein_ids,
                        protein_mask=protein_mask,
                        ppo_trainer=ppo_trainer,
                        teacher_forcing_prob=tf_prob,
                        ppo_weight=ppo_weight
                    )
                    num_rl_batches += 1
                else:
                    # Standard reconstruction loss
                    loss_dict = model.compute_loss(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        protein_ids=protein_ids,
                        protein_mask=protein_mask
                    )
                
                loss = loss_dict['loss'] / args.grad_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % args.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Update metrics - use loss_scalar if available (from hybrid RL), otherwise convert
            if 'loss_scalar' in loss_dict:
                loss_value = loss_dict['loss_scalar']
            else:
                loss_value = loss_dict['loss'].item() if isinstance(loss_dict['loss'], torch.Tensor) else loss_dict['loss']
            
            total_loss += float(loss_value)
            
            # Track reconstruction metrics (available from both standard and RL training)
            if 'reconstruction_loss' in loss_dict:
                recon_value = loss_dict['reconstruction_loss']
                total_recon_loss += recon_value.item() if isinstance(recon_value, torch.Tensor) else float(recon_value)
            
            # Track base metrics (perplexity, accuracy, entropy)
            if 'perplexity' in loss_dict:
                total_perplexity += float(loss_dict['perplexity'])
            if 'accuracy' in loss_dict:
                total_accuracy += float(loss_dict['accuracy'])
            if 'entropy' in loss_dict:
                total_entropy += float(loss_dict['entropy'])
            
            # Track RL-specific metrics
            if use_ppo_this_batch and 'ppo_validity_rate' in loss_dict:
                num_rl_batches += 1
                total_ppo_loss += float(loss_dict.get('ppo_ppo_loss', 0.0))
                total_validity += float(loss_dict.get('ppo_validity_rate', 0.0))
                total_qed += float(loss_dict.get('ppo_avg_qed', 0.0))
                total_sa += float(loss_dict.get('ppo_avg_sa', 0.0))
                
                # Update running averages for display
                running_validity = total_validity / num_rl_batches
                running_qed = total_qed / num_rl_batches
                running_sa = total_sa / num_rl_batches
            
            num_batches += 1
            
            # Update progress bar
            postfix = {
                'loss': f"{loss_value:.3f}",
            }
            
            # Show RL metrics if any RL batches have been processed
            if num_rl_batches > 0 and ppo_weight > 0:
                postfix['valid'] = f"{running_validity:.2%}"
                # Only show QED and SA if they're enabled (non-zero weights)
                if args.reward_qed_weight > 0:
                    postfix['qed'] = f"{running_qed:.3f}"
                if args.reward_sa_weight > 0:
                    postfix['sa'] = f"{running_sa:.3f}"
            else:
                # Show accuracy when RL hasn't started yet
                if 'accuracy' in loss_dict:
                    postfix['acc'] = f"{loss_dict['accuracy']:.3f}"
            
            progress_bar.set_postfix(postfix)
            
        except Exception as batch_error:
            logging.error(f"Error processing batch {i}: {batch_error}")
            import traceback
            traceback.print_exc()
            raise
    
    # Calculate averages
    metrics = {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches if total_recon_loss > 0 else total_loss / num_batches,
        'perplexity': total_perplexity / num_batches if total_perplexity > 0 else 0.0,
        'accuracy': total_accuracy / num_batches if total_accuracy > 0 else 0.0,
        'entropy': total_entropy / num_batches if total_entropy > 0 else 0.0,
        'ppo_weight': ppo_weight,
        'tf_prob': tf_prob
    }
    
    # Add RL metrics if we used RL this epoch
    if num_rl_batches > 0:
        metrics['validity_rate'] = total_validity / num_rl_batches
        metrics['avg_qed'] = total_qed / num_rl_batches
        metrics['avg_sa'] = total_sa / num_rl_batches
    
    return metrics

def _validate(model: SMILESGPTDecoder, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    """Validate model with reconstruction loss."""
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_entropy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        try:
            for batch in tqdm(loader, desc="Validation"):
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Get protein data if available
                    protein_ids = batch.get('protein_ids')
                    protein_mask = batch.get('protein_mask')
                    if protein_ids is not None:
                        protein_ids = protein_ids.to(device)
                        protein_mask = protein_mask.to(device)
                    
                    with autocast(device.type, enabled=args.use_amp):
                        loss_dict = model.compute_loss(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            protein_ids=protein_ids,
                            protein_mask=protein_mask
                        )
                    
                    total_loss += loss_dict['loss'].item()
                    total_perplexity += loss_dict['perplexity'].item()
                    total_accuracy += loss_dict['accuracy'].item()
                    total_entropy += loss_dict['entropy'].item()
                    num_batches += 1
                except Exception as batch_error:
                    logging.error(f"Error processing validation batch: {batch_error}")
                    logging.error(f"Batch keys: {batch.keys()}")
                    if 'input_ids' in batch:
                        logging.error(f"Input shape: {batch['input_ids'].shape}")
                    if 'protein_ids' in batch:
                        logging.error(f"Protein shape: {batch['protein_ids'].shape}")
                    import traceback
                    traceback.print_exc()
                    raise
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise
    
    # Safeguard against division by zero
    if num_batches == 0:
        logging.error("No batches were processed during validation!")
        raise RuntimeError("Validation failed: No batches processed. Check data loader and model compatibility.")
    
    return {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'accuracy': total_accuracy / num_batches,
        'entropy': total_entropy / num_batches
    }

def generate_molecules(
    model: SMILESGPTDecoder, 
    tokenizer: SMILESTokenizer, 
    device: torch.device, 
    args,
    protein_ids: Optional[torch.Tensor] = None,
    protein_mask: Optional[torch.Tensor] = None
) -> List[str]:
    """Generate molecules using the model's built-in generation method with repetition control."""
    model.eval()
    
    # Generate multiple sequences with repetition control
    generated_ids = model.generate(
        prompt_ids=None,  # Start from BOS
        protein_ids=protein_ids,
        protein_mask=protein_mask,
        max_length=args.max_seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_generated,
        repetition_penalty=getattr(args, 'repetition_penalty', 1.2),
        ngram_block_size=getattr(args, 'ngram_block_size', 3),
        apply_repetition_control=True
    )
    
    # Decode to SMILES strings
    generated_molecules = []
    for seq_ids in generated_ids:
        smiles_str = tokenizer.decode(seq_ids.tolist(), skip_special_tokens=True)
        generated_molecules.append(smiles_str)
    
    return generated_molecules

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, run_dir, phase: Optional[str] = None):
    """Save a complete training checkpoint with epoch-specific naming."""
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    
    # Save epoch-specific checkpoint
    phase_suffix = f"_phase{phase}" if phase else ""
    epoch_checkpoint_path = run_dir / f"model{phase_suffix}_ep{epoch:03d}.pt"
    torch.save(checkpoint, epoch_checkpoint_path)
    
    # Also save as latest checkpoint for easy resuming
    latest_checkpoint_path = run_dir / f"checkpoint{phase_suffix}_latest.pt"
    torch.save(checkpoint, latest_checkpoint_path)
    
    logging.info(f"Saved checkpoint for epoch {epoch} to {epoch_checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load a training checkpoint and return the epoch and best validation loss."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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

def save_generated_molecules(molecules: List[str], run_dir: Path, epoch: int, phase: Optional[str] = None):
    """Save generated molecules to file."""
    phase_suffix = f"_phase{phase}" if phase else ""
    molecules_file = run_dir / f"generated_molecules{phase_suffix}_ep{epoch:03d}.txt"
    with open(molecules_file, 'w') as f:
        for i, mol in enumerate(molecules):
            f.write(f"{i+1}: {mol}\n")
    logging.info(f"Saved {len(molecules)} generated molecules to {molecules_file}")

def transfer_phase1_weights_to_phase2(phase1_checkpoint_path: Path, phase2_model: SMILESGPTDecoder, device: torch.device):
    """
    Transfer Phase 1 decoder weights to Phase 2 model with protein conditioning.
    
    This enables transfer learning by loading pretrained SMILES decoder weights
    while leaving protein encoder and cross-attention layers randomly initialized.
    
    Args:
        phase1_checkpoint_path: Path to Phase 1 checkpoint
        phase2_model: Phase 2 model with protein conditioning enabled
        device: Device for loading checkpoint
    """
    logging.info(f"Loading Phase 1 weights from {phase1_checkpoint_path}")
    
    # Load Phase 1 checkpoint (weights_only=False for custom objects like ModelConfig)
    checkpoint = torch.load(phase1_checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state dict (handle both full checkpoint and weights-only formats)
    if 'model_state_dict' in checkpoint:
        phase1_state_dict = checkpoint['model_state_dict']
    else:
        phase1_state_dict = checkpoint
    
    # Get Phase 2 model state dict
    phase2_state_dict = phase2_model.state_dict()
    
    # Transfer compatible weights (decoder components)
    transferred_keys = []
    skipped_keys = []
    
    for key, value in phase1_state_dict.items():
        # Skip protein-specific components (not in Phase 1)
        if 'protein_encoder' in key or 'cross_attention' in key:
            skipped_keys.append(key)
            continue
        
        # Transfer if key exists in Phase 2 model and shapes match
        if key in phase2_state_dict and phase2_state_dict[key].shape == value.shape:
            phase2_state_dict[key] = value
            transferred_keys.append(key)
        else:
            skipped_keys.append(key)
    
    # Load the updated state dict
    phase2_model.load_state_dict(phase2_state_dict)
    
    logging.info(f"Transferred {len(transferred_keys)} weight tensors from Phase 1")
    logging.info(f"Skipped {len(skipped_keys)} keys (protein-specific or incompatible)")
    logging.info("Phase 2 model initialized with Phase 1 decoder weights")

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

# region: Multi-Phase Training Functions
# ==============================================================================

def train_phase1(args, tokenizer, device, run_dir: Path) -> Path:
    """
    Phase 1: SMILES pretraining on large SMILES dataset.
    
    Returns:
        Path to the final Phase 1 checkpoint
    """
    logging.info("="*60)
    logging.info("PHASE 1: SMILES Pretraining")
    logging.info("="*60)
    logging.info(f"Training for {args.phase1_epochs} epochs on SMILES dataset")
    
    # Config WITHOUT protein conditioning
    if args.model_size == "small": config = ModelConfig.small_config()
    elif args.model_size == "standard": config = ModelConfig.standard_config()
    else: config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size
    config.use_protein_conditioning = False  # Phase 1 is SMILES-only
    
    # SMILES dataset with optional subset
    total_lines = args.dataset_subset_size or count_lines(args.data_path)
    val_size = min(args.val_set_size, int(total_lines * 0.2))
    train_size = total_lines - val_size
    train_split_ratio = train_size / total_lines if total_lines > 0 else 0.0
    logging.info(f"Phase 1 Dataset size: {total_lines} (Train: {train_size}, Val: {val_size})")
    
    train_dataset = SMILESDataset(args.data_path, tokenizer, config.max_seq_len, 
                                  total_lines=total_lines, split='train', split_ratio=train_split_ratio)
    val_dataset = SMILESDataset(args.data_path, tokenizer, config.max_seq_len, 
                                total_lines=total_lines, split='val', split_ratio=train_split_ratio)
    
    # Collate function for autoregressive training
    from functools import partial
    collate_fn_impl = partial(_autoregressive_collate_fn, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                              pin_memory=True, collate_fn=collate_fn_impl)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                            pin_memory=True, collate_fn=collate_fn_impl)
    
    # Model, optimizer, scaler
    model = SMILESGPTDecoder(config).to(device)
    model.set_tokenizer(tokenizer)
    model.label_smoothing = args.label_smoothing
    
    logging.info(f"Phase 1 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3, threshold=0.01, min_lr=1e-6
    )
    
    # Initialize RL components
    reward_calculator = None
    ppo_trainer = None
    scheduled_sampler = None
    
    if args.use_rl_training:
        logging.info("[Phase 1] Initializing RL components (PPO + Scheduled Sampling)")
        try:
            reward_calculator = MolecularRewardCalculator(
                validity_weight=args.reward_validity_weight,
                qed_weight=args.reward_qed_weight,
                sa_weight=args.reward_sa_weight
            )
            ppo_trainer = PPOTrainer(
                model=model,
                tokenizer=tokenizer,
                reward_calculator=reward_calculator,
                clip_epsilon=args.ppo_clip_epsilon,
                value_coef=args.ppo_value_coef,
                entropy_coef=args.ppo_entropy_coef,
                max_rollout_length=100
            )
            scheduled_sampler = ScheduledSamplingScheduler(
                total_epochs=args.phase1_epochs,
                schedule_type=args.scheduled_sampling_type,
                warmup_epochs=args.rl_start_epoch
            )
            logging.info(f"[Phase 1] RL initialized: {scheduled_sampler}")
        except Exception as e:
            logging.warning(f"[Phase 1] Failed to initialize RL components: {e}")
            logging.warning("[Phase 1] Falling back to standard training without RL")
            args.use_rl_training = False
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.phase1_epochs):
        logging.info(f"[Phase 1] Epoch {epoch + 1}/{args.phase1_epochs}")
        
        # Use hybrid RL training if enabled, otherwise standard training
        if args.use_rl_training and ppo_trainer is not None:
            train_metrics = _train_epoch_hybrid_rl(
                model, train_loader, optimizer, scaler, device, args,
                ppo_trainer, scheduled_sampler, epoch
            )
        else:
            train_metrics = _train_epoch(model, train_loader, optimizer, scaler, device, args)
        val_metrics = _validate(model, val_loader, device, args)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"[Phase 1] Epoch {epoch + 1}")
        
        # Build training metrics string
        train_str = f"  Train - Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}, Acc: {train_metrics['accuracy']:.3f}"
        if 'validity_rate' in train_metrics:
            train_str += f", Valid: {train_metrics['validity_rate']:.2%}"
        logging.info(train_str)
        
        logging.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}, "
                    f"Acc: {val_metrics['accuracy']:.3f}")
        
        save_training_metrics(run_dir, epoch + 1, train_metrics, val_metrics, current_lr, phase="1")
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = run_dir / "best_model_phase1.pt"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"[Phase 1] Saved best model with validation loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, run_dir, phase="1")
        
        # Generate samples
        if (epoch + 1) % args.generate_interval == 0:
            logging.info("[Phase 1] Generating molecules...")
            molecules = generate_molecules(model, tokenizer, device, args)
            save_generated_molecules(molecules, run_dir, epoch + 1, phase="1")
            for i, mol in enumerate(molecules[:3]):
                logging.info(f"  Generated {i+1}: {mol}")
    
    # Save final checkpoint
    phase1_final = run_dir / "phase1_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_loss': best_val_loss
    }, phase1_final)
    
    logging.info(f"[Phase 1] Complete! Final checkpoint: {phase1_final}")
    return phase1_final


def train_phase2(args, tokenizer, device, run_dir: Path, phase1_checkpoint: Path):
    """
    Phase 2: Protein-conditioned fine-tuning on protein-ligand pairs.
    
    Args:
        phase1_checkpoint: Path to Phase 1 final checkpoint for weight transfer
    """
    logging.info("="*60)
    logging.info("PHASE 2: Protein Conditioning")
    logging.info("="*60)
    logging.info(f"Training for {args.phase2_epochs} epochs on protein-ligand data")
    
    # Initialize protein tokenizer
    from molecule_utils.protein_tokenizer import ProteinTokenizer
    protein_tokenizer = ProteinTokenizer()
    logging.info(f"Initialized ProteinTokenizer with vocab size: {protein_tokenizer.vocab_size}")
    
    # Config WITH protein conditioning
    if args.model_size == "small": config = ModelConfig.small_config()
    elif args.model_size == "standard": config = ModelConfig.standard_config()
    else: config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size
    config.use_protein_conditioning = True
    config.protein_vocab_size = protein_tokenizer.vocab_size
    config.protein_max_seq_len = args.protein_max_seq_len
    config.protein_encoder_layers = args.protein_encoder_layers
    config.protein_encoder_heads = args.protein_encoder_heads
    config.use_cross_attention = True
    config.cross_attention_freq = args.cross_attention_freq
    
    # Protein-ligand dataset (USE ALL DATA)
    from molecule_utils.protein_ligand_dataset import ProteinLigandDataset, count_protein_ligand_pairs
    
    total_lines = count_protein_ligand_pairs(args.protein_ligand_data_path)
    val_size = min(args.val_set_size, int(total_lines * 0.2))
    train_size = total_lines - val_size
    train_split_ratio = train_size / total_lines if total_lines > 0 else 0.0
    logging.info(f"Phase 2 Dataset size: {total_lines} (Train: {train_size}, Val: {val_size})")
    
    train_dataset = ProteinLigandDataset(
        file_path=args.protein_ligand_data_path,
        smiles_tokenizer=tokenizer,
        protein_tokenizer=protein_tokenizer,
        max_smiles_len=config.max_seq_len,
        max_protein_len=config.protein_max_seq_len,
        total_lines=total_lines,
        split='train',
        split_ratio=train_split_ratio
    )
    val_dataset = ProteinLigandDataset(
        file_path=args.protein_ligand_data_path,
        smiles_tokenizer=tokenizer,
        protein_tokenizer=protein_tokenizer,
        max_smiles_len=config.max_seq_len,
        max_protein_len=config.protein_max_seq_len,
        total_lines=total_lines,
        split='val',
        split_ratio=train_split_ratio
    )
    
    # Collate function for protein-conditioned training
    from functools import partial
    collate_fn_impl = partial(
        protein_conditioned_collate_fn,
        smiles_tokenizer=tokenizer,
        protein_tokenizer=protein_tokenizer,
        max_smiles_len=config.max_seq_len,
        max_protein_len=config.protein_max_seq_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn_impl)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn_impl)
    
    # Model with protein conditioning
    model = SMILESGPTDecoder(config).to(device)
    model.set_tokenizer(tokenizer)
    model.label_smoothing = args.label_smoothing
    
    # Transfer Phase 1 weights
    transfer_phase1_weights_to_phase2(phase1_checkpoint, model, device)
    
    logging.info(f"Phase 2 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # RESET optimizer and scheduler with fresh learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3, threshold=0.01, min_lr=1e-6
    )
    
    logging.info(f"Reset optimizer with learning rate: {args.learning_rate}")
    
    # Initialize RL components
    reward_calculator = None
    ppo_trainer = None
    scheduled_sampler = None
    
    if args.use_rl_training:
        logging.info("[Phase 2] Initializing RL components (PPO + Scheduled Sampling)")
        try:
            reward_calculator = MolecularRewardCalculator(
                validity_weight=args.reward_validity_weight,
                qed_weight=args.reward_qed_weight,
                sa_weight=args.reward_sa_weight
            )
            ppo_trainer = PPOTrainer(
                model=model,
                tokenizer=tokenizer,
                reward_calculator=reward_calculator,
                clip_epsilon=args.ppo_clip_epsilon,
                value_coef=args.ppo_value_coef,
                entropy_coef=args.ppo_entropy_coef,
                max_rollout_length=100
            )
            scheduled_sampler = ScheduledSamplingScheduler(
                total_epochs=args.phase2_epochs,
                schedule_type=args.scheduled_sampling_type,
                warmup_epochs=args.rl_start_epoch
            )
            logging.info(f"[Phase 2] RL initialized: {scheduled_sampler}")
        except Exception as e:
            logging.warning(f"[Phase 2] Failed to initialize RL components: {e}")
            logging.warning("[Phase 2] Falling back to standard training without RL")
            args.use_rl_training = False
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.phase2_epochs):
        logging.info(f"[Phase 2] Epoch {epoch + 1}/{args.phase2_epochs}")
        
        # Use hybrid RL training if enabled, otherwise standard training
        if args.use_rl_training and ppo_trainer is not None:
            train_metrics = _train_epoch_hybrid_rl(
                model, train_loader, optimizer, scaler, device, args,
                ppo_trainer, scheduled_sampler, epoch
            )
        else:
            train_metrics = _train_epoch(model, train_loader, optimizer, scaler, device, args)
        val_metrics = _validate(model, val_loader, device, args)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"[Phase 2] Epoch {epoch + 1}")
        
        # Build training metrics string
        train_str = f"  Train - Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}, Acc: {train_metrics['accuracy']:.3f}"
        if 'validity_rate' in train_metrics:
            train_str += f", Valid: {train_metrics['validity_rate']:.2%}"
        logging.info(train_str)
        
        logging.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}, "
                    f"Acc: {val_metrics['accuracy']:.3f}")
        
        save_training_metrics(run_dir, epoch + 1, train_metrics, val_metrics, current_lr, phase="2")
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = run_dir / "best_model_phase2.pt"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"[Phase 2] Saved best model with validation loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, run_dir, phase="2")
        
        # Generate samples
        if (epoch + 1) % args.generate_interval == 0:
            logging.info("[Phase 2] Generating protein-conditioned molecules...")
            # Use first protein from validation set
            val_sample = next(iter(val_loader))
            protein_ids = val_sample['protein_ids'][:1].to(device)
            protein_mask = val_sample['protein_mask'][:1].to(device)
            
            molecules = generate_molecules(model, tokenizer, device, args, 
                                          protein_ids=protein_ids, protein_mask=protein_mask)
            save_generated_molecules(molecules, run_dir, epoch + 1, phase="2")
            for i, mol in enumerate(molecules[:3]):
                logging.info(f"  Generated {i+1}: {mol}")
    
    # Save final checkpoint
    phase2_final = run_dir / "phase2_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_loss': best_val_loss
    }, phase2_final)
    
    logging.info(f"[Phase 2] Complete! Final checkpoint: {phase2_final}")
    logging.info("="*60)
    logging.info("Multi-Phase Training Complete!")
    logging.info("="*60)


def multi_phase_train(args, tokenizer, device):
    """
    Orchestrator for multi-phase training.
    
    Phase 1: SMILES pretraining
    Phase 2: Protein conditioning
    """
    # Setup run directory
    run_dir = setup_run_directory(args.output_dir)
    save_run_config(args, run_dir, tokenizer)
    
    logging.info("Starting Multi-Phase Training Pipeline")
    logging.info(f"Phase 1 Epochs: {args.phase1_epochs}")
    logging.info(f"Phase 2 Epochs: {args.phase2_epochs}")
    
    # Phase 1: SMILES pretraining
    phase1_checkpoint = train_phase1(args, tokenizer, device, run_dir)
    
    # Phase 2: Protein conditioning
    train_phase2(args, tokenizer, device, run_dir, phase1_checkpoint)
    
    logging.info(f"All results saved to: {run_dir}")

# endregion

def train_single_phase(args, tokenizer, device):
    """Single-phase training (legacy mode for backward compatibility)."""
    # Setup run directory and logging
    run_dir = setup_run_directory(args.output_dir)
    save_run_config(args, run_dir, tokenizer)
    
    phase_name = "protein_conditioning" if args.use_protein_conditioning else "smiles_pretraining"
    logging.info(f"Starting Single-Phase Training: {phase_name}")
    
    # Initialize protein tokenizer if using protein conditioning
    protein_tokenizer = None
    if args.use_protein_conditioning:
        from molecule_utils.protein_tokenizer import ProteinTokenizer
        protein_tokenizer = ProteinTokenizer()
        logging.info(f"Initialized ProteinTokenizer with vocab size: {protein_tokenizer.vocab_size}")
    
    # Config
    if args.model_size == "small": config = ModelConfig.small_config()
    elif args.model_size == "standard": config = ModelConfig.standard_config()
    else: config = ModelConfig.large_config()
    config.vocab_size = tokenizer.vocab_size
    
    # Add protein conditioning config
    if args.use_protein_conditioning:
        config.use_protein_conditioning = True
        config.protein_vocab_size = protein_tokenizer.vocab_size
        config.protein_max_seq_len = args.protein_max_seq_len
        config.protein_encoder_layers = args.protein_encoder_layers
        config.protein_encoder_heads = args.protein_encoder_heads
        config.use_cross_attention = True
        config.cross_attention_freq = args.cross_attention_freq
        logging.info(f"Protein conditioning enabled with {args.protein_encoder_layers} encoder layers")

    # Datasets and loaders
    if args.use_protein_conditioning:
        from molecule_utils.protein_ligand_dataset import ProteinLigandDataset, count_protein_ligand_pairs
        
        # Use protein-ligand dataset (ALWAYS use ALL data - no subset)
        total_lines = count_protein_ligand_pairs(args.protein_ligand_data_path)
        val_size = min(args.val_set_size, int(total_lines * 0.2))
        train_size = total_lines - val_size
        train_split_ratio = train_size / total_lines if total_lines > 0 else 0.0
        logging.info(f"Protein-Ligand Dataset size: {total_lines} (Train: {train_size}, Val: {val_size})")
        
        train_dataset = ProteinLigandDataset(
            file_path=args.protein_ligand_data_path,
            smiles_tokenizer=tokenizer,
            protein_tokenizer=protein_tokenizer,
            max_smiles_len=config.max_seq_len,
            max_protein_len=config.protein_max_seq_len,
            total_lines=total_lines,
            split='train',
            split_ratio=train_split_ratio
        )
        val_dataset = ProteinLigandDataset(
            file_path=args.protein_ligand_data_path,
            smiles_tokenizer=tokenizer,
            protein_tokenizer=protein_tokenizer,
            max_smiles_len=config.max_seq_len,
            max_protein_len=config.protein_max_seq_len,
            total_lines=total_lines,
            split='val',
            split_ratio=train_split_ratio
        )
        
        # Use protein-conditioned collate function
        from functools import partial
        collate_fn_impl = partial(
            protein_conditioned_collate_fn,
            smiles_tokenizer=tokenizer,
            protein_tokenizer=protein_tokenizer,
            max_smiles_len=config.max_seq_len,
            max_protein_len=config.protein_max_seq_len
        )
    else:
        # Regular SMILES-only dataset
        total_lines = args.dataset_subset_size or count_lines(args.data_path)
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
        collate_fn_impl = partial(_autoregressive_collate_fn, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    
    # Note: IterableDataset handles shuffling internally, so we don't use shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_impl)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_impl)
    
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
    data_group.add_argument("--use_atomwise", action="store_true", help="Use atomwise tokenization instead of SPE (simpler, faster, ~50 tokens)")
    data_group.add_argument("--val_set_size", type=int, default=10000, help="Size of the validation set")
    data_group.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    data_group.add_argument("--dataset_subset_size", type=int, default=None, help="Use a subset of the dataset")

    # Training arguments
    train_group = parser.add_argument_group('Training settings')
    train_group.add_argument("--num_epochs", type=int, default=50, help="Number of epochs (single-phase training)")
    train_group.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    train_group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    train_group.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for reconstruction loss")
    
    # Multi-phase training arguments
    multiphase_group = parser.add_argument_group('Multi-phase training')
    multiphase_group.add_argument("--enable_multiphase", action="store_true", 
                                  help="Enable two-phase training (Phase 1: SMILES pretraining, Phase 2: Protein conditioning)")
    multiphase_group.add_argument("--phase1_epochs", type=int, default=20, 
                                  help="Number of epochs for Phase 1 (SMILES pretraining)")
    multiphase_group.add_argument("--phase2_epochs", type=int, default=30,
                                  help="Number of epochs for Phase 2 (protein conditioning)")

    
    # Protein conditioning arguments
    protein_group = parser.add_argument_group('Protein conditioning')
    protein_group.add_argument("--use_protein_conditioning", action="store_true", help="Enable protein conditioning")
    protein_group.add_argument("--protein_ligand_data_path", type=str, default="data/output/protein_ligand_training.csv", help="Path to protein-ligand training data CSV")
    protein_group.add_argument("--protein_vocab_path", type=str, default=None, help="Path to protein vocabulary (optional)")
    protein_group.add_argument("--protein_max_seq_len", type=int, default=512, help="Maximum protein sequence length")
    protein_group.add_argument("--protein_encoder_layers", type=int, default=6, help="Number of protein encoder layers")
    protein_group.add_argument("--protein_encoder_heads", type=int, default=8, help="Number of attention heads in protein encoder")
    protein_group.add_argument("--cross_attention_freq", type=int, default=1, help="Apply cross-attention every N decoder layers")

    # RL Training arguments
    rl_group = parser.add_argument_group('Reinforcement Learning')
    rl_group.add_argument("--use_rl_training", action="store_true", default=True, help="Enable RL training with PPO")
    rl_group.add_argument("--rl_start_epoch", type=int, default=5, help="Epoch to start RL training")
    rl_group.add_argument("--rl_max_weight", type=float, default=0.5, help="Maximum weight for RL loss")
    rl_group.add_argument("--rl_weight_schedule", type=str, default="progressive", choices=["progressive", "fixed"], help="RL weight schedule")
    rl_group.add_argument("--ppo_clip_epsilon", type=float, default=0.2, help="PPO clipping parameter")
    rl_group.add_argument("--ppo_value_coef", type=float, default=0.5, help="PPO value loss coefficient")
    rl_group.add_argument("--ppo_entropy_coef", type=float, default=0.01, help="PPO entropy coefficient")
    rl_group.add_argument("--ppo_num_rollouts", type=int, default=4, help="Number of rollouts per batch")
    rl_group.add_argument("--reward_validity_weight", type=float, default=1.0, help="Weight for validity in reward")
    rl_group.add_argument("--reward_qed_weight", type=float, default=0.0, help="Weight for QED in reward (0.0 = disabled for speed)")
    rl_group.add_argument("--reward_sa_weight", type=float, default=0.0, help="Weight for SA score in reward (0.0 = disabled for speed)")
    rl_group.add_argument("--scheduled_sampling_type", type=str, default="inverse_sigmoid", choices=["linear", "exponential", "inverse_sigmoid"], help="Scheduled sampling strategy")

    # Generation arguments
    gen_group = parser.add_argument_group('Generation settings')
    gen_group.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty for generation")
    gen_group.add_argument("--ngram_block_size", type=int, default=3, help="N-gram size for blocking repetitions")
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

    tokenization_mode = "Atomwise" if args.use_atomwise else "SPE (Substructure)"
    print(f"Tokenization mode: {tokenization_mode}")
    
    if not vocab_path.exists():
        print(f"Building vocabulary from {args.data_path}...")
        tokenizer = SMILESTokenizer(data_path=args.data_path, use_atomwise=args.use_atomwise)
        tokenizer.save_vocabulary(str(vocab_path))
        print(f"Vocabulary saved to {vocab_path}")
    else:
        print(f"Loading vocabulary from {vocab_path}")
        tokenizer = SMILESTokenizer(vocab_path=str(vocab_path), use_atomwise=args.use_atomwise)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Tokenization strategy: {'Atomwise (simple)' if tokenizer.use_atomwise or not tokenizer.use_spe else 'SPE (substructure)'}")

    # Analyze sequence length distribution to choose a reasonable cap
    analyze_sequence_lengths(args.data_path, tokenizer)

    # Run training (multi-phase or single-phase)
    if args.enable_multiphase:
        multi_phase_train(args, tokenizer, device)
    else:
        train_single_phase(args, tokenizer, device)
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main() 