#!/usr/bin/env python3
"""
Diagnostic script to test if the SMILES GPT model can overfit on a single batch.
This helps identify fundamental issues with the model architecture or data pipeline.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path to import your modules
sys.path.append(str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.decoder import SMILESGPTDecoder
from molecule_utils.tokenizer import SMILESTokenizer
from molecule_utils.dataset import SMILESDataset, collate_fn
from torch.utils.data import DataLoader


def load_model_and_tokenizer(checkpoint_path, vocab_path, device):
    """Load model from checkpoint and tokenizer."""
    print(f"Loading tokenizer from {vocab_path}")
    tokenizer = SMILESTokenizer(vocab_path=vocab_path)
    
    print(f"Loading model from {checkpoint_path}")
    
    # Create config
    config = ModelConfig.standard_config()
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = SMILESGPTDecoder(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, tokenizer


def generate_from_model(model, tokenizer, device, prompt_tokens=None, max_length=50, temperature=0.1):
    """Generate sequence using the model's forward pass."""
    model.eval()
    
    with torch.no_grad():
        # Start with BOS token or provided prompt
        if prompt_tokens is None:
            current_sequence = [tokenizer.bos_token_id]
        else:
            current_sequence = prompt_tokens.tolist()
        
        for _ in range(max_length - len(current_sequence)):
            # Prepare input
            input_ids = torch.tensor([current_sequence], device=device)
            
            # Forward pass
            output = model(input_ids)
            logits = output['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if EOS
            if next_token == tokenizer.eos_token_id:
                break
                
            current_sequence.append(next_token)
    
    return current_sequence


def run_overfit_test(model, tokenizer, data_path, device, 
                     num_steps=100, learning_rate=1e-3, 
                     batch_size=32, max_seq_len=256):
    """Run overfit test on a single batch."""
    
    print("\n=== Setting up data loader ===")
    # Create minimal dataset
    dataset = SMILESDataset(
        file_path=data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        total_lines=10000,  # Just need a small amount
        split='train',
        split_ratio=1.0,
        shuffle_buffer_size=1000
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token: {tokenizer.pad_token_id}")
    print(f"BOS token: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token_id}")
    
    # Get single batch
    print("\n=== Loading single batch ===")
    single_batch = next(iter(train_loader))
    input_ids = single_batch['input_ids'].to(device)
    attention_mask = single_batch['attention_mask'].to(device)
    
    print(f"Batch shape: {input_ids.shape}")
    print(f"First sequence tokens: {input_ids[0][:20].tolist()}")
    print(f"First sequence decoded: {tokenizer.decode(input_ids[0].tolist())[:100]}...")
    
    # Setup training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n=== Running overfit test for {num_steps} steps ===")
    print(f"Learning rate: {learning_rate}")
    print(f"If working correctly, loss should decrease significantly\n")
    
    losses = []
    
    for i in range(num_steps):
        # Forward pass
        output = model(input_ids, attention_mask)
        logits = output['logits']
        
        # Calculate loss - shift by one for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, tokenizer.vocab_size),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
        if i % 20 == 0:
            print(f"Step {i:3d}, Loss: {loss.item():.4f}")
            
            # Additional debugging info
            if i == 0:
                print(f"    Logits shape: {logits.shape}")
                print(f"    Shift logits shape: {shift_logits.shape}")
                print(f"    Shift labels shape: {shift_labels.shape}")
                print(f"    Unique labels: {len(torch.unique(shift_labels))}")
    
    # Final analysis
    print("\n=== Analysis ===")
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Reduction: {reduction:.1f}%")
    
    if final_loss < 0.5:
        print("\n✓ Model CAN overfit - architecture is likely correct")
    elif final_loss < 2.0:
        print("\n⚠ Model partially overfits - might have issues")
    else:
        print("\n✗ Model CANNOT overfit - fundamental issue exists!")
    
    # Test generation on the overfitted sequence
    print("\n=== Testing generation on memorized sequence ===")
    
    # Use first 10 tokens as prompt
    prompt = input_ids[0][:10]
    generated_sequence = generate_from_model(
        model, tokenizer, device, 
        prompt_tokens=prompt,
        max_length=50,
        temperature=0.1
    )
    
    print(f"Original: {tokenizer.decode(input_ids[0].tolist())[:100]}...")
    print(f"Generated: {tokenizer.decode(generated_sequence)[:100]}...")
    
    # Also test free generation
    print("\n=== Testing free generation (no prompt) ===")
    free_generated = generate_from_model(
        model, tokenizer, device,
        prompt_tokens=None,
        max_length=50,
        temperature=0.9
    )
    print(f"Free generation: {tokenizer.decode(free_generated)}")


def main():
    parser = argparse.ArgumentParser(
        description="Test if SMILES GPT model can overfit on single batch"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to SMILES CSV file"
    )
    
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="checkpoints/vocab.json",
        help="Path to vocabulary JSON"
    )
    
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for overfitting"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint, 
        args.vocab_path,
        device
    )
    
    # Run test
    run_overfit_test(
        model,
        tokenizer,
        args.data_path,
        device,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()