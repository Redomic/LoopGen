#!/usr/bin/env python3
"""
Protein-Conditioned Molecular Generation Script

Generate drug-like molecules conditioned on protein binding site sequences.
This script takes an amino acid sequence and generates molecules that are
tailored for binding to that specific protein pocket.

Usage:
    python generate_molecules.py \
        --protein_sequence "MKTAYIAKQRQISFVKSHFSRQLE" \
        --checkpoint checkpoints/run_XXX/best_model.pt \
        --vocab checkpoints/vocab.json \
        --num_samples 10

Author: LoopGen Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import torch
import pandas as pd

from model.config import ModelConfig
from model.decoder import SMILESGPTDecoder
from molecule_utils.tokenizer import SMILESTokenizer
from molecule_utils.protein_tokenizer import ProteinTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import RDKit for validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Molecular property calculations disabled.")
    RDKIT_AVAILABLE = False


def load_protein_conditioned_model(
    checkpoint_path: str,
    vocab_path: str,
    model_size: str = 'standard',
    device: torch.device = None
) -> tuple:
    """
    Load a trained protein-conditioned molecular generation model.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        vocab_path: Path to SMILES vocabulary JSON
        model_size: Model size (small/standard/large)
        device: PyTorch device
    
    Returns:
        Tuple of (model, smiles_tokenizer, protein_tokenizer)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model on device: {device}")
    
    # Load tokenizers
    logger.info(f"Loading SMILES vocabulary from {vocab_path}")
    smiles_tokenizer = SMILESTokenizer(vocab_path=vocab_path)
    
    logger.info("Initializing protein tokenizer")
    protein_tokenizer = ProteinTokenizer()
    
    # Create model configuration
    logger.info(f"Creating {model_size} model configuration")
    if model_size == 'small':
        config = ModelConfig.small_config()
    elif model_size == 'large':
        config = ModelConfig.large_config()
    else:
        config = ModelConfig.standard_config()
    
    # Configure for protein conditioning
    config.vocab_size = smiles_tokenizer.vocab_size
    config.use_protein_conditioning = True
    config.protein_vocab_size = protein_tokenizer.vocab_size
    
    # Initialize model
    logger.info("Initializing model architecture")
    model = SMILESGPTDecoder(config).to(device)
    model.set_tokenizer(smiles_tokenizer)
    
    # Load checkpoint weights
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model weights")
    
    model.eval()
    logger.info(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    return model, smiles_tokenizer, protein_tokenizer


def validate_protein_sequence(sequence: str) -> bool:
    """
    Validate that the protein sequence contains only standard amino acids.
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        True if valid, False otherwise
    """
    standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    sequence_upper = sequence.upper()
    
    invalid_chars = set(sequence_upper) - standard_amino_acids
    if invalid_chars:
        logger.error(f"Invalid amino acids found: {invalid_chars}")
        logger.error("Only standard 20 amino acids are supported: ACDEFGHIKLMNPQRSTVWY")
        return False
    
    if len(sequence) < 10:
        logger.warning(f"Protein sequence is very short ({len(sequence)} residues). "
                      "Typical binding pockets are 50-200 residues.")
    
    if len(sequence) > 1000:
        logger.warning(f"Protein sequence is very long ({len(sequence)} residues). "
                      "It will be truncated to max_seq_len during generation.")
    
    return True


def calculate_molecular_properties(smiles: str) -> Optional[Dict]:
    """
    Calculate basic molecular properties using RDKit.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary of molecular properties or None if invalid
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),  # Hydrogen bond donors
            'hba': Lipinski.NumHAcceptors(mol),  # Hydrogen bond acceptors
            'tpsa': Descriptors.TPSA(mol),  # Topological polar surface area
            'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'num_rings': Lipinski.RingCount(mol),
            'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'qed': QED.qed(mol),  # Quantitative Estimate of Drug-likeness
        }
        
        # Lipinski's Rule of Five compliance
        lipinski_violations = 0
        if properties['molecular_weight'] > 500:
            lipinski_violations += 1
        if properties['logp'] > 5:
            lipinski_violations += 1
        if properties['hbd'] > 5:
            lipinski_violations += 1
        if properties['hba'] > 10:
            lipinski_violations += 1
        
        properties['lipinski_violations'] = lipinski_violations
        properties['is_drug_like'] = lipinski_violations <= 1
        
        return properties
    
    except Exception as e:
        logger.debug(f"Error calculating properties for {smiles}: {e}")
        return None


def generate_molecules_for_protein(
    model: SMILESGPTDecoder,
    protein_sequence: str,
    smiles_tokenizer: SMILESTokenizer,
    protein_tokenizer: ProteinTokenizer,
    device: torch.device,
    num_samples: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_length: int = 256
) -> List[str]:
    """
    Generate molecules conditioned on a protein binding site sequence.
    
    Args:
        model: Trained protein-conditioned model
        protein_sequence: Amino acid sequence of binding pocket
        smiles_tokenizer: SMILES tokenizer
        protein_tokenizer: Protein tokenizer
        device: Computation device
        num_samples: Number of molecules to generate
        temperature: Sampling temperature (higher = more diverse)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        max_length: Maximum SMILES length
    
    Returns:
        List of generated SMILES strings
    """
    logger.info(f"Generating {num_samples} molecules for protein sequence (length: {len(protein_sequence)})")
    logger.info(f"Sampling parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    
    # Tokenize protein sequence
    protein_tokens = protein_tokenizer.encode(protein_sequence.upper(), add_special_tokens=True)
    protein_ids = torch.tensor([protein_tokens], dtype=torch.long, device=device)
    protein_mask = torch.ones_like(protein_ids)
    
    logger.info(f"Protein tokenized: {len(protein_tokens)} tokens")
    
    # Generate molecules
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids=None,  # Start from BOS token
            protein_ids=protein_ids,
            protein_mask=protein_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_samples
        )
    
    # Decode to SMILES
    generated_smiles = []
    for seq_ids in generated_ids:
        smiles = smiles_tokenizer.decode(seq_ids.tolist(), skip_special_tokens=True)
        generated_smiles.append(smiles)
    
    logger.info(f"Generated {len(generated_smiles)} SMILES strings")
    
    return generated_smiles


def validate_and_analyze_molecules(smiles_list: List[str]) -> Dict:
    """
    Validate generated molecules and calculate statistics.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        Dictionary with validation statistics and molecule data
    """
    results = {
        'total_generated': len(smiles_list),
        'valid_molecules': [],
        'invalid_smiles': [],
        'properties': [],
        'summary': {}
    }
    
    for i, smiles in enumerate(smiles_list):
        if not RDKIT_AVAILABLE:
            # Without RDKit, assume all are valid
            results['valid_molecules'].append(smiles)
            continue
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                results['valid_molecules'].append(smiles)
                
                # Calculate properties
                props = calculate_molecular_properties(smiles)
                if props:
                    props['smiles'] = smiles
                    props['index'] = i + 1
                    results['properties'].append(props)
            else:
                results['invalid_smiles'].append(smiles)
        except:
            results['invalid_smiles'].append(smiles)
    
    # Calculate summary statistics
    results['summary']['validity'] = len(results['valid_molecules']) / len(smiles_list) if smiles_list else 0
    results['summary']['num_valid'] = len(results['valid_molecules'])
    results['summary']['num_invalid'] = len(results['invalid_smiles'])
    results['summary']['uniqueness'] = len(set(results['valid_molecules'])) / len(results['valid_molecules']) if results['valid_molecules'] else 0
    results['summary']['num_unique'] = len(set(results['valid_molecules']))
    
    # Property statistics
    if results['properties']:
        results['summary']['avg_molecular_weight'] = sum(p['molecular_weight'] for p in results['properties']) / len(results['properties'])
        results['summary']['avg_logp'] = sum(p['logp'] for p in results['properties']) / len(results['properties'])
        results['summary']['avg_qed'] = sum(p['qed'] for p in results['properties']) / len(results['properties'])
        results['summary']['drug_like_percentage'] = sum(1 for p in results['properties'] if p['is_drug_like']) / len(results['properties'])
    
    return results


def save_results(results: Dict, output_path: str, protein_sequence: str):
    """
    Save generation results to file.
    
    Args:
        results: Results dictionary from validate_and_analyze_molecules
        output_path: Path to output file
        protein_sequence: Original protein sequence
    """
    output_path = Path(output_path)
    
    # Save as CSV if properties are available
    if results['properties']:
        df = pd.DataFrame(results['properties'])
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} molecules with properties to {csv_path}")
    
    # Save all SMILES (valid and invalid) to text file
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write(f"# Protein-Conditioned Molecular Generation Results\n")
        f.write(f"# Protein Sequence: {protein_sequence[:50]}{'...' if len(protein_sequence) > 50 else ''}\n")
        f.write(f"# Protein Length: {len(protein_sequence)} residues\n")
        f.write(f"# Generated: {results['total_generated']} molecules\n")
        f.write(f"# Valid: {results['summary']['num_valid']} ({results['summary']['validity']:.1%})\n")
        f.write(f"# Unique: {results['summary']['num_unique']} ({results['summary']['uniqueness']:.1%})\n")
        f.write(f"#\n")
        
        f.write(f"\n# Valid SMILES ({len(results['valid_molecules'])} molecules)\n")
        for i, smiles in enumerate(results['valid_molecules'], 1):
            f.write(f"{i}. {smiles}\n")
        
        if results['invalid_smiles']:
            f.write(f"\n# Invalid SMILES ({len(results['invalid_smiles'])} molecules)\n")
            for i, smiles in enumerate(results['invalid_smiles'], 1):
                f.write(f"{i}. {smiles}\n")
    
    logger.info(f"Saved all SMILES to {txt_path}")
    
    # Save summary JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            'protein_sequence': protein_sequence,
            'protein_length': len(protein_sequence),
            'summary': results['summary'],
            'valid_smiles': results['valid_molecules'],
            'invalid_smiles': results['invalid_smiles']
        }, f, indent=2)
    
    logger.info(f"Saved summary to {json_path}")


def main():
    """Main entry point for molecule generation."""
    parser = argparse.ArgumentParser(
        description="Generate molecules conditioned on protein binding sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 molecules for a protein sequence
  python generate_molecules.py \\
    --protein_sequence "MKTAYIAKQRQISFVKSHFSRQLE" \\
    --checkpoint checkpoints/run_XXX/best_model.pt \\
    --vocab checkpoints/vocab.json \\
    --num_samples 10

  # Generate with higher temperature for more diversity
  python generate_molecules.py \\
    --protein_sequence "GLITVQAPILSRVGDGTQDNLSG" \\
    --checkpoint checkpoints/run_XXX/best_model.pt \\
    --vocab checkpoints/vocab.json \\
    --num_samples 20 \\
    --temperature 1.2 \\
    --output results/protein1_molecules
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--protein_sequence',
        type=str,
        required=True,
        help='Amino acid sequence of the protein binding pocket (20 standard amino acids)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to SMILES vocabulary JSON file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of molecules to generate (default: 10)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature - higher values increase diversity (default: 1.0)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling parameter (default: 50, 0 to disable)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Nucleus sampling threshold (default: 0.95)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help='Maximum SMILES length (default: 256)'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='standard',
        choices=['small', 'standard', 'large'],
        help='Model size configuration (default: standard)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='generated_molecules',
        help='Output file path (without extension, default: generated_molecules)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for generation (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    logger.info("="*60)
    logger.info("Protein-Conditioned Molecular Generation")
    logger.info("="*60)
    
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.vocab).exists():
        logger.error(f"Vocabulary file not found: {args.vocab}")
        sys.exit(1)
    
    if not validate_protein_sequence(args.protein_sequence):
        sys.exit(1)
    
    logger.info(f"Protein sequence: {args.protein_sequence[:60]}{'...' if len(args.protein_sequence) > 60 else ''}")
    logger.info(f"Protein length: {len(args.protein_sequence)} residues")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model, smiles_tokenizer, protein_tokenizer = load_protein_conditioned_model(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            model_size=args.model_size,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate molecules
    logger.info("="*60)
    logger.info("Starting Generation")
    logger.info("="*60)
    
    try:
        generated_smiles = generate_molecules_for_protein(
            model=model,
            protein_sequence=args.protein_sequence,
            smiles_tokenizer=smiles_tokenizer,
            protein_tokenizer=protein_tokenizer,
            device=device,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=args.max_length
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate and analyze
    logger.info("="*60)
    logger.info("Validation and Analysis")
    logger.info("="*60)
    
    results = validate_and_analyze_molecules(generated_smiles)
    
    # Print summary
    logger.info(f"\nGeneration Summary:")
    logger.info(f"  Total generated: {results['summary']['num_valid'] + results['summary']['num_invalid']}")
    logger.info(f"  Valid molecules: {results['summary']['num_valid']} ({results['summary']['validity']:.1%})")
    logger.info(f"  Invalid SMILES: {results['summary']['num_invalid']}")
    logger.info(f"  Unique molecules: {results['summary']['num_unique']} ({results['summary']['uniqueness']:.1%})")
    
    if results['properties']:
        logger.info(f"\nMolecular Properties (average):")
        logger.info(f"  Molecular Weight: {results['summary']['avg_molecular_weight']:.1f} Da")
        logger.info(f"  LogP: {results['summary']['avg_logp']:.2f}")
        logger.info(f"  QED (drug-likeness): {results['summary']['avg_qed']:.3f}")
        logger.info(f"  Drug-like (Lipinski): {results['summary']['drug_like_percentage']:.1%}")
    
    # Show some examples
    logger.info(f"\nExample Generated Molecules:")
    for i, smiles in enumerate(results['valid_molecules'][:5], 1):
        logger.info(f"  {i}. {smiles}")
    
    if len(results['valid_molecules']) > 5:
        logger.info(f"  ... and {len(results['valid_molecules']) - 5} more")
    
    # Save results
    logger.info("="*60)
    logger.info("Saving Results")
    logger.info("="*60)
    
    save_results(results, args.output, args.protein_sequence)
    
    logger.info("="*60)
    logger.info("Generation Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

