#!/usr/bin/env python3
"""
Evaluation script for protein-conditioned molecular generation.

This script evaluates the quality and specificity of molecules generated
when conditioned on specific protein pocket sequences.
"""

import argparse
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from collections import defaultdict

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
    from rdkit.Chem import Descriptors, AllChem
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Chemical validity checks disabled.")
    RDKIT_AVAILABLE = False


def load_model(
    checkpoint_path: str,
    vocab_path: str,
    model_size: str = 'standard',
    device: torch.device = None
):
    """Load trained model and tokenizers."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizers
    logger.info(f"Loading SMILES tokenizer from {vocab_path}")
    smiles_tokenizer = SMILESTokenizer(vocab_path=vocab_path)
    
    logger.info("Initializing protein tokenizer")
    protein_tokenizer = ProteinTokenizer()
    
    # Create config
    logger.info(f"Creating {model_size} model config")
    if model_size == 'small':
        config = ModelConfig.small_config()
    elif model_size == 'large':
        config = ModelConfig.large_config()
    else:
        config = ModelConfig.standard_config()
    
    config.vocab_size = smiles_tokenizer.vocab_size
    config.use_protein_conditioning = True
    config.protein_vocab_size = protein_tokenizer.vocab_size
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = SMILESGPTDecoder(config).to(device)
    model.set_tokenizer(smiles_tokenizer)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, smiles_tokenizer, protein_tokenizer


def generate_for_protein(
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
    Generate molecules conditioned on a specific protein pocket sequence.
    
    Args:
        model: Trained protein-conditioned model
        protein_sequence: Amino acid sequence of binding pocket
        smiles_tokenizer: SMILES tokenizer
        protein_tokenizer: Protein tokenizer
        device: Device for computation
        num_samples: Number of molecules to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        max_length: Maximum SMILES length
    
    Returns:
        List of generated SMILES strings
    """
    # Tokenize protein sequence
    protein_tokens = protein_tokenizer.encode(protein_sequence, add_special_tokens=True)
    protein_ids = torch.tensor([protein_tokens], dtype=torch.long, device=device)
    protein_mask = torch.ones_like(protein_ids)
    
    # Generate molecules
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids=None,
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
    
    return generated_smiles


def evaluate_binding_specificity(
    model: SMILESGPTDecoder,
    test_data_path: str,
    smiles_tokenizer: SMILESTokenizer,
    protein_tokenizer: ProteinTokenizer,
    device: torch.device,
    num_proteins: int = 10,
    samples_per_protein: int = 10
) -> Dict:
    """
    Evaluate if generated molecules are specific to target proteins.
    
    Metrics:
    - Validity of generated SMILES
    - Uniqueness within each protein
    - Diversity (chemical fingerprint similarity)
    - Inter-protein diversity (are molecules for different proteins different?)
    
    Args:
        model: Trained model
        test_data_path: Path to test CSV with protein-ligand pairs
        smiles_tokenizer: SMILES tokenizer
        protein_tokenizer: Protein tokenizer
        device: Computation device
        num_proteins: Number of proteins to test
        samples_per_protein: Molecules to generate per protein
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating binding specificity on {num_proteins} proteins")
    
    # Load test data
    try:
        df = pd.read_csv(test_data_path)
        logger.info(f"Loaded {len(df)} test pairs")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return {}
    
    # Sample unique proteins
    unique_proteins = df['pocket_sequence'].unique()[:num_proteins]
    logger.info(f"Testing on {len(unique_proteins)} unique protein pockets")
    
    results = {
        'protein_results': [],
        'overall_validity': 0.0,
        'overall_uniqueness': 0.0,
        'inter_protein_diversity': 0.0
    }
    
    all_molecules = []
    protein_molecules = defaultdict(list)
    
    for i, protein_seq in enumerate(unique_proteins):
        logger.info(f"Generating for protein {i+1}/{len(unique_proteins)} (length: {len(protein_seq)})")
        
        # Generate molecules
        generated = generate_for_protein(
            model=model,
            protein_sequence=protein_seq,
            smiles_tokenizer=smiles_tokenizer,
            protein_tokenizer=protein_tokenizer,
            device=device,
            num_samples=samples_per_protein
        )
        
        # Evaluate validity
        valid_smiles = []
        if RDKIT_AVAILABLE:
            for smi in generated:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        valid_smiles.append(smi)
                except:
                    pass
        else:
            valid_smiles = generated  # Assume all valid if no RDKit
        
        validity = len(valid_smiles) / len(generated) if generated else 0.0
        uniqueness = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
        
        protein_results = {
            'protein_length': len(protein_seq),
            'generated': len(generated),
            'valid': len(valid_smiles),
            'validity': validity,
            'unique': len(set(valid_smiles)),
            'uniqueness': uniqueness,
            'examples': valid_smiles[:3]
        }
        
        results['protein_results'].append(protein_results)
        all_molecules.extend(valid_smiles)
        protein_molecules[i] = valid_smiles
        
        logger.info(f"  Validity: {validity:.2%}, Uniqueness: {uniqueness:.2%}")
    
    # Overall statistics
    if results['protein_results']:
        results['overall_validity'] = sum(r['validity'] for r in results['protein_results']) / len(results['protein_results'])
        results['overall_uniqueness'] = sum(r['uniqueness'] for r in results['protein_results']) / len(results['protein_results'])
    
    # Calculate inter-protein diversity (how different are molecules for different proteins?)
    if RDKIT_AVAILABLE and len(protein_molecules) > 1:
        similarities = []
        protein_ids = list(protein_molecules.keys())
        
        for i in range(len(protein_ids)):
            for j in range(i+1, len(protein_ids)):
                mols_i = protein_molecules[protein_ids[i]][:5]  # Sample
                mols_j = protein_molecules[protein_ids[j]][:5]
                
                # Calculate average Tanimoto similarity
                for smi_i in mols_i:
                    for smi_j in mols_j:
                        try:
                            mol_i = Chem.MolFromSmiles(smi_i)
                            mol_j = Chem.MolFromSmiles(smi_j)
                            if mol_i and mol_j:
                                fp_i = AllChem.GetMorganFingerprint(mol_i, 2)
                                fp_j = AllChem.GetMorganFingerprint(mol_j, 2)
                                sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                                similarities.append(sim)
                        except:
                            pass
        
        if similarities:
            results['inter_protein_diversity'] = 1.0 - (sum(similarities) / len(similarities))
            logger.info(f"Inter-protein diversity: {results['inter_protein_diversity']:.3f}")
    
    return results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate protein-conditioned molecular generation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to SMILES vocabulary")
    parser.add_argument("--test_data", required=True, help="Path to test CSV with protein-ligand pairs")
    parser.add_argument("--model_size", default="standard", choices=["small", "standard", "large"])
    parser.add_argument("--num_proteins", type=int, default=10, help="Number of proteins to test")
    parser.add_argument("--samples_per_protein", type=int, default=10, help="Molecules per protein")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, smiles_tokenizer, protein_tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        model_size=args.model_size,
        device=device
    )
    
    # Run evaluation
    results = evaluate_binding_specificity(
        model=model,
        test_data_path=args.test_data,
        smiles_tokenizer=smiles_tokenizer,
        protein_tokenizer=protein_tokenizer,
        device=device,
        num_proteins=args.num_proteins,
        samples_per_protein=args.samples_per_protein
    )
    
    # Print summary
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Overall Validity: {results.get('overall_validity', 0):.2%}")
    logger.info(f"Overall Uniqueness: {results.get('overall_uniqueness', 0):.2%}")
    logger.info(f"Inter-protein Diversity: {results.get('inter_protein_diversity', 0):.3f}")
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()



