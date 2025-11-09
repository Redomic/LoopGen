#!/usr/bin/env python3
"""
Dummy Molecular Generator for Demonstration

Generates random drug-like molecules with realistic properties.
Mimics the real generator's output format without requiring trained models.

Usage:
    python generate_molecule.py \
        --protein_sequence "MKTAYIAKQRQISFVKSHFSRQLE" \
        --num_samples 10
"""

import argparse
import random
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# Setup logging - will be configured in main() with file output
logger = logging.getLogger(__name__)

# Try to import RDKit for validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Using mock property calculations.")
    RDKIT_AVAILABLE = False


# Drug-like SMILES templates (all valid, drug-like molecules)
DRUG_TEMPLATES = [
    # Simple drug-like molecules
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen-like
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like
    "c1ccc2c(c1)ccc1c2cccc1",  # Anthracene (aromatic)
    "CC(C)NCC(COc1ccccc1)O",  # Beta-blocker-like
    "Cc1ccc(cc1)S(=O)(=O)N",  # Sulfonamide-like
    "c1ccc(cc1)C(=O)c2ccccc2",  # Benzophenone-like
    "CCN(CC)C(=O)c1ccccc1",  # Amide-like
    "COc1ccc(cc1)CCN",  # Phenethylamine-like
    "c1ccc2c(c1)nc(s2)N",  # Benzothiazole-like
    
    # More complex drug-like scaffolds
    "CC(C)Cc1ccc(cc1)C(C)CC(=O)O",
    "CN1CCN(CC1)c2ccc(cc2)OC",
    "Cc1ccc(cc1)c2cc(nn2c3ccccc3)C(F)(F)F",
    "COc1ccc(cc1)C(=O)NCc2ccccc2",
    "c1ccc2c(c1)c(c(n2)C)CCN(C)C",
    "CC(C)NCC(c1ccc(cc1)O)O",
    "Cc1ccc(cc1)C(=O)Nc2ccccc2",
    "CCOc1ccc(cc1)C(=O)N2CCCC2",
    "c1ccc(cc1)CNC(=O)c2ccccc2Cl",
    "COc1ccccc1CCN(C)C",
    
    # Heterocycles
    "c1ccc2c(c1)nc(o2)N",
    "Cc1nc(cs1)Nc2ccccc2",
    "c1ccc(nc1)c2ccccn2",
    "Cc1cc(no1)c2ccccc2",
    "c1cnc2c(c1)cccn2",
    
    # With stereochemistry
    "C[C@H](Cc1ccccc1)NC",
    "C[C@@H](O)c1ccccc1",
    "C[C@H]1CCCO1",
    "CC(=O)N[C@@H](Cc1ccccc1)C(=O)O",
    "O[C@H]1CCNC1",
]


def generate_random_smiles(num_samples: int, seed: Optional[int] = None) -> List[str]:
    """
    Generate random drug-like SMILES strings.
    
    Args:
        num_samples: Number of molecules to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of SMILES strings
    """
    if seed is not None:
        random.seed(seed)
    
    molecules = []
    for _ in range(num_samples):
        # Pick a random template and optionally add variations
        base_smiles = random.choice(DRUG_TEMPLATES)
        
        # Occasionally add small modifications
        if random.random() < 0.3:
            # Add a methyl group or similar simple modification
            modifications = ["C", "CC", "F", "Cl"]
            # This is simplified - real modifications would need careful SMILES editing
            base_smiles = base_smiles  # Keep as is for validity
        
        molecules.append(base_smiles)
    
    return molecules


def calculate_molecular_properties(smiles: str) -> Optional[Dict]:
    """
    Calculate basic molecular properties using RDKit.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary of molecular properties or mock values if RDKit unavailable
    """
    if not RDKIT_AVAILABLE:
        # Return mock properties
        return {
            'valid': True,
            'molecular_weight': round(random.uniform(200, 450), 1),
            'logp': round(random.uniform(1.0, 4.5), 2),
            'hbd': random.randint(1, 3),
            'hba': random.randint(2, 6),
            'tpsa': round(random.uniform(40, 120), 1),
            'num_rotatable_bonds': random.randint(2, 8),
            'num_rings': random.randint(1, 4),
            'num_aromatic_rings': random.randint(1, 3),
            'qed': round(random.uniform(0.45, 0.85), 3),
            'lipinski_violations': random.randint(0, 1),
            'is_drug_like': True
        }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {
            'valid': True,
            'molecular_weight': round(Descriptors.MolWt(mol), 1),
            'logp': round(Crippen.MolLogP(mol), 2),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'tpsa': round(Descriptors.TPSA(mol), 1),
            'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'num_rings': Lipinski.RingCount(mol),
            'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'qed': round(QED.qed(mol), 3),
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


def print_molecule_properties(smiles: str, idx: int, props: Dict):
    """Print formatted molecule properties."""
    print(f"\n{'='*70}")
    print(f"Molecule {idx + 1}")
    print(f"{'='*70}")
    print(f"SMILES: {smiles}")
    
    if props:
        print(f"\nMolecular Properties:")
        print(f"  Molecular Weight: {props['molecular_weight']:.1f} Da")
        print(f"  LogP: {props['logp']:.2f}")
        print(f"  Hydrogen Bond Donors: {props['hbd']}")
        print(f"  Hydrogen Bond Acceptors: {props['hba']}")
        print(f"  TPSA: {props['tpsa']:.1f} Ų")
        print(f"  Rotatable Bonds: {props['num_rotatable_bonds']}")
        print(f"  Aromatic Rings: {props['num_aromatic_rings']}")
        print(f"  QED (Drug-likeness): {props['qed']:.3f}")
        print(f"\n  Lipinski Rule of Five:")
        print(f"    Violations: {props['lipinski_violations']}")
        print(f"    Drug-like: {'Yes' if props['is_drug_like'] else 'No'}")
    else:
        print("  [Invalid molecule - could not calculate properties]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate drug-like molecules (dummy version for demonstration)"
    )
    
    parser.add_argument(
        "--protein_sequence",
        type=str,
        default="MKTAYIAKQRQISFVKSHFSRQLE",
        help="Protein sequence (for demonstration purposes, not actually used)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of molecules to generate (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save generated SMILES (optional)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path (default: generation_YYYYMMDD_HHMMSS.log)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed properties for each molecule"
    )
    
    args = parser.parse_args()
    
    # Setup logging with file output
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"generation_{timestamp}.log"
    
    # Create log directory if needed
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Print header
    logger.info("="*70)
    logger.info("Protein-Conditioned Molecular Generation (Demo)")
    logger.info("="*70)
    logger.info(f"Protein sequence: {args.protein_sequence[:30]}{'...' if len(args.protein_sequence) > 30 else ''}")
    logger.info(f"Number of samples: {args.num_samples}")
    if args.seed is not None:
        logger.info(f"Random seed: {args.seed}")
    logger.info("")
    
    # Generate molecules
    logger.info(f"Generating {args.num_samples} drug-like molecules...")
    molecules = generate_random_smiles(args.num_samples, seed=args.seed)
    
    # Calculate properties and display
    all_properties = []
    valid_count = 0
    qed_scores = []
    
    for idx, smiles in enumerate(molecules):
        props = calculate_molecular_properties(smiles)
        all_properties.append(props)
        
        if props and props.get('valid', False):
            valid_count += 1
            qed_scores.append(props['qed'])
        
        if args.verbose:
            print_molecule_properties(smiles, idx, props)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Generation Summary")
    print(f"{'='*70}")
    print(f"Total molecules generated: {len(molecules)}")
    print(f"Valid molecules: {valid_count}/{len(molecules)} ({100*valid_count/len(molecules):.1f}%)")
    
    if qed_scores:
        print(f"Average QED: {sum(qed_scores)/len(qed_scores):.3f}")
        print(f"QED range: [{min(qed_scores):.3f}, {max(qed_scores):.3f}]")
    
    # Count drug-like molecules
    drug_like_count = sum(1 for p in all_properties if p and p.get('is_drug_like', False))
    print(f"Drug-like molecules (Lipinski): {drug_like_count}/{valid_count} ({100*drug_like_count/valid_count:.1f}%)")
    
    # Print molecules list
    print(f"\n{'='*70}")
    print("Generated Molecules (SMILES)")
    print(f"{'='*70}")
    for idx, smiles in enumerate(molecules):
        qed = all_properties[idx]['qed'] if all_properties[idx] else 0.0
        print(f"{idx+1:2d}. {smiles:50s} (QED: {qed:.3f})")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            for idx, smiles in enumerate(molecules):
                f.write(f"{idx+1}: {smiles}\n")
        logger.info(f"\nSaved molecules to: {args.output}")
    
    print(f"\n{'='*70}")
    logger.info("Generation complete!")
    logger.info(f"Log file saved to: {args.log_file}")


if __name__ == "__main__":
    main()

