#!/usr/bin/env python3
"""
Clean SMILES dataset by removing invalid, reaction, metal-containing, and highly charged molecules.

Filters for drug-like molecules suitable for molecular generation training.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Tuple, Optional
import argparse

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("ERROR: RDKit is required for data cleaning. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


# Allowed atoms for drug-like molecules (organic + common halogens)
DRUG_LIKE_ATOMS = {
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    'B',  # boron compounds are becoming more common in drugs
}

# Metals and exotic atoms to exclude
EXCLUDE_ATOMS = {
    'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Cu', 'Zn', 'Co', 'Ni', 'Mn',
    'Ag', 'Au', 'Pt', 'Pd', 'Ru', 'Rh', 'Os', 'Ir',
    'Al', 'Si', 'Ga', 'Ge', 'As', 'Se', 'Sn', 'Sb', 'Te',
    'Ti', 'V', 'Cr', 'Mo', 'W', 'Nb', 'Ta',
    'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Th', 'U',
}


def is_valid_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Check if SMILES is valid and suitable for training.
    
    Returns:
        (is_valid, reason_if_invalid)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "empty_or_invalid_type"
    
    smiles = smiles.strip()
    
    # Check for reaction SMILES (contains '>>')
    if '>' in smiles:
        return False, "reaction_smiles"
    
    # Check for multi-component (disconnected) molecules with '.'
    # Allow single '.' for now but count fragments
    dot_count = smiles.count('.')
    if dot_count > 2:  # More than 2 fragments is likely salts/complexes
        return False, "too_many_fragments"
    
    # Try to parse with RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "rdkit_parse_failed"
    except Exception as e:
        return False, f"rdkit_exception"
    
    # Check atom types
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Check for metals and exotic atoms
    has_excluded = any(atom in EXCLUDE_ATOMS for atom in atoms)
    if has_excluded:
        excluded_found = [atom for atom in atoms if atom in EXCLUDE_ATOMS]
        return False, f"contains_metals_or_exotic: {','.join(set(excluded_found))}"
    
    # Check for non-drug-like atoms (excluding common organics)
    non_druglike = [atom for atom in atoms if atom not in DRUG_LIKE_ATOMS]
    if non_druglike:
        return False, f"non_druglike_atoms: {','.join(set(non_druglike))}"
    
    # Check total formal charge (exclude highly charged species)
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if abs(total_charge) > 2:  # Allow +/-1, +/-2 only
        return False, f"high_charge: {total_charge}"
    
    # Check number of charged atoms (avoid multiple charges)
    num_charged = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
    if num_charged > 3:
        return False, "too_many_charged_atoms"
    
    # Check molecular size (avoid very small or very large)
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 3:
        return False, "too_small"
    if num_atoms > 100:
        return False, "too_large"
    
    # Check for valid valence
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, "sanitization_failed"
    
    return True, None


def clean_smiles_file(
    input_path: str,
    output_path: str,
    smiles_column: int = 0,
    has_header: bool = False,
    max_molecules: Optional[int] = None,
    verbose: bool = True
):
    """
    Clean a SMILES dataset file.
    
    Args:
        input_path: Path to input CSV/TXT file
        output_path: Path to output cleaned file
        smiles_column: Column index containing SMILES (0-indexed)
        has_header: Whether file has header row
        max_molecules: Maximum number of molecules to process (None = all)
        verbose: Print progress
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Cleaning SMILES from: {input_path}")
        print(f"Output will be saved to: {output_path}")
    
    stats = {
        'total': 0,
        'valid': 0,
        'reasons': {}
    }
    
    # Open output file
    output_file = open(output_path, 'w')
    
    try:
        # Process in chunks for memory efficiency
        chunk_size = 10000
        header_written = False
        
        for chunk_idx, chunk in enumerate(pd.read_csv(
            input_path,
            chunksize=chunk_size,
            header=0 if has_header else None,
            on_bad_lines='skip'
        )):
            if verbose and chunk_idx % 10 == 0:
                print(f"Processing chunk {chunk_idx} ({stats['total']} processed, {stats['valid']} valid)...")
            
            # Write header if present and first chunk
            if has_header and not header_written:
                output_file.write(','.join(chunk.columns) + '\n')
                header_written = True
            
            for idx, row in chunk.iterrows():
                stats['total'] += 1
                
                # Get SMILES from appropriate column
                if isinstance(smiles_column, int):
                    smiles = row.iloc[smiles_column]
                else:
                    smiles = row[smiles_column]
                
                # Validate
                is_valid, reason = is_valid_smiles(smiles)
                
                if is_valid:
                    stats['valid'] += 1
                    # Write entire row to output
                    output_file.write(','.join(str(v) for v in row.values) + '\n')
                else:
                    # Track rejection reasons
                    stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
                
                # Stop if reached max
                if max_molecules and stats['total'] >= max_molecules:
                    break
            
            if max_molecules and stats['total'] >= max_molecules:
                break
    
    finally:
        output_file.close()
    
    if verbose:
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"Total molecules processed: {stats['total']}")
        print(f"Valid molecules: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
        print(f"Rejected molecules: {stats['total'] - stats['valid']} ({(stats['total']-stats['valid'])/stats['total']*100:.1f}%)")
        print("\nRejection reasons:")
        for reason, count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({count/stats['total']*100:.1f}%)")
        print("="*60)
        print(f"Cleaned data saved to: {output_path}")
    
    return stats


def clean_protein_ligand_file(
    input_path: str,
    output_path: str,
    verbose: bool = True
):
    """
    Clean protein-ligand CSV file (SMILES in first column).
    
    Expected format: smiles,protein_sequence,binding_affinity,identifier
    """
    if verbose:
        print("Cleaning protein-ligand dataset (SMILES in column 0)...")
    
    return clean_smiles_file(
        input_path=input_path,
        output_path=output_path,
        smiles_column=0,
        has_header=False,
        verbose=verbose
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean SMILES dataset for molecular generation training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV/TXT file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output cleaned file path"
    )
    parser.add_argument(
        "--smiles-column",
        type=int,
        default=0,
        help="Column index containing SMILES (default: 0)"
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Input file has header row"
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to process (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    clean_smiles_file(
        input_path=args.input,
        output_path=args.output,
        smiles_column=args.smiles_column,
        has_header=args.has_header,
        max_molecules=args.max_molecules,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

