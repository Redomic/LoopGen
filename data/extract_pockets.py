"""
Extract binding pocket sequences from CrossDock protein-ligand structures.

This script processes CrossDock2020 dataset to extract amino acid sequences
from protein binding pockets (residues within a cutoff distance of the ligand).
Generates a CSV file with SMILES, pocket sequences, and binding affinity data.
"""

import os
import sys
import logging
import csv
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Set
from collections import defaultdict
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import molecular processing libraries
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available - install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

# Try BioPython for PDB parsing (better for proteins)
try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.Polypeptide import three_to_one, is_aa
    BIOPYTHON_AVAILABLE = True
except ImportError:
    logger.warning("BioPython not available - install with: pip install biopython")
    BIOPYTHON_AVAILABLE = False


def extract_pocket_sequence_biopython(
    protein_pdb_path: str, 
    ligand_pdb_path: str, 
    cutoff: float = 10.0
) -> Optional[str]:
    """
    Extract protein pocket sequence using BioPython (preferred method).
    
    Args:
        protein_pdb_path: Path to protein PDB file
        ligand_pdb_path: Path to ligand PDB/SDF file
        cutoff: Distance cutoff in Angstroms for pocket definition
    
    Returns:
        String of amino acid letters in sequence order, or None if extraction fails
    """
    if not BIOPYTHON_AVAILABLE or not RDKIT_AVAILABLE:
        return None
    
    try:
        # Parse protein structure
        parser = PDBParser(QUIET=True)
        protein_structure = parser.get_structure('protein', protein_pdb_path)
        
        # Get ligand atoms and their coordinates
        ligand_mol = Chem.MolFromPDBFile(ligand_pdb_path)
        if ligand_mol is None:
            # Try SDF format
            ligand_mol = Chem.MolFromMolFile(ligand_pdb_path)
        
        if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
            return None
        
        ligand_conf = ligand_mol.GetConformer()
        ligand_coords = [
            ligand_conf.GetAtomPosition(i) 
            for i in range(ligand_mol.GetNumAtoms())
        ]
        
        # Find pocket residues (within cutoff of any ligand atom)
        pocket_residues = set()
        
        for model in protein_structure:
            for chain in model:
                for residue in chain:
                    # Skip non-amino acids (water, ligands, etc.)
                    if not is_aa(residue, standard=True):
                        continue
                    
                    # Check if any atom in residue is within cutoff
                    for atom in residue:
                        atom_coord = atom.get_coord()
                        
                        for lig_pos in ligand_coords:
                            distance = ((atom_coord[0] - lig_pos.x)**2 + 
                                       (atom_coord[1] - lig_pos.y)**2 + 
                                       (atom_coord[2] - lig_pos.z)**2) ** 0.5
                            
                            if distance <= cutoff:
                                pocket_residues.add((
                                    chain.id,
                                    residue.id[1],  # Residue number
                                    residue.resname
                                ))
                                break
        
        if not pocket_residues:
            return None
        
        # Sort residues by chain and sequence number
        sorted_residues = sorted(pocket_residues, key=lambda x: (x[0], x[1]))
        
        # Convert to one-letter amino acid codes
        sequence = []
        for chain_id, res_num, res_name in sorted_residues:
            try:
                one_letter = three_to_one(res_name)
                sequence.append(one_letter)
            except KeyError:
                # Non-standard residue - skip or use 'X'
                logger.debug(f"Non-standard residue: {res_name}")
                continue
        
        return ''.join(sequence)
    
    except Exception as e:
        logger.debug(f"Error extracting pocket from {protein_pdb_path}: {e}")
        return None


def extract_pocket_sequence_rdkit(
    protein_pdb_path: str, 
    ligand_pdb_path: str, 
    cutoff: float = 10.0
) -> Optional[str]:
    """
    Extract protein pocket sequence using RDKit (fallback method).
    
    Args:
        protein_pdb_path: Path to protein PDB file
        ligand_pdb_path: Path to ligand PDB/SDF file
        cutoff: Distance cutoff in Angstroms
    
    Returns:
        Pocket sequence string or None
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        # Load protein
        protein_mol = Chem.MolFromPDBFile(protein_pdb_path, removeHs=False)
        if protein_mol is None:
            return None
        
        # Load ligand
        ligand_mol = Chem.MolFromPDBFile(ligand_pdb_path)
        if ligand_mol is None:
            ligand_mol = Chem.MolFromMolFile(ligand_pdb_path)
        
        if ligand_mol is None:
            return None
        
        # Get conformers
        if protein_mol.GetNumConformers() == 0 or ligand_mol.GetNumConformers() == 0:
            return None
        
        protein_conf = protein_mol.GetConformer()
        ligand_conf = ligand_mol.GetConformer()
        
        # Find pocket atoms (protein atoms near ligand)
        pocket_atom_indices = set()
        
        for lig_idx in range(ligand_mol.GetNumAtoms()):
            lig_pos = ligand_conf.GetAtomPosition(lig_idx)
            
            for prot_idx in range(protein_mol.GetNumAtoms()):
                prot_pos = protein_conf.GetAtomPosition(prot_idx)
                distance = lig_pos.Distance(prot_pos)
                
                if distance <= cutoff:
                    pocket_atom_indices.add(prot_idx)
        
        if not pocket_atom_indices:
            return None
        
        # Extract residue information from pocket atoms
        # This is a simplified approach - BioPython is better for this
        pocket_residues = set()
        for atom_idx in pocket_atom_indices:
            atom = protein_mol.GetAtomWithIdx(atom_idx)
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info:
                res_name = pdb_info.GetResidueName().strip()
                res_num = pdb_info.GetResidueNumber()
                chain_id = pdb_info.GetChainId()
                pocket_residues.add((chain_id, res_num, res_name))
        
        # Sort and convert to sequence
        sorted_residues = sorted(pocket_residues, key=lambda x: (x[0], x[1]))
        
        # Map 3-letter to 1-letter codes
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = []
        for chain_id, res_num, res_name in sorted_residues:
            if res_name in aa_map:
                sequence.append(aa_map[res_name])
        
        return ''.join(sequence)
    
    except Exception as e:
        logger.debug(f"Error extracting pocket from {protein_pdb_path}: {e}")
        return None


def extract_pocket_sequence(
    protein_pdb_path: str, 
    ligand_pdb_path: str, 
    cutoff: float = 10.0
) -> Optional[str]:
    """
    Extract binding pocket sequence from protein-ligand complex.
    
    Tries BioPython first (more accurate), falls back to RDKit.
    
    Args:
        protein_pdb_path: Path to protein structure file
        ligand_pdb_path: Path to ligand structure file
        cutoff: Distance cutoff in Angstroms for pocket definition
    
    Returns:
        Amino acid sequence of pocket residues, or None if extraction fails
    """
    # Try BioPython method first
    if BIOPYTHON_AVAILABLE:
        sequence = extract_pocket_sequence_biopython(protein_pdb_path, ligand_pdb_path, cutoff)
        if sequence:
            return sequence
    
    # Fall back to RDKit method
    if RDKIT_AVAILABLE:
        sequence = extract_pocket_sequence_rdkit(protein_pdb_path, ligand_pdb_path, cutoff)
        if sequence:
            return sequence
    
    return None


def extract_smiles_from_ligand(ligand_path: str) -> Optional[str]:
    """Extract SMILES string from ligand structure file."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        # Try PDB format
        mol = Chem.MolFromPDBFile(ligand_path)
        if mol is None:
            # Try SDF format
            mol = Chem.MolFromMolFile(ligand_path)
        if mol is None:
            # Try MOL2 format
            mol = Chem.MolFromMol2File(ligand_path)
        
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            return smiles
        
        return None
    
    except Exception as e:
        logger.debug(f"Error extracting SMILES from {ligand_path}: {e}")
        return None


def process_crossdock_for_pockets(
    crossdock_csv: str = "data/output/positive_pairs.csv",
    output_csv: str = "data/output/protein_ligand_training.csv",
    max_pairs: Optional[int] = None,
    cutoff: float = 10.0,
    min_pocket_length: int = 10,
    max_pocket_length: int = 500
):
    """
    Process CrossDock dataset to extract protein-ligand training data.
    
    Creates a CSV file with columns:
        - SMILES: Ligand molecular structure
        - pocket_sequence: Amino acid sequence of binding pocket
        - affinity: Binding affinity (if available)
        - pair_id: Unique identifier
    
    Args:
        crossdock_csv: Input CSV with protein-ligand pair information
        output_csv: Output CSV path
        max_pairs: Maximum number of pairs to process (None = all)
        cutoff: Distance cutoff for pocket definition (Angstroms)
        min_pocket_length: Minimum pocket sequence length
        max_pocket_length: Maximum pocket sequence length
    """
    logger.info("Starting CrossDock pocket extraction...")
    logger.info(f"Input: {crossdock_csv}")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Cutoff distance: {cutoff}Å")
    
    # Check dependencies
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required. Install with: conda install -c conda-forge rdkit")
        return
    
    if not BIOPYTHON_AVAILABLE:
        logger.warning("BioPython not available. Using RDKit only (less accurate).")
        logger.warning("Install with: pip install biopython")
    
    # Read input CSV
    try:
        df = pd.read_csv(crossdock_csv)
        logger.info(f"Loaded {len(df)} pairs from input file")
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        return
    
    # Limit number of pairs if specified
    if max_pairs:
        df = df.head(max_pairs)
        logger.info(f"Processing first {max_pairs} pairs")
    
    # Process each pair
    results = []
    successful = 0
    failed_pocket = 0
    failed_smiles = 0
    invalid_length = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} pairs "
                       f"(success: {successful}, failed: {failed_pocket + failed_smiles})")
        
        protein_file = row.get('protein_file', '')
        ligand_file = row.get('ligand_file', '')
        pair_id = row.get('pair_id', f'pair_{idx}')
        affinity = row.get('affinity', None)
        
        # Check files exist
        if not os.path.exists(protein_file) or not os.path.exists(ligand_file):
            logger.debug(f"Files not found for pair {pair_id}")
            failed_pocket += 1
            continue
        
        # Extract pocket sequence
        pocket_seq = extract_pocket_sequence(protein_file, ligand_file, cutoff)
        if not pocket_seq:
            logger.debug(f"Failed to extract pocket for pair {pair_id}")
            failed_pocket += 1
            continue
        
        # Check pocket length
        if len(pocket_seq) < min_pocket_length or len(pocket_seq) > max_pocket_length:
            logger.debug(f"Pocket length {len(pocket_seq)} out of range for pair {pair_id}")
            invalid_length += 1
            continue
        
        # Extract SMILES
        smiles = extract_smiles_from_ligand(ligand_file)
        if not smiles:
            logger.debug(f"Failed to extract SMILES for pair {pair_id}")
            failed_smiles += 1
            continue
        
        # Add to results
        results.append({
            'pair_id': pair_id,
            'SMILES': smiles,
            'pocket_sequence': pocket_seq,
            'affinity': affinity if pd.notna(affinity) else '',
            'pocket_length': len(pocket_seq)
        })
        successful += 1
    
    # Save results
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        logger.info(f"\nSuccessfully processed {successful} pairs")
        logger.info(f"Output saved to: {output_csv}")
    else:
        logger.error("No pairs were successfully processed!")
    
    # Summary statistics
    logger.info("\n--- Processing Summary ---")
    logger.info(f"Total pairs processed: {len(df)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed (pocket extraction): {failed_pocket}")
    logger.info(f"Failed (SMILES extraction): {failed_smiles}")
    logger.info(f"Failed (invalid length): {invalid_length}")
    logger.info(f"Success rate: {successful/len(df)*100:.1f}%")
    
    if results:
        pocket_lengths = [r['pocket_length'] for r in results]
        logger.info(f"\nPocket length statistics:")
        logger.info(f"  Mean: {sum(pocket_lengths)/len(pocket_lengths):.1f}")
        logger.info(f"  Min: {min(pocket_lengths)}, Max: {max(pocket_lengths)}")


def main():
    """Main entry point for pocket extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract binding pocket sequences from CrossDock dataset"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/output/positive_pairs.csv",
        help="Input CSV file with protein-ligand pairs"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/output/protein_ligand_training.csv",
        help="Output CSV file with SMILES and pocket sequences"
    )
    parser.add_argument(
        "--max-pairs", "-n",
        type=int,
        default=None,
        help="Maximum number of pairs to process"
    )
    parser.add_argument(
        "--cutoff", "-c",
        type=float,
        default=10.0,
        help="Distance cutoff for pocket definition (Angstroms)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum pocket sequence length"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=500,
        help="Maximum pocket sequence length"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    process_crossdock_for_pockets(
        crossdock_csv=args.input,
        output_csv=args.output,
        max_pairs=args.max_pairs,
        cutoff=args.cutoff,
        min_pocket_length=args.min_length,
        max_pocket_length=args.max_length
    )


if __name__ == "__main__":
    main()



