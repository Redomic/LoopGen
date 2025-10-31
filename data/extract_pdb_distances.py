"""
Extract C-alpha atom coordinates and compute distance matrices from PDB files.

This module provides utilities for extracting C-alpha coordinates from protein
PDB structures and computing pairwise distance matrices for topological analysis.
Used for binding site topology encoding in molecular generation.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Import BioPython for PDB parsing
try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Polypeptide import is_aa
    BIOPYTHON_AVAILABLE = True
except ImportError:
    logger.warning("BioPython not available - install with: pip install biopython")
    BIOPYTHON_AVAILABLE = False

# Import RDKit for ligand handling
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available - install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False


def extract_c_alpha_coordinates(pdb_file: str) -> Optional[np.ndarray]:
    """
    Extract C-alpha atom coordinates from a protein PDB file.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Array of C-alpha coordinates [n_residues, 3] or None if extraction fails
    """
    if not BIOPYTHON_AVAILABLE:
        logger.error("BioPython is required for PDB parsing")
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        c_alpha_coords = []
        residue_info = []  # Store (chain_id, res_num, res_name) for tracking
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Only process standard amino acids
                    if not is_aa(residue, standard=True):
                        continue
                    
                    # Extract C-alpha atom
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        coord = ca_atom.get_coord()
                        c_alpha_coords.append(coord)
                        residue_info.append((
                            chain.id,
                            residue.id[1],  # Residue number
                            residue.resname
                        ))
        
        if not c_alpha_coords:
            logger.debug(f"No C-alpha atoms found in {pdb_file}")
            return None
        
        coords_array = np.array(c_alpha_coords, dtype=np.float32)
        logger.debug(f"Extracted {len(c_alpha_coords)} C-alpha atoms from {pdb_file}")
        
        return coords_array
    
    except Exception as e:
        logger.debug(f"Error extracting C-alpha coordinates from {pdb_file}: {e}")
        return None


def extract_pocket_c_alpha_coordinates(
    protein_pdb_file: str,
    ligand_pdb_file: str,
    cutoff: float = 10.0
) -> Optional[Tuple[np.ndarray, List[Tuple]]]:
    """
    Extract C-alpha coordinates for pocket residues near the ligand.
    
    Args:
        protein_pdb_file: Path to protein PDB file
        ligand_pdb_file: Path to ligand PDB file
        cutoff: Distance cutoff in Angstroms for pocket definition
        
    Returns:
        Tuple of (coordinates array [n_pocket_residues, 3], residue_info list)
        or None if extraction fails
    """
    if not BIOPYTHON_AVAILABLE or not RDKIT_AVAILABLE:
        logger.error("BioPython and RDKit are required")
        return None
    
    try:
        # Parse protein structure
        parser = PDBParser(QUIET=True)
        protein_structure = parser.get_structure('protein', protein_pdb_file)
        
        # Get ligand coordinates
        ligand_mol = Chem.MolFromPDBFile(ligand_pdb_file)
        if ligand_mol is None:
            ligand_mol = Chem.MolFromMolFile(ligand_pdb_file)
        
        if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
            logger.debug(f"Could not load ligand from {ligand_pdb_file}")
            return None
        
        # Extract ligand atom positions
        ligand_conf = ligand_mol.GetConformer()
        ligand_coords = np.array([
            [ligand_conf.GetAtomPosition(i).x,
             ligand_conf.GetAtomPosition(i).y,
             ligand_conf.GetAtomPosition(i).z]
            for i in range(ligand_mol.GetNumAtoms())
        ], dtype=np.float32)
        
        # Find pocket residues (C-alpha within cutoff of any ligand atom)
        pocket_residues = []
        
        for model in protein_structure:
            for chain in model:
                for residue in chain:
                    # Only standard amino acids
                    if not is_aa(residue, standard=True):
                        continue
                    
                    # Check C-alpha distance to ligand
                    if 'CA' not in residue:
                        continue
                    
                    ca_coord = residue['CA'].get_coord()
                    
                    # Compute minimum distance to any ligand atom
                    distances = np.linalg.norm(ligand_coords - ca_coord, axis=1)
                    min_distance = np.min(distances)
                    
                    if min_distance <= cutoff:
                        pocket_residues.append({
                            'coord': ca_coord,
                            'chain': chain.id,
                            'res_num': residue.id[1],
                            'res_name': residue.resname
                        })
        
        if not pocket_residues:
            logger.debug(f"No pocket residues found within {cutoff}Å")
            return None
        
        # Sort by chain and residue number for consistent ordering
        pocket_residues.sort(key=lambda x: (x['chain'], x['res_num']))
        
        # Extract coordinates and info
        coords = np.array([res['coord'] for res in pocket_residues], dtype=np.float32)
        residue_info = [(res['chain'], res['res_num'], res['res_name']) 
                       for res in pocket_residues]
        
        logger.debug(f"Extracted {len(pocket_residues)} pocket residues")
        
        return coords, residue_info
    
    except Exception as e:
        logger.debug(f"Error extracting pocket coordinates: {e}")
        return None


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix from 3D coordinates.
    
    Args:
        coords: Array of 3D coordinates [n_points, 3]
        
    Returns:
        Symmetric distance matrix [n_points, n_points]
    """
    # Efficient vectorized computation
    # dist(i,j) = sqrt(sum((coords[i] - coords[j])^2))
    n = coords.shape[0]
    
    # Expand dims for broadcasting: [n, 1, 3] - [1, n, 3] = [n, n, 3]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    # Compute Euclidean distances
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return dist_matrix.astype(np.float32)


def save_distance_matrix(distance_matrix: np.ndarray, output_path: str):
    """
    Save distance matrix to disk in compressed format.
    
    Args:
        distance_matrix: Distance matrix to save
        output_path: Path to save the matrix (.npz format)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, distance_matrix=distance_matrix)
    logger.debug(f"Saved distance matrix to {output_path}")


def load_distance_matrix(input_path: str) -> Optional[np.ndarray]:
    """
    Load distance matrix from disk.
    
    Args:
        input_path: Path to distance matrix file (.npz format)
        
    Returns:
        Distance matrix or None if loading fails
    """
    try:
        data = np.load(input_path)
        return data['distance_matrix']
    except Exception as e:
        logger.debug(f"Error loading distance matrix from {input_path}: {e}")
        return None


def process_pdb_to_distance_matrix(
    protein_pdb_file: str,
    ligand_pdb_file: Optional[str] = None,
    cutoff: float = 10.0,
    output_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    End-to-end pipeline: PDB file -> C-alpha coords -> distance matrix.
    
    Args:
        protein_pdb_file: Path to protein PDB file
        ligand_pdb_file: Optional path to ligand file (for pocket extraction)
        cutoff: Distance cutoff for pocket definition (if ligand provided)
        output_path: Optional path to save the distance matrix
        
    Returns:
        Distance matrix or None if processing fails
    """
    # Extract coordinates
    if ligand_pdb_file:
        result = extract_pocket_c_alpha_coordinates(
            protein_pdb_file, 
            ligand_pdb_file, 
            cutoff
        )
        if result is None:
            return None
        coords, residue_info = result
    else:
        coords = extract_c_alpha_coordinates(protein_pdb_file)
        if coords is None:
            return None
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(coords)
    
    # Save if output path provided
    if output_path:
        save_distance_matrix(distance_matrix, output_path)
    
    return distance_matrix


def batch_process_pdb_files(
    pdb_pairs: List[Tuple[str, str]],
    output_dir: str,
    cutoff: float = 10.0,
    overwrite: bool = False
) -> Dict[str, bool]:
    """
    Process multiple protein-ligand pairs in batch.
    
    Args:
        pdb_pairs: List of (protein_pdb_path, ligand_pdb_path) tuples
        output_dir: Directory to save distance matrices
        cutoff: Distance cutoff for pocket definition
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary mapping pair_id to success status
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, (protein_file, ligand_file) in enumerate(pdb_pairs):
        # Generate output filename based on input files
        protein_stem = Path(protein_file).stem
        ligand_stem = Path(ligand_file).stem
        pair_id = f"{protein_stem}_{ligand_stem}"
        output_path = os.path.join(output_dir, f"{pair_id}.npz")
        
        # Skip if already exists and not overwriting
        if os.path.exists(output_path) and not overwrite:
            logger.debug(f"Skipping {pair_id} - already exists")
            results[pair_id] = True
            skipped += 1
            continue
        
        # Process the pair
        distance_matrix = process_pdb_to_distance_matrix(
            protein_file,
            ligand_file,
            cutoff,
            output_path
        )
        
        if distance_matrix is not None:
            results[pair_id] = True
            successful += 1
        else:
            results[pair_id] = False
            failed += 1
        
        # Log progress
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(pdb_pairs)} pairs "
                       f"(success: {successful}, failed: {failed}, skipped: {skipped})")
    
    logger.info(f"\nBatch processing complete:")
    logger.info(f"  Total: {len(pdb_pairs)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped: {skipped}")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract C-alpha coordinates and compute distance matrices from PDB files"
    )
    parser.add_argument("--protein", type=str, help="Path to protein PDB file")
    parser.add_argument("--ligand", type=str, help="Path to ligand PDB file (optional)")
    parser.add_argument("--output", type=str, help="Output path for distance matrix")
    parser.add_argument("--cutoff", type=float, default=10.0, 
                       help="Distance cutoff for pocket (Angstroms)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if not args.protein:
        print("Please provide --protein argument")
        exit(1)
    
    # Process PDB file
    distance_matrix = process_pdb_to_distance_matrix(
        args.protein,
        args.ligand,
        args.cutoff,
        args.output
    )
    
    if distance_matrix is not None:
        print(f"Successfully computed distance matrix:")
        print(f"  Shape: {distance_matrix.shape}")
        print(f"  Min distance: {np.min(distance_matrix[distance_matrix > 0]):.2f} Å")
        print(f"  Max distance: {np.max(distance_matrix):.2f} Å")
        print(f"  Mean distance: {np.mean(distance_matrix):.2f} Å")
    else:
        print("Failed to compute distance matrix")
        exit(1)


