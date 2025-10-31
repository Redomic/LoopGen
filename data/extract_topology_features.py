#!/usr/bin/env python3
"""
Standalone script to extract topology features (distance matrices) from CrossDock PDB files.

This script processes protein-ligand pairs from CrossDock2020 dataset and extracts
C-alpha distance matrices for topological analysis using persistent homology.

Usage:
    python extract_topology_features.py --input crossdock_dir --output topology_dir
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import distance extraction module
try:
    from extract_pdb_distances import batch_process_pdb_files
except ImportError:
    logger.error("Could not import extract_pdb_distances module")
    logger.error("Make sure extract_pdb_distances.py is in the same directory")
    sys.exit(1)


def collect_pdb_pairs_from_directory(
    crossdock_dir: str,
    max_pairs: int = None
) -> List[Tuple[str, str]]:
    """
    Collect protein-ligand PDB file pairs from CrossDock directory structure.
    
    Args:
        crossdock_dir: Root directory of CrossDock dataset
        max_pairs: Maximum number of pairs to collect
        
    Returns:
        List of (protein_pdb_path, ligand_pdb_path) tuples
    """
    logger.info(f"Scanning CrossDock directory: {crossdock_dir}")
    
    pairs = []
    crossdock_path = Path(crossdock_dir)
    
    # CrossDock structure: crossdock_dir/pocket_name/receptor.pdb, ligand.pdb
    for pocket_dir in crossdock_path.iterdir():
        if not pocket_dir.is_dir():
            continue
        
        # Look for receptor and ligand files
        receptor_files = list(pocket_dir.glob("*_rec.pdb"))
        ligand_files = list(pocket_dir.glob("*_lig.pdb"))
        
        # Match receptor-ligand pairs
        for receptor in receptor_files:
            for ligand in ligand_files:
                # Simple matching: same prefix
                if receptor.stem.replace("_rec", "") == ligand.stem.replace("_lig", ""):
                    pairs.append((str(receptor), str(ligand)))
        
        # Check max pairs limit
        if max_pairs and len(pairs) >= max_pairs:
            pairs = pairs[:max_pairs]
            break
    
    logger.info(f"Found {len(pairs)} protein-ligand pairs")
    return pairs


def collect_pdb_pairs_from_csv(
    csv_file: str,
    max_pairs: int = None
) -> List[Tuple[str, str]]:
    """
    Collect protein-ligand PDB file pairs from a CSV file.
    
    Expected CSV format:
        protein_file,ligand_file
        /path/to/protein1.pdb,/path/to/ligand1.pdb
        /path/to/protein2.pdb,/path/to/ligand2.pdb
    
    Args:
        csv_file: Path to CSV file with protein-ligand pairs
        max_pairs: Maximum number of pairs to collect
        
    Returns:
        List of (protein_pdb_path, ligand_pdb_path) tuples
    """
    logger.info(f"Loading pairs from CSV: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'protein_file' not in df.columns or 'ligand_file' not in df.columns:
            logger.error("CSV must have 'protein_file' and 'ligand_file' columns")
            return []
        
        pairs = list(zip(df['protein_file'], df['ligand_file']))
        
        # Limit pairs if specified
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        logger.info(f"Loaded {len(pairs)} protein-ligand pairs")
        return pairs
    
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []


def main():
    """Main entry point for topology feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract topology features (distance matrices) from CrossDock PDB files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to CrossDock directory"
    )
    input_group.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with protein-ligand pairs"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for distance matrices"
    )
    
    # Processing options
    parser.add_argument(
        "--cutoff", "-c",
        type=float,
        default=10.0,
        help="Distance cutoff for pocket definition (Angstroms)"
    )
    parser.add_argument(
        "--max-pairs", "-n",
        type=int,
        default=None,
        help="Maximum number of pairs to process"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing distance matrices"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect protein-ligand pairs
    if args.input:
        pairs = collect_pdb_pairs_from_directory(args.input, args.max_pairs)
    else:
        pairs = collect_pdb_pairs_from_csv(args.csv, args.max_pairs)
    
    if not pairs:
        logger.error("No protein-ligand pairs found")
        sys.exit(1)
    
    logger.info(f"\n{'='*70}")
    logger.info("EXTRACTING TOPOLOGY FEATURES (DISTANCE MATRICES)")
    logger.info(f"{'='*70}")
    logger.info(f"Total pairs: {len(pairs)}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Cutoff distance: {args.cutoff}Å")
    logger.info(f"Overwrite existing: {args.overwrite}")
    
    # Process pairs in batch
    results = batch_process_pdb_files(
        pdb_pairs=pairs,
        output_dir=args.output,
        cutoff=args.cutoff,
        overwrite=args.overwrite
    )
    
    # Summary
    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    logger.info(f"\n{'='*70}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(results)*100:.1f}%")
    logger.info(f"\nDistance matrices saved to: {args.output}")


if __name__ == "__main__":
    main()


