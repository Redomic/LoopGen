"""
Pipeline for processing CrossDock2020 protein-ligand dataset.

This module provides functionality to parse extracted structure files from the CrossDock2020
dataset and generate CSV files containing protein-ligand pair information.
"""

import os
import sys
import logging
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
import time
import argparse
from collections import defaultdict

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import rdkit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem import rdMolTransforms
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Some molecular processing features will be limited.")

# Try to import pandas for data handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Using basic CSV handling.")


@dataclass
class DatasetStats:
    """Data structure for dataset statistics."""
    total_pairs: int = 0
    positive_pairs: int = 0
    negative_pairs: int = 0
    unique_pockets: set = field(default_factory=set)
    protein_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, label: int, pocket_id: str, protein_id: Optional[str]):
        """Update statistics with data from a new pair."""
        self.total_pairs += 1
        if label == 1:
            self.positive_pairs += 1
        else:
            self.negative_pairs += 1
        self.unique_pockets.add(pocket_id)
        if protein_id:
            self.protein_counts[protein_id] += 1

    def summary(self) -> str:
        """Generate a string summary of the statistics."""
        if self.total_pairs == 0:
            return "No data to generate statistics."
            
        summary_lines = [
            f"Total pairs: {self.total_pairs:,}",
            f"  - Positive examples (RMSD <= 2A): {self.positive_pairs:,} ({self.positive_pairs / self.total_pairs:.1%})",
            f"  - Negative examples (RMSD > 2A): {self.negative_pairs:,} ({self.negative_pairs / self.total_pairs:.1%})",
            f"Unique binding pockets: {len(self.unique_pockets):,}",
        ]
        
        most_common_proteins = sorted(self.protein_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        summary_lines.append("Most frequent proteins:")
        for protein, count in most_common_proteins:
            summary_lines.append(f"  - {protein}: {count:,} pairs")

        return "\n".join(summary_lines)


@dataclass
class ProteinLigandPair:
    """Data structure representing a protein-ligand pair."""
    pair_id: str
    protein_file: str
    ligand_file: str
    affinity: Optional[float] = None
    pocket_center: Optional[Tuple[float, float, float]] = None
    protein_sequence: Optional[str] = None
    ligand_molecular_weight: Optional[float] = None
    interaction_type: Optional[str] = None
    source_pdb: Optional[str] = None
    resolution: Optional[float] = None
    pocket_pdb_block: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV writing."""
        return {
            'pair_id': self.pair_id,
            'protein_file': self.protein_file,
            'ligand_file': self.ligand_file,
            'affinity': self.affinity or '',
            'pocket_center_x': self.pocket_center[0] if self.pocket_center else '',
            'pocket_center_y': self.pocket_center[1] if self.pocket_center else '',
            'pocket_center_z': self.pocket_center[2] if self.pocket_center else '',
            'protein_sequence': self.protein_sequence or '',
            'ligand_molecular_weight': self.ligand_molecular_weight or '',
            'interaction_type': self.interaction_type or '',
            'source_pdb': self.source_pdb or '',
            'resolution': self.resolution or '',
            'pocket_pdb_block': self.pocket_pdb_block or '',
        }


class CrossDockProcessingError(Exception):
    """Custom exception for CrossDock processing errors."""
    pass


class StructureFileLoader:
    """Loader for individual structure files from extracted CrossDocked2020 archive."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
    def load_molecule(self, file_path: Path) -> Optional[Chem.Mol]:
        """Load a molecule from a structure file (PDB, SDF, MOL2, etc.)."""
        if not RDKIT_AVAILABLE:
            logger.error("RDKit required for loading structure files")
            return None
            
        if not file_path.exists():
            if self.debug:
                logger.debug(f"File not found: {file_path}")
            return None
        
        try:
            # Determine file format and load appropriately
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdb':
                mol = Chem.MolFromPDBFile(str(file_path))
            elif suffix == '.sdf':
                mol = Chem.MolFromMolFile(str(file_path))
            elif suffix == '.mol':
                mol = Chem.MolFromMolFile(str(file_path))
            elif suffix == '.mol2':
                # RDKit doesn't have direct mol2 support, try as SDF
                mol = Chem.MolFromMolFile(str(file_path))
            else:
                # Try as PDB first, then SDF
                mol = Chem.MolFromPDBFile(str(file_path))
                if mol is None:
                    mol = Chem.MolFromMolFile(str(file_path))
            
            if mol is None and self.debug:
                logger.debug(f"Failed to parse molecule from {file_path}")
                
            return mol
            
        except Exception as e:
            if self.debug:
                logger.debug(f"Error loading {file_path}: {e}")
            return None


class TypesFileParser:
    """Parser for CrossDock2020 types files that specify protein-ligand pairs."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug

    def parse_types_file(self, types_dir: Path) -> Tuple[Dict[str, Dict], DatasetStats]:
        """
        Parse types files to extract protein-ligand pair metadata.
        
        Args:
            types_dir: Directory containing types files
            
        Returns:
            Tuple of (metadata_dict, dataset_stats)
        """
        metadata = {}
        stats = DatasetStats()
        
        # Find types files
        types_files = list(types_dir.glob("**/*.types"))
        if not types_files:
            logger.warning(f"No .types files found in {types_dir}")
            return metadata, stats
        
        logger.info(f"Found {len(types_files)} types files")
        
        for types_file in types_files:
            logger.info(f"Processing types file: {types_file.name}")
            
            try:
                with open(types_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        pair_data = self._parse_types_line(line)
                        if pair_data:
                            pair_id = f"{types_file.stem}_{line_num}"
                            metadata[pair_id] = pair_data
                            
                            # Update statistics
                            stats.update(
                                label=pair_data.get('label', 0),
                                pocket_id=pair_data.get('protein_file', ''),
                                protein_id=pair_data.get('protein_file', '')
                            )
                            
            except Exception as e:
                logger.error(f"Error processing {types_file}: {e}")
                
        logger.info(f"Parsed {len(metadata)} pairs from types files")
        return metadata, stats
    
    def _parse_types_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single line from a types file.
        
        Expected format: label weight affinity receptor_file ligand_file [additional_affinity]
        Example: 0 0.0000 7.6830 1a42_A_rec_1a42_260_lig_cg_CHEMBL1637013.gninatypes 1a42_A_rec_1a42_260_lig_CHEMBL1637013.gninatypes 7.6830
        """
        try:
            parts = line.split()
            if len(parts) < 5:
                if self.debug:
                    logger.debug(f"Skipping malformed line (too few parts): {line}")
                return None
            
            label = int(parts[0])
            weight = float(parts[1])
            affinity = float(parts[2])
            receptor_file = parts[3]
            ligand_file = parts[4]
            
            # Additional affinity value if present
            additional_affinity = None
            if len(parts) > 5:
                try:
                    additional_affinity = float(parts[5])
                except ValueError:
                    pass
            
            return {
                'label': label,
                'weight': weight,
                'affinity': affinity,
                'protein_file': receptor_file,
                'ligand_file': ligand_file,
                'additional_affinity': additional_affinity,
                'interaction_type': 'active' if label == 1 else 'inactive'
            }
            
        except (ValueError, IndexError) as e:
            if self.debug:
                logger.debug(f"Failed to parse line: {line} - Error: {e}")
            return None


class CrossDockDatasetProcessor:
    """Main processor for CrossDock2020 dataset."""
    
    def __init__(
        self,
        data_dir: str = "data",
        cache_subdir: str = "crossdocked",
        output_subdir: str = "output",
        debug: bool = False
    ):
        self.base_dir = Path(data_dir)
        self.cache_dir = self.base_dir / cache_subdir
        self.output_dir = self.base_dir / output_subdir
        self.debug = debug
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parsers
        self.structure_loader = StructureFileLoader(debug=debug)
        self.types_parser = TypesFileParser(debug=debug)
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check that required dependencies are available."""
        if not RDKIT_AVAILABLE:
            raise CrossDockProcessingError("RDKit is required for molecular processing. Install with: pip install rdkit")
    
    def _update_progress(
        self,
        current: int,
        total: int,
        start_time: float,
        prefix: str = 'Progress'
    ) -> None:
        """Display progress bar."""
        if total == 0:
            return
            
        elapsed = time.time() - start_time
        percent = 100 * (current / total)
        filled_length = int(50 * current // total)
        bar = '█' * filled_length + '-' * (50 - filled_length)
        
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        
        sys.stderr.write(f'\\r{prefix}: |{bar}| {percent:.1f}% {current:,}/{total:,} '
                        f'[{elapsed:.1f}s<{eta:.1f}s, {rate:.1f}it/s]')
        sys.stderr.flush()
    
    def _find_structure_files(self, extracted_dir: Path) -> Dict[str, Path]:
        """Find all structure files in the extracted directory and create a filename lookup."""
        structure_files = {}
        
        # Common structure file extensions
        extensions = ['.pdb', '.sdf', '.mol', '.mol2', '.gninatypes']
        
        for ext in extensions:
            for file_path in extracted_dir.rglob(f"*{ext}"):
                # Use filename as key (without extension for .gninatypes compatibility)
                if ext == '.gninatypes':
                    key = file_path.stem  # Remove .gninatypes extension
                else:
                    key = file_path.name
                structure_files[key] = file_path
                
        logger.info(f"Found {len(structure_files)} structure files")
        if self.debug:
            sample_files = list(structure_files.keys())[:10]
            logger.debug(f"Sample structure file names: {sample_files}")
            
        return structure_files
    
    def _define_pocket_from_ligand(self, protein_mol: Chem.Mol, ligand_mol: Chem.Mol, cutoff: float = 10.0) -> Optional[str]:
        """
        Define a binding pocket by identifying protein residues near a ligand.
        """
        try:
            if not protein_mol.GetNumConformers() or not ligand_mol.GetNumConformers():
                return None
                
            ligand_conformer = ligand_mol.GetConformer()
            protein_conformer = protein_mol.GetConformer()
            
            # Find all atoms in the protein that are within the cutoff distance of any ligand atom
            pocket_atom_indices = set()
            for lig_atom in ligand_mol.GetAtoms():
                lig_pos = ligand_conformer.GetAtomPosition(lig_atom.GetIdx())
                for prot_atom in protein_mol.GetAtoms():
                    prot_pos = protein_conformer.GetAtomPosition(prot_atom.GetIdx())
                    if lig_pos.Distance(prot_pos) <= cutoff:
                        pocket_atom_indices.add(prot_atom.GetIdx())
            
            if not pocket_atom_indices:
                return None
            
            # Generate a PDB block for the pocket
            pocket_pdb_block = Chem.MolToPDBBlock(protein_mol, atomIds=list(pocket_atom_indices))
            return pocket_pdb_block

        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to define pocket: {e}")
            return None

    def process_dataset(
        self,
        output_filename: str = "protein_ligand_pairs.csv",
        extract_types: bool = True,
        num_workers: int = 1
    ) -> str:
        """
        Process the CrossDock2020 dataset and generate CSV output.
        """
        logger.info("Starting CrossDock2020 dataset processing...")
        
        # Parse metadata from types file
        metadata, stats = {}, DatasetStats()
        if extract_types:
            types_dir = self.cache_dir / "types"
            logger.info("Parsing types file metadata...")
            
            metadata, stats = self.types_parser.parse_types_file(types_dir)
            logger.info(f"Found metadata for {len(metadata)} pairs")
            
            if len(metadata) == 0:
                logger.error("No metadata was parsed from types files!")
                # List all files in cache directory for debugging
                all_files = list(types_dir.glob("**/*"))
                logger.error(f"All files in cache directory: {[f.name for f in all_files if f.is_file()]}")
                
            logger.info("--- Dataset Statistics ---")
            logger.info(stats.summary())
            logger.info("--------------------------")

        # Find extracted structure files in crossdocked subdirectory
        crossdocked_dir = self.cache_dir / "crossdocked"
        if not crossdocked_dir.exists():
            raise CrossDockProcessingError(f"Crossdocked directory not found: {crossdocked_dir}")
        
        # Look for CrossDocked2020 directory within crossdocked folder
        extracted_dirs = list(crossdocked_dir.glob("CrossDocked2020*"))
        if not extracted_dirs:
            # If no subdirectory, use crossdocked folder directly
            structure_dir = crossdocked_dir
        else:
            structure_dir = extracted_dirs[0]  # Use the first found directory
        
        logger.info(f"Using structure directory: {structure_dir}")
        
        structure_files = self._find_structure_files(structure_dir)
        
        if not structure_files:
            raise CrossDockProcessingError(f"No structure files found in {structure_dir}")

        # Process protein-ligand pairs
        positive_pairs = []
        negative_pairs = []

        logger.info("Processing and filtering protein-ligand pairs...")
        total_pairs = len(metadata)
        start_time = time.time()

        match_stats = {'found': 0, 'missing_protein': 0, 'missing_ligand': 0, 'missing_both': 0}

        for i, (pair_id, pair_data) in enumerate(metadata.items()):
            # Get file paths from metadata
            protein_file_path = pair_data['protein_file']
            ligand_file_path = pair_data['ligand_file']
            
            # Try to find corresponding structure files
            protein_file = structure_files.get(protein_file_path) or structure_files.get(Path(protein_file_path).stem)
            ligand_file = structure_files.get(ligand_file_path) or structure_files.get(Path(ligand_file_path).stem)
            
            # Load molecules
            protein_mol = self.structure_loader.load_molecule(protein_file) if protein_file else None
            ligand_mol = self.structure_loader.load_molecule(ligand_file) if ligand_file else None

            # Update match statistics
            if protein_mol and ligand_mol:
                match_stats['found'] += 1
            elif not protein_mol and not ligand_mol:
                match_stats['missing_both'] += 1
            elif not protein_mol:
                match_stats['missing_protein'] += 1
            else:
                match_stats['missing_ligand'] += 1

            if not protein_mol or not ligand_mol:
                if self.debug and i < 5:  # Show first few failures
                    logger.debug(f"Could not find structures for pair {pair_id}")
                    logger.debug(f"  Protein file: {protein_file_path} -> {protein_file}")
                    logger.debug(f"  Ligand file: {ligand_file_path} -> {ligand_file}")
                continue

            # Create pair object
            pair = ProteinLigandPair(
                pair_id=pair_id,
                protein_file=str(protein_file) if protein_file else protein_file_path,
                ligand_file=str(ligand_file) if ligand_file else ligand_file_path,
                affinity=pair_data.get('affinity'),
                interaction_type=pair_data.get('interaction_type'),
            )

            # Calculate additional properties if possible
            try:
                if ligand_mol:
                    pair.ligand_molecular_weight = Descriptors.MolWt(ligand_mol)
                
                # Define pocket if both molecules are available
                if protein_mol and ligand_mol:
                    pair.pocket_pdb_block = self._define_pocket_from_ligand(protein_mol, ligand_mol)
                    
            except Exception as e:
                if self.debug:
                    logger.debug(f"Error calculating properties for pair {pair_id}: {e}")

            # Categorize as positive or negative
            if pair_data.get('label', 0) == 1:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)

            # Update progress
            if (i + 1) % 1000 == 0 or i == total_pairs - 1:
                self._update_progress(i + 1, total_pairs, start_time, "Processing pairs")

        self._update_progress(total_pairs, total_pairs, start_time, "Processing pairs")
        sys.stderr.write('\\n')

        # Log matching statistics
        logger.info("Molecule matching statistics:")
        logger.info(f"  Successfully matched: {match_stats['found']:,}")
        logger.info(f"  Missing proteins: {match_stats['missing_protein']:,}")
        logger.info(f"  Missing ligands: {match_stats['missing_ligand']:,}")
        logger.info(f"  Missing both: {match_stats['missing_both']:,}")

        # Write CSV outputs
        positive_output_path = self.output_dir / "positive_pairs.csv"
        negative_output_path = self.output_dir / "negative_pairs.csv"

        self._write_csv_output(positive_pairs, positive_output_path)
        self._write_csv_output(negative_pairs, negative_output_path)

        logger.info(f"Generated {len(positive_pairs):,} positive pairs and {len(negative_pairs):,} negative pairs.")
        logger.info(f"Output saved to: {self.output_dir}")
        
        return str(self.output_dir)

    def _write_csv_output(self, pairs: List[ProteinLigandPair], output_path: Path) -> None:
        """Write protein-ligand pairs to CSV file."""
        if not pairs:
            logger.warning(f"No pairs to write to {output_path}")
            return
            
        logger.info(f"Writing {len(pairs):,} pairs to {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=pairs[0].to_dict().keys())
            writer.writeheader()
            for pair in pairs:
                writer.writerow(pair.to_dict())

    def cleanup_cache(self) -> None:
        """Clean up downloaded cache files to save disk space."""
        logger.info("Cleaning up cache directory...")
        
        # Remove large archive files but keep extracted content
        for pattern in ["*.tgz", "*.tar.gz", "*.zip"]:
            for file_path in self.cache_dir.glob(pattern):
                try:
                    file_path.unlink()
                    logger.info(f"Removed: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

    def get_cache_size(self) -> int:
        """Get total size of cache directory in bytes."""
        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size


def main():
    """Main entry point for the CrossDock2020 processing pipeline."""
    parser = argparse.ArgumentParser(description="Process CrossDock2020 protein-ligand dataset")
    parser.add_argument("--output", "-o", default="protein_ligand_pairs.csv",
                        help="Output CSV filename")
    parser.add_argument("--data-dir", "-d", default="data",
                        help="Data directory containing cache and output subdirectories")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of worker processes")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up cache files after processing")
    parser.add_argument("--no-extract-types", action="store_true",
                        help="Skip extracting and processing types file")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        processor = CrossDockDatasetProcessor(
            data_dir=args.data_dir,
            debug=args.debug
        )
        
        output_dir = processor.process_dataset(
            output_filename=args.output,
            extract_types=not args.no_extract_types,
            num_workers=args.workers
        )
        
        if args.cleanup:
            processor.cleanup_cache()
        
        logger.info(f"Processing complete! Output files are in: {output_dir}")
        
        # Print cache size info
        cache_size_gb = processor.get_cache_size() / (1024**3)
        logger.info(f"Cache directory size: {cache_size_gb:.2f} GB")
        
    except CrossDockProcessingError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 