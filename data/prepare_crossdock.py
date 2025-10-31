#!/usr/bin/env python3
"""
Pipeline for processing CrossDock2020 protein-ligand dataset from PDB files.

This module reads CrossDock PDB structures directly from the tar archive without
full extraction, generates CSV with protein pocket sequences, ligand SMILES, and affinities.
"""

import os
import sys
import logging
import csv
import tarfile
import pickle
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import time
import argparse
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import rdkit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.error("RDKit is required. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


class CrossDockProcessingError(Exception):
    """Custom exception for CrossDock processing errors."""
    pass


def parse_affinity_from_types_archive(
    types_archive_path: str, 
    target_pairs: List[Tuple[str, str]] = None,
    debug: bool = False, 
    cache_dir: str = None
) -> Dict[str, float]:
    """
    Parse binding affinity data directly from types tar archive without extracting.
    
    OPTIMIZED: If target_pairs provided, only extracts affinities for those specific pairs.
    This reduces memory usage from 44M entries to ~100K entries.
    
    Types file format:
        label weight affinity receptor_file ligand_file [additional_affinity]
        
    Example:
        0 0.0000 7.6830 1a42_A_rec_1a42_260_lig_cg_CHEMBL1637013.gninatypes 1a42_A_rec_1a42_260_lig_CHEMBL1637013.gninatypes 7.6830
    
    Args:
        types_archive_path: Path to types.tgz archive
        target_pairs: Optional list of (protein_path, ligand_path) tuples to filter for
        debug: Enable debug logging
        cache_dir: Directory to store cache file (default: same as archive)
        
    Returns:
        Dictionary mapping (receptor_file, ligand_file) tuple to affinity value
    """
    affinity_map = {}
    
    if not os.path.exists(types_archive_path):
        logger.warning(f"Types archive not found at {types_archive_path}. Will use default affinity values.")
        return affinity_map
    
    # Build target lookup set if filtering for specific pairs
    target_lookup = None
    target_pockets = None
    if target_pairs:
        logger.info(f"Building lookup set for {len(target_pairs)} target pairs...")
        target_lookup = set()
        target_pockets = set()
        
        for protein_path, ligand_path in target_pairs:
            # Extract pocket name (parent directory)
            pocket_name = Path(protein_path).parent.name
            target_pockets.add(pocket_name)
            
            # Extract base filenames for matching
            p_name = Path(protein_path).name  # e.g., 1i9y_A_rec.pdb
            l_name = Path(ligand_path).name   # e.g., 1i9z_2ip_lig.pdb
            p_stem = Path(protein_path).stem  # e.g., 1i9y_A_rec
            l_stem = Path(ligand_path).stem   # e.g., 1i9z_2ip_lig
            
            # Build keys with pocket name (types file format: POCKET/file.gninatypes)
            target_lookup.add((pocket_name, p_stem))  # Match on pocket + receptor
            target_lookup.add((pocket_name, l_stem))  # Match on pocket + ligand base
            
            # Also add just the receptor/ligand stems for broader matching
            target_lookup.add(p_stem)
            target_lookup.add(l_stem)
        
        logger.info(f"✓ Lookup set built: {len(target_pockets)} pockets, {len(target_lookup)} key variations")
        logger.info(f"OPTIMIZED MODE: Only parsing affinities for target pairs (saves memory!)")
    else:
        logger.warning("No target pairs provided - will parse ALL affinities (may use lots of memory)")
    
    # Get file size for progress tracking
    archive_size_mb = os.path.getsize(types_archive_path) / (1024 * 1024)
    logger.info(f"Parsing affinity data from {types_archive_path} ({archive_size_mb:.1f} MB)")
    if target_lookup:
        logger.info("Fast parsing mode - should complete in 1-2 minutes...")
    else:
        logger.info("Full parsing mode - will take ~5-10 minutes...")
    
    start_time = time.time()
    
    try:
        with tarfile.open(types_archive_path, 'r:gz') as tar:
            types_files = [m for m in tar.getmembers() if m.name.endswith('.types')]
            logger.info(f"Found {len(types_files)} types files in archive")
            
            entries_parsed = 0
            for file_idx, member in enumerate(types_files):
                if (file_idx + 1) % 5 == 0 or file_idx == 0:
                    elapsed = time.time() - start_time
                    rate = (file_idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(types_files) - file_idx - 1) / rate if rate > 0 else 0
                    mem_mb = len(affinity_map) * 0.0001  # Rough estimate
                    logger.info(f"Processing types file {file_idx + 1}/{len(types_files)} "
                               f"({entries_parsed:,} entries, ~{mem_mb:.0f}MB dict, "
                               f"{rate:.1f} files/s, ETA: {eta:.0f}s)")
                
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                # Stream line-by-line instead of loading all content
                for line in f:
                    line = line.decode('utf-8', errors='ignore').strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        
                        # CrossDock types format can vary:
                        # Format 1: label weight affinity receptor_file ligand_file [optional]
                        # Format 2: label affinity receptor_file ligand_file
                        # We need to detect which format by checking if parts[2] is a file path
                        
                        affinity = None
                        receptor_file = None
                        ligand_file = None
                        
                        if len(parts) >= 5:
                            # Try format 1: label weight affinity receptor ligand
                            # Check if parts[3] looks like a file path (contains '/' or ends with extension)
                            if '/' in parts[3] or parts[3].endswith('.gninatypes'):
                                # Format 2: label affinity receptor ligand (no weight column)
                                affinity = float(parts[1])
                                receptor_file = parts[2]
                                ligand_file = parts[3]
                            else:
                                # Format 1: label weight affinity receptor ligand
                                affinity = float(parts[2])
                                receptor_file = parts[3]
                                ligand_file = parts[4]
                        elif len(parts) == 4:
                            # Format 2: label affinity receptor ligand
                            affinity = float(parts[1])
                            receptor_file = parts[2]
                            ligand_file = parts[3]
                        else:
                            continue
                        
                        if affinity is None or receptor_file is None or ligand_file is None:
                            continue
                        
                        # Strip .gninatypes extension and create key
                        receptor_base = receptor_file.replace('.gninatypes', '').replace('_rec_', '_rec.')
                        ligand_base = ligand_file.replace('.gninatypes', '').replace('_lig_', '_lig.')
                        
                        # If filtering, check if this pair is in our target set
                        if target_lookup:
                            # Extract pocket name from types file path (e.g., "POCKET/file.gninatypes")
                            rec_pocket = receptor_file.split('/')[0] if '/' in receptor_file else None
                            
                            # Check if this pocket is in our target set
                            if rec_pocket and rec_pocket not in target_pockets:
                                continue  # Not a target pocket, skip
                            
                            # Extract base file names (without extensions)
                            rec_base_clean = receptor_base.split('/')[-1]  # Get filename only
                            lig_base_clean = ligand_base.split('/')[-1]
                            
                            # Check if receptor or ligand stem matches (types files have variations like _0, _docked_0, etc.)
                            # Match on the base pattern before variants
                            is_target = False
                            for lookup_key in target_lookup:
                                if isinstance(lookup_key, str):
                                    # Simple stem match
                                    if rec_base_clean.startswith(lookup_key) or lig_base_clean.startswith(lookup_key):
                                        is_target = True
                                        break
                                elif isinstance(lookup_key, tuple) and len(lookup_key) == 2:
                                    # (pocket, stem) match
                                    pocket, stem = lookup_key
                                    if rec_pocket == pocket and (rec_base_clean.startswith(stem) or lig_base_clean.startswith(stem)):
                                        is_target = True
                                        break
                            
                            if not is_target:
                                continue  # Skip this entry, not in our target pairs
                        
                        # Store key formats for matching
                        affinity_map[(receptor_base, ligand_base)] = affinity
                        
                        # Store filename-only version for matching
                        rec_name = receptor_file.split('/')[-1] if '/' in receptor_file else receptor_file
                        lig_name = ligand_file.split('/')[-1] if '/' in ligand_file else ligand_file
                        if (rec_name, lig_name) != (receptor_base, ligand_base):
                            affinity_map[(rec_name, lig_name)] = affinity
                        
                        entries_parsed += 1
                        
                    except (ValueError, IndexError) as e:
                        if debug:
                            logger.debug(f"Failed to parse line: {line[:100]}... - Error: {e}")
                        continue
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Parsed affinity data for {len(affinity_map):,} pair combinations in {elapsed:.1f}s")
        logger.info(f"  Total entries scanned: {entries_parsed:,}")
        logger.info(f"  Affinity map size: {len(affinity_map):,} entries")
        
        # Only cache if we parsed all affinities (not filtered)
        if not target_lookup:
            try:
                logger.info(f"Saving full affinity cache to {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(affinity_map, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("✓ Cache saved successfully - future runs will be instant!")
            except Exception as cache_err:
                logger.warning(f"Failed to save cache (non-fatal): {cache_err}")
        else:
            logger.info("Skipping cache (filtered mode - cache not useful for different pair sets)")
        
    except Exception as e:
        logger.error(f"Error parsing types archive: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    return affinity_map


def find_affinity_for_pair(protein_file: str, ligand_file: str, affinity_map: Dict) -> Optional[float]:
    """
    Find affinity value for a protein-ligand pair.
    
    Args:
        protein_file: Path to protein file (can be archive member name)
        ligand_file: Path to ligand file (can be archive member name)
        affinity_map: Dictionary from parse_affinity_from_types_archive
        
    Returns:
        Affinity value or None if not found
    """
    # Extract basenames for matching
    protein_name = Path(protein_file).name
    ligand_name = Path(ligand_file).name
    protein_base = Path(protein_file).stem
    ligand_base = Path(ligand_file).stem
    
    # Try different key formats
    possible_keys = [
        # Full paths
        (protein_file, ligand_file),
        # Just filenames
        (protein_name, ligand_name),
        # Basenames (no extension)
        (protein_base, ligand_base),
        # With .pdb extension
        (protein_base + '.pdb', ligand_base + '.pdb'),
        # With .gninatypes extension (types file format)
        (protein_base + '.gninatypes', ligand_base + '.gninatypes'),
        # Try variations with _rec_ format
        (protein_base.replace('_rec', '_rec_'), ligand_base.replace('_lig', '_lig_')),
        # Try removing crossdocked/ prefix if present
        (protein_file.replace('crossdocked/', ''), ligand_file.replace('crossdocked/', '')),
    ]
    
    for key in possible_keys:
        if key in affinity_map:
            return affinity_map[key]
    
    return None


def extract_pocket_sequence(protein_file: str, ligand_file: str, cutoff: float = 10.0) -> Optional[str]:
    """
    Extract protein pocket sequence (amino acids within cutoff distance of ligand).
    
    Args:
        protein_file: Path to protein PDB file
        ligand_file: Path to ligand PDB/SDF file
        cutoff: Distance cutoff in Angstroms
        
    Returns:
        Pocket sequence as a string of amino acids, or None if extraction fails
    """
    try:
        # Load molecules
        protein_mol = Chem.MolFromPDBFile(protein_file, sanitize=False)
        ligand_mol = Chem.MolFromPDBFile(ligand_file, sanitize=False)
        
        if not protein_mol or not ligand_mol:
            return None
            
        if protein_mol.GetNumConformers() == 0 or ligand_mol.GetNumConformers() == 0:
            return None
        
        protein_conf = protein_mol.GetConformer()
        ligand_conf = ligand_mol.GetConformer()
        
        # Find protein atoms near ligand
        pocket_residues = set()
        
        for lig_atom_idx in range(ligand_mol.GetNumAtoms()):
            lig_pos = ligand_conf.GetAtomPosition(lig_atom_idx)
            
            for prot_atom_idx in range(protein_mol.GetNumAtoms()):
                prot_pos = protein_conf.GetAtomPosition(prot_atom_idx)
                distance = lig_pos.Distance(prot_pos)
                
                if distance <= cutoff:
                    # Get residue info
                    atom = protein_mol.GetAtomWithIdx(prot_atom_idx)
                    residue_info = atom.GetPDBResidueInfo()
                    if residue_info:
                        res_name = residue_info.GetResidueName().strip()
                        res_num = residue_info.GetResidueNumber()
                        chain = residue_info.GetChainId()
                        pocket_residues.add((chain, res_num, res_name))
        
        if not pocket_residues:
            return None

        # Sort by chain and residue number
        sorted_residues = sorted(pocket_residues, key=lambda x: (x[0], x[1]))
        
        # Convert 3-letter codes to 1-letter codes
        three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        sequence = []
        for _, _, res_name in sorted_residues:
            one_letter = three_to_one.get(res_name, 'X')
            sequence.append(one_letter)
        
        return ''.join(sequence)
        
    except Exception as e:
        logger.debug(f"Failed to extract pocket from {protein_file}: {e}")
        return None


def extract_smiles_from_ligand(ligand_file: str) -> Optional[str]:
    """
    Extract SMILES string from ligand file.
    
    Args:
        ligand_file: Path to ligand file
        
    Returns:
        SMILES string or None if extraction fails
    """
    try:
        # Try loading as PDB first
        mol = Chem.MolFromPDBFile(ligand_file, sanitize=True, removeHs=False)
        
        if not mol:
            # Try as SDF
            mol = Chem.MolFromMolFile(ligand_file, sanitize=True, removeHs=False)
        
        if not mol:
            return None
                
        # Generate SMILES
        smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # Validate SMILES
        if smiles and len(smiles) > 0:
            # Quick validation
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol:
                return smiles
        
        return None

    except Exception as e:
        logger.debug(f"Failed to extract SMILES from {ligand_file}: {e}")
        return None


def find_protein_ligand_pairs(crossdock_dir: Path, max_pairs: int = None) -> List[Tuple[str, str]]:
    """
    Find protein-ligand pairs from extracted CrossDock directory structure.
    
    CrossDock structure:
        POCKET_NAME/
            XXXX_A_rec.pdb  (receptor/protein)
            YYYY_lig.pdb     (ligand)
            
    Args:
        crossdock_dir: Path to extracted crossdocked directory
        max_pairs: Maximum pairs to process (None = all)
    
    Returns:
        List of (protein_path, ligand_path) tuples
    """
    pairs = []
    
    logger.info(f"Scanning {crossdock_dir} for protein-ligand pairs...")
    
    # Find all pocket directories
    pocket_dirs = [d for d in crossdock_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(pocket_dirs)} pocket directories")
    
    for pocket_dir in pocket_dirs:
        if max_pairs and len(pairs) >= max_pairs:
            break
        
        # Find receptor (protein) files
        rec_files = list(pocket_dir.glob("*_rec.pdb"))
        
        # Find ligand files  
        lig_files = list(pocket_dir.glob("*_lig.pdb"))
        
        if not rec_files or not lig_files:
            continue
        
        # Create pairs - typically each receptor with each ligand in the pocket
        for rec_file in rec_files:
            for lig_file in lig_files:
                pairs.append((str(rec_file), str(lig_file)))
                
                if max_pairs and len(pairs) >= max_pairs:
                    break
            if max_pairs and len(pairs) >= max_pairs:
                break
    
    logger.info(f"Found {len(pairs)} protein-ligand pairs")
    return pairs


def find_protein_ligand_pairs_from_archive(
    structures_archive: str, 
    max_pairs: int = None
) -> List[Tuple[str, str, str]]:
    """
    Find protein-ligand pairs by scanning the tar archive WITHOUT extracting.
    
    CrossDock structure in archive:
        crossdocked/POCKET_NAME/XXXX_A_rec.pdb  (receptor/protein)
        crossdocked/POCKET_NAME/YYYY_lig.pdb     (ligand)
            
    Args:
        structures_archive: Path to CrossDock structures .tgz file
        max_pairs: Maximum pairs to find (None = all)
    
    Returns:
        List of (pocket_name, receptor_member_name, ligand_member_name) tuples
    """
    pairs = []
    
    logger.info(f"Scanning archive {structures_archive} for protein-ligand pairs...")
    logger.info("This scans the archive index without extracting files")
    
    if not os.path.exists(structures_archive):
        raise FileNotFoundError(f"Archive not found: {structures_archive}")
    
    # Group files by pocket directory
    pockets = defaultdict(lambda: {'receptors': [], 'ligands': []})
    
    with tarfile.open(structures_archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            
            path_parts = Path(member.name).parts
            if len(path_parts) < 3:
                continue
            
            # Expect: crossdocked/POCKET_NAME/file.pdb
            if path_parts[0] == 'crossdocked':
                pocket_name = path_parts[1]
                filename = path_parts[2]
                
                if filename.endswith('_rec.pdb'):
                    pockets[pocket_name]['receptors'].append(member.name)
                elif filename.endswith('_lig.pdb'):
                    pockets[pocket_name]['ligands'].append(member.name)
    
    logger.info(f"Found {len(pockets)} pocket directories in archive")
    
    # Create pairs
    for pocket_name, files in pockets.items():
        if max_pairs and len(pairs) >= max_pairs:
            break
        
        receptors = files['receptors']
        ligands = files['ligands']
        
        if not receptors or not ligands:
            continue
        
        for rec in receptors:
            for lig in ligands:
                pairs.append((pocket_name, rec, lig))
                
                if max_pairs and len(pairs) >= max_pairs:
                    break
            if max_pairs and len(pairs) >= max_pairs:
                break
    
    logger.info(f"Found {len(pairs)} protein-ligand pairs")
    return pairs


def extract_members_to_temp(
    tar: tarfile.TarFile, 
    member_names: List[str], 
    temp_dir: str
) -> Dict[str, str]:
    """
    Extract specific tar members to temporary directory.
    
    Args:
        tar: Open tarfile object
        member_names: List of member names to extract
        temp_dir: Temporary directory path
        
    Returns:
        Dict mapping member name -> extracted file path
    """
    extracted = {}
    for member_name in member_names:
        try:
            member = tar.getmember(member_name)
            tar.extract(member, temp_dir)
            extracted_path = os.path.join(temp_dir, member_name)
            extracted[member_name] = extracted_path
        except Exception as e:
            logger.debug(f"Failed to extract {member_name}: {e}")
    return extracted


def process_pairs_to_training_data(
    pairs: List[Tuple[str, str]],
    output_csv: str,
    affinity_map: Dict = None,
    cutoff: float = 10.0,
    min_pocket_length: int = 10,
    max_pocket_length: int = 500,
    default_affinity: float = 5.0
):
    """
    Process protein-ligand pairs from extracted files into training CSV.
    
    Args:
        pairs: List of (protein_path, ligand_path) tuples
        output_csv: Output CSV path
        affinity_map: Dictionary mapping protein-ligand pairs to affinity values
        cutoff: Distance cutoff for pocket definition
        min_pocket_length: Minimum pocket sequence length
        max_pocket_length: Maximum pocket sequence length
        default_affinity: Default affinity value if not found in map
    """
    logger.info(f"Processing {len(pairs)} pairs from extracted files...")
    
    if affinity_map is None:
        affinity_map = {}
    
    results = []
    successful = 0
    failed_pocket = 0
    failed_smiles = 0
    invalid_length = 0
    affinity_found = 0
    affinity_missing = 0
    
    start_time = time.time()
    
    for i, (protein_file, ligand_file) in enumerate(pairs):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / rate if rate > 0 else 0
            logger.info(f"Processed {i + 1}/{len(pairs)} pairs "
                       f"(success: {successful}, failed: {failed_pocket + failed_smiles}, "
                       f"rate: {rate:.1f}it/s, ETA: {eta:.0f}s)")
        
        # Extract pocket sequence
        pocket_seq = extract_pocket_sequence(protein_file, ligand_file, cutoff)
        if not pocket_seq:
            failed_pocket += 1
            continue

        # Check pocket length
        if len(pocket_seq) < min_pocket_length or len(pocket_seq) > max_pocket_length:
            invalid_length += 1
            continue
        
        # Extract SMILES
        smiles = extract_smiles_from_ligand(ligand_file)
        if not smiles:
            failed_smiles += 1
            continue
        
        # Find affinity value
        affinity = find_affinity_for_pair(protein_file, ligand_file, affinity_map)
        if affinity is None:
            affinity = default_affinity
            affinity_missing += 1
        else:
            affinity_found += 1
        
        # Create pair ID from file paths
        pair_id = f"{Path(protein_file).parent.name}_{Path(ligand_file).stem}"
        
        results.append({
            'SMILES': smiles,
            'pocket_sequence': pocket_seq,
            'affinity': affinity,
            'pocket_length': len(pocket_seq),
            'pair_id': pair_id,
            'protein_file': protein_file,
            'ligand_file': ligand_file
        })
        successful += 1
    
    # Save results - headerless CSV for training
    # Format: SMILES,pocket_sequence,affinity,pair_id
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # No header - dataset expects headerless CSV
            for result in results:
                # Output: SMILES, pocket_sequence, affinity, pair_id
                writer.writerow([
                    result['SMILES'],
                    result['pocket_sequence'],
                    result['affinity'],
                    result['pair_id']
                ])
        
        logger.info(f"\n✓ Successfully processed {successful} pairs")
        logger.info(f"✓ Output saved to: {output_csv}")
    else:
        logger.error("✗ No pairs were successfully processed!")
        return
            
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total pairs attempted:       {len(pairs):,}")
    logger.info(f"Successful:                  {successful:,} ({successful/len(pairs)*100:.1f}%)")
    logger.info(f"Failed (pocket extraction):  {failed_pocket:,}")
    logger.info(f"Failed (SMILES extraction):  {failed_smiles:,}")
    logger.info(f"Failed (invalid length):     {invalid_length:,}")
    
    # Affinity statistics
    logger.info(f"\nAffinity statistics:")
    logger.info(f"  Found in database:     {affinity_found:,} ({affinity_found/successful*100:.1f}%)")
    logger.info(f"  Using default (5.0):   {affinity_missing:,} ({affinity_missing/successful*100:.1f}%)")
    
    if results:
        affinities = [r['affinity'] for r in results]
        logger.info(f"  Mean affinity: {sum(affinities)/len(affinities):.2f}")
        logger.info(f"  Min affinity:  {min(affinities):.2f}")
        logger.info(f"  Max affinity:  {max(affinities):.2f}")
        
        pocket_lengths = [r['pocket_length'] for r in results]
        logger.info(f"\nPocket length statistics:")
        logger.info(f"  Mean: {sum(pocket_lengths)/len(pocket_lengths):.1f}")
        logger.info(f"  Min:  {min(pocket_lengths)}")
        logger.info(f"  Max:  {max(pocket_lengths)}")
        
        # Sample output
        logger.info(f"\nSample entries (first 3):")
        for i, result in enumerate(results[:3]):
            logger.info(f"  {i+1}. SMILES: {result['SMILES'][:50]}...")
            logger.info(f"     Pocket: {result['pocket_sequence'][:50]}... (len={result['pocket_length']})")
            logger.info(f"     Affinity: {result['affinity']:.2f}")
    
    logger.info("="*80)


def process_pairs_from_archive_to_training_data(
    structures_archive: str,
    pairs: List[Tuple[str, str, str]],
    output_csv: str,
    affinity_map: Dict = None,
    cutoff: float = 10.0,
    min_pocket_length: int = 10,
    max_pocket_length: int = 500,
    default_affinity: float = 5.0
):
    """
    Process protein-ligand pairs from archive into training CSV.
    Extracts files on-demand to avoid disk space issues.
    
    Args:
        structures_archive: Path to structures .tgz file
        pairs: List of (pocket_name, receptor_member, ligand_member) tuples
        output_csv: Output CSV path
        affinity_map: Dictionary mapping protein-ligand pairs to affinity values
        cutoff: Distance cutoff for pocket definition
        min_pocket_length: Minimum pocket sequence length
        max_pocket_length: Maximum pocket sequence length
        default_affinity: Default affinity value if not found in map
    """
    logger.info(f"Processing {len(pairs)} pairs from archive...")
    logger.info("Extracting files on-demand to temporary directory")
    
    if affinity_map is None:
        affinity_map = {}
    
    results = []
    successful = 0
    failed_pocket = 0
    failed_smiles = 0
    invalid_length = 0
    affinity_found = 0
    affinity_missing = 0
    
    start_time = time.time()
    
    # Create temporary directory for extraction
    temp_base = tempfile.mkdtemp(prefix='crossdock_extract_')
    logger.info(f"Using temporary directory: {temp_base}")
    
    try:
        with tarfile.open(structures_archive, 'r:gz') as tar:
            for i, (pocket_name, rec_member, lig_member) in enumerate(pairs):
                if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(pairs) - i - 1) / rate if rate > 0 else 0
                    logger.info(f"Processed {i + 1}/{len(pairs)} pairs "
                               f"(success: {successful}, failed: {failed_pocket + failed_smiles}, "
                               f"rate: {rate:.1f}it/s, ETA: {eta:.0f}s)")
                
                # Extract current pair to temp directory
                temp_pair_dir = os.path.join(temp_base, f"pair_{i}")
                os.makedirs(temp_pair_dir, exist_ok=True)
                
                extracted = extract_members_to_temp(tar, [rec_member, lig_member], temp_pair_dir)
                
                if rec_member not in extracted or lig_member not in extracted:
                    failed_pocket += 1
                    # Clean up this pair's temp files
                    shutil.rmtree(temp_pair_dir, ignore_errors=True)
                    continue
                
                protein_file = extracted[rec_member]
                ligand_file = extracted[lig_member]
                
                # Extract pocket sequence
                pocket_seq = extract_pocket_sequence(protein_file, ligand_file, cutoff)
                if not pocket_seq:
                    failed_pocket += 1
                    shutil.rmtree(temp_pair_dir, ignore_errors=True)
                    continue

                # Check pocket length
                if len(pocket_seq) < min_pocket_length or len(pocket_seq) > max_pocket_length:
                    invalid_length += 1
                    shutil.rmtree(temp_pair_dir, ignore_errors=True)
                    continue
                
                # Extract SMILES
                smiles = extract_smiles_from_ligand(ligand_file)
                if not smiles:
                    failed_smiles += 1
                    shutil.rmtree(temp_pair_dir, ignore_errors=True)
                    continue
                
                # Find affinity value using original member names
                affinity = find_affinity_for_pair(rec_member, lig_member, affinity_map)
                if affinity is None:
                    affinity = default_affinity
                    affinity_missing += 1
                else:
                    affinity_found += 1
                
                # Create pair ID
                pair_id = f"{pocket_name}_{Path(lig_member).stem}"
                
                results.append({
                    'SMILES': smiles,
                    'pocket_sequence': pocket_seq,
                    'affinity': affinity,
                    'pocket_length': len(pocket_seq),
                    'pair_id': pair_id,
                    'protein_file': rec_member,
                    'ligand_file': lig_member
                })
                successful += 1
                
                # Clean up this pair's temp files immediately
                shutil.rmtree(temp_pair_dir, ignore_errors=True)
    
    finally:
        # Clean up temp directory
        logger.info(f"Cleaning up temporary directory: {temp_base}")
        shutil.rmtree(temp_base, ignore_errors=True)
    
    # Save results - headerless CSV for training
    # Format: SMILES,pocket_sequence,affinity
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # No header - dataset expects headerless CSV
            for result in results:
                # Output: SMILES, pocket_sequence, affinity
                writer.writerow([
                    result['SMILES'],
                    result['pocket_sequence'],
                    result['affinity']
                ])
        
        logger.info(f"\n✓ Successfully processed {successful} pairs")
        logger.info(f"✓ Output saved to: {output_csv}")
    else:
        logger.error("✗ No pairs were successfully processed!")
        return
            
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total pairs attempted:       {len(pairs):,}")
    logger.info(f"Successful:                  {successful:,} ({successful/len(pairs)*100:.1f}%)")
    logger.info(f"Failed (pocket extraction):  {failed_pocket:,}")
    logger.info(f"Failed (SMILES extraction):  {failed_smiles:,}")
    logger.info(f"Failed (invalid length):     {invalid_length:,}")
    
    # Affinity statistics
    logger.info(f"\nAffinity statistics:")
    logger.info(f"  Found in database:     {affinity_found:,} ({affinity_found/successful*100:.1f}%)")
    logger.info(f"  Using default (5.0):   {affinity_missing:,} ({affinity_missing/successful*100:.1f}%)")
    
    if results:
        affinities = [r['affinity'] for r in results]
        logger.info(f"  Mean affinity: {sum(affinities)/len(affinities):.2f}")
        logger.info(f"  Min affinity:  {min(affinities):.2f}")
        logger.info(f"  Max affinity:  {max(affinities):.2f}")
        
        pocket_lengths = [r['pocket_length'] for r in results]
        logger.info(f"\nPocket length statistics:")
        logger.info(f"  Mean: {sum(pocket_lengths)/len(pocket_lengths):.1f}")
        logger.info(f"  Min:  {min(pocket_lengths)}")
        logger.info(f"  Max:  {max(pocket_lengths)}")
        
        # Sample output
        logger.info(f"\nSample entries (first 3):")
        for i, result in enumerate(results[:3]):
            logger.info(f"  {i+1}. SMILES: {result['SMILES'][:50]}...")
            logger.info(f"     Pocket: {result['pocket_sequence'][:50]}... (len={result['pocket_length']})")
            logger.info(f"     Affinity: {result['affinity']:.2f}")
    
    logger.info("="*80)


def main():
    """Main entry point for the CrossDock2020 processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process CrossDock2020 protein-ligand dataset from PDB files"
    )
    parser.add_argument(
        "--output", "-o",
        default="protein_ligand_training.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Data directory containing crossdocked subdirectory"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10000,
        help="Maximum number of pairs to process (0 = all)"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff for pocket definition (Angstroms)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (unused, for compatibility)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up cache files after processing"
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Skip affinity extraction (use pocket length as proxy)"
    )
    parser.add_argument(
        "--extract-topology",
        action="store_true",
        help="Extract topology features (distance matrices) during processing"
    )
    parser.add_argument(
        "--topology-cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff for topology pocket definition (Angstroms)"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for RDKit
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required but not available!")
        sys.exit(1)
    
    try:
        base_dir = Path(args.data_dir)
        cache_dir = base_dir / "crossdocked"
        output_dir = base_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / args.output
        
        # Check if structures are already extracted (much faster!)
        crossdocked_dir = cache_dir / "crossdocked"
        structures_archive = cache_dir / "CrossDocked2020_v1.3.tgz"
        
        use_extracted = False
        if crossdocked_dir.exists():
            # Check if there are pocket directories
            pocket_dirs = list(crossdocked_dir.glob("*/"))
            if len(pocket_dirs) > 100:
                use_extracted = True
                logger.info(f"Found {len(pocket_dirs)} extracted pocket directories")
                logger.info(f"Using extracted files from: {crossdocked_dir}")
            else:
                logger.warning(f"Only {len(pocket_dirs)} directories found, will use archive instead")
        
        if not use_extracted:
            if not structures_archive.exists():
                raise CrossDockProcessingError(
                    f"Structures archive not found at {structures_archive}\n"
                    f"Run download_crossdock.sh first to download the dataset"
                )
            logger.info(f"Using CrossDock archive: {structures_archive}")
            logger.info("Processing directly from archive - no full extraction needed")
        
        # Find protein-ligand pairs FIRST (before parsing affinities)
        max_pairs = None if args.max_pairs == 0 else args.max_pairs
        
        if use_extracted:
            # Use extracted files (FASTER!)
            logger.info("Using extracted pocket directories")
            pairs = find_protein_ligand_pairs(crossdocked_dir, max_pairs)
            
            if not pairs:
                logger.error("No protein-ligand pairs found in extracted directory!")
                sys.exit(1)
            
            logger.info(f"Found {len(pairs)} pairs to process")
            
            # Parse affinity data ONLY for the pairs we're processing (unless disabled)
            affinity_map = {}
            if not args.no_affinity:
                types_archive = cache_dir / "CrossDocked2020_v1.3_types.tgz"
                if not types_archive.exists():
                    logger.error(f"Types archive not found: {types_archive}")
                    logger.error("Affinities are required. Download with download_crossdock.sh")
                    sys.exit(1)
                
                # OPTIMIZATION: Pass target pairs to only parse those affinities
                affinity_map = parse_affinity_from_types_archive(
                    str(types_archive),
                    target_pairs=pairs,  # Only parse for these pairs!
                    debug=args.debug,
                    cache_dir=str(output_dir)
                )
            else:
                logger.warning("Skipping affinity extraction (--no-affinity flag set)")
            
            # Extract topology features if requested (BEFORE processing to CSV)
            if args.extract_topology:
                topology_dir = output_dir / "topology_features"
                logger.info(f"\nExtracting topology features...")
                extract_and_cache_distance_matrices(
                    protein_ligand_pairs=pairs,
                    output_dir=str(topology_dir),
                    cutoff=args.topology_cutoff,
                    max_pairs=None  # Process all found pairs
                )
            
            # Process with regular function (files on disk)
            process_pairs_to_training_data(
                pairs,
                str(output_path),
                affinity_map=affinity_map,
                cutoff=args.cutoff
            )
        else:
            # Read from archive (SAVES DISK SPACE)
            logger.info("Reading pairs directly from archive")
            pairs = find_protein_ligand_pairs_from_archive(str(structures_archive), max_pairs)
            
            if not pairs:
                logger.error("No protein-ligand pairs found in archive!")
                sys.exit(1)
            
            logger.info(f"Found {len(pairs)} pairs to process")
            
            # Parse affinity data ONLY for the pairs we're processing (unless disabled)
            affinity_map = {}
            if not args.no_affinity:
                types_archive = cache_dir / "CrossDocked2020_v1.3_types.tgz"
                if not types_archive.exists():
                    logger.error(f"Types archive not found: {types_archive}")
                    logger.error("Affinities are required. Download with download_crossdock.sh")
                    sys.exit(1)
                
                # Note: archive pairs are (pocket_name, rec_member, lig_member) format
                # Convert to (protein_path, ligand_path) for lookup
                lookup_pairs = [(rec, lig) for (_, rec, lig) in pairs]
                
                affinity_map = parse_affinity_from_types_archive(
                    str(types_archive),
                    target_pairs=lookup_pairs,  # Only parse for these pairs!
                    debug=args.debug,
                    cache_dir=str(output_dir)
                )
            else:
                logger.warning("Skipping affinity extraction (--no-affinity flag set)")
            
            # Process from archive
            process_pairs_from_archive_to_training_data(
                str(structures_archive),
                pairs,
                str(output_path),
                affinity_map=affinity_map,
                cutoff=args.cutoff
            )
        
        # Cleanup if requested
        if args.cleanup:
            logger.info("Cleaning up archive files...")
            for pattern in ["*.tgz", "*.tar.gz"]:
                for file_path in cache_dir.glob(pattern):
                    try:
                        file_path.unlink()
                        logger.info(f"Removed: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
        
        logger.info("\n✓ Processing complete!")
        logger.info(f"\nNext steps:")
        logger.info(f"  python train.py --use_protein_conditioning \\")
        logger.info(f"    --protein_ligand_data_path {output_path} \\")
        logger.info(f"    --model_size standard --batch_size 128 --use_amp")
        
    except CrossDockProcessingError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def extract_and_cache_distance_matrices(
    protein_ligand_pairs: List[Tuple[str, str]],
    output_dir: str,
    cutoff: float = 10.0,
    max_pairs: Optional[int] = None
) -> Dict[str, bool]:
    """
    Extract C-alpha distance matrices from protein-ligand pairs for topology encoding.
    
    This function processes PDB files to extract C-alpha coordinates from binding
    pockets and computes pairwise distance matrices for topological analysis.
    
    Args:
        protein_ligand_pairs: List of (protein_pdb_path, ligand_pdb_path) tuples
        output_dir: Directory to save distance matrices (.npz files)
        cutoff: Distance cutoff for pocket definition (Angstroms)
        max_pairs: Maximum number of pairs to process (None = all)
        
    Returns:
        Dictionary mapping pair_id to success status
    """
    logger.info("\n" + "="*70)
    logger.info("EXTRACTING TOPOLOGY FEATURES (DISTANCE MATRICES)")
    logger.info("="*70)
    
    # Import distance extraction module
    try:
        from extract_pdb_distances import batch_process_pdb_files
    except ImportError:
        logger.warning("Could not import extract_pdb_distances module")
        logger.warning("Topology features will not be available")
        return {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cutoff distance: {cutoff}Å")
    
    # Limit pairs if specified
    if max_pairs:
        protein_ligand_pairs = protein_ligand_pairs[:max_pairs]
        logger.info(f"Processing first {max_pairs} pairs")
    
    # Process pairs in batch
    results = batch_process_pdb_files(
        pdb_pairs=protein_ligand_pairs,
        output_dir=output_dir,
        cutoff=cutoff,
        overwrite=False
    )
    
    return results


if __name__ == "__main__":
    main() 
