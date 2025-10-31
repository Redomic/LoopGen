"""
Dataset for protein-conditioned molecular generation.

This module provides a dataset class for training models on protein-ligand pairs,
where each example consists of a SMILES string (ligand) and a protein pocket sequence.
Optionally loads precomputed topology features from distance matrices.
"""

import torch
from torch.utils.data import IterableDataset
from typing import Optional, Iterator, Dict
import pandas as pd
import math
import random
import numpy as np
import os
import logging

from .tokenizer import SMILESTokenizer
from .protein_tokenizer import ProteinTokenizer

logger = logging.getLogger(__name__)


class ProteinLigandDataset(IterableDataset):
    """
    Iterable dataset for protein-ligand pairs with pocket sequences.
    
    Reads CSV file with columns:
        - SMILES: Ligand molecular structure
        - pocket_sequence: Amino acid sequence of binding pocket
        - affinity: Binding affinity (optional)
        - pair_id: Unique identifier (optional)
    
    Optionally loads precomputed topology features from distance matrices.
    
    Yields dictionaries with tokenized SMILES and protein sequences,
    plus topology features if available.
    
    Args:
        file_path: Path to CSV file with protein-ligand data
        smiles_tokenizer: Tokenizer for SMILES strings
        protein_tokenizer: Tokenizer for protein sequences
        max_smiles_len: Maximum SMILES sequence length
        max_protein_len: Maximum protein sequence length
        total_lines: Total number of lines in file (for split calculation)
        split: 'train' or 'val'
        split_ratio: Ratio for train/val split (default 0.8)
        shuffle_buffer_size: Size of shuffle buffer for randomization
        topology_dir: Optional directory with precomputed topology features
        use_topology: Whether to load and use topology features
    """
    
    def __init__(
        self,
        file_path: str,
        smiles_tokenizer: SMILESTokenizer,
        protein_tokenizer: ProteinTokenizer,
        max_smiles_len: int = 256,
        max_protein_len: int = 512,
        total_lines: int = None,
        split: str = 'train',
        split_ratio: float = 0.8,
        shuffle_buffer_size: int = 10000,
        topology_dir: Optional[str] = None,
        use_topology: bool = False
    ):
        super().__init__()
        self.file_path = file_path
        self.smiles_tokenizer = smiles_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_protein_len = max_protein_len
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.topology_dir = topology_dir
        self.use_topology = use_topology
        
        # Initialize topology feature extractor if needed
        self.topology_extractor = None
        if self.use_topology:
            try:
                from model.topology_encoder import TopologyFeatureExtractor
                self.topology_extractor = TopologyFeatureExtractor(
                    homology_dimensions=[0, 1, 2],
                    n_bins=50,
                    representation='image'
                )
                logger.info("Initialized topology feature extraction for dataset")
            except ImportError as e:
                logger.warning(f"Could not initialize topology extraction: {e}")
                self.use_topology = False
        
        # Calculate total lines if not provided
        if total_lines is None:
            total_lines = self._count_lines()
        self.total_lines = total_lines
        
        # Calculate split ranges
        if split == 'train':
            self.start_line = 0
            self.end_line = math.floor(total_lines * split_ratio)
        elif split == 'val':
            self.start_line = math.floor(total_lines * split_ratio)
            self.end_line = total_lines
        else:
            raise ValueError("split must be 'train' or 'val'")
        
        self.length = self.end_line - self.start_line
    
    def _count_lines(self) -> int:
        """Count total lines in CSV file (no header in our format)."""
        try:
            with open(self.file_path, 'r') as f:
                return sum(1 for _ in f)  # No header to subtract
        except FileNotFoundError:
            return 0
    
    def __len__(self):
        return self.length
    
    def _line_iterator(self) -> Iterator[Dict]:
        """Stream lines from the CSV file for this split."""
        try:
            num_rows_to_read = self.end_line - self.start_line
            if num_rows_to_read <= 0:
                return
            
            # Read CSV in chunks (no header in the file)
            chunk_iterator = pd.read_csv(
                self.file_path,
                chunksize=1000,
                header=None,  # No header row in the CSV
                names=['SMILES', 'pocket_sequence', 'affinity', 'pair_id'],  # Assign column names
                skiprows=range(0, self.start_line) if self.start_line > 0 else None,
                nrows=num_rows_to_read,
                on_bad_lines='skip'
            )
            
            for chunk in chunk_iterator:
                for _, row in chunk.iterrows():
                    smiles = row.get('SMILES', None)
                    pocket_seq = row.get('pocket_sequence', None)
                    affinity = row.get('affinity', None)
                    pair_id = row.get('pair_id', None)
                    
                    # Skip invalid rows
                    if not isinstance(smiles, str) or not isinstance(pocket_seq, str):
                        continue
                    
                    yield {
                        'smiles': smiles.strip(),
                        'pocket_sequence': pocket_seq.strip(),
                        'affinity': affinity if pd.notna(affinity) else None,
                        'pair_id': pair_id.strip() if isinstance(pair_id, str) else None
                    }
        
        except FileNotFoundError:
            return
        except Exception as e:
            print(f"Error reading dataset: {e}")
            import traceback
            traceback.print_exc()
            return
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over protein-ligand pairs, returning tokenized sequences.
        
        Handles multi-worker data loading by splitting data among workers.
        
        Yields:
            Dictionary with:
                - smiles_ids: Tokenized SMILES [seq_len]
                - protein_ids: Tokenized protein sequence [protein_seq_len]
                - affinity: Binding affinity (optional)
        """
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multiple workers: split the data
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Calculate this worker's range
            per_worker = self.length // num_workers
            remainder = self.length % num_workers
            
            # Distribute remainder among first workers
            if worker_id < remainder:
                worker_start = self.start_line + worker_id * (per_worker + 1)
                worker_length = per_worker + 1
            else:
                worker_start = self.start_line + worker_id * per_worker + remainder
                worker_length = per_worker
            
            worker_end = worker_start + worker_length
            
            # Ensure we don't go beyond our split
            worker_end = min(worker_end, self.end_line)
            
            # Temporarily override the split range for this worker
            original_start = self.start_line
            original_end = self.end_line
            self.start_line = worker_start
            self.end_line = worker_end
        else:
            original_start = None
            original_end = None
        
        buffer = []
        
        try:
            for pair_data in self._line_iterator():
                buffer.append(pair_data)
                
                # Shuffle and yield when buffer is full
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    
                    for item in buffer:
                        tokenized = self._tokenize_pair(item)
                        if tokenized is not None:
                            yield tokenized
                    
                    buffer = []
            
            # Yield remaining items
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    tokenized = self._tokenize_pair(item)
                    if tokenized is not None:
                        yield tokenized
        
        finally:
            # Restore original range if we're in a worker
            if original_start is not None:
                self.start_line = original_start
                self.end_line = original_end
    
    def _load_topology_features(self, pair_id: Optional[str]) -> Optional[np.ndarray]:
        """
        Load precomputed topology features for a pair.
        
        Args:
            pair_id: Identifier for the protein-ligand pair
            
        Returns:
            Topology feature vector or None if not available
        """
        if not self.use_topology or not self.topology_dir or pair_id is None:
            return None
        
        try:
            # Try to load distance matrix
            distance_matrix_path = os.path.join(self.topology_dir, f"{pair_id}.npz")
            
            if not os.path.exists(distance_matrix_path):
                return None
            
            # Load distance matrix
            data = np.load(distance_matrix_path)
            distance_matrix = data['distance_matrix']
            
            # Extract topology features
            if self.topology_extractor is not None:
                features = self.topology_extractor.extract_features(distance_matrix)
                return features
            
            return None
        
        except Exception as e:
            logger.debug(f"Error loading topology features for {pair_id}: {e}")
            return None
    
    def _tokenize_pair(self, pair_data: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize a single protein-ligand pair and optionally load topology features.
        
        Args:
            pair_data: Dictionary with 'smiles', 'pocket_sequence', etc.
        
        Returns:
            Dictionary with tokenized sequences and optional topology features,
            or None if tokenization fails
        """
        try:
            # Tokenize SMILES
            smiles_tokens = self.smiles_tokenizer.encode(
                pair_data['smiles'],
                add_special_tokens=True
            )
            
            # Tokenize protein sequence
            protein_tokens = self.protein_tokenizer.encode(
                pair_data['pocket_sequence'],
                add_special_tokens=True
            )
            
            # Check if sequences are valid length
            if len(smiles_tokens) == 0 or len(protein_tokens) == 0:
                return None
            
            # Convert to tensors (no padding here - done in collate_fn)
            result = {
                'smiles_ids': torch.tensor(smiles_tokens, dtype=torch.long),
                'protein_ids': torch.tensor(protein_tokens, dtype=torch.long)
            }
            
            # Add affinity if available
            if pair_data.get('affinity') is not None:
                result['affinity'] = torch.tensor(pair_data['affinity'], dtype=torch.float)
            
            # Load topology features if enabled
            if self.use_topology:
                topology_features = self._load_topology_features(pair_data.get('pair_id'))
                if topology_features is not None:
                    result['topology_features'] = torch.from_numpy(topology_features).float()
                else:
                    # Use zero features if topology not available
                    feature_dim = (50 ** 2) * 3  # Default: 50 bins, 3 homology dims
                    result['topology_features'] = torch.zeros(feature_dim, dtype=torch.float)
            
            return result
        
        except Exception as e:
            # Skip pairs that fail tokenization
            logger.debug(f"Error tokenizing pair: {e}")
            return None


def count_protein_ligand_pairs(file_path: str) -> int:
    """Count total number of protein-ligand pairs in CSV file (no header)."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)  # No header to subtract
    except FileNotFoundError:
        return 0



