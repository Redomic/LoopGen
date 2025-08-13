"""
Molecular augmentation utilities for SELFIES.

This module provides SMILES enumeration-based augmentation that can be used
for data augmentation during training. The augmentations preserve molecular
validity by using RDKit's SMILES randomization.
"""

import random
import logging
import gc
from typing import List, Tuple, Optional, Set
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import warnings

# Suppress RDKit warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SELFIESAugmenter:
    """Generates augmented SELFIES representations using SMILES enumeration only."""
    
    def __init__(self, tokenizer):
        """
        Initialize the SELFIES augmenter.
        
        Args:
            tokenizer: SELFIESTokenizer instance (kept for compatibility)
        """
        self.tokenizer = tokenizer

    def augment(self, selfies_string: str, n_augmentations: int = 10) -> List[str]:
        """
        Generate augmentations using only SMILES enumeration (gold standard).
        
        Args:
            selfies_string: Input SELFIES string
            n_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented SELFIES strings (including original)
        """
        # Step 1: SELFIES → SMILES
        try:
            canonical_smiles = sf.decoder(selfies_string)
        except Exception:
            return [selfies_string] * n_augmentations
        
        # Step 2: Create RDKit molecule
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return [selfies_string] * n_augmentations
        
        try:
            # Step 3: Generate augmentations using SMILES enumeration only
            augmented_selfies = set([selfies_string])  # Always include original
            
            attempts = 0
            max_attempts = n_augmentations * 10
            
            while len(augmented_selfies) < n_augmentations and attempts < max_attempts:
                # Generate random SMILES
                random_smiles = self._generate_random_smiles(mol, attempts)
                
                # Convert back to SELFIES
                try:
                    random_selfies = sf.encoder(random_smiles)
                    augmented_selfies.add(random_selfies)
                except Exception:
                    pass
                
                attempts += 1
            
            # Convert to list and pad if needed
            result = list(augmented_selfies)
            while len(result) < n_augmentations:
                result.append(selfies_string)
            
            return result[:n_augmentations]
            
        finally:
            # Ensure molecule is always cleaned up
            try:
                del mol
            except:
                pass

    def _generate_random_smiles(self, mol: Chem.Mol, attempt: int) -> str:
        """
        Generate randomized SMILES with different strategies based on attempt number.
        """
        if attempt < 10:
            # Standard randomization
            return Chem.MolToSmiles(mol, doRandom=True)
        elif attempt < 20:
            # Try with specific starting atom
            n_atoms = mol.GetNumAtoms()
            if n_atoms > 1:
                start_atom = random.randint(0, n_atoms - 1)
                return Chem.MolToSmiles(mol, doRandom=True, rootedAtAtom=start_atom)
            else:
                return Chem.MolToSmiles(mol, doRandom=True)
        else:
            # Try with different canonical settings
            return Chem.MolToSmiles(mol, 
                                  doRandom=True, 
                                  canonical=False,
                                  allBondsExplicit=random.choice([True, False]))

    def augment_batch(self, selfies_list: List[str], n_augmentations: int = 10) -> List[str]:
        """
        Augment a batch of SELFIES strings.
        
        This can be used for data augmentation during training to increase
        the diversity of training examples.
        
        Args:
            selfies_list: List of SELFIES strings
            n_augmentations: Number of augmentations per molecule
            
        Returns:
            List of augmented SELFIES strings
        """
        augmented_batch = []
        
        for selfies in selfies_list:
            augmentations = self.augment(selfies, n_augmentations)
            augmented_batch.extend(augmentations)
        
        return augmented_batch

# Note: Removed create_contrastive_batch function as it's no longer needed
# for the pure autoregressive model


 