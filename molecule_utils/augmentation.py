"""
SMILES Data Augmentation for Robust Molecular Generation.

Based on "SMILES enumeration as data augmentation for molecular generation" (Bjerrum, 2017).
Randomized SMILES significantly improves model robustness and reduces overfitting.
"""

from typing import List, Optional, Set
from rdkit import Chem
from rdkit import RDLogger
import random


# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')


def randomize_smiles(smiles: str, num_variants: int = 5, canonical: bool = False) -> List[str]:
    """
    Generate randomized SMILES variants by random atom ordering.
    
    This is a SOTA data augmentation technique that improves model robustness
    by exposing the model to different valid representations of the same molecule.
    
    Args:
        smiles: Input SMILES string
        num_variants: Number of random variants to generate (default: 5)
        canonical: If True, include canonical SMILES as first variant (default: False)
    
    Returns:
        List of unique SMILES variants (including original if canonical=True)
        Returns empty list if SMILES is invalid
    
    Example:
        >>> variants = randomize_smiles("CC(C)CCO", num_variants=3)
        >>> print(variants)
        ['CC(C)CCO', 'OCC(C)C', 'C(C)CCO']
    
    Reference:
        Bjerrum, E. J. (2017). SMILES Enumeration as Data Augmentation for Neural
        Network Modeling of Molecules. arXiv:1703.07076
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return []
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return []
        
        variants = []
        
        # Add canonical SMILES if requested
        if canonical:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            variants.append(canonical_smiles)
        
        # Generate random variants
        attempts = 0
        max_attempts = num_variants * 3  # Allow some failed attempts
        
        while len(variants) < num_variants and attempts < max_attempts:
            # Generate random SMILES with random atom ordering
            random_smiles = Chem.MolToSmiles(
                mol,
                doRandom=True,
                canonical=False,
                isomericSmiles=True  # Preserve stereochemistry
            )
            
            # Only add if unique
            if random_smiles not in variants:
                variants.append(random_smiles)
            
            attempts += 1
        
        return variants
    
    except Exception:
        return []


def augment_smiles_dataset(
    smiles_list: List[str],
    augmentation_factor: int = 3,
    include_original: bool = True,
    shuffle: bool = True
) -> List[str]:
    """
    Augment a dataset of SMILES strings with randomized variants.
    
    Args:
        smiles_list: List of input SMILES strings
        augmentation_factor: Number of variants per SMILES (default: 3)
        include_original: Whether to include original SMILES (default: True)
        shuffle: Whether to shuffle the augmented dataset (default: True)
    
    Returns:
        List of augmented SMILES (original + variants)
    
    Example:
        >>> smiles = ["CCO", "c1ccccc1"]
        >>> augmented = augment_smiles_dataset(smiles, augmentation_factor=2)
        >>> print(len(augmented))  # ~6 (2 original + 2*2 variants)
    """
    augmented = []
    
    for smiles in smiles_list:
        # Add original if requested
        if include_original:
            augmented.append(smiles)
        
        # Generate and add variants
        variants = randomize_smiles(
            smiles,
            num_variants=augmentation_factor,
            canonical=False
        )
        augmented.extend(variants)
    
    # Shuffle to mix originals and variants
    if shuffle:
        random.shuffle(augmented)
    
    return augmented


def canonicalize_smiles(smiles: str, remove_stereo: bool = False) -> Optional[str]:
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles: Input SMILES string
        remove_stereo: If True, remove stereochemistry information (default: False)
    
    Returns:
        Canonical SMILES string, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=not remove_stereo
        )
        return canonical
    
    except Exception:
        return None


def remove_stereochemistry(smiles: str) -> Optional[str]:
    """
    Remove stereochemistry markers from SMILES.
    
    Useful when stereochemistry causes generation issues but you still
    want to train on the molecular scaffold.
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        SMILES without stereochemistry, or None if invalid
    """
    return canonicalize_smiles(smiles, remove_stereo=True)


def deduplicate_smiles(smiles_list: List[str], canonicalize: bool = True) -> List[str]:
    """
    Remove duplicate SMILES from a list.
    
    Args:
        smiles_list: List of SMILES strings
        canonicalize: If True, canonicalize before deduplication (default: True)
                     This ensures different representations of the same molecule
                     are treated as duplicates
    
    Returns:
        List of unique SMILES strings
    """
    if canonicalize:
        # Canonicalize and deduplicate
        seen = set()
        unique = []
        
        for smiles in smiles_list:
            canonical = canonicalize_smiles(smiles)
            if canonical and canonical not in seen:
                seen.add(canonical)
                unique.append(smiles)  # Keep original form
        
        return unique
    else:
        # Simple string-based deduplication
        return list(dict.fromkeys(smiles_list))


class SMILESAugmenter:
    """
    Stateful SMILES augmenter for use in data pipelines.
    
    Example:
        >>> augmenter = SMILESAugmenter(augmentation_factor=3)
        >>> for smiles in dataset:
        ...     augmented = augmenter.augment(smiles)
        ...     # Process augmented SMILES
    """
    
    def __init__(
        self,
        augmentation_factor: int = 3,
        include_original: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize SMILES augmenter.
        
        Args:
            augmentation_factor: Number of variants per SMILES
            include_original: Whether to include original SMILES
            cache_size: Maximum cache size for generated variants (0 = no cache)
        """
        self.augmentation_factor = augmentation_factor
        self.include_original = include_original
        self.cache_size = cache_size
        self._cache = {}
    
    def augment(self, smiles: str) -> List[str]:
        """
        Generate augmented variants for a single SMILES.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            List of SMILES variants (including original if include_original=True)
        """
        # Check cache
        if self.cache_size > 0 and smiles in self._cache:
            return self._cache[smiles]
        
        result = []
        
        if self.include_original:
            result.append(smiles)
        
        variants = randomize_smiles(smiles, num_variants=self.augmentation_factor)
        result.extend(variants)
        
        # Update cache
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                # Simple FIFO cache eviction
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[smiles] = result
        
        return result
    
    def clear_cache(self):
        """Clear the augmentation cache."""
        self._cache.clear()


def validate_augmentation_quality(
    original_smiles: str,
    augmented_smiles: List[str]
) -> dict:
    """
    Validate that augmented SMILES represent the same molecule.
    
    Args:
        original_smiles: Original SMILES string
        augmented_smiles: List of augmented SMILES variants
    
    Returns:
        Dictionary with validation results:
        - all_valid: bool - All variants are valid SMILES
        - all_equivalent: bool - All variants represent the same molecule
        - num_unique: int - Number of unique representations
    """
    try:
        original_mol = Chem.MolFromSmiles(original_smiles)
        if original_mol is None:
            return {
                'all_valid': False,
                'all_equivalent': False,
                'num_unique': 0
            }
        
        original_canonical = Chem.MolToSmiles(original_mol, canonical=True)
        
        all_valid = True
        all_equivalent = True
        canonical_set = set()
        
        for variant in augmented_smiles:
            mol = Chem.MolFromSmiles(variant)
            if mol is None:
                all_valid = False
                all_equivalent = False
                continue
            
            canonical = Chem.MolToSmiles(mol, canonical=True)
            canonical_set.add(canonical)
            
            if canonical != original_canonical:
                all_equivalent = False
        
        return {
            'all_valid': all_valid,
            'all_equivalent': all_equivalent,
            'num_unique': len(canonical_set)
        }
    
    except Exception:
        return {
            'all_valid': False,
            'all_equivalent': False,
            'num_unique': 0
        }
