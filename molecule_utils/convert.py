"""
Molecular utility functions for SMILES and SELFIES conversion.

This module provides robust conversion functions between SMILES and SELFIES
representations of molecules, with comprehensive error handling, validation,
and filtering for problematic structures.
"""

import logging
import re
from typing import Optional, Union, List, Tuple, Dict, Set
import warnings
from enum import Enum
from dataclasses import dataclass

# Suppress RDKit warnings during import
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logger = logging.getLogger(__name__)


class MolecularFilterReason(Enum):
    """Enumeration of reasons for molecular filtering."""
    VALID = "valid"
    EMPTY_MOLECULE = "empty_molecule"
    INVALID_SMILES = "invalid_smiles"
    DISCONNECTED_STRUCTURE = "disconnected_structure"
    CONTAINS_METAL = "contains_metal"
    RADICAL_STRUCTURE = "radical_structure"
    CHARGED_STRUCTURE = "charged_structure"
    EXPLICIT_HYDROGEN = "explicit_hydrogen"
    MOLECULAR_WEIGHT_OUT_OF_RANGE = "molecular_weight_out_of_range"
    LOGP_OUT_OF_RANGE = "logp_out_of_range"
    TOO_MANY_ATOMS = "too_many_atoms"
    STANDARDIZATION_FAILED = "standardization_failed"
    CONVERSION_FAILED = "conversion_failed"
    ROUND_TRIP_FAILED = "round_trip_failed"


@dataclass
class MolecularFilterConfig:
    """Configuration for molecular filtering parameters."""
    remove_salts: bool = True
    remove_stereochemistry: bool = False
    remove_charges: bool = True
    remove_isotopes: bool = True
    remove_explicit_hydrogens: bool = True
    remove_fragments: bool = True
    min_molecular_weight: float = 150.0
    max_molecular_weight: float = 500.0
    max_logp: float = 5.0
    max_num_atoms: int = 200
    allowed_elements: Optional[Set[str]] = None
    
    def __post_init__(self):
        if self.allowed_elements is None:
            # Default allowed elements (organic molecules)
            self.allowed_elements = {
                'H', 'C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br', 'I', 'B', 'Si'
            }


class MolecularConversionError(Exception):
    """Custom exception for molecular conversion errors."""
    pass


class InvalidSMILESError(MolecularConversionError):
    """Exception raised when SMILES string is invalid."""
    pass


class MolecularStandardizer:
    """Handles molecular standardization and cleaning operations."""
    
    def __init__(self, config: MolecularFilterConfig):
        self.config = config
        self.salt_remover = SaltRemover.SaltRemover()
        self.uncharger = rdMolStandardize.Uncharger()
        
    def standardize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Standardize a molecule according to the configuration.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Standardized molecule or None if standardization fails
        """
        try:
            # Remove explicit hydrogens first to avoid issues
            if self.config.remove_explicit_hydrogens:
                mol = Chem.RemoveHs(mol, sanitize=False)
            
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            
            # Use the Cleanup function for standardization
            mol = rdMolStandardize.Cleanup(mol)
            
            # Remove salts
            if self.config.remove_salts:
                mol = self.salt_remover.StripMol(mol)
            
            # Remove charges
            if self.config.remove_charges:
                mol = self.uncharger.uncharge(mol)
            
            # Remove stereochemistry
            if self.config.remove_stereochemistry:
                Chem.RemoveStereochemistry(mol)
            
            # Remove isotopes
            if self.config.remove_isotopes:
                for atom in mol.GetAtoms():
                    atom.SetIsotope(0)
            
            # Final sanitization
            Chem.SanitizeMol(mol)
            
            return mol
            
        except Exception as e:
            logger.debug(f"Standardization failed: {str(e)}")
            return None


class MolecularFilter:
    """Comprehensive molecular filtering and validation."""
    
    def __init__(self, config: MolecularFilterConfig):
        self.config = config
        self.standardizer = MolecularStandardizer(config)
        
        # Compile regex patterns for efficiency
        self.metal_pattern = re.compile(
            r'\[(?:Li|Na|K|Rb|Cs|Fr|Be|Mg|Ca|Sr|Ba|Ra|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|'
            r'Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|'
            r'Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|'
            r'No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og)'
        )
        
    def filter_smiles(self, smiles: str) -> Tuple[bool, MolecularFilterReason, Optional[str]]:
        """
        Filter and validate a SMILES string.
        
        Args:
            smiles: SMILES string to filter
            
        Returns:
            Tuple of (is_valid, reason, processed_smiles)
        """
        # Basic validation
        if not smiles or not smiles.strip():
            return False, MolecularFilterReason.EMPTY_MOLECULE, None
            
        smiles = smiles.strip()
        
        # Check for disconnected structures (multiple molecules)
        if '.' in smiles and self.config.remove_fragments:
            # Try to get the largest fragment
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, MolecularFilterReason.INVALID_SMILES, None
                    
                # Get largest fragment
                fragments = Chem.GetMolFrags(mol, asMols=True)
                if len(fragments) > 1:
                    largest_fragment = max(fragments, key=lambda x: x.GetNumAtoms())
                    smiles = Chem.MolToSmiles(largest_fragment)
                    mol = largest_fragment
            except Exception:
                return False, MolecularFilterReason.DISCONNECTED_STRUCTURE, None
        else:
            # Parse molecule
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, MolecularFilterReason.INVALID_SMILES, None
            except Exception:
                return False, MolecularFilterReason.INVALID_SMILES, None
        
        # Check for metals
        if self.metal_pattern.search(smiles):
            return False, MolecularFilterReason.CONTAINS_METAL, None
        
        # Check for radicals
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return False, MolecularFilterReason.RADICAL_STRUCTURE, None
        
        # Check allowed elements
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in self.config.allowed_elements:
                return False, MolecularFilterReason.CONTAINS_METAL, None
        
        # Standardize molecule
        standardized_mol = self.standardizer.standardize_molecule(mol)
        if standardized_mol is None:
            return False, MolecularFilterReason.STANDARDIZATION_FAILED, None
        
        # Check molecular properties
        mol_weight = Descriptors.MolWt(standardized_mol)
        if mol_weight < self.config.min_molecular_weight or mol_weight > self.config.max_molecular_weight:
            return False, MolecularFilterReason.MOLECULAR_WEIGHT_OUT_OF_RANGE, None
        
        logp = Descriptors.MolLogP(standardized_mol)
        if logp >= self.config.max_logp:
            return False, MolecularFilterReason.LOGP_OUT_OF_RANGE, None
        
        num_atoms = standardized_mol.GetNumAtoms()
        if num_atoms > self.config.max_num_atoms:
            return False, MolecularFilterReason.TOO_MANY_ATOMS, None
        
        # Check for remaining charges after standardization
        if self.config.remove_charges:
            total_charge = sum(atom.GetFormalCharge() for atom in standardized_mol.GetAtoms())
            if total_charge != 0:
                return False, MolecularFilterReason.CHARGED_STRUCTURE, None
        
        # Convert back to canonical SMILES
        try:
            canonical_smiles = Chem.MolToSmiles(standardized_mol, canonical=True)
            return True, MolecularFilterReason.VALID, canonical_smiles
        except Exception:
            return False, MolecularFilterReason.CONVERSION_FAILED, None


# Global default filter configuration
DEFAULT_FILTER_CONFIG = MolecularFilterConfig()


def validate_smiles(smiles: str, use_filter: bool = False) -> bool:
    """
    Validate if a SMILES string represents a valid molecule.
    
    Args:
        smiles: SMILES string to validate
        use_filter: Whether to use comprehensive filtering
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if use_filter:
            filter_instance = MolecularFilter(DEFAULT_FILTER_CONFIG)
            is_valid, _, _ = filter_instance.filter_smiles(smiles)
            return is_valid
        else:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
    except Exception:
        return False


def preprocess_smiles_for_selfies(smiles: str, config: Optional[MolecularFilterConfig] = None) -> Optional[str]:
    """
    Preprocess SMILES string to prepare it for SELFIES conversion.
    
    Args:
        smiles: SMILES string to preprocess
        config: Filter configuration (uses default if None)
        
    Returns:
        Preprocessed SMILES string or None if preprocessing fails
    """
    if config is None:
        config = DEFAULT_FILTER_CONFIG
    
    filter_instance = MolecularFilter(config)
    is_valid, reason, processed_smiles = filter_instance.filter_smiles(smiles)
    
    if is_valid:
        return processed_smiles
    else:
        logger.debug(f"SMILES preprocessing failed: {reason.value}")
        return None


def convert_smiles_to_selfies(
    smiles: str, 
    validate_input: bool = True,
    preprocess: bool = True,
    filter_config: Optional[MolecularFilterConfig] = None,
    return_validation_info: bool = False
) -> Union[str, Tuple[str, dict]]:
    """
    Convert SMILES string to SELFIES representation with robust error handling.
    
    Args:
        smiles: SMILES string to convert
        validate_input: Whether to validate SMILES before conversion
        preprocess: Whether to preprocess SMILES (recommended to avoid warnings)
        filter_config: Configuration for molecular filtering
        return_validation_info: Whether to return additional validation information
        
    Returns:
        SELFIES string, or tuple of (SELFIES, validation_info) if return_validation_info=True
        
    Raises:
        InvalidSMILESError: If SMILES string is invalid
        MolecularConversionError: If conversion fails
    """
    if not isinstance(smiles, str):
        raise ValueError("SMILES must be a string")
    
    if not smiles.strip():
        raise ValueError("SMILES string cannot be empty")
    
    smiles = smiles.strip()
    validation_info = {
        'original_smiles': smiles,
        'preprocessed': False,
        'filter_reason': None
    }
    
    # Preprocess SMILES if requested
    if preprocess:
        if filter_config is None:
            filter_config = DEFAULT_FILTER_CONFIG
            
        preprocessed_smiles = preprocess_smiles_for_selfies(smiles, filter_config)
        if preprocessed_smiles is None:
            raise InvalidSMILESError(f"SMILES failed preprocessing: {smiles}")
        
        validation_info['preprocessed'] = True
        validation_info['preprocessed_smiles'] = preprocessed_smiles
        smiles = preprocessed_smiles
    
    # Validate SMILES if requested
    if validate_input:
        if not validate_smiles(smiles, use_filter=False):
            raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")
        
        # Get molecular properties for validation info
        if return_validation_info:
            mol = Chem.MolFromSmiles(smiles)
            validation_info.update({
                'molecular_weight': Descriptors.MolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'is_valid': True
            })
    
    try:
        # Suppress warnings during conversion
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Convert to SELFIES
            selfies = sf.encoder(smiles)
            
            # Verify the conversion by decoding back
            decoded_smiles = sf.decoder(selfies)
            
            # Check if round-trip conversion is consistent
            round_trip_consistent = False
            try:
                if validate_input:
                    canonical_original = Chem.CanonSmiles(smiles)
                    canonical_decoded = Chem.CanonSmiles(decoded_smiles)
                    round_trip_consistent = (canonical_original == canonical_decoded)
                    
                    if not round_trip_consistent:
                        logger.debug(f"Round-trip conversion inconsistency: {smiles} -> {decoded_smiles}")
            except Exception:
                pass
        
        if return_validation_info:
            validation_info['round_trip_consistent'] = round_trip_consistent
            validation_info['selfies'] = selfies
            validation_info['decoded_smiles'] = decoded_smiles
            return selfies, validation_info
        
        return selfies
        
    except Exception as e:
        raise MolecularConversionError(f"Failed to convert SMILES to SELFIES: {str(e)}")


def convert_selfies_to_smiles(
    selfies: str,
    validate_output: bool = True,
    postprocess: bool = True,
    filter_config: Optional[MolecularFilterConfig] = None
) -> str:
    """
    Convert SELFIES string to SMILES representation with validation.
    
    Args:
        selfies: SELFIES string to convert
        validate_output: Whether to validate the resulting SMILES
        postprocess: Whether to postprocess the resulting SMILES
        filter_config: Configuration for molecular filtering
        
    Returns:
        SMILES string
        
    Raises:
        MolecularConversionError: If conversion fails
        InvalidSMILESError: If resulting SMILES is invalid
    """
    if not isinstance(selfies, str):
        raise ValueError("SELFIES must be a string")
    
    if not selfies.strip():
        raise ValueError("SELFIES string cannot be empty")
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            smiles = sf.decoder(selfies.strip())
        
        if validate_output and not validate_smiles(smiles, use_filter=False):
            raise InvalidSMILESError(f"Decoded SMILES is invalid: {smiles}")
        
        # Postprocess if requested
        if postprocess and filter_config is not None:
            processed_smiles = preprocess_smiles_for_selfies(smiles, filter_config)
            if processed_smiles is not None:
                smiles = processed_smiles
        
        return smiles
        
    except Exception as e:
        raise MolecularConversionError(f"Failed to convert SELFIES to SMILES: {str(e)}")


def get_molecular_info(smiles: str, preprocess: bool = True) -> dict:
    """
    Get comprehensive molecular information from a SMILES string.
    
    Args:
        smiles: SMILES string to analyze
        preprocess: Whether to preprocess the SMILES
        
    Returns:
        Dictionary containing molecular properties
        
    Raises:
        InvalidSMILESError: If SMILES string is invalid
    """
    # Preprocess if requested
    original_smiles = smiles
    if preprocess:
        processed_smiles = preprocess_smiles_for_selfies(smiles)
        if processed_smiles is None:
            raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")
        smiles = processed_smiles
    
    if not validate_smiles(smiles):
        raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")
    
    mol = Chem.MolFromSmiles(smiles)
    
    try:
        selfies = convert_smiles_to_selfies(smiles, preprocess=False)
    except Exception:
        selfies = None
    
    info = {
        'original_smiles': original_smiles,
        'smiles': smiles,
        'canonical_smiles': Chem.CanonSmiles(smiles),
        'selfies': selfies,
        'molecular_weight': Descriptors.MolWt(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': Descriptors.RingCount(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'molecular_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'num_hetero_atoms': len([atom for atom in mol.GetAtoms() if atom.GetSymbol() != 'C']),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
    }
    
    # Add filter validation info
    filter_instance = MolecularFilter(DEFAULT_FILTER_CONFIG)
    is_valid, reason, _ = filter_instance.filter_smiles(original_smiles)
    info['passes_default_filter'] = is_valid
    info['filter_reason'] = reason.value
    
    return info


def is_valid_molecule(smiles: str, use_comprehensive_check: bool = True) -> bool:
    """
    Check if a SMILES string represents a chemically valid molecule.
    
    Args:
        smiles: SMILES string to validate
        use_comprehensive_check: Whether to use comprehensive filtering
        
    Returns:
        True if the molecule is valid, False otherwise
    """
    return validate_smiles(smiles, use_filter=use_comprehensive_check)


def batch_filter_smiles(
    smiles_list: List[str],
    filter_config: Optional[MolecularFilterConfig] = None,
    return_reasons: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, List[str]]]]:
    """
    Filter a batch of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings to filter
        filter_config: Configuration for filtering
        return_reasons: Whether to return filtering reasons
        
    Returns:
        List of valid SMILES, or tuple of (valid_smiles, reason_dict) if return_reasons=True
    """
    if filter_config is None:
        filter_config = DEFAULT_FILTER_CONFIG
    
    filter_instance = MolecularFilter(filter_config)
    valid_smiles = []
    reason_dict = {reason.value: [] for reason in MolecularFilterReason}
    
    for smiles in smiles_list:
        is_valid, reason, processed_smiles = filter_instance.filter_smiles(smiles)
        
        if is_valid and processed_smiles:
            valid_smiles.append(processed_smiles)
        
        if return_reasons:
            reason_dict[reason.value].append(smiles)
    
    if return_reasons:
        # Remove empty reason lists
        reason_dict = {k: v for k, v in reason_dict.items() if v}
        return valid_smiles, reason_dict
    
    return valid_smiles


# Convenience aliases for the main conversion functions
smiles_to_selfies = convert_smiles_to_selfies
selfies_to_smiles = convert_selfies_to_smiles

# Export filter configuration for external use
__all__ = [
    'MolecularConversionError',
    'InvalidSMILESError',
    'MolecularFilterConfig',
    'MolecularFilterReason',
    'convert_smiles_to_selfies',
    'convert_selfies_to_smiles',
    'validate_smiles',
    'preprocess_smiles_for_selfies',
    'get_molecular_info',
    'is_valid_molecule',
    'batch_filter_smiles',
    'smiles_to_selfies',
    'selfies_to_smiles',
    'DEFAULT_FILTER_CONFIG'
]
