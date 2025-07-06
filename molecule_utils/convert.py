"""
Molecular utility functions for SMILES and SELFIES conversion.

This module provides robust conversion functions between SMILES and SELFIES
representations of molecules, with comprehensive error handling and validation.
"""
import logging
from typing import Optional, Union, List, Tuple
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configure logging
logger = logging.getLogger(__name__)


class MolecularConversionError(Exception):
    """Custom exception for molecular conversion errors."""
    pass


class InvalidSMILESError(MolecularConversionError):
    """Exception raised when SMILES string is invalid."""
    pass


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string represents a valid molecule.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def convert_smiles_to_selfies(
    smiles: str, 
    validate_input: bool = True,
    return_validation_info: bool = False
) -> Union[str, Tuple[str, dict]]:
    """
    Convert SMILES string to SELFIES representation with robust error handling.
    
    Args:
        smiles: SMILES string to convert
        validate_input: Whether to validate SMILES before conversion
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
    validation_info = {}
    
    # Validate SMILES if requested
    if validate_input:
        if not validate_smiles(smiles):
            raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")
        
        # Get molecular properties for validation info
        if return_validation_info:
            mol = Chem.MolFromSmiles(smiles)
            validation_info = {
                'molecular_weight': Descriptors.MolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'is_valid': True
            }
    
    try:
        # Convert to SELFIES
        selfies = sf.encoder(smiles)
        
        # Verify the conversion by decoding back
        decoded_smiles = sf.decoder(selfies)
        
        # Check if round-trip conversion is consistent
        if validate_input and not Chem.CanonicalSmiles(smiles) == Chem.CanonicalSmiles(decoded_smiles):
            logger.warning(f"Round-trip conversion inconsistency detected for SMILES: {smiles}")
        
        if return_validation_info:
            validation_info['round_trip_consistent'] = (
                Chem.CanonicalSmiles(smiles) == Chem.CanonicalSmiles(decoded_smiles)
            )
            return selfies, validation_info
        
        return selfies
        
    except Exception as e:
        raise MolecularConversionError(f"Failed to convert SMILES to SELFIES: {str(e)}")


def convert_selfies_to_smiles(
    selfies: str,
    validate_output: bool = True
) -> str:
    """
    Convert SELFIES string to SMILES representation with validation.
    
    Args:
        selfies: SELFIES string to convert
        validate_output: Whether to validate the resulting SMILES
        
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
        smiles = sf.decoder(selfies.strip())
        
        if validate_output and not validate_smiles(smiles):
            raise InvalidSMILESError(f"Decoded SMILES is invalid: {smiles}")
        
        return smiles
        
    except Exception as e:
        raise MolecularConversionError(f"Failed to convert SELFIES to SMILES: {str(e)}")


def batch_convert_smiles_to_selfies(
    smiles_list: List[str],
    validate_input: bool = True,
    skip_invalid: bool = False
) -> List[Optional[str]]:
    """
    Convert a batch of SMILES strings to SELFIES with error handling.
    
    Args:
        smiles_list: List of SMILES strings to convert
        validate_input: Whether to validate each SMILES before conversion
        skip_invalid: If True, invalid SMILES will return None instead of raising error
        
    Returns:
        List of SELFIES strings (or None for invalid entries if skip_invalid=True)
        
    Raises:
        ValueError: If input is not a list
        InvalidSMILESError: If any SMILES is invalid and skip_invalid=False
    """
    if not isinstance(smiles_list, list):
        raise ValueError("Input must be a list of SMILES strings")
    
    results = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            selfies = convert_smiles_to_selfies(smiles, validate_input=validate_input)
            results.append(selfies)
            
        except (InvalidSMILESError, MolecularConversionError) as e:
            if skip_invalid:
                logger.warning(f"Skipping invalid SMILES at index {i}: {smiles} - {str(e)}")
                results.append(None)
            else:
                raise
    
    return results


def get_molecular_info(smiles: str) -> dict:
    """
    Get comprehensive molecular information from a SMILES string.
    
    Args:
        smiles: SMILES string to analyze
        
    Returns:
        Dictionary containing molecular properties
        
    Raises:
        InvalidSMILESError: If SMILES string is invalid
    """
    if not validate_smiles(smiles):
        raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")
    
    mol = Chem.MolFromSmiles(smiles)
    selfies = convert_smiles_to_selfies(smiles)
    
    return {
        'smiles': smiles,
        'canonical_smiles': Chem.CanonicalSmiles(smiles),
        'selfies': selfies,
        'molecular_weight': Descriptors.MolWt(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': Descriptors.RingCount(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'molecular_formula': Chem.Descriptors.MolecularFormula(mol),
    }


def is_valid_molecule(smiles: str) -> bool:
    """
    Check if a SMILES string represents a chemically valid molecule.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if the molecule is valid, False otherwise
    """
    return validate_smiles(smiles)


# Convenience alias for the main conversion function
smiles_to_selfies = convert_smiles_to_selfies
selfies_to_smiles = convert_selfies_to_smiles
