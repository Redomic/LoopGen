"""
SMILES Validator for Post-Generation Quality Checking.

Comprehensive SMILES validation following OpenSMILES specification.
Used to analyze and debug generation quality.
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re


class SMILESValidator:
    """
    Comprehensive SMILES syntax validator following OpenSMILES spec.
    
    Checks for common SMILES violations that cause RDKit parsing failures:
    1. Bracket balance
    2. Parenthesis balance
    3. Ring closure completeness
    4. Aromatic atoms in valid contexts
    5. Stereochemistry placement
    6. Bond symbol positions
    7. Valid token sequences
    
    Example:
        >>> validator = SMILESValidator()
        >>> is_valid, violations = validator.validate("CCO")
        >>> print(is_valid)  # True
        >>> is_valid, violations = validator.validate("/Br/[OH+]")
        >>> print(violations)  # ['starts_with_stereo']
    """
    
    def __init__(self):
        """Initialize validator with grammar rules."""
        # Define valid tokens
        self.atoms = set(['B', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
        self.aromatic_atoms = set(['b', 'c', 'n', 'o', 's', 'p'])
        self.bonds = set(['=', '#', '-'])
        self.stereo = set(['/', '\\'])
        
    def validate(self, smiles: str) -> Tuple[bool, List[str]]:
        """
        Validate SMILES syntax.
        
        Args:
            smiles: SMILES string to validate
        
        Returns:
            Tuple of (is_valid, list_of_violations)
            is_valid is True if no violations found
            list_of_violations contains string identifiers for each violation
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return False, ['empty_or_invalid']
        
        smiles = smiles.strip()
        violations = []
        
        # Check for basic syntax violations
        violations.extend(self._check_brackets(smiles))
        violations.extend(self._check_parentheses(smiles))
        violations.extend(self._check_ring_closures(smiles))
        violations.extend(self._check_bond_positions(smiles))
        violations.extend(self._check_stereochemistry(smiles))
        violations.extend(self._check_start_end(smiles))
        violations.extend(self._check_aromatic_atoms(smiles))
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _check_brackets(self, smiles: str) -> List[str]:
        """Check bracket balance."""
        violations = []
        
        balance = 0
        for char in smiles:
            if char == '[':
                balance += 1
            elif char == ']':
                balance -= 1
                if balance < 0:
                    violations.append('bracket_close_without_open')
                    break
        
        if balance > 0:
            violations.append('unclosed_brackets')
        elif balance < 0:
            violations.append('extra_bracket_close')
        
        # Check for empty brackets
        if '[]' in smiles:
            violations.append('empty_brackets')
        
        return violations
    
    def _check_parentheses(self, smiles: str) -> List[str]:
        """Check parenthesis balance."""
        violations = []
        
        balance = 0
        for char in smiles:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                if balance < 0:
                    violations.append('paren_close_without_open')
                    break
        
        if balance > 0:
            violations.append('unclosed_parentheses')
        elif balance < 0:
            violations.append('extra_paren_close')
        
        # Check for empty parentheses
        if '()' in smiles:
            violations.append('empty_parentheses')
        
        return violations
    
    def _check_ring_closures(self, smiles: str) -> List[str]:
        """Check ring closures are properly matched."""
        violations = []
        
        # Find all ring digits (including %10-%99)
        ring_pattern = r'%?\d+'
        open_rings = defaultdict(int)
        
        for match in re.finditer(ring_pattern, smiles):
            digit_str = match.group()
            digit = int(digit_str[1:]) if digit_str.startswith('%') else int(digit_str)
            open_rings[digit] += 1
        
        # Check for unclosed or duplicate rings
        for digit, count in open_rings.items():
            if count == 1:
                violations.append(f'unclosed_ring_{digit}')
            elif count > 2:
                violations.append(f'duplicate_ring_{digit}')
        
        return violations
    
    def _check_bond_positions(self, smiles: str) -> List[str]:
        """Check bond symbols are in valid positions."""
        violations = []
        
        # Check for bonds at start
        if smiles and smiles[0] in self.bonds:
            violations.append('starts_with_bond')
        
        # Check for bonds at end
        if smiles and smiles[-1] in self.bonds:
            violations.append('ends_with_bond')
        
        # Check for consecutive bonds
        for bond in self.bonds:
            if bond + bond in smiles:
                violations.append('consecutive_bonds')
                break
        
        # Check for bonds after opening paren
        if '(=' in smiles or '(#' in smiles or '(-' in smiles:
            violations.append('bond_after_open_paren')
        
        # Check for bonds before closing paren
        if '=)' in smiles or '#)' in smiles or '-)' in smiles:
            violations.append('bond_before_close_paren')
        
        return violations
    
    def _check_stereochemistry(self, smiles: str) -> List[str]:
        """Check stereochemistry markers (/ and \) placement."""
        violations = []
        
        # Check for stereo at start
        if smiles and (smiles[0] == '/' or smiles[0] == '\\'):
            violations.append('starts_with_stereo')
        
        # Check for stereo at end
        if smiles and (smiles[-1] == '/' or smiles[-1] == '\\'):
            violations.append('ends_with_stereo')
        
        # Check for consecutive stereo
        if '//' in smiles or '\\\\' in smiles or '/\\' in smiles or '\\/' in smiles:
            violations.append('consecutive_stereo')
        
        return violations
    
    def _check_start_end(self, smiles: str) -> List[str]:
        """Check valid start and end tokens."""
        violations = []
        
        if not smiles:
            return violations
        
        # Valid start characters: atoms (including aromatic and brackets)
        valid_start = self.atoms | self.aromatic_atoms | {'['}
        
        if smiles[0] not in valid_start and not smiles[0].isupper():
            violations.append('invalid_start_token')
        
        # Check for closing brackets/parens at start
        if smiles[0] in {']', ')'}:
            violations.append('starts_with_closure')
        
        # Check for ring digits at start
        if smiles[0].isdigit() or smiles.startswith('%'):
            violations.append('starts_with_ring_digit')
        
        return violations
    
    def _check_aromatic_atoms(self, smiles: str) -> List[str]:
        """Check aromatic atoms (rough heuristic - full validation requires parsing)."""
        violations = []
        
        # This is a heuristic check - full validation requires molecular graph
        # We just check if aromatic atoms appear without ring context
        
        # Count aromatic atoms
        aromatic_count = sum(1 for char in smiles if char in self.aromatic_atoms)
        
        # Count ring openings (digits)
        ring_count = len(re.findall(r'\d', smiles))
        
        # Heuristic: if we have aromatic atoms but no rings, likely invalid
        if aromatic_count > 3 and ring_count == 0:
            violations.append('aromatic_without_rings')
        
        return violations
    
    def get_violation_description(self, violation: str) -> str:
        """Get human-readable description of a violation."""
        descriptions = {
            'empty_or_invalid': 'SMILES is empty or not a string',
            'bracket_close_without_open': 'Closing bracket ] without matching [',
            'unclosed_brackets': 'Opening bracket [ not closed',
            'extra_bracket_close': 'Extra closing bracket ]',
            'empty_brackets': 'Empty brackets [] found',
            'paren_close_without_open': 'Closing parenthesis ) without matching (',
            'unclosed_parentheses': 'Opening parenthesis ( not closed',
            'extra_paren_close': 'Extra closing parenthesis )',
            'empty_parentheses': 'Empty parentheses () found',
            'starts_with_bond': 'SMILES starts with bond symbol (=, #, -)',
            'ends_with_bond': 'SMILES ends with bond symbol',
            'consecutive_bonds': 'Consecutive bond symbols found',
            'bond_after_open_paren': 'Bond symbol immediately after (',
            'bond_before_close_paren': 'Bond symbol immediately before )',
            'starts_with_stereo': 'SMILES starts with stereochemistry marker (/, \\)',
            'ends_with_stereo': 'SMILES ends with stereochemistry marker',
            'consecutive_stereo': 'Consecutive stereochemistry markers found',
            'invalid_start_token': 'SMILES starts with invalid token',
            'starts_with_closure': 'SMILES starts with closing bracket or paren',
            'starts_with_ring_digit': 'SMILES starts with ring digit',
            'aromatic_without_rings': 'Aromatic atoms present without ring structures',
        }
        
        # Handle ring-specific violations
        if violation.startswith('unclosed_ring_'):
            return f'Ring closure {violation.split("_")[-1]} is unclosed'
        elif violation.startswith('duplicate_ring_'):
            return f'Ring digit {violation.split("_")[-1]} used more than twice'
        
        return descriptions.get(violation, f'Unknown violation: {violation}')
    
    def generate_report(self, smiles_list: List[str]) -> Dict:
        """
        Generate validation report for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            Dictionary with:
            - total: Total number of SMILES
            - valid: Number of valid SMILES
            - invalid: Number of invalid SMILES
            - validity_rate: Percentage of valid SMILES
            - violation_counts: Dictionary of violation types and their counts
            - examples: Dictionary mapping violations to example SMILES
        """
        total = len(smiles_list)
        valid = 0
        violation_counts = defaultdict(int)
        violation_examples = defaultdict(list)
        
        for smiles in smiles_list:
            is_valid, violations = self.validate(smiles)
            
            if is_valid:
                valid += 1
            else:
                for violation in violations:
                    violation_counts[violation] += 1
                    # Store up to 3 examples per violation type
                    if len(violation_examples[violation]) < 3:
                        violation_examples[violation].append(smiles)
        
        invalid = total - valid
        validity_rate = (valid / total * 100) if total > 0 else 0.0
        
        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'validity_rate': validity_rate,
            'violation_counts': dict(violation_counts),
            'violation_examples': dict(violation_examples)
        }
    
    def print_report(self, report: Dict):
        """Print a formatted validation report."""
        print("=" * 60)
        print("SMILES Validation Report")
        print("=" * 60)
        print(f"Total SMILES: {report['total']}")
        print(f"Valid: {report['valid']} ({report['validity_rate']:.1f}%)")
        print(f"Invalid: {report['invalid']} ({100-report['validity_rate']:.1f}%)")
        print()
        
        if report['violation_counts']:
            print("Violation Breakdown:")
            print("-" * 60)
            
            # Sort violations by frequency
            sorted_violations = sorted(
                report['violation_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for violation, count in sorted_violations:
                pct = (count / report['total'] * 100)
                desc = self.get_violation_description(violation)
                print(f"  {violation}: {count} ({pct:.1f}%)")
                print(f"    {desc}")
                
                # Show examples
                if violation in report['violation_examples']:
                    examples = report['violation_examples'][violation]
                    for i, example in enumerate(examples[:2], 1):
                        print(f"    Example {i}: {example[:60]}...")
                
                print()
        
        print("=" * 60)

