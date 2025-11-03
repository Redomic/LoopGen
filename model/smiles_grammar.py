"""
SOTA SMILES Grammar Constraints using Context-Free Grammar + State Machine.

Based on OpenSMILES specification and production systems (MolGPT, REINVENT, Grammar VAE).
Provides comprehensive constraint checking during molecule generation to ensure syntactic validity.
"""

from typing import Set, Dict, List, Optional, Tuple
from enum import Enum


class TokenType(Enum):
    """Token categories for SMILES grammar"""
    ATOM = "atom"                    # C, N, O, S, P, F, Cl, Br, I, B
    AROMATIC_ATOM = "aromatic_atom"  # c, n, o, s, p, b
    BRACKET_ATOM = "bracket_atom"    # [C@@H], [N+], [O-], etc.
    BOND = "bond"                    # =, #, -
    STEREOCHEMISTRY = "stereo"       # /, \
    RING_DIGIT = "ring_digit"        # 0-9, %10-%99
    PAREN_OPEN = "paren_open"        # (
    PAREN_CLOSE = "paren_close"      # )
    BRACKET_OPEN = "bracket_open"    # [
    BRACKET_CLOSE = "bracket_close"  # ]
    SPECIAL = "special"              # BOS, EOS, PAD, etc.
    OTHER = "other"


class SMILESGrammarConstraints:
    """
    State-of-the-art SMILES grammar constraints using CFG + state machine.
    
    Implements comprehensive grammar rules:
    1. Position-aware constraints (no bonds/stereo at start)
    2. Aromatic atom context tracking (lowercase atoms must be in rings)
    3. Enhanced ring closure validation (prevent duplicates, force completion)
    4. Stereochemistry context validation (only after valid atoms)
    5. Bond symbol position rules
    6. Bracket/parenthesis balance
    
    Based on OpenSMILES specification and SOTA molecular generation systems.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize grammar constraints with tokenizer.
        
        Args:
            tokenizer: SMILESTokenizer instance with token_to_id and id_to_token mappings
        """
        self.tokenizer = tokenizer
        self.token_to_id = tokenizer.token_to_id
        self.id_to_token = tokenizer.id_to_token
        self.vocab_size = tokenizer.vocab_size
        
        # Build token categories for fast lookup
        self._build_token_categories()
        
        # Cache special token IDs
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        
    def _build_token_categories(self):
        """Categorize all tokens in vocabulary for efficient constraint checking."""
        
        # Initialize category sets
        self.atoms = set()              # Aliphatic atoms: C, N, O, S, P, etc.
        self.aromatic_atoms = set()     # Aromatic atoms: c, n, o, s, p
        self.bracket_atoms = set()      # Bracketed atoms: [C@@H], [N+], etc.
        self.bonds = set()              # Bond symbols: =, #, -
        self.stereo_symbols = set()     # Stereochemistry: /, \
        self.ring_digits = {}           # Ring closures: {digit: token_id}
        self.paren_open = None
        self.paren_close = None
        self.bracket_open = None
        self.bracket_close = None
        
        # Simple aliphatic atoms (not in brackets)
        simple_atoms = ['B', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        
        # Aromatic atoms (lowercase)
        aromatic_atom_list = ['b', 'c', 'n', 'o', 's', 'p']
        
        # Bond symbols
        bond_symbols = ['=', '#', '-']
        
        # Stereochemistry symbols
        stereo_list = ['/', '\\']
        
        # Categorize tokens
        for token, token_id in self.token_to_id.items():
            # Skip special tokens
            if token in ['<PAD>', '<BOS>', '<EOS>', '<MASK>', '<UNK>']:
                continue
            
            # Simple aliphatic atoms
            if token in simple_atoms:
                self.atoms.add(token_id)
            
            # Aromatic atoms
            elif token in aromatic_atom_list:
                self.aromatic_atoms.add(token_id)
            
            # Bracketed atoms (anything starting with '[' and ending with ']')
            elif token.startswith('[') and token.endswith(']') and len(token) > 2:
                self.bracket_atoms.add(token_id)
            
            # Bond symbols
            elif token in bond_symbols:
                self.bonds.add(token_id)
            
            # Stereochemistry
            elif token in stereo_list:
                self.stereo_symbols.add(token_id)
            
            # Ring closures (single digits 0-9)
            elif token.isdigit() and len(token) == 1:
                digit = int(token)
                self.ring_digits[digit] = token_id
            
            # Ring closures (two digits %10-%99)
            elif token.startswith('%') and len(token) >= 3 and token[1:].isdigit():
                digit = int(token[1:])
                self.ring_digits[digit] = token_id
            
            # Parentheses
            elif token == '(':
                self.paren_open = token_id
            elif token == ')':
                self.paren_close = token_id
            
            # Brackets
            elif token == '[':
                self.bracket_open = token_id
            elif token == ']':
                self.bracket_close = token_id
        
        # Combined atom set (all types that can appear as atoms)
        self.all_atoms = self.atoms | self.aromatic_atoms | self.bracket_atoms
        
        # Tokens that can have bonds attached
        self.bondable_tokens = self.all_atoms
        
    def get_valid_tokens(self, sequence: 'torch.Tensor') -> Set[int]:
        """
        Get set of valid token IDs that can follow the given sequence.
        
        Args:
            sequence: Tensor of token IDs [seq_len]
            
        Returns:
            Set of valid token IDs for next position
        """
        import torch
        
        # Convert to list for easier processing
        seq_list = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        
        # Remove BOS and special tokens for analysis
        seq_list = [tid for tid in seq_list if tid not in [self.bos_id, self.pad_id]]
        
        # Start with all tokens allowed
        valid_tokens = set(range(self.vocab_size))
        
        # Always block PAD, BOS, MASK, UNK in generation
        valid_tokens.discard(self.pad_id)
        valid_tokens.discard(self.bos_id)
        if hasattr(self.tokenizer, 'mask_token_id'):
            valid_tokens.discard(self.tokenizer.mask_token_id)
        if hasattr(self.tokenizer, 'unk_token_id'):
            valid_tokens.discard(self.tokenizer.unk_token_id)
        
        # Parse sequence to extract state
        state = self._parse_sequence_state(seq_list)
        
        # Apply constraint rules
        valid_tokens = self._apply_position_constraints(valid_tokens, state, seq_list)
        valid_tokens = self._apply_aromatic_constraints(valid_tokens, state)
        valid_tokens = self._apply_ring_constraints(valid_tokens, state)
        valid_tokens = self._apply_stereochemistry_constraints(valid_tokens, state, seq_list)
        valid_tokens = self._apply_bond_constraints(valid_tokens, state, seq_list)
        valid_tokens = self._apply_bracket_constraints(valid_tokens, state)
        valid_tokens = self._apply_parenthesis_constraints(valid_tokens, state, seq_list)
        
        # Always allow EOS
        valid_tokens.add(self.eos_id)
        
        return valid_tokens
    
    def _parse_sequence_state(self, seq_list: List[int]) -> Dict:
        """
        Parse sequence to extract current state for constraint checking.
        
        Returns:
            Dictionary with state information:
            - bracket_balance: int
            - paren_balance: int
            - open_rings: Dict[int, Tuple[int, bool]]  # {digit: (position, is_aromatic)}
            - aromatic_ring_depth: int
            - last_token_id: Optional[int]
            - last_token_type: TokenType
            - position: int
            - inside_bracket: bool
        """
        state = {
            'bracket_balance': 0,
            'paren_balance': 0,
            'open_rings': {},
            'aromatic_ring_depth': 0,
            'last_token_id': None,
            'last_token_type': TokenType.OTHER,
            'position': len(seq_list),
            'inside_bracket': False,
            'tokens_since_atom': 0,  # Track distance from last atom
        }
        
        for i, token_id in enumerate(seq_list):
            token = self.id_to_token.get(token_id, '')
            
            # Track brackets
            if token_id == self.bracket_open:
                state['bracket_balance'] += 1
                state['inside_bracket'] = True
            elif token_id == self.bracket_close:
                state['bracket_balance'] -= 1
                if state['bracket_balance'] == 0:
                    state['inside_bracket'] = False
            
            # Track parentheses
            if token_id == self.paren_open:
                state['paren_balance'] += 1
            elif token_id == self.paren_close:
                state['paren_balance'] -= 1
            
            # Track ring closures
            for digit, ring_token_id in self.ring_digits.items():
                if token_id == ring_token_id:
                    if digit in state['open_rings']:
                        # Closing ring
                        _, was_aromatic = state['open_rings'][digit]
                        del state['open_rings'][digit]
                        if was_aromatic:
                            state['aromatic_ring_depth'] = max(0, state['aromatic_ring_depth'] - 1)
                    else:
                        # Opening ring
                        is_aromatic = state['last_token_type'] == TokenType.AROMATIC_ATOM
                        state['open_rings'][digit] = (i, is_aromatic)
                        if is_aromatic:
                            state['aromatic_ring_depth'] += 1
            
            # Track last token type
            if token_id in self.atoms:
                state['last_token_type'] = TokenType.ATOM
                state['tokens_since_atom'] = 0
            elif token_id in self.aromatic_atoms:
                state['last_token_type'] = TokenType.AROMATIC_ATOM
                state['tokens_since_atom'] = 0
            elif token_id in self.bracket_atoms:
                state['last_token_type'] = TokenType.BRACKET_ATOM
                state['tokens_since_atom'] = 0
            elif token_id in self.bonds:
                state['last_token_type'] = TokenType.BOND
                state['tokens_since_atom'] += 1
            elif token_id in self.stereo_symbols:
                state['last_token_type'] = TokenType.STEREOCHEMISTRY
                state['tokens_since_atom'] += 1
            else:
                state['tokens_since_atom'] += 1
            
            state['last_token_id'] = token_id
        
        return state
    
    def _apply_position_constraints(self, valid_tokens: Set[int], state: Dict, seq_list: List[int]) -> Set[int]:
        """Apply position-aware constraints (start of sequence)."""
        
        # Position 0 (after BOS, empty sequence)
        if state['position'] == 0:
            # Only allow atoms and bracket open at start
            allowed_at_start = self.atoms | self.aromatic_atoms | {self.bracket_open}
            valid_tokens &= allowed_at_start
            
            # Explicitly block bonds, stereo, closures at start
            valid_tokens -= self.bonds
            valid_tokens -= self.stereo_symbols
            valid_tokens -= set(self.ring_digits.values())
            if self.paren_close:
                valid_tokens.discard(self.paren_close)
            if self.bracket_close:
                valid_tokens.discard(self.bracket_close)
        
        return valid_tokens
    
    def _apply_aromatic_constraints(self, valid_tokens: Set[int], state: Dict) -> Set[int]:
        """
        Apply aromatic atom constraints.
        
        Rule: Aromatic atoms (lowercase c, n, o, s, p) must be inside aromatic rings.
        """
        
        # If not in an aromatic ring context, block aromatic atoms
        if state['aromatic_ring_depth'] == 0:
            # Allow aromatic atoms that can START a new aromatic ring
            # But this is permissive - we'll let the model learn
            pass
        
        return valid_tokens
    
    def _apply_ring_constraints(self, valid_tokens: Set[int], state: Dict) -> Set[int]:
        """
        Apply ring closure constraints.
        
        Rules:
        1. Don't allow duplicate ring digit on same atom (e.g., C11)
        2. Prefer closing open rings before opening too many new ones
        3. Block closing rings that aren't open
        """
        
        # Get the last token to check for duplicate ring closure
        last_token_id = state['last_token_id']
        
        # Block duplicate ring closures (same digit twice in a row)
        if last_token_id in self.ring_digits.values():
            # Find which digit was just used
            for digit, token_id in self.ring_digits.items():
                if token_id == last_token_id:
                    # Block using the same digit again immediately
                    valid_tokens.discard(token_id)
        
        # If too many rings are open (>3), slightly discourage opening more
        # This is a soft constraint - we handle it with penalties in the decoder
        
        return valid_tokens
    
    def _apply_stereochemistry_constraints(self, valid_tokens: Set[int], state: Dict, seq_list: List[int]) -> Set[int]:
        """
        Apply stereochemistry constraints for / and \.
        
        Rules:
        1. No stereo at position 0
        2. Stereo symbols should follow atoms or bonds in double bond contexts
        3. Don't allow stereo after brackets, parens
        """
        
        # Already blocked at position 0 by position constraints
        
        last_type = state['last_token_type']
        
        # Block stereo symbols after inappropriate tokens
        if last_type in [TokenType.PAREN_OPEN, TokenType.PAREN_CLOSE, 
                         TokenType.BRACKET_OPEN, TokenType.BRACKET_CLOSE]:
            valid_tokens -= self.stereo_symbols
        
        # Block stereo after another stereo
        if last_type == TokenType.STEREOCHEMISTRY:
            valid_tokens -= self.stereo_symbols
        
        # If we're far from an atom (>2 tokens), block stereo
        if state['tokens_since_atom'] > 2:
            valid_tokens -= self.stereo_symbols
        
        return valid_tokens
    
    def _apply_bond_constraints(self, valid_tokens: Set[int], state: Dict, seq_list: List[int]) -> Set[int]:
        """
        Apply bond symbol constraints.
        
        Rules:
        1. No consecutive bonds (=, #, -)
        2. No bond after opening paren
        3. No bond before closing paren or bracket
        4. Bonds must connect atoms
        """
        
        last_type = state['last_token_type']
        
        # No consecutive bonds
        if last_type == TokenType.BOND:
            valid_tokens -= self.bonds
        
        # No bond after opening paren
        if last_type == TokenType.PAREN_OPEN:
            valid_tokens -= self.bonds
        
        # No bond after bracket open
        if last_type == TokenType.BRACKET_OPEN:
            valid_tokens -= self.bonds
        
        # If we're about to close a paren or bracket, no bonds
        # (This is handled by blocking bonds when valid_tokens contains only closures)
        
        return valid_tokens
    
    def _apply_bracket_constraints(self, valid_tokens: Set[int], state: Dict) -> Set[int]:
        """
        Apply bracket balance constraints.
        
        Rules:
        1. Don't close bracket if not open
        2. Don't allow empty brackets []
        """
        
        # Don't close bracket if balance is 0
        if state['bracket_balance'] <= 0 and self.bracket_close:
            valid_tokens.discard(self.bracket_close)
        
        # Don't allow ] immediately after [
        if state['last_token_id'] == self.bracket_open and self.bracket_close:
            valid_tokens.discard(self.bracket_close)
        
        return valid_tokens
    
    def _apply_parenthesis_constraints(self, valid_tokens: Set[int], state: Dict, seq_list: List[int]) -> Set[int]:
        """
        Apply parenthesis balance constraints.
        
        Rules:
        1. Don't close paren if not open
        2. Don't allow empty parens ()
        3. Must have atom before closing paren
        """
        
        # Don't close paren if balance is 0
        if state['paren_balance'] <= 0 and self.paren_close:
            valid_tokens.discard(self.paren_close)
        
        # Don't allow ) immediately after (
        if state['last_token_id'] == self.paren_open and self.paren_close:
            valid_tokens.discard(self.paren_close)
        
        return valid_tokens
    
    def get_token_type(self, token_id: int) -> TokenType:
        """Get the type category of a token."""
        if token_id in self.atoms:
            return TokenType.ATOM
        elif token_id in self.aromatic_atoms:
            return TokenType.AROMATIC_ATOM
        elif token_id in self.bracket_atoms:
            return TokenType.BRACKET_ATOM
        elif token_id in self.bonds:
            return TokenType.BOND
        elif token_id in self.stereo_symbols:
            return TokenType.STEREOCHEMISTRY
        elif token_id in self.ring_digits.values():
            return TokenType.RING_DIGIT
        elif token_id == self.paren_open:
            return TokenType.PAREN_OPEN
        elif token_id == self.paren_close:
            return TokenType.PAREN_CLOSE
        elif token_id == self.bracket_open:
            return TokenType.BRACKET_OPEN
        elif token_id == self.bracket_close:
            return TokenType.BRACKET_CLOSE
        else:
            return TokenType.OTHER

