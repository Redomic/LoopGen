import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Union
from collections import Counter
import pandas as pd
import selfies as sf
import re

class SELFIESTokenizer:
    """
    Production-ready SELFIES tokenizer using official library methods.
    
    This tokenizer follows the official SELFIES patterns:
    - Uses sf.get_alphabet_from_selfies() for vocabulary construction
    - Uses sf.split_selfies() for tokenization
    - Supports the official [nop] padding token
    """
    
    # Standard special tokens
    PAD_TOKEN = '[PAD]'
    START_TOKEN = '[BOS]'
    END_TOKEN = '[EOS]'
    MASK_TOKEN = '[MASK]'
    NOP_TOKEN = '[nop]'  # Official SELFIES no-op token
    
    def __init__(self, vocab_path: Optional[str] = None, data_path: Optional[str] = None):
        self.special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.MASK_TOKEN]
        
        # Build vocabulary
        if vocab_path and Path(vocab_path).exists():
            self.vocabulary = self._load_vocabulary(vocab_path)
        elif data_path:
            self.vocabulary = self._build_vocabulary_from_data(data_path)
        else:
            self.vocabulary = self._get_default_vocabulary()

        # Create bidirectional mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}
        
        # Cache grammar rules for generation
        self._cache_grammar_rules()

    def _build_vocabulary_from_data(self, data_path: str, max_size: int = 1000, max_molecules: Optional[int] = None) -> List[str]:
        """Build vocabulary using official SELFIES alphabet extraction."""
        print(f"Building vocabulary from {data_path}")
        
        selfies_strings = []
        row_count = 0
        
        try:
            for chunk in pd.read_csv(data_path, usecols=['SELFIES'], chunksize=10000, on_bad_lines='skip'):
                for selfies_str in chunk['SELFIES']:
                    if isinstance(selfies_str, str) and selfies_str.strip():
                        try:
                            # Validate by tokenizing
                            list(sf.split_selfies(selfies_str))
                            selfies_strings.append(selfies_str.strip())
                            row_count += 1
                        except Exception:
                            continue
                            
                        if row_count % 100000 == 0:
                            print(f"Processed {row_count:,} molecules")
                            
                        # Only apply limit if specified
                        if max_molecules and len(selfies_strings) >= max_molecules:
                            print(f"Reached molecule limit of {max_molecules:,}")
                            break
                
                if max_molecules and len(selfies_strings) >= max_molecules:
                    break
            
            print(f"Collected {len(selfies_strings):,} valid SELFIES strings")
            
            # Extract alphabet using official method
            alphabet = sf.get_alphabet_from_selfies(selfies_strings)
            print(f"Found {len(alphabet)} unique tokens")
            
            # Add official padding token
            alphabet.add(self.NOP_TOKEN)
            
            # Build final vocabulary
            selfies_tokens = sorted(list(alphabet))
            vocabulary = self.special_tokens + selfies_tokens
            
            # Limit size if needed
            if len(vocabulary) > max_size:
                vocabulary = vocabulary[:max_size]
                print(f"Truncated vocabulary to {len(vocabulary)} tokens")
            
            print(f"Final vocabulary size: {len(vocabulary)}")
            return vocabulary
            
        except Exception as e:
            print(f"Error building vocabulary: {e}")
            return self._get_default_vocabulary()

    def _get_default_vocabulary(self) -> List[str]:
        """Get default vocabulary using official robust alphabet."""
        try:
            robust_alphabet = sf.get_semantic_robust_alphabet()
            selfies_tokens = sorted(list(robust_alphabet))
            vocabulary = self.special_tokens + [self.NOP_TOKEN] + selfies_tokens
            print(f"Using robust alphabet: {len(vocabulary)} tokens")
            return vocabulary
        except Exception as e:
            print(f"Error getting robust alphabet: {e}")
            return self._get_minimal_vocabulary()

    def _get_minimal_vocabulary(self) -> List[str]:
        """Minimal fallback vocabulary for testing."""
        atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
        bonds = ['=', '#']
        rings = ['Ring1', 'Ring2']
        branches = ['Branch1', 'Branch2']
        charges = ['N+', 'O-', 'C+', 'C-']
        
        tokens = []
        for group in [atoms, bonds, rings, branches, charges]:
            tokens.extend([f'[{token}]' for token in group])
        
        tokens.append(self.NOP_TOKEN)
        return self.special_tokens + tokens

    def _load_vocabulary(self, vocab_path: str) -> List[str]:
        """Load vocabulary from JSON file."""
        try:
            with open(vocab_path, 'r') as f:
                data = json.load(f)
                return data.get('vocabulary', self._get_default_vocabulary())
        except Exception as e:
            print(f"Error loading vocabulary from {vocab_path}: {e}")
            return self._get_default_vocabulary()

    def save_vocabulary(self, save_path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            'vocabulary': self.vocabulary,
            'size': len(self.vocabulary),
            'special_tokens': self.special_tokens,
            'nop_token': self.NOP_TOKEN
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Vocabulary saved to {save_path}")

    def _cache_grammar_rules(self) -> None:
        """Cache token categories for grammar-aware generation."""
        self.atom_tokens = set()
        self.bond_tokens = set()
        self.ring_tokens = set()
        self.branch_tokens = set()
        
        for token in self.vocabulary:
            if token in self.special_tokens or token == self.NOP_TOKEN:
                continue
                
            if 'Ring' in token:
                self.ring_tokens.add(token)
            elif 'Branch' in token:
                self.branch_tokens.add(token)
            elif token in ['[=]', '[#]', '[=O]', '[=N]', '[=C]', '[=S]']:
                self.bond_tokens.add(token)
            elif re.match(r'\[[A-Za-z][A-Za-z0-9]*[+-]?\]', token):
                self.atom_tokens.add(token)

    def get_valid_next_tokens(self, current_sequence: List[int]) -> Set[int]:
        """Get valid next tokens based on SELFIES grammar."""
        if not current_sequence:
            valid_tokens = {self.start_token_id}
            valid_tokens.update(self.token_to_id.get(token, -1) for token in self.atom_tokens)
            return {tid for tid in valid_tokens if tid != -1}
        
        last_token_id = current_sequence[-1]
        last_token = self.id_to_token.get(last_token_id, '')
        
        valid_ids = set()
        valid_ids.add(self.end_token_id)
        
        if self.NOP_TOKEN in self.token_to_id:
            valid_ids.add(self.token_to_id[self.NOP_TOKEN])
        
        if last_token in self.atom_tokens:
            valid_ids.update(self.token_to_id.get(token, -1) for token in 
                           self.atom_tokens | self.bond_tokens | self.ring_tokens | self.branch_tokens)
        elif last_token in self.bond_tokens:
            valid_ids.update(self.token_to_id.get(token, -1) for token in self.atom_tokens)
        else:
            valid_ids.update(self.token_to_id.get(token, -1) for token in 
                           self.atom_tokens | self.bond_tokens)
        
        return {tid for tid in valid_ids if tid != -1 and tid < len(self.vocabulary)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def start_token_id(self) -> int:
        return self.token_to_id[self.START_TOKEN]

    @property
    def end_token_id(self) -> int:
        return self.token_to_id[self.END_TOKEN]
    
    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.MASK_TOKEN]
    
    @property
    def nop_token_id(self) -> int:
        return self.token_to_id.get(self.NOP_TOKEN, self.pad_token_id)

    # Legacy property names for backward compatibility
    @property
    def bos_token_id(self) -> int:
        return self.start_token_id

    @property
    def eos_token_id(self) -> int:
        return self.end_token_id

    def tokenize(self, selfies_string: str) -> List[str]:
        """Tokenize SELFIES string using official method."""
        if not isinstance(selfies_string, str) or not selfies_string.strip():
            return []
        
        try:
            return list(sf.split_selfies(selfies_string.strip()))
        except Exception:
            return []

    def encode(self, selfies_string: str, add_special_tokens: bool = True) -> List[int]:
        """Convert SELFIES string to token IDs."""
        tokens = self.tokenize(selfies_string)
        
        if add_special_tokens:
            tokens = [self.START_TOKEN] + tokens + [self.END_TOKEN]
        
        token_ids = []
        for token in tokens:
            token_ids.append(self.token_to_id.get(token, self.pad_token_id))
        
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to SELFIES string."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                if token == self.NOP_TOKEN:
                    continue  # Always skip nop tokens
                tokens.append(token)
        
        return "".join(tokens)

    def encode_with_padding(self, selfies_string: str, max_length: int) -> List[int]:
        """Encode with padding to fixed length."""
        token_ids = self.encode(selfies_string, add_special_tokens=True)
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.nop_token_id] * (max_length - len(token_ids)))
        
        return token_ids

    def build_vocabulary_from_data(self, data_path: str, save_path: Optional[str] = None, max_molecules: Optional[int] = None) -> None:
        """Build vocabulary from training data."""
        vocabulary = self._build_vocabulary_from_data(data_path, max_molecules=max_molecules)
        self.vocabulary = vocabulary
        
        # Rebuild mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}
        
        # Rebuild grammar cache
        self._cache_grammar_rules()
        
        if save_path:
            self.save_vocabulary(save_path)

    def get_alphabet_from_data(self, data_path: str) -> Set[str]:
        """Extract unique tokens from dataset."""
        selfies_strings = []
        
        try:
            for chunk in pd.read_csv(data_path, usecols=['SELFIES'], chunksize=10000, on_bad_lines='skip'):
                for selfies_str in chunk['SELFIES']:
                    if isinstance(selfies_str, str) and selfies_str.strip():
                        try:
                            list(sf.split_selfies(selfies_str))
                            selfies_strings.append(selfies_str.strip())
                        except Exception:
                            continue
                            
                        if len(selfies_strings) >= 100000:
                            break
                
                if len(selfies_strings) >= 100000:
                    break
            
            return sf.get_alphabet_from_selfies(selfies_strings)
            
        except Exception as e:
            print(f"Error extracting alphabet: {e}")
            return set()

    def is_valid_selfies(self, selfies_string: str) -> bool:
        """Validate SELFIES string."""
        try:
            tokens = list(sf.split_selfies(selfies_string))
            return len(tokens) > 0
        except Exception:
            return False 