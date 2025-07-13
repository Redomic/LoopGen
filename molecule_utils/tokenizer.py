import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter
import pandas as pd
import selfies as sf
import re

class SELFIETokenizer:
    """
    A tokenizer for SELFIES strings that builds vocabulary from actual training data
    using the official `selfies` library.
    """
    def __init__(self, vocab_file: Optional[str] = None, training_data_path: Optional[str] = None):
        self.special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]']
        
        # Load or build vocabulary
        if vocab_file and Path(vocab_file).exists():
            self.vocab = self._load_vocab(vocab_file)
        elif training_data_path:
            self.vocab = self._build_vocab_from_data(training_data_path)
        else:
            # Fallback to minimal working vocab for testing
            self.vocab = self._get_minimal_working_vocab()

        self.token_to_id: Dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: token for i, token in enumerate(self.vocab)}
        
        # Build grammar validation sets for better generation
        self._build_grammar_constraints()

    def _build_vocab_from_data(self, data_path: str, min_frequency: int = 5, max_vocab_size: int = 512) -> List[str]:
        """Build vocabulary from actual SELFIES tokens in the training data."""
        print(f"Building vocabulary from {data_path}...")
        
        token_counter = Counter()
        processed_lines = 0
        
        try:
            # Process data in chunks to handle large files
            chunk_iter = pd.read_csv(data_path, usecols=['SELFIES'], chunksize=10000, on_bad_lines='skip')
            
            for chunk in chunk_iter:
                for selfies_string in chunk['SELFIES']:
                    if isinstance(selfies_string, str) and selfies_string.strip():
                        try:
                            # Use official SELFIES tokenizer
                            tokens = list(sf.split_selfies(selfies_string))
                            token_counter.update(tokens)
                            processed_lines += 1
                        except Exception:
                            continue # Skip lines that cause tokenization errors
                            
                        if processed_lines % 100000 == 0:
                            print(f"Processed {processed_lines:,} molecules...")
            
            print(f"Finished processing {processed_lines:,} molecules")
            print(f"Found {len(token_counter)} unique tokens")
            
            # Filter tokens by frequency and build final vocab
            frequent_tokens = [
                token for token, count in token_counter.most_common()
                if count >= min_frequency and token not in self.special_tokens
            ]
            
            # Limit vocab size
            if len(frequent_tokens) > max_vocab_size - len(self.special_tokens):
                frequent_tokens = frequent_tokens[:max_vocab_size - len(self.special_tokens)]
                print(f"Limited vocabulary to {len(frequent_tokens)} most frequent tokens")
            
            final_vocab = self.special_tokens + frequent_tokens
            print(f"Final vocabulary size: {len(final_vocab)}")
            
            return final_vocab
            
        except Exception as e:
            print(f"Error building vocabulary from data: {e}")
            print("Falling back to minimal working vocabulary")
            return self._get_minimal_working_vocab()

    def _get_minimal_working_vocab(self) -> List[str]:
        """Provides a minimal but functional SELFIES vocabulary for testing."""
        # Core atoms
        atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
        
        # Bonds and structural elements  
        bonds = ['=', '#']
        structure = ['Ring1', 'Ring2', 'Branch1', 'Branch2']
        
        # Charges and common variants
        charges = ['N+', 'O-', 'C+', 'C-']
        
        vocab_tokens = []
        
        # Add bracketed tokens
        for token_set in [atoms, bonds, structure, charges]:
            vocab_tokens.extend([f'[{token}]' for token in token_set])
        
        return self.special_tokens + vocab_tokens

    def _load_vocab(self, vocab_file: str) -> List[str]:
        """Load vocabulary from JSON file."""
        try:
            with open(vocab_file, 'r') as f:
                vocab_data = json.load(f)
                return vocab_data.get('vocab', self._get_minimal_working_vocab())
        except Exception as e:
            print(f"Error loading vocabulary from {vocab_file}: {e}")
            return self._get_minimal_working_vocab()

    def _save_vocab(self, vocab: List[str], save_path: str) -> None:
        """Save vocabulary to JSON file."""
        try:
            vocab_data = {
                'vocab': vocab,
                'vocab_size': len(vocab),
                'special_tokens': self.special_tokens
            }
            with open(save_path, 'w') as f:
                json.dump(vocab_data, f, indent=2)
            print(f"Vocabulary saved to {save_path}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")

    def _build_grammar_constraints(self) -> None:
        """Build grammar constraint sets for better generation."""
        # Atoms that can appear in molecules
        self.atom_tokens = {token for token in self.vocab if re.match(r'\[[A-Z][a-z]?[+\-]?\]', token)}
        
        # Ring tokens
        self.ring_tokens = {token for token in self.vocab if 'Ring' in token}
        
        # Branch tokens  
        self.branch_tokens = {token for token in self.vocab if 'Branch' in token}
        
        # Bond tokens
        self.bond_tokens = {token for token in self.vocab if token in ['[=]', '[#]']}

    def get_valid_next_tokens(self, current_sequence: List[int]) -> Set[int]:
        """
        Get valid next tokens based on SELFIES grammar rules.
        This is a simplified implementation - can be expanded with more sophisticated rules.
        """
        if not current_sequence:
            # At the start, we can begin with atoms or special tokens
            valid_tokens = {self.bos_token_id}
            valid_tokens.update(self.token_to_id.get(token, -1) for token in self.atom_tokens)
            return {tid for tid in valid_tokens if tid != -1}
        
        last_token_id = current_sequence[-1]
        last_token = self.id_to_token.get(last_token_id, '')
        
        valid_token_ids = set()
        
        # Always allow EOS
        valid_token_ids.add(self.eos_token_id)
        
        # Basic rules (can be expanded)
        if last_token in self.atom_tokens:
            # After atoms, we can have bonds, rings, branches, or more atoms
            valid_token_ids.update(self.token_to_id.get(token, -1) for token in 
                                 self.atom_tokens | self.bond_tokens | self.ring_tokens | self.branch_tokens)
        elif last_token in self.bond_tokens:
            # After bonds, we typically expect atoms
            valid_token_ids.update(self.token_to_id.get(token, -1) for token in self.atom_tokens)
        else:
            # Default: allow most tokens
            valid_token_ids.update(self.token_to_id.get(token, -1) for token in 
                                 self.atom_tokens | self.bond_tokens)
        
        # Remove invalid token IDs
        return {tid for tid in valid_token_ids if tid != -1 and tid < len(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id['[PAD]']

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id['[BOS]']

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id['[EOS]']
    
    @property
    def mask_token_id(self) -> int:
        return self.token_to_id['[MASK]']

    def tokenize(self, selfies_string: str) -> List[str]:
        """Splits a SELFIES string into a list of tokens using the official library."""
        if not isinstance(selfies_string, str):
            return []
        try:
            return list(sf.split_selfies(selfies_string))
        except Exception:
            return [] # Return empty list for invalid SELFIES

    def encode(self, selfies_string: str, add_special_tokens: bool = True) -> List[int]:
        """Converts a SELFIES string into a list of token IDs."""
        tokens = self.tokenize(selfies_string)
        
        if add_special_tokens:
            tokens = ['[BOS]'] + tokens + ['[EOS]']
        
        # Map tokens to IDs, using PAD for unknown tokens
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Log unknown tokens for debugging
                token_ids.append(self.pad_token_id)
        
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converts a list of token IDs back into a SELFIES string."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        return "".join(tokens)

    def build_vocab_from_training_data(self, data_path: str, save_path: Optional[str] = None) -> None:
        """Public method to rebuild vocabulary from training data."""
        self.vocab = self._build_vocab_from_data(data_path)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self._build_grammar_constraints()
        
        if save_path:
            self._save_vocab(self.vocab, save_path) 