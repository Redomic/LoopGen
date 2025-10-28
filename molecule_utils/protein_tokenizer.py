"""
Protein tokenizer for amino acid sequences.

This module provides tokenization for protein sequences, converting amino acid
letters into token IDs suitable for neural network input. Handles the 20 standard
amino acids plus special tokens for padding, boundaries, and masking.
"""

from typing import List, Optional
import json


class ProteinTokenizer:
    """
    Tokenizer for protein sequences (amino acids).
    
    Converts protein sequences into token IDs for neural network processing.
    Supports the 20 standard amino acids plus special tokens.
    
    Vocabulary:
        - Special tokens: <PAD>, <BOS>, <EOS>, <UNK>, <MASK>
        - Amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    
    Example:
        >>> tokenizer = ProteinTokenizer()
        >>> tokens = tokenizer.encode("MKTAYIAK")
        >>> print(tokens)  # [1, 15, 14, 5, 24, 13, 5, 12, 2]
        >>> sequence = tokenizer.decode(tokens)
        >>> print(sequence)  # "MKTAYIAK"
    """
    
    # Standard 20 amino acids (sorted alphabetically)
    AMINO_ACIDS = [
        'A',  # Alanine
        'C',  # Cysteine
        'D',  # Aspartic acid
        'E',  # Glutamic acid
        'F',  # Phenylalanine
        'G',  # Glycine
        'H',  # Histidine
        'I',  # Isoleucine
        'K',  # Lysine
        'L',  # Leucine
        'M',  # Methionine
        'N',  # Asparagine
        'P',  # Proline
        'Q',  # Glutamine
        'R',  # Arginine
        'S',  # Serine
        'T',  # Threonine
        'V',  # Valine
        'W',  # Tryptophan
        'Y'   # Tyrosine
    ]
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    MASK_TOKEN = '<MASK>'
    
    def __init__(self):
        """Initialize the protein tokenizer with vocabulary."""
        # Build vocabulary: special tokens first, then amino acids
        self.special_tokens = [
            self.PAD_TOKEN, 
            self.BOS_TOKEN, 
            self.EOS_TOKEN,
            self.UNK_TOKEN, 
            self.MASK_TOKEN
        ]
        
        self.vocabulary = self.special_tokens + self.AMINO_ACIDS
        
        # Create bidirectional mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}
        
        # Cache special token IDs
        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_token_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]
        self.mask_token_id = self.token_to_id[self.MASK_TOKEN]
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocabulary)
    
    def encode(
        self, 
        protein_sequence: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Convert protein sequence string to token IDs.
        
        Args:
            protein_sequence: String of amino acid letters (e.g., "MKTAYIAK")
            add_special_tokens: Whether to add BOS and EOS tokens
            max_length: Maximum sequence length (truncates if exceeded)
        
        Returns:
            List of token IDs
        
        Example:
            >>> tokenizer.encode("MKTAY")
            [1, 15, 14, 5, 24, 2]  # [BOS, M, K, T, A, Y, EOS]
        """
        # Convert sequence to uppercase and remove whitespace
        sequence = protein_sequence.upper().strip()
        
        # Tokenize each amino acid
        tokens = []
        for aa in sequence:
            if aa in self.token_to_id:
                tokens.append(self.token_to_id[aa])
            else:
                # Unknown amino acid - use UNK token
                tokens.append(self.unk_token_id)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        # Truncate if max_length specified
        if max_length is not None and len(tokens) > max_length:
            if add_special_tokens:
                # Keep BOS and EOS, truncate middle
                tokens = [tokens[0]] + tokens[1:max_length-1] + [tokens[-1]]
            else:
                tokens = tokens[:max_length]
        
        return tokens
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Convert token IDs back to protein sequence string.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Protein sequence string (e.g., "MKTAYIAK")
        
        Example:
            >>> tokenizer.decode([1, 15, 14, 5, 24, 2])
            "MKTA Y"  # (with skip_special_tokens=True)
        """
        sequence = []
        
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                continue  # Skip invalid token IDs
            
            token = self.id_to_token[token_id]
            
            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            sequence.append(token)
        
        return ''.join(sequence)
    
    def batch_encode(
        self, 
        sequences: List[str], 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_attention_mask: bool = False
    ) -> dict:
        """
        Encode multiple sequences at once.
        
        Args:
            sequences: List of protein sequence strings
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            return_attention_mask: Whether to return attention masks
        
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        all_token_ids = []
        
        # Encode each sequence
        for seq in sequences:
            tokens = self.encode(seq, add_special_tokens, max_length)
            all_token_ids.append(tokens)
        
        # Pad if requested
        if padding and max_length is not None:
            padded_ids = []
            attention_masks = []
            
            for tokens in all_token_ids:
                # Pad to max_length
                padding_length = max_length - len(tokens)
                if padding_length > 0:
                    padded = tokens + [self.pad_token_id] * padding_length
                    mask = [1] * len(tokens) + [0] * padding_length
                else:
                    padded = tokens[:max_length]
                    mask = [1] * max_length
                
                padded_ids.append(padded)
                attention_masks.append(mask)
            
            result = {'input_ids': padded_ids}
            if return_attention_mask:
                result['attention_mask'] = attention_masks
            
            return result
        else:
            return {'input_ids': all_token_ids}
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to JSON file."""
        vocab_dict = {
            'vocabulary': self.vocabulary,
            'token_to_id': self.token_to_id,
            'special_tokens': {
                'pad_token': self.PAD_TOKEN,
                'bos_token': self.BOS_TOKEN,
                'eos_token': self.EOS_TOKEN,
                'unk_token': self.UNK_TOKEN,
                'mask_token': self.MASK_TOKEN
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
    
    @classmethod
    def from_vocabulary(cls, filepath: str):
        """Load tokenizer from saved vocabulary file."""
        with open(filepath, 'r') as f:
            vocab_dict = json.load(f)
        
        tokenizer = cls()
        # Vocabulary is already initialized in __init__
        # This method is here for consistency with SMILES tokenizer
        return tokenizer



