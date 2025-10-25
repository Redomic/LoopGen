import json
import codecs
from pathlib import Path
from typing import List, Dict, Optional, Set, Union
from collections import Counter
import pandas as pd

# Import SmilesPE for substructure tokenization
try:
    from SmilesPE.tokenizer import SPE_Tokenizer
    from SmilesPE.pretokenizer import atomwise_tokenizer
    SMILES_PE_AVAILABLE = True
except ImportError:
    SMILES_PE_AVAILABLE = False
    print("Warning: SmilesPE not installed. Install with: pip install SmilesPE")

class SMILESTokenizer:
    """
    Production-ready SMILES tokenizer using SMILES Pair Encoding (SPE).
    
    This tokenizer uses learned substructure tokens from SPE_ChEMBL vocabulary:
    - Substructure-aware: benzene ring 'c1ccccc1' is a single token
    - Efficient: ~50% fewer tokens per molecule vs character-level
    - SOTA: Used in ChemBERTa, MolGPT, and other leading models
    - Falls back to atom-level tokenization if SPE vocab not available
    """
    
    # Standard special tokens
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<BOS>'
    END_TOKEN = '<EOS>'
    MASK_TOKEN = '<MASK>'
    UNK_TOKEN = '<UNK>'  # For unseen tokens
    
    # Default SPE vocabulary path
    DEFAULT_SPE_VOCAB_PATH = 'checkpoints/SPE_ChEMBL.txt'
    
    def __init__(self, vocab_path: Optional[str] = None, data_path: Optional[str] = None, 
                 spe_vocab_path: Optional[str] = None):
        """
        Initialize SMILES tokenizer with SPE support.
        
        Args:
            vocab_path: Path to saved vocabulary JSON (legacy support)
            data_path: Path to SMILES data for building vocab (legacy support)
            spe_vocab_path: Path to SPE vocabulary file (defaults to checkpoints/SPE_ChEMBL.txt)
        """
        self.special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, 
                             self.MASK_TOKEN, self.UNK_TOKEN]
        
        # Determine SPE vocabulary path
        if spe_vocab_path is None:
            spe_vocab_path = self.DEFAULT_SPE_VOCAB_PATH
        
        self.spe_vocab_path = spe_vocab_path
        self.spe_tokenizer = None
        self.use_spe = False
        
        # Try to load SPE tokenizer
        if SMILES_PE_AVAILABLE and Path(spe_vocab_path).exists():
            try:
                print(f"Loading SPE vocabulary from {spe_vocab_path}")
                spe_vocab_file = codecs.open(spe_vocab_path, 'r', encoding='utf-8')
                self.spe_tokenizer = SPE_Tokenizer(spe_vocab_file)
                self.use_spe = True
                print("✓ Successfully loaded SPE tokenizer")
            except Exception as e:
                print(f"Warning: Failed to load SPE tokenizer: {e}")
                print("Falling back to atom-level tokenization")
                self.use_spe = False
        else:
            if not SMILES_PE_AVAILABLE:
                print("Warning: SmilesPE library not available")
                print("Install with: pip install SmilesPE")
            elif not Path(spe_vocab_path).exists():
                print(f"Warning: SPE vocabulary not found at {spe_vocab_path}")
                print("Download with:")
                print(f"  wget -O {spe_vocab_path} https://github.com/XinhaoLi74/SmilesPE/raw/master/SPE_ChEMBL.txt")
            print("Falling back to atom-level tokenization")
        
        # Build vocabulary
        if vocab_path and Path(vocab_path).exists():
            self.vocabulary = self._load_vocabulary(vocab_path)
        elif self.use_spe:
            self.vocabulary = self._build_vocabulary_from_spe()
        elif data_path:
            self.vocabulary = self._build_vocabulary_from_data(data_path)
        else:
            self.vocabulary = self._get_default_vocabulary()

        # Create bidirectional mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}

    def _build_vocabulary_from_spe(self) -> List[str]:
        """
        Build vocabulary from SPE tokenizer.
        
        SPE_ChEMBL.txt contains merge operations, not final tokens.
        We need to build vocabulary from actual tokenization output.
        """
        print("Building vocabulary from SPE tokenizer")
        
        # Get vocabulary directly from SPE tokenizer's vocab
        # The SPE tokenizer maintains its own vocabulary internally
        spe_vocab = []
        
        try:
            # Get the vocabulary from SPE tokenizer
            # SPE stores vocab as a dict with token -> frequency
            if hasattr(self.spe_tokenizer, 'vocab'):
                spe_vocab = list(self.spe_tokenizer.vocab.keys())
            else:
                # Fallback: extract from the vocab file by parsing merge operations
                # Read all merge pairs and extract unique tokens
                unique_tokens = set()
                with codecs.open(self.spe_vocab_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Each line is a merge: "token1 token2" -> "token1token2"
                        parts = line.strip().split()
                        if len(parts) == 2:
                            # Add both the merged result and original tokens
                            merged = ''.join(parts)
                            unique_tokens.add(merged)
                            unique_tokens.update(parts)
                
                spe_vocab = sorted(list(unique_tokens))
            
            print(f"Extracted {len(spe_vocab)} unique tokens from SPE")
            
            # Combine special tokens + SPE tokens
            vocabulary = self.special_tokens + spe_vocab
            print(f"Final vocabulary size: {len(vocabulary)}")
            return vocabulary
            
        except Exception as e:
            print(f"Error building vocabulary from SPE: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_vocabulary()

    def _build_vocabulary_from_data(self, data_path: str, max_size: int = 500, 
                                   max_molecules: Optional[int] = None) -> List[str]:
        """Build vocabulary from SMILES strings using atom-level tokenization."""
        print(f"Building atom-level vocabulary from {data_path}")
        
        token_counter = Counter()
        row_count = 0
        
        try:
            for chunk in pd.read_csv(data_path, usecols=[0], header=None, chunksize=10000, on_bad_lines='skip'):
                for smiles_str in chunk[0]:
                    if isinstance(smiles_str, str) and smiles_str.strip():
                        try:
                            # Use atom-level tokenization
                            tokens = self._atomwise_tokenize(smiles_str.strip())
                            if tokens:
                                token_counter.update(tokens)
                                row_count += 1
                        except Exception:
                            continue
                            
                        if row_count % 100000 == 0:
                            print(f"Processed {row_count:,} molecules")
                            
                        if max_molecules and row_count >= max_molecules:
                            print(f"Reached molecule limit of {max_molecules:,}")
                            break
                
                if max_molecules and row_count >= max_molecules:
                    break
            
            print(f"Collected tokens from {row_count:,} valid SMILES strings")
            print(f"Found {len(token_counter)} unique tokens")
            
            # Build vocabulary from most common tokens
            most_common_tokens = [token for token, _ in token_counter.most_common(max_size)]
            vocabulary = self.special_tokens + sorted(most_common_tokens)
            
            print(f"Final vocabulary size: {len(vocabulary)}")
            return vocabulary
            
        except Exception as e:
            print(f"Error building vocabulary: {e}")
            return self._get_default_vocabulary()

    def _get_default_vocabulary(self) -> List[str]:
        """Get default atom-level SMILES vocabulary."""
        print("Using default atom-level SMILES vocabulary")
        
        # Common SMILES tokens (atom-level)
        organic_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'H']
        aromatic_atoms = ['c', 'n', 'o', 's', 'p']
        bonds = ['=', '#', '-', '/', '\\']
        structure = ['(', ')', '[', ']', '@', '@@']
        ring_numbers = [str(i) for i in range(10)]  # 0-9
        ring_closures = ['%10', '%11', '%12', '%13', '%14', '%15']
        
        # Common bracket atoms
        bracket_atoms = [
            '[NH]', '[NH2]', '[NH3+]', '[N+]', '[N-]',
            '[O-]', '[O+]', '[OH]', '[OH2+]',
            '[S-]', '[S+]', '[SH]',
            '[C@H]', '[C@@H]', '[C@]', '[C@@]',
            '[nH]', '[cH]', '[sH]'
        ]
        
        # Combine all tokens
        smiles_tokens = (
            organic_atoms + aromatic_atoms + bonds + structure + 
            ring_numbers + ring_closures + bracket_atoms
        )
        
        vocabulary = self.special_tokens + sorted(set(smiles_tokens))
        print(f"Default vocabulary size: {len(vocabulary)}")
        return vocabulary

    def _load_vocabulary(self, vocab_path: str) -> List[str]:
        """Load vocabulary from JSON file."""
        try:
            with open(vocab_path, 'r') as f:
                data = json.load(f)
                
                # Check if this is an SPE vocabulary
                if 'spe_vocab_path' in data:
                    spe_path = data['spe_vocab_path']
                    if Path(spe_path).exists() and SMILES_PE_AVAILABLE:
                        print(f"Loading SPE vocabulary from {spe_path}")
                        spe_vocab_file = codecs.open(spe_path, 'r', encoding='utf-8')
                        self.spe_tokenizer = SPE_Tokenizer(spe_vocab_file)
                        self.use_spe = True
                        self.spe_vocab_path = spe_path
                        return data.get('vocabulary', self._get_default_vocabulary())
                
                return data.get('vocabulary', self._get_default_vocabulary())
        except Exception as e:
            print(f"Error loading vocabulary from {vocab_path}: {e}")
            return self._get_default_vocabulary()

    def save_vocabulary(self, save_path: str) -> None:
        """Save vocabulary to JSON file with SPE metadata."""
        data = {
            'vocabulary': self.vocabulary,
            'size': len(self.vocabulary),
            'special_tokens': self.special_tokens,
            'tokenizer_type': 'SPE' if self.use_spe else 'atom_level',
        }
        
        # Add SPE vocab path if using SPE
        if self.use_spe:
            data['spe_vocab_path'] = self.spe_vocab_path
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Vocabulary saved to {save_path}")

    def _atomwise_tokenize(self, smiles: str) -> List[str]:
        """Atom-level tokenization fallback using SmilesPE or regex."""
        if SMILES_PE_AVAILABLE:
            try:
                return atomwise_tokenizer(smiles)
            except:
                pass
        
        # Regex fallback
        import re
        pattern = r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"""
        return re.findall(pattern, smiles)

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
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    # Legacy property names for backward compatibility
    @property
    def bos_token_id(self) -> int:
        return self.start_token_id

    @property
    def eos_token_id(self) -> int:
        return self.end_token_id

    def tokenize(self, smiles_string: str) -> List[str]:
        """
        Tokenize SMILES string using SPE or atom-level tokenization.
        
        Returns list of tokens (substructures if SPE, atoms if fallback).
        """
        if not isinstance(smiles_string, str) or not smiles_string.strip():
            return []
        
        try:
            if self.use_spe and self.spe_tokenizer is not None:
                # SPE returns space-separated tokens
                tokenized_smiles = self.spe_tokenizer.tokenize(smiles_string.strip())
                tokens = tokenized_smiles.split()
                return tokens
            else:
                # Fall back to atom-level tokenization
                return self._atomwise_tokenize(smiles_string.strip())
        except Exception as e:
            print(f"Tokenization error: {e}")
            return []

    def encode(self, smiles_string: str, add_special_tokens: bool = True) -> List[int]:
        """Convert SMILES string to token IDs."""
        tokens = self.tokenize(smiles_string)
        
        if add_special_tokens:
            tokens = [self.START_TOKEN] + tokens + [self.END_TOKEN]
        
        token_ids = []
        for token in tokens:
            # Use UNK token for unknown tokens
            token_ids.append(self.token_to_id.get(token, self.unk_token_id))
        
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to SMILES string."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        # Join tokens (SPE tokens already have proper spacing embedded)
        if self.use_spe:
            # SPE tokens need to be concatenated directly
            return "".join(tokens)
        else:
            # Atom-level tokens also concatenate directly
            return "".join(tokens)

    def encode_with_padding(self, smiles_string: str, max_length: int) -> List[int]:
        """Encode with padding to fixed length."""
        token_ids = self.encode(smiles_string, add_special_tokens=True)
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        return token_ids

    def build_vocabulary_from_data(self, data_path: str, save_path: Optional[str] = None, 
                                   max_molecules: Optional[int] = None) -> None:
        """Build vocabulary from training data."""
        vocabulary = self._build_vocabulary_from_data(data_path, max_molecules=max_molecules)
        self.vocabulary = vocabulary
        
        # Rebuild mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}
        
        if save_path:
            self.save_vocabulary(save_path)

    def get_tokens_from_data(self, data_path: str, max_molecules: int = 100000) -> Set[str]:
        """Extract unique tokens from dataset."""
        all_tokens = set()
        row_count = 0
        
        try:
            for chunk in pd.read_csv(data_path, usecols=['SMILES'], chunksize=10000, on_bad_lines='skip'):
                for smiles_str in chunk['SMILES']:
                    if isinstance(smiles_str, str) and smiles_str.strip():
                        try:
                            tokens = self.tokenize(smiles_str.strip())
                            all_tokens.update(tokens)
                            row_count += 1
                        except Exception:
                            continue
                            
                        if row_count >= max_molecules:
                            break
                
                if row_count >= max_molecules:
                    break
            
            return all_tokens
            
        except Exception as e:
            print(f"Error extracting tokens: {e}")
            return set()

    def is_valid_smiles(self, smiles_string: str) -> bool:
        """Validate SMILES string by checking if it can be tokenized."""
        try:
            tokens = self.tokenize(smiles_string)
            return len(tokens) > 0
        except Exception:
            return False
