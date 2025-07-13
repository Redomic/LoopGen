import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, Iterator
import pandas as pd
import math
import random

from .tokenizer import SELFIETokenizer

def count_lines(file_path: str) -> int:
    """Counts the number of lines in a file, skipping the header."""
    try:
        with open(file_path, 'r') as f:
            # Efficiently count lines without loading the file into memory
            return sum(1 for _ in f) - 1 # Subtract 1 for header
    except FileNotFoundError:
        return 0

class SELFIESDataset(IterableDataset):
    """
    An iterable dataset for reading a large CSV file of SELFIES strings
    without loading the entire file into memory. It can serve a train or
    validation split from the same file and shuffles data in a streaming manner.
    """
    def __init__(
        self, 
        file_path: str, 
        tokenizer: SELFIETokenizer, 
        max_length: int,
        total_lines: int,
        split: str = 'train',
        split_ratio: float = 0.8,
        shuffle_buffer_size: int = 100_000
    ):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.total_lines = total_lines
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Calculate the start and end line for this split
        if split == 'train':
            self.start_line = 1 # Skip header
            self.end_line = math.floor(total_lines * split_ratio)
        elif split == 'validation':
            self.start_line = math.floor(total_lines * split_ratio) + 1
            self.end_line = total_lines + 1
        else:
            raise ValueError("split must be 'train' or 'validation'")
        
        # For iterable datasets, the length is an estimate
        self.length = self.end_line - self.start_line

    def __len__(self):
        return self.length

    def _line_iterator(self) -> Iterator[str]:
        """An internal iterator to stream lines from the correct split of the CSV."""
        try:
            chunk_iterator = pd.read_csv(
                self.file_path,
                usecols=['SELFIES'],
                chunksize=10000,
                header=0,
                skiprows=range(1, self.start_line),
                on_bad_lines='skip'
            )
            line_num = self.start_line - 1
            for chunk in chunk_iterator:
                for selfies_string in chunk['SELFIES']:
                    line_num += 1
                    if line_num >= self.end_line:
                        return
                    if isinstance(selfies_string, str):
                        yield selfies_string
        except FileNotFoundError:
            return

    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer = []
        for line in self._line_iterator():
            buffer.append(line)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                for selfies_string in buffer:
                    encoded = self.tokenizer.encode(selfies_string, add_special_tokens=True)
                    if len(encoded) > self.max_length:
                        encoded = encoded[:self.max_length]
                    else:
                        encoded += [self.tokenizer.pad_token_id] * (self.max_length - len(encoded))
                    yield torch.tensor(encoded, dtype=torch.long)
                buffer = []
        
        # Yield remaining items
        if buffer:
            random.shuffle(buffer)
            for selfies_string in buffer:
                encoded = self.tokenizer.encode(selfies_string, add_special_tokens=True)
                if len(encoded) > self.max_length:
                    encoded = encoded[:self.max_length]
                else:
                    encoded += [self.tokenizer.pad_token_id] * (self.max_length - len(encoded))
                yield torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch, pad_token_id=0):
    """
    Collate function to create input_ids and attention_mask.
    """
    input_ids = torch.stack(batch)
    attention_mask = (input_ids != pad_token_id).long()
    return {'input_ids': input_ids, 'attention_mask': attention_mask} 

class FixedSizeSELFIESDataset(Dataset):
    """
    A map-style dataset for a fixed-size validation set.
    Reads a portion of the CSV file into memory.
    """
    def __init__(self, file_path: str, tokenizer: SELFIETokenizer, max_length: int, num_samples: int, total_lines: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        try:
            # Read the last `num_samples` from the file, skipping the header and the training part
            skip_rows = max(1, total_lines - num_samples)
            df = pd.read_csv(
                file_path,
                usecols=['SELFIES'],
                skiprows=range(1, skip_rows),
                nrows=num_samples,
                header=0,
                on_bad_lines='skip'
            )
            self.data = df['SELFIES'].dropna().tolist()
        except FileNotFoundError:
            print(f"Warning: Data file not found at {file_path}. The dataset will be empty.")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        selfies_string = self.data[idx]
        encoded = self.tokenizer.encode(selfies_string, add_special_tokens=True)
        
        # Pad or truncate
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded += [self.tokenizer.pad_token_id] * (self.max_length - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long) 