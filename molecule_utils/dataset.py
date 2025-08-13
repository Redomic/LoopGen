import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from typing import Optional, Iterator
import pandas as pd
import math
import random

from .tokenizer import SELFIESTokenizer

def count_lines(file_path: str) -> int:
    """Counts the number of lines in a file, skipping the header."""
    try:
        with open(file_path, 'r') as f:
            # Efficiently count lines without loading the file into memory
            return sum(1 for _ in f) - 1 # Subtract 1 for header
    except FileNotFoundError:
        return 0

def get_curriculum_max_length(epoch: int,
                              total_epochs: int,
                              min_len: int = 50,
                              max_len: int = 256) -> int:
    """Compute curriculum max length that increases linearly over training.

    Args:
        epoch: Current epoch index (1-based or 0-based supported).
        total_epochs: Total number of epochs planned.
        min_len: Starting maximum length.
        max_len: Final maximum length.

    Returns:
        Integer maximum length to use for this epoch.
    """
    if total_epochs <= 0:
        return int(max_len)
    # Support either 0-based or 1-based epoch indexing
    clamped_epoch = max(0, min(epoch, total_epochs))
    progress = clamped_epoch / float(total_epochs)
    return int(min_len + (max_len - min_len) * progress)

class SELFIESDataset(IterableDataset):
    """
    An iterable dataset for reading a large CSV file of SELFIES strings
    without loading the entire file into memory. It can serve a train or
    validation split from the same file and shuffles data in a streaming manner.
    """
    def __init__(
        self, 
        file_path: str, 
        tokenizer: SELFIESTokenizer, 
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
        elif split == 'val':
            self.start_line = math.floor(total_lines * split_ratio) + 1
            self.end_line = total_lines + 1
        else:
            raise ValueError("split must be 'train' or 'val'")
        
        # For iterable datasets, the length is an estimate
        self.length = self.end_line - self.start_line

    def __len__(self):
        return self.length

    def set_max_length(self, new_max_length: int) -> None:
        """Update maximum sequence length used for truncation/padding."""
        self.max_length = int(new_max_length)

    def _line_iterator(self) -> Iterator[str]:
        """An internal iterator to stream lines from the correct split of the CSV."""
        worker_info = get_worker_info()

        try:
            num_rows_to_read = self.end_line - self.start_line
            if num_rows_to_read <= 0:
                return

            chunk_iterator = pd.read_csv(
                self.file_path,
                usecols=['SELFIES'],
                chunksize=10000,
                header=0,
                skiprows=range(1, self.start_line),
                nrows=num_rows_to_read,
                on_bad_lines='skip'
            )
            
            line_idx = -1
            # The `nrows` argument correctly constrains the reader to the intended split.
            for chunk in chunk_iterator:
                for selfies_string in chunk['SELFIES']:
                    line_idx += 1

                    # If in a worker process, skip lines not assigned to this worker.
                    if worker_info is not None:
                        if line_idx % worker_info.num_workers != worker_info.id:
                            continue
                    
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
                    # Do not pad or truncate here; return actual sequence length
                    yield torch.tensor(encoded, dtype=torch.long)
                buffer = []
        
        # Yield remaining items
        if buffer:
            random.shuffle(buffer)
            for selfies_string in buffer:
                encoded = self.tokenizer.encode(selfies_string, add_special_tokens=True)
                # Do not pad or truncate here; return actual sequence length
                yield torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch, pad_token_id=0):
    """
    Dynamic padding to actual sequence lengths in the batch.
    """
    # Find max length in this batch
    max_len = max(len(seq) for seq in batch)
    
    padded_batch = []
    for seq in batch:
        if len(seq) < max_len:
            padded = torch.cat([
                seq,
                torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)
            ])
        else:
            padded = seq[:max_len]
        padded_batch.append(padded)
    
    input_ids = torch.stack(padded_batch)
    attention_mask = (input_ids != pad_token_id).long()
    return {'input_ids': input_ids, 'attention_mask': attention_mask} 

class FixedSizeSELFIESDataset(Dataset):
    """
    A map-style dataset for a fixed-size validation set.
    Reads a portion of the CSV file into memory.
    """
    def __init__(self, file_path: str, tokenizer: SELFIESTokenizer, max_length: int, num_samples: int, total_lines: int):
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

    def set_max_length(self, new_max_length: int) -> None:
        """Update maximum sequence length used for truncation/padding."""
        self.max_length = int(new_max_length)