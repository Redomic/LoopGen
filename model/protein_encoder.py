"""
Protein encoder for conditioning molecular generation on protein binding sites.

This module implements a transformer-based encoder that processes protein pocket
sequences (amino acid sequences) and produces contextual embeddings that can be
used to condition molecule generation via cross-attention.
"""

import torch
import torch.nn as nn
from typing import Optional
from .config import ModelConfig


class ProteinEncoder(nn.Module):
    """
    Transformer encoder for protein pocket sequences.
    
    Takes amino acid sequences from binding pockets and produces contextual
    embeddings that capture the biochemical properties and spatial constraints
    of the binding site.
    
    Architecture:
        - Token embeddings for 20 amino acids + special tokens
        - Positional embeddings for sequence order
        - Multi-layer transformer encoder
        - Layer normalization
    
    Args:
        config: ModelConfig containing protein encoder parameters
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings for amino acids
        self.token_embedding = nn.Embedding(
            config.protein_vocab_size, 
            config.d_model
        )
        
        # Positional embeddings for sequence positions
        self.position_embedding = nn.Embedding(
            config.protein_max_seq_len, 
            config.d_model
        )
        
        # Input dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.protein_encoder_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-layer normalization like in decoder
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.protein_encoder_layers
        )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        protein_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode protein pocket sequence into contextual embeddings.
        
        Args:
            protein_ids: Amino acid token IDs [batch_size, protein_seq_len]
            attention_mask: Mask for padding tokens [batch_size, protein_seq_len]
                           1 for real tokens, 0 for padding
        
        Returns:
            Protein embeddings [batch_size, protein_seq_len, d_model]
        """
        batch_size, seq_len = protein_ids.size()
        device = protein_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(protein_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer
        # TransformerEncoder expects: True for positions to mask (padding)
        if attention_mask is not None:
            # Convert from (1 = real, 0 = pad) to (True = mask, False = real)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply transformer encoder
        encoded = self.encoder(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Final layer norm
        encoded = self.layer_norm(encoded)
        
        return encoded



