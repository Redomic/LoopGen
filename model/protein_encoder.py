"""
Protein encoder for conditioning molecular generation on protein binding sites.

This module implements a transformer-based encoder that processes protein pocket
sequences (amino acid sequences) and produces contextual embeddings that can be
used to condition molecule generation via cross-attention.

Optionally integrates topological features from persistent homology for enhanced
geometric understanding of binding sites.
"""

import torch
import torch.nn as nn
from typing import Optional
from .config import ModelConfig
import logging

logger = logging.getLogger(__name__)


class ProteinEncoder(nn.Module):
    """
    Transformer encoder for protein pocket sequences with optional topology encoding.
    
    Takes amino acid sequences from binding pockets and produces contextual
    embeddings that capture the biochemical properties and spatial constraints
    of the binding site. Optionally fuses topological features from persistent
    homology for enhanced geometric understanding.
    
    Architecture:
        - Token embeddings for 20 amino acids + special tokens
        - Positional embeddings for sequence order
        - Multi-layer transformer encoder
        - Optional topology feature fusion
        - Layer normalization
    
    Args:
        config: ModelConfig containing protein encoder parameters
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_topology = config.use_topology_encoding
        
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
        
        # Optional topology encoding
        self.topology_encoder = None
        self.topology_fusion = None
        
        if self.use_topology:
            try:
                from .topology_encoder import TopologyEncoder, TopologyFusionLayer
                
                self.topology_encoder = TopologyEncoder(
                    d_model=config.d_model,
                    homology_dimensions=config.topology_persistence_dims,
                    n_bins=config.topology_n_bins,
                    dropout=config.dropout,
                    representation=config.topology_representation
                )
                
                self.topology_fusion = TopologyFusionLayer(
                    d_model=config.d_model,
                    fusion_method=config.topology_fusion_method,
                    dropout=config.dropout
                )
                
                logger.info("Initialized topology encoding with persistent homology")
            except ImportError as e:
                logger.warning(f"Could not initialize topology encoding: {e}")
                logger.warning("Falling back to sequence-only encoding")
                self.use_topology = False
                self.topology_encoder = None
                self.topology_fusion = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        protein_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        topology_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode protein pocket sequence into contextual embeddings.
        
        Args:
            protein_ids: Amino acid token IDs [batch_size, protein_seq_len]
            attention_mask: Mask for padding tokens [batch_size, protein_seq_len]
                           1 for real tokens, 0 for padding
            topology_features: Pre-extracted topology features [batch_size, feature_dim]
                             (optional, only used if use_topology=True)
        
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
        
        # Fuse with topology features if available
        if self.use_topology and topology_features is not None:
            if self.topology_encoder is not None and self.topology_fusion is not None:
                # Project topology features to model dimension
                topology_embeddings = self.topology_encoder(topology_features)
                
                # Fuse with sequence embeddings
                encoded = self.topology_fusion(encoded, topology_embeddings)
        
        return encoded



