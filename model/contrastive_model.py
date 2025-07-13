"""
Contrastive learning model for molecular representation learning.
Wraps the existing decoder to learn robust molecular embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

from .decoder import SELFIESGPTDecoder
from .config import ModelConfig


class ProjectionHead(nn.Module):
    """Projects encoder representations to a space suitable for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveSELFIESModel(nn.Module):
    """
    Contrastive learning wrapper for SELFIES model.
    Combines contrastive pre-training with generative capabilities.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Main model (encoder-decoder)
        self.model = SELFIESGPTDecoder(config)
        
        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=128  # Standard contrastive embedding size
        )
        
        # Temperature parameter for contrastive loss
        self.temperature = 0.07
        
    def get_molecular_embedding(self, input_ids: torch.Tensor, 
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract molecular embedding from the model.
        Uses mean pooling over non-padded tokens.
        """
        # Get hidden states from the model (before the LM head)
        with torch.no_grad():
            # Temporarily remove hooks if any
            hidden_states = self.model(
                input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # For now, we'll modify the decoder to return hidden states
        # In practice, you'd modify the forward method to optionally return them
        # Here we'll use a workaround by getting the output before lm_head
        
        # Forward through transformer blocks
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Embeddings
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        tok_embeds = self.model.wte(input_ids)
        pos_embeds = self.model.wpe(pos_ids)
        hidden_states = tok_embeds + pos_embeds
        hidden_states = self.model.drop(hidden_states)
        
        # Pass through transformer blocks
        for block in self.model.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        
        hidden_states = self.model.ln_f(hidden_states)
        
        # Mean pooling over sequence (excluding padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            mean_embeddings = hidden_states.mean(dim=1)
        
        return mean_embeddings
    
    def forward_contrastive(self, input_ids: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor] = None,
                          labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            input_ids: Augmented molecular sequences [batch_size * num_aug, seq_len]
            attention_mask: Attention masks
            labels: Which sequences are augmentations of the same molecule
            
        Returns:
            Dictionary with 'loss' and 'embeddings'
        """
        # Get molecular embeddings
        embeddings = self.get_molecular_embedding(input_ids, attention_mask)
        
        # Project to contrastive space
        projections = self.projection(embeddings)
        projections = F.normalize(projections, dim=1)
        
        # Compute contrastive loss if labels provided
        if labels is not None:
            loss = self.nt_xent_loss(projections, labels)
        else:
            loss = None
        
        return {
            'loss': loss,
            'embeddings': embeddings,
            'projections': projections
        }
    
    def forward_generative(self, input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard forward pass for generation."""
        return self.model(input_ids, attention_mask=attention_mask)
    
    def nt_xent_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        Also known as InfoNCE loss.
        
        Args:
            features: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Integer labels indicating which samples are positive pairs
            
        Returns:
            Scalar loss
        """
        batch_size = features.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Mask out self-similarity
        mask.fill_diagonal_(0)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Diversity loss to prevent mode collapse.
        Encourages the model to use the full embedding space.
        """
        # Normalize embeddings
        normalized = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(normalized, normalized.T)
        
        # We want to minimize the average similarity (excluding diagonal)
        batch_size = embeddings.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # Average similarity between different samples
        avg_similarity = similarity[mask].mean()
        
        # Loss encourages diversity (lower similarity)
        return avg_similarity
    
    def combined_loss(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None,
                     alpha: float = 1.0,
                     beta: float = 0.1,
                     gamma: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Combined loss for contrastive pre-training.
        
        Args:
            input_ids: Input sequences
            attention_mask: Attention masks
            labels: Contrastive labels
            alpha: Weight for contrastive loss
            beta: Weight for reconstruction loss
            gamma: Weight for diversity loss
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Get contrastive loss and embeddings
        contrastive_output = self.forward_contrastive(input_ids, attention_mask, labels)
        contrastive_loss = contrastive_output['loss']
        embeddings = contrastive_output['embeddings']
        
        # Get reconstruction loss (standard LM loss)
        logits = self.forward_generative(input_ids, attention_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        reconstruction_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0  # Assuming pad_token_id is 0
        )
        
        # Get diversity loss
        diversity_loss = self.diversity_loss(embeddings)
        
        # Combine losses
        total_loss = (alpha * contrastive_loss + 
                     beta * reconstruction_loss + 
                     gamma * diversity_loss)
        
        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'reconstruction_loss': reconstruction_loss,
            'diversity_loss': diversity_loss
        } 