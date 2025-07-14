import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional, Dict, Set, List, Tuple, Any
from .config import ModelConfig

class RobustLoss(nn.Module):
    """
    Robust loss function that prevents model collapse by enforcing
    minimum sequence lengths and penalizing trivial solutions.
    """
    def __init__(self, tokenizer, min_length: int = 20, length_penalty_weight: float = 5.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.length_penalty_weight = length_penalty_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute robust cross-entropy loss with anti-collapse penalties.
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            input_ids: Original input sequence [batch, seq_len]
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean'
        )
        
        # Length enforcement: penalize sequences that are too short
        actual_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).float()
        length_shortfall = torch.clamp(self.min_length - actual_lengths, min=0.0)
        length_penalty = length_shortfall.mean() * self.length_penalty_weight
        
        # Early EOS penalty: discourage premature sequence termination
        eos_mask = (input_ids == self.tokenizer.eos_token_id)
        if eos_mask.any():
            # Find first EOS position for each sequence
            eos_positions = torch.where(eos_mask, torch.arange(input_ids.size(1), device=input_ids.device), input_ids.size(1))
            first_eos = eos_positions.min(dim=1)[0].float()
            early_eos_penalty = torch.clamp(self.min_length - first_eos, min=0.0).mean() * self.length_penalty_weight
        else:
            early_eos_penalty = torch.tensor(0.0, device=logits.device)
        
        total_loss = ce_loss + length_penalty + early_eos_penalty
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'length_penalty': length_penalty,
            'early_eos_penalty': early_eos_penalty
        }

class AntiCollapseRegularizer(nn.Module):
    """
    Regularization module that prevents embedding collapse and enforces diversity.
    """
    def __init__(self, min_embedding_std: float = 0.1, max_similarity: float = 0.8):
        super().__init__()
        self.min_embedding_std = min_embedding_std
        self.max_similarity = max_similarity
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute anti-collapse regularization losses.
        
        Args:
            embeddings: Molecular embeddings [batch_size, embedding_dim]
        """
        batch_size, embed_dim = embeddings.shape
        
        # Prevent embedding collapse: ensure sufficient standard deviation
        embedding_std = torch.std(embeddings, dim=0)  # [embed_dim]
        std_shortfall = torch.clamp(self.min_embedding_std - embedding_std, min=0.0)
        std_penalty = std_shortfall.mean() * 10.0
        
        # Prevent excessive similarity between different molecular embeddings
        if batch_size > 1:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = F.normalize(embeddings, dim=1)
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
            
            # Mask out diagonal (self-similarity)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
            off_diagonal_similarities = similarity_matrix[mask]
            
            # Penalize if average similarity is too high
            avg_similarity = off_diagonal_similarities.mean()
            similarity_excess = torch.clamp(avg_similarity - self.max_similarity, min=0.0)
            similarity_penalty = similarity_excess * 5.0
        else:
            similarity_penalty = torch.tensor(0.0, device=embeddings.device)
        
        total_penalty = std_penalty + similarity_penalty
        
        return {
            'anti_collapse_loss': total_penalty,
            'std_penalty': std_penalty,
            'similarity_penalty': similarity_penalty,
            'avg_embedding_std': embedding_std.mean(),
            'avg_similarity': similarity_penalty  # Reuse for logging
        }

class TrainingStabilizer:
    """
    Production-grade training stability monitor and intervention system.
    
    Monitors training health and applies conservative interventions only when
    genuine pathological behavior is detected, not normal convergence.
    """
    
    def __init__(self, 
                 patience: int = 200,
                 min_loss_threshold: float = 1e-6,
                 max_gradient_norm: float = 10.0,
                 stability_window: int = 50):
        """
        Args:
            patience: Steps to wait before considering intervention
            min_loss_threshold: Absolute minimum loss before considering pathological
            max_gradient_norm: Maximum allowed gradient norm
            stability_window: Window size for stability analysis
        """
        self.patience = patience
        self.min_loss_threshold = min_loss_threshold
        self.max_gradient_norm = max_gradient_norm
        self.stability_window = stability_window
        
        # Training history
        self.loss_history = []
        self.gradient_norms = []
        self.lr_history = []
        
        # State tracking
        self.steps_since_improvement = 0
        self.best_loss = float('inf')
        self.intervention_count = 0
        self.max_interventions = 3  # Limit interventions per training
        
    def update_metrics(self, 
                      total_loss: float, 
                      gradient_norm: float, 
                      learning_rate: float) -> Dict[str, bool]:
        """
        Update training metrics and assess stability.
        
        Returns:
            Dictionary with stability assessment and recommended actions
        """
        self.loss_history.append(total_loss)
        self.gradient_norms.append(gradient_norm)
        self.lr_history.append(learning_rate)
        
        # Maintain window size
        if len(self.loss_history) > self.stability_window * 4:
            self.loss_history = self.loss_history[-self.stability_window * 2:]
            self.gradient_norms = self.gradient_norms[-self.stability_window * 2:]
            self.lr_history = self.lr_history[-self.stability_window * 2:]
        
        # Track improvement
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        # Assess stability (only after sufficient history)
        if len(self.loss_history) < self.stability_window:
            return {'requires_intervention': False, 'reason': 'insufficient_history'}
        
        stability_assessment = self._assess_training_stability()
        return stability_assessment
    
    def _assess_training_stability(self) -> Dict[str, Any]:
        """Assess if training is genuinely unstable (not just converged)."""
        recent_losses = self.loss_history[-self.stability_window:]
        recent_gradients = self.gradient_norms[-self.stability_window:]
        
        # Check for genuine pathological behavior
        pathological_indicators = []
        
        # 1. Extremely low loss (potential numerical instability)
        if all(loss < self.min_loss_threshold for loss in recent_losses[-10:]):
            pathological_indicators.append('numerical_instability')
        
        # 2. Exploding gradients
        if any(grad > self.max_gradient_norm for grad in recent_gradients[-10:]):
            pathological_indicators.append('exploding_gradients')
        
        # 3. Loss becoming NaN or infinite
        if any(not torch.isfinite(torch.tensor(loss)) for loss in recent_losses[-5:]):
            pathological_indicators.append('non_finite_loss')
        
        # 4. Complete stagnation (no change for extended period)
        if (self.steps_since_improvement > self.patience and 
            len(set(f"{loss:.8f}" for loss in recent_losses[-20:])) <= 2):
            pathological_indicators.append('complete_stagnation')
        
        # Determine if intervention is needed
        requires_intervention = (
            len(pathological_indicators) > 0 and 
            self.intervention_count < self.max_interventions
        )
        
        return {
            'requires_intervention': requires_intervention,
            'pathological_indicators': pathological_indicators,
            'steps_since_improvement': self.steps_since_improvement,
            'recent_loss_trend': self._calculate_loss_trend(),
            'gradient_stability': self._assess_gradient_stability()
        }
    
    def _calculate_loss_trend(self) -> str:
        """Calculate recent loss trend."""
        if len(self.loss_history) < 20:
            return 'insufficient_data'
        
        recent = self.loss_history[-20:]
        early_avg = sum(recent[:10]) / 10
        late_avg = sum(recent[-10:]) / 10
        
        if late_avg < early_avg * 0.95:
            return 'improving'
        elif late_avg > early_avg * 1.05:
            return 'degrading'
        else:
            return 'stable'
    
    def _assess_gradient_stability(self) -> str:
        """Assess gradient norm stability."""
        if len(self.gradient_norms) < 20:
            return 'insufficient_data'
        
        recent_grads = self.gradient_norms[-20:]
        grad_std = torch.std(torch.tensor(recent_grads)).item()
        grad_mean = torch.mean(torch.tensor(recent_grads)).item()
        
        if grad_std / (grad_mean + 1e-8) > 2.0:
            return 'unstable'
        else:
            return 'stable'
    
    def apply_intervention(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, str]:
        """
        Apply conservative intervention to stabilize training.
        
        Returns:
            Dictionary describing actions taken
        """
        if self.intervention_count >= self.max_interventions:
            return {'action': 'max_interventions_reached'}
        
        self.intervention_count += 1
        actions_taken = []
        
        # Conservative learning rate reduction (not increase!)
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] *= 0.5  # Reduce, don't increase
            actions_taken.append(f"reduced_lr_from_{old_lr:.2e}_to_{param_group['lr']:.2e}")
        
        # Reset improvement tracking
        self.steps_since_improvement = 0
        
        return {
            'action': 'intervention_applied',
            'details': actions_taken,
            'intervention_count': self.intervention_count
        }

class ReconstructionLoss(nn.Module):
    """
    Clean reconstruction loss without overly aggressive anti-collapse mechanisms.
    
    Focuses on proper loss computation rather than trying to prevent normal convergence.
    """
    
    def __init__(self, pad_token_id: int, min_sequence_length: int = 5):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.min_sequence_length = min_sequence_length
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss with minimal intervention.
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
        """
        # Standard cross-entropy loss
        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction='mean'
        )
        
        # Mild length encouragement (not aggressive penalty)
        sequence_lengths = (targets != self.pad_token_id).sum(dim=1).float()
        length_bonus = torch.clamp(sequence_lengths - self.min_sequence_length, min=0.0).mean() * 0.01
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'length_bonus': length_bonus,
            'average_sequence_length': sequence_lengths.mean()
        }

class DiversityRegularizer(nn.Module):
    """
    Clean diversity regularization without excessive constraints.
    """
    
    def __init__(self, regularization_strength: float = 0.1):
        super().__init__()
        self.regularization_strength = regularization_strength
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute diversity regularization.
        
        Args:
            embeddings: Molecular embeddings [batch_size, embedding_dim]
        """
        batch_size = embeddings.size(0)
        
        if batch_size <= 1:
            return {
                'diversity_loss': torch.tensor(0.0, device=embeddings.device),
                'embedding_variance': torch.tensor(0.0, device=embeddings.device)
            }
        
        # Encourage diversity through cosine similarity penalty
        normalized_embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Mask diagonal and compute off-diagonal similarity
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        off_diagonal_similarities = similarity_matrix[mask]
        
        # Penalize high similarity (encourage diversity)
        diversity_loss = off_diagonal_similarities.mean() * self.regularization_strength
        
        # Monitor embedding variance
        embedding_variance = torch.var(embeddings, dim=0).mean()
        
        return {
            'diversity_loss': diversity_loss,
            'embedding_variance': embedding_variance,
            'average_similarity': off_diagonal_similarities.mean()
        }

class SwiGLU(nn.Module):
    """ SwiGLU activation function: SiLU(x) * Linear(x) """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Calculates the slopes for ALiBi heads."""
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        return torch.cat([
            torch.tensor(get_slopes_power_of_2(closest_power_of_2)),
            torch.tensor(get_slopes_power_of_2(2*closest_power_of_2))[0::2][:n_heads-closest_power_of_2]
        ])

class GrammarLSTM(nn.Module):
    """
    LSTM-based grammar state tracker for SELFIES molecules.
    Tracks structural context like branch depth, ring states, and bond connectivity.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = 128  # Smaller hidden size for grammar tracking
        
        # LSTM for tracking sequential grammar state
        self.lstm = nn.LSTM(
            input_size=config.vocab_size,  # One-hot token inputs
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Track different grammar states
        self.branch_depth_tracker = nn.Linear(self.hidden_size, config.max_branch_depth)
        self.ring_state_tracker = nn.Linear(self.hidden_size, 8)  # Max 8 rings
        self.bond_context_tracker = nn.Linear(self.hidden_size, 16)  # Bond context states
        
        # Grammar validity predictor
        self.validity_head = nn.Linear(self.hidden_size, config.vocab_size)
        
        # Special token tracking
        self.register_buffer("special_tokens", torch.tensor([0, 1, 2, 3]))  # PAD, BOS, EOS, MASK
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask indicating which tokens are grammatically valid at each position.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Convert to one-hot for LSTM input
        one_hot = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()
        
        # Initialize LSTM hidden state
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(one_hot, (h0, c0))
        
        # Predict validity for next tokens
        validity_logits = self.validity_head(lstm_out)  # [batch, seq_len, vocab_size]
        
        # Create grammar mask (True = invalid, False = valid)
        grammar_mask = torch.sigmoid(validity_logits) < 0.5
        
        # Always allow special tokens
        for special_token_id in self.special_tokens:
            if special_token_id < self.config.vocab_size:
                grammar_mask[:, :, special_token_id] = False
        
        return grammar_mask

class ChemicalValidityModule(nn.Module):
    """
    Module for ensuring chemical validity of generated SELFIES.
    Tracks valence, formal charges, and molecular stability constraints.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Atom property embeddings
        self.atom_embedding = nn.Embedding(config.vocab_size, 64)
        
        # Valence tracking network
        self.valence_tracker = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # Max valence of 8
            nn.Softmax(dim=-1)
        )
        
        # Formal charge tracker
        self.charge_tracker = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # Charges from -3 to +3
        )
        
        # Molecular stability predictor
        self.stability_predictor = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Chemical rule constraints
        self.register_chemical_rules()
        
    def register_chemical_rules(self):
        """Register basic chemical validity rules."""
        # Common atom valences (simplified)
        self.atom_valences = {
            'C': [4],           # Carbon: valence 4
            'N': [3, 5],        # Nitrogen: valence 3 or 5
            'O': [2],           # Oxygen: valence 2
            'S': [2, 4, 6],     # Sulfur: valence 2, 4, or 6
            'P': [3, 5],        # Phosphorus: valence 3 or 5
            'F': [1],           # Fluorine: valence 1
            'Cl': [1],          # Chlorine: valence 1
            'Br': [1],          # Bromine: valence 1
            'I': [1]            # Iodine: valence 1
        }
        
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns stability scores and validity constraints.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Predict molecular stability
        stability_scores = self.stability_predictor(hidden_states)  # [batch, seq_len, 1]
        
        # Get atom embeddings for chemical analysis
        atom_embeds = self.atom_embedding(input_ids)  # [batch, seq_len, 64]
        
        # Predict valence requirements
        valence_probs = self.valence_tracker(atom_embeds)  # [batch, seq_len, 8]
        
        # Predict formal charges
        charge_logits = self.charge_tracker(atom_embeds)  # [batch, seq_len, 7]
        
        # Create validity mask based on chemical rules
        validity_mask = self._apply_chemical_rules(input_ids, valence_probs, charge_logits)
        
        return stability_scores, validity_mask
    
    def _apply_chemical_rules(self, input_ids: torch.Tensor, valence_probs: torch.Tensor, 
                            charge_logits: torch.Tensor) -> torch.Tensor:
        """Apply basic chemical validity rules."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize as all valid
        validity_mask = torch.zeros(batch_size, seq_len, self.config.vocab_size, 
                                  dtype=torch.bool, device=device)
        
        # Apply valence rules (simplified implementation)
        # In practice, this would involve more sophisticated chemical logic
        
        return validity_mask

class Attention(nn.Module):
    """ Multi-Head Self-Attention with ALiBi and Flash Attention. """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model)
        self.c_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_alibi = config.use_alibi
        if self.use_alibi:
            self.register_buffer("slopes", get_alibi_slopes(self.n_heads))
        
        # Flash Attention is used via F.scaled_dot_product_attention
        self.flash_available = hasattr(F, 'scaled_dot_product_attention') and config.use_flash_attention
        if not self.flash_available and config.use_flash_attention:
            print("Warning: Flash Attention not available. Using manual implementation.")

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        m = self.slopes.to(device)
        # Create a causal mask and combine with relative position encodings
        relative_positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        alibi = m.unsqueeze(1).unsqueeze(1) * relative_positions.unsqueeze(0)
        return alibi

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.flash_available:
            # We must construct the final attention mask manually because the `is_causal`
            # flag in `scaled_dot_product_attention` is mutually exclusive with `attn_mask`.
            final_attn_mask = None
            if self.use_alibi:
                # ALiBi is an additive float mask that replaces the boolean causal mask.
                final_attn_mask = self._get_alibi_bias(seq_len, x.device)
                if attention_mask is not None:
                    # Combine with padding mask (0 for pad, so 1-mask = 1 for pad)
                    padding_mask_float = (1.0 - attention_mask) * -1e9
                    final_attn_mask = final_attn_mask.unsqueeze(0) + padding_mask_float.view(batch_size, 1, 1, seq_len)
            else:
                # Build a boolean mask. True means "do not attend".
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
                final_attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) # Expand to (1, 1, T, T)
                if attention_mask is not None:
                    padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
                    final_attn_mask = final_attn_mask | padding_mask # Combine causal and padding

            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=final_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            # Manual implementation for fallback
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            # Build additive mask for manual implementation
            final_additive_mask = None
            if self.use_alibi:
                final_additive_mask = self._get_alibi_bias(seq_len, x.device)
            else:
                final_additive_mask = torch.triu(torch.full((seq_len, seq_len), -1e9, device=x.device), diagonal=1)

            if attention_mask is not None:
                padding_mask_float = (1.0 - attention_mask) * -1e9
                final_additive_mask = final_additive_mask + padding_mask_float.view(batch_size, 1, 1, seq_len)
            
            attn_weights = attn_weights + final_additive_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            y = torch.matmul(attn_weights, v)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    """ Feed-Forward Network with SwiGLU activation. """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff * 2) # *2 for SwiGLU
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    """ A single Transformer Decoder block with Pre-Layer Normalization and Stochastic Depth. """
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)
        
        # Stochastic Depth (drop path)
        # Linearly increase drop prob from 0 to the target prob over layers
        self.drop_path_prob = config.stochastic_depth_prob * (layer_idx / (config.n_layers - 1))

    def drop_path(self, x: torch.Tensor, drop_prob: float = 0.) -> torch.Tensor:
        """Drop connections with a given probability."""
        if drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: Norm -> Attention -> Add, with drop path
        x = x + self.drop_path(self.attn(self.ln_1(x), attention_mask=attention_mask), self.drop_path_prob)
        # Pre-LN: Norm -> MLP -> Add, with drop path
        x = x + self.drop_path(self.mlp(self.ln_2(x)), self.drop_path_prob)
        return x

class SELFIESGPTDecoder(nn.Module):
    """
    GPT-style decoder model for SELFIES generation and representation learning.
    This model can be used for both generative and contrastive training.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        # Positional embeddings (always needed for base embeddings)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([DecoderBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Final language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        # ----- Contrastive Learning Components -----
        # Projection head for projecting hidden states to a contrastive space
        self.projection_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(config.d_model, 256),  
            nn.ReLU(),
            nn.Linear(256, 256)  
        )

        self.temperature = 0.1 # For NT-Xent loss
        # -----------------------------------------

        # Optional grammar and chemical validity modules
        self.grammar_lstm = GrammarLSTM(config) if config.use_grammar_constraint else None
        self.chem_validity = ChemicalValidityModule(config) if config.use_grammar_constraint else None

        # Robust loss components (initialized separately with tokenizer)
        self.robust_loss_fn = None
        self.anti_collapse_regularizer = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
        branch_depths: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        apply_constraints: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # 1. Get embeddings
        tok_embeds = self.wte(input_ids)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_embeds = self.wpe(pos_ids)
        hidden_states = tok_embeds + pos_embeds

        if branch_depths is not None and self.branch_embed is not None:
             branch_embeds = self.branch_embed(branch_depths)
             hidden_states = hidden_states + branch_embeds

        hidden_states = self.drop(hidden_states)
        
        # 2. Transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
        
        # 3. Language model head
        lm_logits = self.lm_head(hidden_states)

        # 4. Apply constraints if specified (for inference)
        if apply_constraints and self.grammar_lstm is not None:
            grammar_mask = self.grammar_lstm(input_ids)
            lm_logits.masked_fill_(grammar_mask, -65504.0)
        
        output = {"logits": lm_logits}
        if return_hidden:
            output["hidden_states"] = hidden_states

        return output

    # ----- Contrastive Loss Functions -----
    def get_molecular_embedding(self, hidden_states: torch.Tensor, 
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract molecular embedding using mean pooling over non-padded tokens."""
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            mean_embeddings = hidden_states.mean(dim=1)
        return mean_embeddings

    def nt_xent_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Normalized Temperature-scaled Cross Entropy Loss (InfoNCE)."""
        batch_size = features.shape[0]
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask.fill_diagonal_(0)
        
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        logits = logits / self.temperature
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Handle cases where there are no positive pairs for a sample
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask_sum, min=1e-9)
        
        loss = -mean_log_prob_pos[mask_sum > 0].mean()
        return loss

    def diversity_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Encourages diversity between DIFFERENT molecules only.
        Does NOT penalize similarity between augmentations of the same molecule.
        """
        normalized = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(normalized, normalized.T)
        batch_size = embeddings.shape[0]
        
        # Create mask for different molecules only
        labels_expanded = labels.unsqueeze(1)  # [batch_size, 1]
        same_molecule_mask = (labels_expanded == labels_expanded.T)  # [batch_size, batch_size]
        
        # Only penalize similarity between DIFFERENT molecules
        different_molecule_mask = ~same_molecule_mask
        # Remove diagonal (self-similarity)
        different_molecule_mask.fill_diagonal_(False)
        
        if different_molecule_mask.any():
            avg_similarity = similarity[different_molecule_mask].mean()
            return avg_similarity
        else:
            # No different molecules to compare
            return torch.tensor(0.0, device=embeddings.device)

    def combined_loss(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None,
                     contrastive_weight: float = 1.0,
                     reconstruction_weight: float = 0.1) -> Dict[str, torch.Tensor]:
        """Simplified loss without diversity - contrastive learning handles diversity naturally."""
        
        # Forward pass
        model_output = self.forward(input_ids, attention_mask, return_hidden=True)
        hidden_states = model_output['hidden_states']
        logits = model_output['logits']

        # 1. Contrastive Loss (main objective)
        molecular_embeddings = self.get_molecular_embedding(hidden_states, attention_mask)
        projections = self.projection_head(molecular_embeddings)
        projections = F.normalize(projections, dim=1)
        
        if labels is not None:
            contrastive_loss = self.nt_xent_loss(projections, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=logits.device)

        # 2. Reconstruction Loss (auxiliary)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        reconstruction_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=getattr(self, 'pad_token_id', 0)
        )
        
        # Total loss
        total_loss = (contrastive_weight * contrastive_loss + 
                     reconstruction_weight * reconstruction_loss)
        
        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'reconstruction_loss': reconstruction_loss,
            'diversity_loss': torch.tensor(0.0, device=logits.device),  # Keep for compatibility
            'length_bonus': torch.tensor(0.0, device=logits.device),
            'average_sequence_length': torch.tensor(0.0, device=logits.device),
            'embedding_variance': molecular_embeddings.var(dim=0).mean(),  # Monitor collapse
            'average_similarity': torch.tensor(0.0, device=logits.device)
        } 