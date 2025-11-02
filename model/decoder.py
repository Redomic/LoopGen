import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from .config import ModelConfig

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

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy to prevent overconfident predictions.
    
    Replaces one-hot targets with smoothed labels to encourage better calibration
    and prevent the model from outputting extreme logits.
    """
    
    def __init__(self, epsilon=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, preds, target):
        """
        Args:
            preds: Model predictions [N, C] where C is number of classes
            target: Target labels [N]
        """
        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        # Create mask for valid (non-ignored) tokens
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index)
            target_masked = target.masked_fill(~valid_mask, 0)  # Replace ignored with 0 for indexing
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
            target_masked = target
        
        # Compute standard NLL loss
        nll = F.nll_loss(log_preds, target_masked, reduction='none')
        
        # Compute uniform loss (entropy regularization)
        uniform_loss = -log_preds.sum(dim=-1)
        
        # Apply mask to exclude ignored tokens
        if self.ignore_index >= 0:
            nll = nll.masked_fill(~valid_mask, 0.0)
            uniform_loss = uniform_loss.masked_fill(~valid_mask, 0.0)
        
        # Combine losses with label smoothing
        loss = (1 - self.epsilon) * nll + self.epsilon * uniform_loss / n
        
        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return loss.sum() / valid_mask.sum().clamp(min=1)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss with label smoothing to prevent overconfident predictions.
    
    Uses label smoothing to encourage better calibration and prevent extreme logits
    that make the model brittle and repetitive.
    """
    
    def __init__(self, pad_token_id: int, min_sequence_length: int = 5, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.min_sequence_length = min_sequence_length
        self.label_smoothing = label_smoothing
        
        # Use label smoothing cross-entropy instead of standard cross-entropy
        self.loss_fn = LabelSmoothingCrossEntropy(
            epsilon=label_smoothing,
            ignore_index=pad_token_id
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss with label smoothing.
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
        """
        # Label smoothing cross-entropy loss
        reconstruction_loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Optional: Add entropy regularization for additional diversity
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Mask out padding tokens for entropy
        valid_mask = (targets != self.pad_token_id)
        if valid_mask.any():
            entropy_reg = entropy[valid_mask].mean() * 0.01  # Small entropy bonus
        else:
            entropy_reg = torch.tensor(0.0, device=logits.device)
        
        # Mild length encouragement
        sequence_lengths = (targets != self.pad_token_id).sum(dim=1).float()
        length_bonus = torch.clamp(sequence_lengths - self.min_sequence_length, min=0.0).mean() * 0.01
        
        return {
            'reconstruction_loss': reconstruction_loss - entropy_reg,  # Subtract because we want higher entropy
            'entropy_regularization': entropy_reg,
            'length_bonus': length_bonus,
            'average_sequence_length': sequence_lengths.mean(),
            'average_entropy': entropy[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=logits.device)
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

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention from molecule hidden states to protein features.
    
    Allows the molecule decoder to attend to protein pocket embeddings,
    enabling protein-conditioned generation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        protein_embeddings: torch.Tensor, 
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention from molecule to protein.
        
        Args:
            hidden_states: Molecule hidden states [batch, mol_seq_len, d_model] (query)
            protein_embeddings: Protein embeddings [batch, prot_seq_len, d_model] (key/value)
            protein_mask: Protein attention mask [batch, prot_seq_len] (1=real, 0=pad)
        
        Returns:
            Cross-attended hidden states [batch, mol_seq_len, d_model]
        """
        # Normalize before attention (pre-norm)
        normed_hidden = self.layer_norm(hidden_states)
        
        # Create key padding mask for cross-attention
        # MultiheadAttention expects: True for positions to ignore (padding)
        if protein_mask is not None:
            key_padding_mask = (protein_mask == 0)
        else:
            key_padding_mask = None
        
        # Cross-attention: query from molecule, key/value from protein
        attn_output, _ = self.cross_attn(
            query=normed_hidden,
            key=protein_embeddings,
            value=protein_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Apply dropout
        attn_output = self.dropout(attn_output)
        
        return attn_output


class DecoderBlock(nn.Module):
    """ A single Transformer Decoder block with Pre-Layer Normalization and Stochastic Depth. """
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)
        
        # Cross-attention to protein (if protein conditioning enabled)
        if hasattr(config, 'use_protein_conditioning') and config.use_protein_conditioning:
            if hasattr(config, 'use_cross_attention') and config.use_cross_attention:
                # Apply cross-attention every N layers
                cross_attn_freq = getattr(config, 'cross_attention_freq', 1)
                if layer_idx % cross_attn_freq == 0:
                    self.cross_attn = CrossAttentionLayer(config)
                else:
                    self.cross_attn = None
            else:
                self.cross_attn = None
        else:
            self.cross_attn = None
        
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

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        protein_embeddings: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN: Norm -> Self-Attention -> Add, with drop path
        x = x + self.drop_path(self.attn(self.ln_1(x), attention_mask=attention_mask), self.drop_path_prob)
        
        # Cross-attention to protein (if available)
        if self.cross_attn is not None and protein_embeddings is not None:
            x = x + self.drop_path(self.cross_attn(x, protein_embeddings, protein_mask), self.drop_path_prob)
        
        # Pre-LN: Norm -> MLP -> Add, with drop path
        x = x + self.drop_path(self.mlp(self.ln_2(x)), self.drop_path_prob)
        return x

class SMILESGPTDecoder(nn.Module):
    """
    Autoregressive GPT-style decoder model for SMILES molecular generation.
    
    This model learns to generate molecules by:
    1. Reconstruction loss - Learning to speak "molecular SMILES"
    2. (Future) Protein conditioning - Learning to listen to "protein language"
    
    The model is purely generative without contrastive learning components.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.orthogonal_(self.wte.weight)
        # Positional embeddings
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        # High dropout on embeddings to encourage robustness
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Fixed positional scaling for stability (keep simple and deterministic)
        self.pos_scale: float = 1.0
        
        # Input layer norm for large models
        if config.d_model > 256:
            self.input_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        else:
            self.input_ln = nn.Identity()
        
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([DecoderBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Final language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight
        
        # Protein encoder for conditioning
        if hasattr(config, 'use_protein_conditioning') and config.use_protein_conditioning:
            from .protein_encoder import ProteinEncoder
            self.protein_encoder = ProteinEncoder(config)
            print(f"Initialized ProteinEncoder with {config.protein_encoder_layers} layers")
        else:
            self.protein_encoder = None

        # Note: Grammar/chemistry constraints could be added for SMILES validation
        if hasattr(config, 'use_grammar_constraint') and config.use_grammar_constraint:
            print("Warning: Grammar/Chemistry constraints not yet implemented for SMILES; set use_grammar_constraint=False.")
        
        # Label smoothing for reconstruction loss
        self.label_smoothing = 0.1
        self.pad_token_id = 0  # Will be set by tokenizer
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Override initialization for embeddings in larger models
        if config.d_model > 256:
            with torch.no_grad():
                # Use smaller std for position embeddings
                nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
                # Keep token embeddings with default init
                nn.init.normal_(self.wte.weight, mean=0.0, std=1.0/math.sqrt(config.d_model))
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Standard transformer initialization
            std = 1.0 / math.sqrt(self.config.d_model)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        protein_ids: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        apply_constraints: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the autoregressive decoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            protein_ids: Protein token IDs [batch_size, protein_seq_len] (optional)
            protein_mask: Protein attention mask [batch_size, protein_seq_len] (optional)
            apply_constraints: Backward-compat flag (no-op). Grammar/chem constraints are removed.
        
        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # 1. Encode protein if provided
        protein_embeddings = None
        if self.protein_encoder is not None and protein_ids is not None:
            protein_embeddings = self.protein_encoder(protein_ids, protein_mask)
        
        # 2. Get molecule embeddings
        tok_embeds = self.embed_dropout(self.wte(input_ids))
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_embeds = self.wpe(pos_ids)
        
        # Scale embeddings for stability
        hidden_states = tok_embeds + self.pos_scale * pos_embeds
        # Add training-time Gaussian noise to improve robustness
        if self.training:
            hidden_states = hidden_states + torch.randn_like(hidden_states) * 0.1
        
        # Apply input layer norm and dropout
        hidden_states = self.input_ln(hidden_states)
        hidden_states = self.drop(hidden_states)
        
        # 3. Transformer blocks with protein cross-attention
        for block in self.h:
            hidden_states = block(
                hidden_states, 
                attention_mask=attention_mask,
                protein_embeddings=protein_embeddings,
                protein_mask=protein_mask
            )
        
        hidden_states = self.ln_f(hidden_states)
        
        # 3. Language model head
        lm_logits = self.lm_head(hidden_states)
        
        # 4. Constraints removed; rely on SELFIES validity and sampling controls
        
        return {"logits": lm_logits, "hidden_states": hidden_states}

    def set_tokenizer(self, tokenizer):
        """Set tokenizer to get special token IDs."""
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id

    def compute_loss(self, 
                    input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    protein_ids: Optional[torch.Tensor] = None,
                    protein_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute autoregressive reconstruction loss for molecular generation.
        
        This is the core training objective: learning to predict the next token
        in a SMILES sequence, optionally conditioned on protein context.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            protein_ids: Protein token IDs [batch_size, protein_seq_len] (optional)
            protein_mask: Protein attention mask [batch_size, protein_seq_len] (optional)
        
        Returns:
            Dictionary with loss components
        """
        # Forward pass
        model_output = self.forward(input_ids, attention_mask, protein_ids, protein_mask)
        logits = model_output['logits']
        logits = torch.clamp(logits, min=-10, max=10)
        
        # Shift for next-token prediction
        # Input: [BOS, tok1, tok2, ...]
        # Target: [tok1, tok2, ..., EOS]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Reconstruction loss with label smoothing
        # Label smoothing prevents overconfident predictions and improves generation diversity
        if self.label_smoothing > 0:
            loss_fn = LabelSmoothingCrossEntropy(
                epsilon=self.label_smoothing,
                ignore_index=self.pad_token_id
            )
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        
        reconstruction_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Calculate metrics and debug statistics
        with torch.no_grad():
            perplexity = torch.exp(reconstruction_loss)
            
            # Calculate sequence lengths for monitoring
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).float()
                avg_seq_length = seq_lengths.mean()
                max_seq_length = seq_lengths.max()
                min_seq_length = seq_lengths.min()
            else:
                avg_seq_length = torch.tensor(float(input_ids.size(1)), device=input_ids.device)
                max_seq_length = avg_seq_length
                min_seq_length = avg_seq_length
            
            # Logits statistics
            logits_mean = shift_logits.mean()
            logits_std = shift_logits.std()
            logits_max = shift_logits.max()
            logits_min = shift_logits.min()
            
            # Prediction confidence (entropy)
            probs = F.softmax(shift_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            avg_entropy = entropy.mean()
            
            # Token prediction accuracy (for monitoring)
            predicted_tokens = shift_logits.argmax(dim=-1)
            correct_predictions = (predicted_tokens == shift_labels).float()
            if attention_mask is not None:
                # Only count non-padded tokens
                valid_mask = attention_mask[:, 1:].bool()  # Skip BOS token
                accuracy = correct_predictions[valid_mask].mean()
            else:
                accuracy = correct_predictions.mean()

            # Additional debug: compare real-token vs padding accuracy and distribution
            pad_mask = (shift_labels == self.pad_token_id)
            non_pad_mask = ~pad_mask
            real_token_acc = correct_predictions[non_pad_mask].mean() if non_pad_mask.any() else torch.tensor(0.0, device=logits.device)
            pad_token_acc = correct_predictions[pad_mask].mean() if pad_mask.any() else torch.tensor(0.0, device=logits.device)
            total_tokens = shift_labels.numel()
            pad_tokens = pad_mask.sum().item()
            real_tokens = non_pad_mask.sum().item()
            
            # Hidden state statistics (from last layer)
            hidden_states = model_output['hidden_states']
            hidden_mean = hidden_states.mean()
            hidden_std = hidden_states.std()
            hidden_norm = hidden_states.norm(dim=-1).mean()
        
        # Enhanced debug output with similarity and normalization stats
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
            
        # Calculate additional debug metrics (similar to old contrastive model)
        with torch.no_grad():
            # Hidden state similarity analysis
            batch_size = hidden_states.size(0)
            if batch_size > 1:
                # Get sequence representations (mean pooling)
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                    sequence_representations = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    sequence_representations = hidden_states.mean(dim=1)
                
                # Normalize and compute similarity
                seq_repr_norm = F.normalize(sequence_representations, dim=1)
                similarity_matrix = torch.matmul(seq_repr_norm, seq_repr_norm.T)
                
                # Remove diagonal for similarity stats
                similarity_no_diag = similarity_matrix[~torch.eye(batch_size, dtype=torch.bool, device=hidden_states.device)]
                similarity_mean = similarity_no_diag.mean()
                similarity_std = similarity_no_diag.std()
                similarity_max = similarity_no_diag.max()
                similarity_min = similarity_no_diag.min()
            else:
                similarity_mean = torch.tensor(0.0, device=hidden_states.device)
                similarity_std = torch.tensor(0.0, device=hidden_states.device)
                similarity_max = torch.tensor(0.0, device=hidden_states.device)
                similarity_min = torch.tensor(0.0, device=hidden_states.device)
            
            # Token embedding analysis
            token_embeddings = self.wte.weight
            token_emb_mean = token_embeddings.mean()
            token_emb_std = token_embeddings.std()
            token_emb_norm = token_embeddings.norm(dim=1).mean()
            
            # Output layer analysis
            output_weight = self.lm_head.weight
            output_weight_mean = output_weight.mean()
            output_weight_std = output_weight.std()
            output_weight_norm = output_weight.norm(dim=1).mean()
            
        # Print detailed stats every N steps (similar to old model)
        if self._debug_step_count % 50 == 0:
            print(f"\n=== Autoregressive Model Debug Stats (Step {self._debug_step_count}) ===")
            print(f"Sequence similarity BEFORE normalization: {similarity_mean:.3f}")
            print(f"Hidden states - Mean: {hidden_mean:.3f}, Std: {hidden_std:.3f}")
            print(f"Hidden state norm: {hidden_norm:.3f}")
            print(f"Similarity stats - Mean: {similarity_mean:.3f}, Std: {similarity_std:.3f}")
            print(f"Max similarity: {similarity_max:.3f}, Min: {similarity_min:.3f}")
            print(f"Hidden representation std: {hidden_std:.4f}")
            print(f"Token embeddings - Mean: {token_emb_mean:.4f}, Std: {token_emb_std:.4f}, Norm: {token_emb_norm:.3f}")
            print(f"Output weights - Mean: {output_weight_mean:.4f}, Std: {output_weight_std:.4f}, Norm: {output_weight_norm:.3f}")
            print(f"Loss: {reconstruction_loss:.4f}, Perplexity: {perplexity:.2f}")
            print(f"Prediction accuracy: {accuracy:.3f}, Avg entropy: {avg_entropy:.3f}")
            print(f"Token distribution: {real_tokens}/{total_tokens} real ({(real_tokens/total_tokens*100 if total_tokens>0 else 0):.1f}%), {pad_tokens}/{total_tokens} padding")
            print(f"Accuracy - Real tokens: {real_token_acc:.3f}, Padding: {pad_token_acc:.3f}")
            print(f"Logits - Mean: {logits_mean:.3f}, Std: {logits_std:.3f}, Range: [{logits_min:.2f}, {logits_max:.2f}]")
        
        return {
            'loss': reconstruction_loss,
            'reconstruction_loss': reconstruction_loss,
            'perplexity': perplexity,
            'average_sequence_length': avg_seq_length,
            'accuracy': accuracy,
            'entropy': avg_entropy,
            'logits_std': logits_std,
            'hidden_std': hidden_std,
            'hidden_norm': hidden_norm
        }
    
    @torch.no_grad()
    def generate(self,
                prompt_ids: Optional[torch.Tensor] = None,
                protein_ids: Optional[torch.Tensor] = None,
                protein_mask: Optional[torch.Tensor] = None,
                max_length: int = 256,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95,
                num_return_sequences: int = 1,
                repetition_penalty: float = 1.2,
                ngram_block_size: int = 3,
                apply_repetition_control: bool = True) -> torch.Tensor:
        """
        Generate molecular SMILES sequences autoregressively with repetition control.
        
        Args:
            prompt_ids: Optional prompt tokens [batch_size, prompt_len]
            protein_ids: Protein token IDs [batch_size, protein_seq_len] (optional)
            protein_mask: Protein attention mask [batch_size, protein_seq_len] (optional)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate
            repetition_penalty: Penalty factor for repeated tokens (default: 1.2)
            ngram_block_size: Size of n-grams to block (default: 3)
            apply_repetition_control: Whether to apply repetition penalties (default: True)
        
        Returns:
            Generated token IDs [num_sequences, seq_len]
        """
        device = next(self.parameters()).device
        
        if prompt_ids is None:
            # Start with BOS token
            prompt_ids = torch.tensor([[self.bos_token_id]], device=device)
        
        # Expand for multiple sequences
        if num_return_sequences > 1:
            prompt_ids = prompt_ids.repeat(num_return_sequences, 1)
            if protein_ids is not None:
                protein_ids = protein_ids.repeat(num_return_sequences, 1)
            if protein_mask is not None:
                protein_mask = protein_mask.repeat(num_return_sequences, 1)
        
        generated = prompt_ids
        finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)
        
        for _ in range(max_length - prompt_ids.size(1)):
            # Get logits for next token
            outputs = self.forward(generated, protein_ids=protein_ids, protein_mask=protein_mask)
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply repetition control if enabled
            if apply_repetition_control:
                # Apply token-level repetition penalty
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, 
                    generated, 
                    repetition_penalty
                )
                
                # Apply n-gram blocking
                ngram_mask = self._get_ngram_blocked_mask(generated, ngram_block_size)
                next_token_logits[ngram_mask] = -float('inf')
            
            # Apply top-k and top-p filtering
            filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
            
            # Sample next token
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS and mark finished sequences
            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
            
            # Replace next_token with PAD for finished sequences
            next_token[finished] = self.pad_token_id
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if finished.all():
                break
        
        return generated

    def _top_k_top_p_filtering(self, logits, top_k=50, top_p=0.95):
        """Apply top-k and nucleus (top-p) filtering to logits."""
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_tokens: torch.Tensor, 
        penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Apply repetition penalty to reduce probability of already-generated tokens.
        
        This helps prevent the model from generating long monotonous sequences
        like "CCCCCCCCCC..." by penalizing tokens that have already appeared.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            generated_tokens: Previously generated tokens [batch_size, seq_len]
            penalty: Penalty factor (>1.0 reduces probability, default: 1.2)
        
        Returns:
            Modified logits with repetition penalty applied
        """
        if penalty <= 1.0 or generated_tokens.size(1) == 0:
            return logits
        
        batch_size = logits.size(0)
        
        for batch_idx in range(batch_size):
            # Get unique tokens that have been generated
            unique_tokens = torch.unique(generated_tokens[batch_idx])
            
            # Apply penalty: divide logits by penalty factor
            # This reduces the probability of these tokens
            logits[batch_idx, unique_tokens] = logits[batch_idx, unique_tokens] / penalty
        
        return logits
    
    def _get_ngram_blocked_mask(
        self, 
        generated_sequence: torch.Tensor, 
        n: int = 3
    ) -> torch.Tensor:
        """
        Create mask for tokens that would create repeated n-grams.
        
        Prevents generating sequences with repeated substructures by blocking
        tokens that would complete an n-gram that already exists in the sequence.
        
        Args:
            generated_sequence: Already generated tokens [batch_size, seq_len]
            n: N-gram size (default: 3 for trigrams)
        
        Returns:
            Boolean mask [batch_size, vocab_size] where True = block this token
        """
        batch_size, seq_len = generated_sequence.shape
        vocab_size = self.config.vocab_size
        device = generated_sequence.device
        
        # Initialize mask (False = allowed, True = blocked)
        block_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
        
        if seq_len < n - 1:
            # Not enough tokens to form n-grams yet
            return block_mask
        
        for batch_idx in range(batch_size):
            # Get the last (n-1) tokens - these will form the prefix
            # We're checking if adding any token would create a repeated n-gram
            prefix = generated_sequence[batch_idx, -(n-1):].tolist()
            
            # Extract all existing n-grams from the sequence
            existing_ngrams = set()
            for i in range(seq_len - n + 1):
                ngram = tuple(generated_sequence[batch_idx, i:i+n].tolist())
                existing_ngrams.add(ngram)
            
            # Check each possible next token
            # Block tokens that would create an n-gram we've already seen
            for token_id in range(vocab_size):
                potential_ngram = tuple(prefix + [token_id])
                if potential_ngram in existing_ngrams:
                    block_mask[batch_idx, token_id] = True
        
        return block_mask
    
    def compute_loss_with_ppo(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        protein_ids: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ppo_trainer = None,
        teacher_forcing_prob: float = 1.0,
        ppo_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss combining teacher forcing reconstruction with PPO.
        
        This is the core of the hybrid training approach:
        1. Standard reconstruction loss (teacher forcing)
        2. PPO reinforcement learning loss (validity-driven)
        3. Weighted combination
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            protein_ids: Protein token IDs (optional)
            protein_mask: Protein mask (optional)
            ppo_trainer: PPOTrainer instance for RL updates
            teacher_forcing_prob: Probability of using teacher forcing (scheduled sampling)
            ppo_weight: Weight for PPO loss component (0-1)
        
        Returns:
            Dictionary with loss and metrics
        """
        # 1. Standard reconstruction loss (teacher forcing)
        recon_loss_dict = self.compute_loss(
            input_ids, attention_mask, protein_ids, protein_mask
        )
        reconstruction_loss = recon_loss_dict['loss']
        
        # 2. PPO loss (if enabled)
        if ppo_weight > 0 and ppo_trainer is not None:
            batch_size = input_ids.size(0)
            ppo_loss, ppo_metrics = ppo_trainer.compute_ppo_loss(
                batch_size=batch_size,
                protein_ids=protein_ids,
                protein_mask=protein_mask,
                num_rollouts=4  # Generate 4 sequences per batch item
            )
        else:
            ppo_loss = torch.tensor(0.0, device=reconstruction_loss.device)
            ppo_metrics = {
                'ppo_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'avg_reward': 0.0,
                'validity_rate': 0.0,
                'avg_qed': 0.0,
                'avg_sa': 0.0
            }
        
        # 3. Combine losses
        total_loss = (1 - ppo_weight) * reconstruction_loss + ppo_weight * ppo_loss
        
        # Combine metrics - keep loss as tensor for backprop, convert others to scalars
        result = {}
        for key, value in recon_loss_dict.items():
            if key == 'loss':
                # Skip the original loss, we'll use total_loss instead
                continue
            elif isinstance(value, torch.Tensor):
                result[key] = value.item()
            else:
                result[key] = value
        
        # IMPORTANT: Keep loss as tensor for backpropagation
        result['loss'] = total_loss
        # Store scalar version for logging
        result['loss_scalar'] = total_loss.item()
        result['reconstruction_loss'] = reconstruction_loss.item()
        result['ppo_weight'] = float(ppo_weight)
        result['teacher_forcing_prob'] = float(teacher_forcing_prob)
        
        # Add PPO metrics - ensure all are Python scalars
        for key, value in ppo_metrics.items():
            if isinstance(value, torch.Tensor):
                result[f'ppo_{key}'] = value.item()
            else:
                result[f'ppo_{key}'] = float(value)
        
        return result 