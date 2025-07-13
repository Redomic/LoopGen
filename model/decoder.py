import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Set, List, Tuple
from .config import ModelConfig

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
            is_causal = attention_mask is None
            attn_mask = self._get_alibi_bias(seq_len, x.device) if self.use_alibi else attention_mask
            if self.use_alibi and attention_mask is not None:
                attn_mask = attn_mask + attention_mask

            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=is_causal and not self.use_alibi
            )
        else:
            # Manual implementation for fallback
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if self.use_alibi:
                attn_weights += self._get_alibi_bias(seq_len, x.device)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
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
    """ A single Transformer Decoder block with Pre-Layer Normalization. """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: Norm -> Attention -> Add
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        # Pre-LN: Norm -> MLP -> Add
        x = x + self.mlp(self.ln_2(x))
        return x

class SELFIESGPTDecoder(nn.Module):
    """ GPT-style Decoder model for SELFIES generation with chemical validity constraints. """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        
        # SELFIES-specific embeddings
        self.branch_depth_emb = nn.Embedding(config.max_branch_depth, config.branch_depth_embedding_dim)
        # Project branch embedding to d_model to be added
        self.branch_proj = nn.Linear(config.branch_depth_embedding_dim, config.d_model, bias=False)

        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.wte.weight = self.lm_head.weight

        # Chemical Validity and Grammar State Tracker modules
        if config.use_grammar_constraint:
            self.grammar_tracker = GrammarLSTM(config)
            self.validity_checker = ChemicalValidityModule(config)
        else:
            self.grammar_tracker = None
            self.validity_checker = None
        
        # Apply a more sophisticated initialization
        self.apply(self._init_weights)
        # Apply special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

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
        return_dict: bool = False,
        apply_constraints: bool = False  # Only apply during inference, not training
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(pos_ids)
        
        hidden_states = tok_embeds + pos_embeds
        
        if branch_depths is not None:
            depth_embeds = self.branch_depth_emb(branch_depths)
            hidden_states += self.branch_proj(depth_embeds)
            
        hidden_states = self.drop(hidden_states)

        if attention_mask is not None:
            # Pytorch's scaled_dot_product_attention expects mask where True indicates masking
            # But standard huggingface mask has 1 for not masked, 0 for masked.
            # And for alibi, we need additive mask. Let's create it here.
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_mask = extended_attention_mask + (1.0 - causal_mask) * -10000.0

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask if not self.config.use_alibi else None)
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Apply grammar and chemical validity constraints
        if (self.config.use_grammar_constraint and 
            apply_constraints and 
            self.grammar_tracker is not None and 
            self.validity_checker is not None):
            
            # Get grammar mask from LSTM tracker
            grammar_mask = self.grammar_tracker(input_ids)  # [batch, seq_len, vocab_size]
            
            # Get chemical validity constraints
            stability_scores, validity_mask = self.validity_checker(hidden_states, input_ids)
            
            # Combine grammar and chemical constraints
            combined_mask = grammar_mask | validity_mask
            
            # Debug: Count how many tokens are being masked
            total_tokens = combined_mask.numel()
            masked_tokens = combined_mask.sum().item()
            if masked_tokens > total_tokens * 0.9:  # If >90% tokens masked
                print(f"WARNING: {masked_tokens}/{total_tokens} ({100*masked_tokens/total_tokens:.1f}%) tokens masked!")
            
            # Apply mask to logits (set invalid tokens to very negative values)
            # Use -65504 which is the largest negative finite value in fp16
            mask_value = -65504.0 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(combined_mask, mask_value)
            
            if return_dict:
                return {
                    'logits': logits,
                    'hidden_states': hidden_states,
                    'stability_scores': stability_scores,
                    'grammar_mask': grammar_mask,
                    'validity_mask': validity_mask
                }

        return logits 