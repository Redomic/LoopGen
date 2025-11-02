import dataclasses
from typing import Optional

@dataclasses.dataclass
class ModelConfig:
    """
    Configuration for the SMILES GPT-style decoder model.
    """
    # Core model architecture
    vocab_size: int = 512
    d_model: int = 768  # Hidden size
    n_heads: int = 12   # Number of attention heads
    n_layers: int = 12  # Number of transformer layers
    d_ff: int = 3072    # Feed-forward intermediate size
    max_seq_len: int = 256  # Maximum sequence length
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.05
    layer_norm_eps: float = 1e-5
    stochastic_depth_prob: float = 0.1
    
    # Advanced architectural features
    use_alibi: bool = True              # ALiBi positional encoding
    use_flash_attention: bool = True    # Flash Attention optimization
    use_swiglu: bool = True            # SwiGLU activation function
    use_pre_layer_norm: bool = True     # Pre-Layer Normalization
    use_cache: bool = True              # Key-value caching
    tie_word_embeddings: bool = False   # Tie input/output embeddings
    
    # SMILES-specific features (grammar constraints optional for SMILES)
    use_grammar_constraint: bool = False
    branch_depth_embedding_dim: int = 64
    max_branch_depth: int = 16
    
    # Training configuration
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 50000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Advanced features (for future implementation)
    use_early_exit: bool = False
    early_exit_threshold: float = 0.95
    use_expert_routing: bool = False
    num_experts: int = 2
    
    # Protein conditioning features
    use_protein_conditioning: bool = False
    protein_max_seq_len: int = 512  # Pocket sequences typically 50-300 residues
    protein_encoder_layers: int = 6
    protein_encoder_heads: int = 8
    protein_vocab_size: int = 25  # 20 amino acids + 5 special tokens
    use_cross_attention: bool = True
    cross_attention_freq: int = 1  # Apply cross-attn every N layers (1 = every layer)
    
    # RL Training Configuration
    use_rl_training: bool = True
    rl_weight_schedule: str = 'progressive'  # 'progressive' or 'fixed'
    rl_start_epoch: int = 5  # Start RL after N epochs
    rl_max_weight: float = 0.5  # Maximum RL loss weight
    ppo_clip_epsilon: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_num_rollouts: int = 4  # Samples per batch
    ppo_max_rollout_length: int = 100  # Max tokens in RL rollouts
    
    # Reward Function Weights
    reward_validity_weight: float = 1.0
    reward_qed_weight: float = 0.0  # Disabled by default for speed (QED calculation is slow)
    reward_sa_weight: float = 0.0   # Disabled by default for speed (SA score is very slow)
    
    # Scheduled Sampling
    scheduled_sampling_type: str = 'inverse_sigmoid'  # 'linear', 'exponential', 'inverse_sigmoid'
    scheduled_sampling_k: Optional[float] = None  # Auto-computed based on total epochs
    scheduled_sampling_warmup: int = 0  # Epochs of full teacher forcing before decay
    
    # Repetition Control (Inference only)
    repetition_penalty: float = 1.2
    ngram_block_size: int = 3

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        # Grammar/chemistry constraints note for SMILES
        if self.use_grammar_constraint:
            print("Warning: use_grammar_constraint not yet implemented for SMILES generation.")

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self):
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def standard_config(cls) -> 'ModelConfig':
        """Returns standard model configuration."""
        return cls(
            vocab_size=512,
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=3072,
            max_seq_len=256
        )

    @classmethod
    def large_config(cls) -> 'ModelConfig':
        """Returns large model configuration, inspired by recent research."""
        return cls(
            vocab_size=512,
            d_model=1280,
            n_heads=16,
            n_layers=24,
            d_ff=1280 * 4,
            max_seq_len=256,
            stochastic_depth_prob=0.1  # Add regularization for the larger model
        )

    @classmethod
    def small_config(cls) -> 'ModelConfig':
        """Returns a small, robust model configuration for rapid prototyping."""
        return cls(
            vocab_size=512,
            d_model=256,
            n_heads=4,
            n_layers=6,
            d_ff=1024,
            max_seq_len=256,
            dropout=0.1,
            stochastic_depth_prob=0.05
        )

# Legacy compatibility functions
def get_standard_config() -> ModelConfig:
    """Returns the standard model configuration."""
    return ModelConfig.standard_config()

def get_large_config() -> ModelConfig:
    """Returns the large model configuration."""
    return ModelConfig.large_config()

def get_small_config() -> ModelConfig:
    """Returns the small model configuration."""
    return ModelConfig.small_config() 