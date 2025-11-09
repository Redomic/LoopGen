# LoopGen Project - Complete Technical Breakdown

## PROJECT OVERVIEW

LoopGen is a protein-conditioned molecular generation system using transformer-based deep learning.
Goal: Generate drug-like molecules tailored to protein binding sites.
Approach: Two-phase training with hybrid MLE + reinforcement learning.

## ARCHITECTURE

### High-Level Design
- Decoder-only transformer (GPT-style) for SMILES generation
- Optional protein encoder (transformer encoder) for conditioning
- Cross-attention between protein and molecule representations
- Hybrid training: supervised learning (MLE) + reinforcement learning (PPO)

### Model Components

1. SMILES Decoder (SMILESGPTDecoder)
   - Input: Token embeddings + positional embeddings
   - Architecture: N transformer blocks with causal attention
   - Output: Next-token logits over vocabulary
   - Key features:
     * Pre-layer normalization
     * SwiGLU activation (optional)
     * ALiBi positional encoding (optional)
     * Stochastic depth / drop path regularization
     * Label smoothing cross-entropy loss
   - Parameters (standard): ~114M parameters
     * d_model=768, n_heads=12, n_layers=12, d_ff=3072
   - Small config: ~20M parameters (d_model=256, n_heads=4, n_layers=6)
   - Large config: ~400M parameters (d_model=1280, n_heads=16, n_layers=24)

2. Protein Encoder (ProteinEncoder)
   - Input: Amino acid token IDs
   - Architecture: 6-layer transformer encoder (bidirectional)
   - Output: Contextualized protein embeddings
   - Vocab: 25 tokens (20 amino acids + 5 special tokens)
   - Used only in Phase 2

3. Cross-Attention Module (DecoderBlock.cross_attn)
   - Query: Molecule decoder hidden states
   - Key/Value: Protein encoder outputs
   - Applied every N layers (default: every layer)
   - Masked to ignore protein padding tokens

4. Language Model Head
   - Linear projection: d_model → vocab_size
   - Optionally tied with input embeddings
   - Produces logits for next-token prediction

## TOKENIZATION

### Three Modes

1. SPE (Substructure Tokenization) - Default
   - Uses SmilesPE library
   - Learns substructure tokens (e.g., benzene ring as single token)
   - Vocabulary: ~3000-5000 tokens
   - More efficient (fewer tokens per molecule)
   - Requires SPE_ChEMBL.txt vocabulary file
   - Command: default behavior

2. Atomwise Tokenization
   - Simple atom-level: C, N, O, [NH3+], =, #, etc.
   - Vocabulary: ~50-100 tokens
   - More intuitive, easier to learn
   - Better validity rates (60-80% vs 1-5% with SPE)
   - Command: --use_atomwise

3. SELFIES (100% Valid)
   - Self-referencing embedded strings
   - Guarantees syntactically valid molecules
   - Larger vocabulary (~170 tokens)
   - Command: --use_selfies
   - Requires: pip install selfies

### Special Tokens
- <PAD>: Padding token (ID 0)
- <BOS>: Begin-of-sequence
- <EOS>: End-of-sequence
- <MASK>: Masking (future use)
- <UNK>: Unknown tokens

### Protein Tokenization
- Simple character-level: one token per amino acid
- 20 standard amino acids: ACDEFGHIKLMNPQRSTVWY
- 5 special tokens (same as SMILES)
- Total vocab size: 25

## TRAINING OBJECTIVES

### Phase 1: SMILES Pretraining

Goal: Learn general molecular SMILES syntax and valid structures.
Data: Large SMILES dataset (e.g., 100K molecules)
Epochs: 5 (configurable with --phase1_epochs)

Loss Components:
1. Reconstruction Loss (MLE)
   - Autoregressive next-token prediction
   - L_recon = -∑ log P(x_t | x_{<t})
   - Label smoothing (ε=0.1) to prevent overconfidence
   - Implementation: LabelSmoothingCrossEntropy
     * loss = (1-ε) * NLL + ε * uniform_loss / vocab_size
   - Ignores padding tokens in loss computation

2. PPO Reinforcement Learning (optional, starts epoch 2)
   - Policy gradient optimization for validity
   - Clipped surrogate objective
   - Value function learning with MSE loss
   - Entropy regularization for exploration
   - Weight: progressively increases from 0.0 to 0.4

Hybrid Loss:
- total_loss = (1 - ppo_weight) * recon_loss + ppo_weight * ppo_loss
- Early epochs: mostly MLE
- Later epochs: balanced MLE + RL

### Phase 2: Protein Conditioning

Goal: Learn to generate molecules conditioned on protein pockets.
Data: Protein-ligand pairs with pocket sequences
Epochs: 5 (configurable with --phase2_epochs)

Weight Transfer:
- Decoder weights from Phase 1 loaded into Phase 2 model
- Protein encoder initialized from scratch
- Cross-attention layers initialized from scratch
- Optimizer reset with fresh learning rate

Loss: Same hybrid MLE + PPO as Phase 1
- Now conditioned on protein context
- Cross-attention activated

## REINFORCEMENT LEARNING DETAILS

### PPO (Proximal Policy Optimization)

Components:
1. Policy (Actor): The decoder model itself
2. Value Network: Separate MLP predicting state values
   - Input: decoder hidden states
   - Architecture: Linear(d_model, d_ff) → ReLU → Linear(d_ff, 1)
3. Reward Calculator: Evaluates generated molecules

Algorithm:
1. Generate rollouts (N=4-8 molecules per batch)
2. Compute rewards for each molecule
3. Calculate advantages using GAE (Generalized Advantage Estimation)
   - δ_t = r_t + γV(s_{t+1}) - V(s_t)
   - A_t = ∑(γλ)^l δ_{t+l}
   - γ=0.99 (discount), λ=0.95 (GAE parameter)
4. Update policy with clipped objective
   - ratio = π_new / π_old
   - L_policy = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
   - ε=0.2 (clip parameter)
5. Update value network
   - L_value = MSE(V(s), returns)
6. Add entropy bonus for exploration
   - L_entropy = -∑ π(a|s) log π(a|s)

Total PPO Loss:
- L_PPO = L_policy + 0.5 * L_value - 0.01 * L_entropy

### Reward Function

Three components (configurable weights):

1. Validity (weight=1.0)
   - Binary: 1.0 if RDKit can parse SMILES, 0.0 otherwise
   - Most important signal

2. QED (weight=0.0-0.2, optional)
   - Quantitative Estimate of Drug-likeness
   - Range: 0-1 (higher = more drug-like)
   - Based on MW, LogP, HBD, HBA, PSA, etc.
   - Computationally expensive (disabled by default)

3. SA Score (weight=0.0, disabled by default)
   - Synthetic Accessibility
   - Raw score: 1 (easy) to 10 (hard)
   - Normalized: (10 - SA) / 9
   - Very expensive to compute

Composite Reward:
- R = w_v * R_validity + w_q * R_QED + w_s * R_SA
- Default: R = 1.0 * validity (focus on validity only)

### Scheduled Sampling

Gradually transitions from teacher forcing to model predictions.

Strategies:
1. Linear: p(t) = 1 - t/T
2. Exponential: p(t) = k^t (k=0.95)
3. Inverse Sigmoid: p(t) = k / (k + exp(t/k)) [recommended]
   - Smooth S-curve transition
   - Stable early training
   - Gradual independence

Default: Inverse sigmoid with warmup
- Full teacher forcing for first 2 epochs
- Then gradual decay

Teacher Forcing Probability (example, 5 epochs):
- Epoch 0-1: 1.0 (100% teacher forcing)
- Epoch 2: 0.5
- Epoch 3: 0.27
- Epoch 4: 0.12
- Epoch 5: 0.05

## GRAMMAR CONSTRAINTS

### SMILESGrammarConstraints

State-machine for enforcing SMILES syntax during generation.

Token Categories:
- Atoms: C, N, O, S, P, F, Cl, Br, I, B
- Aromatic atoms: c, n, o, s, p, b
- Bracket atoms: [C@@H], [NH3+], [O-], etc.
- Bonds: =, #, - (single bond implicit)
- Stereochemistry: /, \
- Ring digits: 0-9, %10-%99
- Parentheses: (, )

Grammar Rules:
1. Start token must be followed by atom
2. Bonds cannot appear at sequence start
3. Bonds must be followed by atoms
4. Ring closures must be paired (opening and closing)
5. Parentheses must be balanced
6. Aromatic atoms must be in rings
7. Stereochemistry only valid after atoms

State Tracking:
- Current position in sequence
- Open ring closures (dict of ring_num → first_occurrence_idx)
- Open parentheses count
- Last token type

Constraint Application:
- During generation: filter logits to valid tokens only
- Invalid token logits set to -inf
- Ensures generated SMILES follow syntax rules

Effectiveness:
- Improves validity from ~60% to ~80% (atomwise tokenization)
- Less effective with SPE tokenization (substructure tokens)

## DATA AUGMENTATION

### SMILES Augmentation (Bjerrum 2017)

Randomized SMILES for robustness.

How it works:
- Same molecule, different atom orderings
- Example: "CC(C)CCO" → ["OCC(C)C", "C(C)CCO", "CC(C)CCO"]
- Uses RDKit's doRandom=True in MolToSmiles

Benefits:
- Reduces overfitting
- Improves generalization
- Model learns canonical molecular representation
- Increases effective dataset size by 2-3x

Usage:
- Command: --use_smiles_augmentation --augmentation_factor 2
- Augmentation factor: number of variants per molecule
- Only applied to training data (not validation)

Implementation:
- On-the-fly augmentation in dataset iterator
- Each epoch sees different SMILES variants
- Original + N random variants

## TRAINING PIPELINE

### Multi-Phase Training

Command:
```bash
python train.py \
  --data_path data/smiles_100k.csv \
  --enable_multiphase \
  --phase1_epochs 5 \
  --phase2_epochs 5 \
  --protein_ligand_data_path data/output/protein_ligand_training.csv
```

Flow:
1. Phase 1: SMILES pretraining
   - Load SMILES dataset
   - Train decoder-only model
   - Save phase1_final.pt

2. Weight Transfer
   - Load Phase 1 decoder weights
   - Initialize Phase 2 model with protein encoder
   - Transfer decoder weights (163 tensors)
   - Protein encoder trained from scratch

3. Phase 2: Protein conditioning
   - Load protein-ligand dataset
   - Train with cross-attention active
   - Save phase2 checkpoints

### Single-Phase Training

Command:
```bash
python train.py \
  --data_path data/smiles_100k.csv \
  --num_epochs 50
```

Just Phase 1 (SMILES-only) without protein conditioning.

### Training Loop Details

Per Epoch:
1. Training
   - Iterate batches from dataloader
   - Forward pass through model
   - Compute hybrid loss (MLE + PPO)
   - Backward pass with gradient scaling (AMP)
   - Gradient accumulation (if configured)
   - Gradient clipping (max_norm=1.0)
   - Optimizer step
   - Scheduler step (ReduceLROnPlateau)

2. Validation
   - Same as training but no backprop
   - Compute metrics: loss, perplexity, accuracy

3. Generation & Evaluation
   - Generate N=10 molecules (every 5 epochs)
   - Calculate validity, uniqueness, QED
   - Save to generated_molecules_phaseX_epYYY.txt

4. Checkpointing
   - Save best model (lowest val loss)
   - Save epoch checkpoint with full state
   - Save metrics to metrics_phaseX.jsonl

### Optimization

Optimizer: AdamW
- Learning rate: 3e-4 (default)
- Weight decay: 0.1
- Betas: (0.9, 0.999)

Scheduler: ReduceLROnPlateau
- Mode: minimize loss
- Factor: 0.8 (reduce by 20%)
- Patience: 3 epochs
- Min LR: 1e-6

Gradient Clipping:
- Max norm: 1.0
- Applied before optimizer step

Mixed Precision (AMP):
- Optional: --use_amp
- Uses torch.amp GradScaler
- Speeds up training ~30-40%

### Training Stabilization

TrainingStabilizer monitors:
- Loss history (smoothness, convergence)
- Gradient norms (exploding gradients)
- Learning rate history

Interventions (if pathological behavior detected):
- Reduce learning rate by 50%
- Increase gradient clipping
- Skip update if gradients too large
- Maximum 3 interventions per run

Pathological Indicators:
- Loss < 1e-6 (numerical instability)
- Gradients > 10.0 (exploding)
- Non-finite loss (NaN/Inf)
- Complete stagnation (200+ steps no improvement)

## GENERATION

### Sampling Strategies

Autoregressive sampling with multiple controls:

1. Temperature Scaling (default=0.85)
   - logits = logits / temperature
   - Higher T: more diverse, less valid
   - Lower T: more valid, less diverse

2. Top-k Sampling (default=50)
   - Keep only top k logits
   - Set others to -inf
   - Filters out unlikely tokens

3. Top-p (Nucleus) Sampling (default=0.9)
   - Keep smallest set of tokens with cumulative prob ≥ p
   - Dynamic vocabulary size
   - More principled than top-k

4. Repetition Penalty (default=1.1)
   - Penalize tokens already in sequence
   - penalty = logits / repetition_penalty^(count)
   - Reduces repetitive patterns

5. N-gram Blocking (default=3)
   - Block tokens that would create repeated n-grams
   - Prevents "CCC..." or "121212..."

6. Grammar Constraints (if enabled)
   - Filter logits to syntactically valid tokens
   - Uses SMILESGrammarConstraints

Generation Process:
1. Start with <BOS> token
2. Forward pass → logits
3. Apply temperature, top-k, top-p
4. Apply repetition penalty
5. Apply n-gram blocking
6. Apply grammar constraints
7. Sample token from filtered distribution
8. Append to sequence
9. Repeat until <EOS> or max_length

### Protein-Conditioned Generation

Script: generate_molecules.py

Command:
```bash
python generate_molecules.py \
  --protein_sequence "MKTAYIAKQRQISFVKSHFSRQLE" \
  --checkpoint checkpoints/run_XXX/phase2_final.pt \
  --vocab checkpoints/vocab.json \
  --num_samples 10
```

Process:
1. Load Phase 2 model with protein encoder
2. Tokenize protein sequence
3. Encode protein → protein_embeddings
4. Generate molecules with cross-attention to protein
5. Validate with RDKit
6. Calculate properties (MW, LogP, QED, etc.)
7. Display results

Properties Calculated:
- Validity (RDKit parsing)
- Molecular weight
- LogP (lipophilicity)
- HBD/HBA (hydrogen bond donors/acceptors)
- TPSA (topological polar surface area)
- Rotatable bonds
- Ring count
- QED (drug-likeness)
- Lipinski violations (Rule of Five)

## EVALUATION METRICS

### During Training

1. Loss
   - Reconstruction loss (cross-entropy)
   - PPO loss (if RL enabled)
   - Total loss (weighted combination)

2. Perplexity
   - exp(loss)
   - Measures model uncertainty
   - Lower is better

3. Accuracy
   - Token-level accuracy
   - Correct predictions / total tokens
   - Ignores padding tokens

4. Entropy
   - Average entropy of output distribution
   - Higher: more uncertain/diverse
   - Lower: more confident/deterministic

5. Validity Rate (RL only)
   - Fraction of generated molecules that RDKit can parse
   - Key metric for RL effectiveness

6. Average QED (RL only)
   - Mean QED of valid generated molecules
   - Range: 0-1

7. Average SA Score (RL only, if enabled)
   - Mean synthetic accessibility
   - Normalized to 0-1

### After Generation

1. Validity
   - Percentage of valid SMILES
   - Target: >60% (atomwise), >80% (with RL+grammar)

2. Uniqueness
   - Percentage of unique molecules
   - Target: >90%

3. Novelty
   - Fraction not in training set
   - Measured via exact SMILES match
   - Target: >90%

4. Diversity
   - Average Tanimoto distance between molecules
   - Based on Morgan fingerprints
   - Range: 0 (identical) to 1 (completely different)

5. QED Distribution
   - Mean, median, range of QED scores
   - Target: mean >0.5 (drug-like)

## FILE STRUCTURE

### Code Organization

```
LoopGen/
├── model/
│   ├── config.py              # ModelConfig dataclass
│   ├── decoder.py             # SMILESGPTDecoder, transformer blocks
│   ├── protein_encoder.py     # ProteinEncoder
│   ├── rl_trainer.py          # PPOTrainer, MolecularRewardCalculator
│   ├── scheduled_sampling.py  # ScheduledSamplingScheduler
│   └── smiles_grammar.py      # SMILESGrammarConstraints
├── molecule_utils/
│   ├── tokenizer.py           # SMILESTokenizer (SPE/atomwise/SELFIES)
│   ├── protein_tokenizer.py   # ProteinTokenizer
│   ├── dataset.py             # SMILESDataset
│   ├── protein_ligand_dataset.py  # ProteinLigandDataset
│   ├── augmentation.py        # SMILES augmentation
│   ├── smiles_validator.py    # Validation utilities
│   └── convert.py             # Format conversions
├── data/
│   ├── prepare_data.py        # Data preprocessing
│   ├── prepare_crossdock.py   # CrossDocked dataset prep
│   └── clean_smiles.py        # SMILES cleaning
├── train.py                   # Main training script
├── generate_molecules.py      # Generation script
├── requirements.txt           # Dependencies
└── README.md
```

### Checkpoint Structure

```
checkpoints/run_20251103_035000/
├── config.json                  # Full run configuration
├── training.log                 # Detailed training logs
├── metrics_phase1.jsonl         # Phase 1 metrics (one JSON per line)
├── metrics_phase2.jsonl         # Phase 2 metrics
├── phase1_final.pt              # Phase 1 final checkpoint
├── phase2_final.pt              # Phase 2 final checkpoint
├── best_model_phase1.pt         # Best Phase 1 model (lowest val loss)
├── best_model_phase2.pt         # Best Phase 2 model
├── model_phase1_ep000.pt        # Epoch checkpoints (Phase 1)
├── model_phase1_ep001.pt
├── ...
├── model_phase2_ep000.pt        # Epoch checkpoints (Phase 2)
├── ...
├── generated_molecules_phase1_ep001.txt  # Generated samples
├── generated_molecules_phase1_ep005.txt
├── generated_molecules_phase2_ep001.txt
└── generated_molecules_phase2_ep005.txt
```

### Checkpoint Contents

Epoch checkpoint (.pt file):
- model_state_dict: Model weights
- optimizer_state_dict: Optimizer state
- scheduler_state_dict: LR scheduler state
- scaler_state_dict: AMP scaler state
- epoch: Current epoch number
- best_val_loss: Best validation loss so far
- config: ModelConfig object

Final checkpoint (phase1_final.pt, phase2_final.pt):
- model_state_dict: Model weights only
- config: ModelConfig object
- best_val_loss: Final best validation loss

## HYPERPARAMETERS

### Model Architecture

Small:
- d_model: 256
- n_heads: 4
- n_layers: 6
- d_ff: 1024
- Parameters: ~20M

Standard:
- d_model: 768
- n_heads: 12
- n_layers: 12
- d_ff: 3072
- Parameters: ~114M

Large:
- d_model: 1280
- n_heads: 16
- n_layers: 24
- d_ff: 5120
- Parameters: ~400M

### Training

Learning rate: 3e-4
Weight decay: 0.1
Batch size: 64 (default)
Grad accumulation: 1 (default)
Grad clip: 1.0
Label smoothing: 0.1
Dropout: 0.1
Attention dropout: 0.05
Stochastic depth: 0.1

### Sequences

Max SMILES length: 256
Max protein length: 512

### RL Hyperparameters

PPO clip epsilon: 0.2
Value coefficient: 0.5
Entropy coefficient: 0.01
RL start epoch: 2
RL max weight: 0.4
Num rollouts: 4-8
Max rollout length: 100

Reward weights:
- Validity: 1.0
- QED: 0.0-0.2
- SA: 0.0 (disabled)

Scheduled sampling:
- Type: inverse_sigmoid
- Warmup epochs: 2

### Generation

Temperature: 0.85
Top-k: 50
Top-p: 0.9
Repetition penalty: 1.1
N-gram block size: 3
Max length: 160

## DATASETS

### SMILES Dataset (Phase 1)

Format: CSV file, one SMILES per line
Example:
```
CC(C)CCO
c1ccccc1
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

Sources:
- PubChem (download_pubchem.sh)
- ChEMBL
- ZINC
- Custom filtered datasets

Preprocessing:
- Remove invalid SMILES (RDKit validation)
- Remove duplicates
- Filter by molecular weight (50-500 Da typical)
- Remove salts, mixtures
- Canonicalize (optional)

### Protein-Ligand Dataset (Phase 2)

Format: CSV with columns: SMILES, pocket_sequence, affinity (optional), pair_id (optional)

Example:
```
SMILES,pocket_sequence,affinity
CC(C)CCO,MKTAYIAKQRQISFVKSHFSRQLE,5.2
c1ccccc1,GARAVTLSNPEF,6.8
```

Sources:
- CrossDocked (docking poses + pocket extraction)
- PDBbind (binding affinity data)
- BindingDB
- Custom protein-ligand pairs

Pocket Extraction:
- Extract residues within 6Å of ligand
- Typical pocket size: 50-200 residues
- Use prepare_crossdock.py script

## IMPLEMENTATION DETAILS

### Key Architectural Choices

1. Pre-Layer Normalization
   - Norm applied before attention/MLP (not after)
   - More stable training
   - Better gradient flow
   - Standard in modern transformers

2. SwiGLU Activation
   - Gated Linear Unit with Swish activation
   - Better than GELU/ReLU for language models
   - Formula: SwiGLU(x) = (Wx ⊙ σ(Vx)) · Ux
   - Used in PaLM, LLaMA

3. Stochastic Depth / Drop Path
   - Randomly drop transformer blocks during training
   - Improves regularization
   - Better than simple dropout for deep models
   - Drop probability increases with depth

4. Label Smoothing
   - Target distribution: (1-ε) * one-hot + ε / vocab_size
   - ε = 0.1
   - Prevents overconfident predictions
   - Improves calibration and diversity

5. Causal Masking
   - Self-attention only sees past tokens
   - Implemented via attention mask
   - Essential for autoregressive generation

6. Cross-Attention
   - Query from molecule decoder
   - Key/Value from protein encoder
   - Applied every N layers (N=1 default)
   - Allows molecule to "attend to" protein

### Training Efficiency Tricks

1. Dynamic Padding
   - Pad to max length in batch (not global max)
   - Reduces wasted computation
   - Implemented in collate_fn

2. Pin Memory
   - pin_memory=True in DataLoader
   - Faster CPU→GPU transfer
   - Recommended for GPU training

3. Mixed Precision (AMP)
   - Float16 computation, Float32 master weights
   - ~30-40% speedup
   - Minimal accuracy loss
   - Use torch.amp.autocast

4. Gradient Accumulation
   - Accumulate gradients over N batches
   - Effective batch size = batch_size * N
   - Useful for large models / limited GPU memory

5. Gradient Checkpointing (not yet implemented)
   - Trade compute for memory
   - Recompute activations during backward
   - Enables larger models

6. Data Streaming
   - IterableDataset for large files
   - Don't load entire dataset into RAM
   - Shuffle buffer for randomization

7. Multi-Worker DataLoader
   - num_workers=4 (default)
   - Parallel data loading
   - Reduces data loading bottleneck

### Numerical Stability

1. Logit Clamping
   - Clamp logits to [-10, 10]
   - Prevents overflow in softmax
   - Applied before loss computation

2. Gradient Clipping
   - Max norm: 1.0
   - Prevents exploding gradients
   - Essential for RL training

3. Layer Norm Epsilon
   - eps=1e-5
   - Prevents division by zero
   - Standard value

4. Loss Scaling (AMP)
   - Dynamic loss scaling in GradScaler
   - Prevents underflow in float16
   - Automatic adjustment

5. Attention Scaling
   - Scale attention logits by 1/√d_k
   - Prevents softmax saturation
   - Standard in transformers

### Memory Management

1. Batch Size Tuning
   - Standard model: batch_size=64 fits on 24GB GPU
   - Small model: batch_size=128
   - Large model: batch_size=32 or use grad accumulation

2. Sequence Length
   - Trim sequences to 95th percentile
   - Most SMILES <100 tokens
   - Max length=256 is conservative

3. RL Memory
   - PPO generates N rollouts per batch
   - Each rollout creates computation graph
   - Apply PPO every 5th batch (not every batch)
   - 40-50% speedup vs every-batch PPO

4. Clear Unused Tensors
   - Explicit del statements in RL code
   - Free intermediate tensors
   - Reduces memory fragmentation

## COMMON ISSUES & SOLUTIONS

### Low Validity

Problem: Generated molecules <10% valid
Causes:
- SPE tokenization (substructure tokens hard to learn)
- No grammar constraints
- Insufficient training

Solutions:
- Use atomwise tokenization (--use_atomwise)
- Enable grammar constraints (--use_strong_grammar)
- Enable RL training (--use_rl_training)
- Train longer (more epochs)
- Use SELFIES (--use_selfies) for 100% validity

### Repetitive Molecules

Problem: Model generates same molecules repeatedly
Causes:
- Model collapsed to mode
- Overconfident predictions
- No diversity pressure

Solutions:
- Increase temperature (0.9-1.0)
- Increase top-p (0.95)
- Use repetition penalty (1.2-1.5)
- Add entropy regularization (already in code)
- Use label smoothing (already enabled)

### Training Instability

Problem: Loss spikes, NaN loss, exploding gradients
Causes:
- Learning rate too high
- RL weight too high too early
- Numerical overflow

Solutions:
- Reduce learning rate (1e-4)
- Use warmup for RL (--rl_start_epoch 5)
- Enable TrainingStabilizer (already enabled)
- Use gradient clipping (already enabled)
- Enable mixed precision carefully

### Out of Memory

Problem: CUDA OOM during training
Causes:
- Batch size too large
- Model too large
- RL rollouts too many

Solutions:
- Reduce batch size (--batch_size 32)
- Use gradient accumulation (--grad_accumulation_steps 2)
- Use smaller model (--model_size small)
- Reduce RL rollouts (--ppo_num_rollouts 4)
- Reduce sequence length (--max_seq_len 128)

### Slow Training

Problem: Training very slow (<1 batch/sec)
Causes:
- Too many data workers
- SA score calculation (very expensive)
- QED calculation (moderately expensive)

Solutions:
- Set num_workers=4 (not too high)
- Disable SA score (--reward_sa_weight 0.0) [default]
- Disable QED (--reward_qed_weight 0.0) if not needed
- Use AMP (--use_amp)
- Apply PPO less frequently (hardcoded: every 5th batch)

## EXPECTED RESULTS

### Phase 1 (SMILES Pretraining)

Epoch 1:
- Loss: ~2.0, Perplexity: ~8.0, Accuracy: ~0.5
- Validity: 10-20%

Epoch 5:
- Loss: ~0.8, Perplexity: ~2.3, Accuracy: ~0.95
- Validity: 70-84% (with RL + grammar)

### Phase 2 (Protein Conditioning)

Epoch 1:
- Loss: ~0.7-0.8 (benefits from Phase 1 transfer)
- Accuracy: ~0.94-0.95
- Validity: 60-70%

Epoch 5:
- Loss: ~0.6-0.7
- Accuracy: ~0.95-0.96
- Validity: 75-84%

### Generation Quality

Validity: 70-84% (atomwise + RL + grammar)
Uniqueness: 85-95%
Novelty: >95% (not in training set)
QED: 0.4-0.7 average (drug-like range: 0.5-0.9)

Sample molecules (Phase 2):
- Should be chemically valid
- Should have appropriate complexity (~20-40 heavy atoms)
- Should follow drug-like rules (Lipinski)

## COMPARING TO BASELINES

### Naive Transformer (No RL, No Grammar)
- Validity: 5-20%
- Uniqueness: 70-80%
- Mostly learns syntax but poor validity

### With RL (No Grammar)
- Validity: 40-60%
- Uniqueness: 80-90%
- RL strongly improves validity

### With Grammar (No RL)
- Validity: 50-70%
- Uniqueness: 85-95%
- Grammar constrains but doesn't optimize

### With RL + Grammar (LoopGen)
- Validity: 70-84%
- Uniqueness: 85-95%
- Best of both: constraints + optimization

### SELFIES (No RL needed)
- Validity: 100% (guaranteed)
- Uniqueness: 80-90%
- Larger vocabulary, different token distribution
- Good baseline for guaranteed validity

## THEORETICAL FOUNDATIONS

### Maximum Likelihood Estimation (MLE)

Objective: Maximize P(X | θ)
- X: sequence of tokens
- θ: model parameters

Loss: -log P(X | θ) = -∑_t log P(x_t | x_{<t}, θ)
- Cross-entropy loss
- Next-token prediction
- Autoregressive factorization

Advantages:
- Simple, stable
- Well-understood
- Good for learning distribution

Disadvantages:
- Exposure bias (teacher forcing)
- No explicit validity optimization
- May generate invalid sequences

### Reinforcement Learning (PPO)

Formulation:
- State s_t: partial sequence x_{<t}
- Action a_t: next token x_t
- Reward R(s, a): delayed, only at end of sequence
- Policy π(a|s): model's token distribution

Objective: Maximize expected reward
- J(θ) = E_{x~π_θ}[R(x)]

PPO Clipped Objective:
- Prevents large policy updates
- Stable training
- Better than vanilla policy gradient

Advantages:
- Direct optimization of task objective (validity)
- Reduces exposure bias
- Can optimize non-differentiable metrics

Disadvantages:
- High variance (Monte Carlo sampling)
- Expensive (multiple rollouts)
- Can be unstable

### Hybrid Training (MLE + PPO)

Combines strengths of both:
- MLE: stable, learns syntax and distribution
- PPO: optimizes validity and drug-likeness

Loss: L = (1-α) * L_MLE + α * L_PPO
- α increases from 0 to 0.4 over epochs
- Early: mostly MLE (learn syntax)
- Late: balanced (optimize validity)

Why it works:
- MLE provides strong base distribution
- PPO fine-tunes for specific objectives
- Scheduled transition prevents instability

### Scheduled Sampling

Addresses exposure bias in MLE.

Problem:
- Training: model sees ground truth tokens
- Inference: model sees its own predictions
- Distribution mismatch → poor generation

Solution:
- Gradually mix ground truth and predictions during training
- Probability of ground truth: p(epoch)
- p(0) = 1.0 (full teacher forcing)
- p(T) = 0.0 (full model predictions)

Inverse Sigmoid Schedule:
- p(t) = k / (k + exp(t/k))
- Smooth transition
- Stable early training
- Better than abrupt switching

### Cross-Attention for Conditioning

Mechanism:
- Query: What does the molecule need?
- Key: What does the protein have?
- Value: Protein features to use

Math:
- Q = W_q * H_mol (molecule hidden states)
- K = W_k * H_protein (protein hidden states)
- V = W_v * H_protein
- Attention(Q, K, V) = softmax(QK^T / √d_k) * V

Why it works:
- Molecule can selectively attend to relevant protein residues
- Differentiable, end-to-end trainable
- Proven effective in machine translation, image captioning

### Transformer Architecture

Self-Attention:
- Captures long-range dependencies
- O(n²) complexity in sequence length
- Parallelizable (unlike RNNs)

Multi-Head Attention:
- Multiple attention "heads"
- Each head learns different patterns
- Heads concatenated and projected

Positional Encoding:
- Transformers have no inherent position info
- Add positional embeddings to input
- Standard: sinusoidal or learned

Feed-Forward Network:
- Point-wise fully connected layers
- Applied after attention
- Expands and contracts hidden dimension

Layer Normalization:
- Normalizes across features (not batch)
- Stabilizes training
- Pre-LN: norm before attention/FFN

Residual Connections:
- Skip connections around each sublayer
- Enables gradient flow
- Allows training very deep models

## PAPER WRITING TIPS

Key points to emphasize:

1. Problem & Motivation
   - De novo drug design is hard
   - Protein-specific molecules needed
   - Existing methods limited

2. Approach
   - Transformer-based generation
   - Two-phase training (pretraining + conditioning)
   - Hybrid MLE + RL optimization
   - Grammar constraints for validity

3. Novel Contributions
   - Cross-attention for protein conditioning
   - Hybrid training schedule (MLE → MLE+PPO)
   - SOTA grammar constraints for SMILES
   - Scheduled sampling integration

4. Results
   - 70-84% validity (strong for SMILES generation)
   - Protein-conditioned molecules
   - Drug-like properties (QED, Lipinski compliance)
   - Comparison to baselines

5. Ablations
   - RL vs no RL
   - Grammar vs no grammar
   - Atomwise vs SPE tokenization
   - Phase 1 only vs Phase 1+2

6. Limitations
   - Validity not 100% (SELFIES can achieve this)
   - Binding affinity not predicted
   - No 3D structure information
   - SA score optimization expensive

7. Future Work
   - Integrate docking scores
   - 3D protein structure (graph neural networks)
   - Multi-objective optimization (affinity + ADMET)
   - Active learning for data efficiency

## VIVA PREPARATION

Common questions:

1. Why transformer instead of RNN?
   - Parallelizable (faster training)
   - Better long-range dependencies
   - Proven effective in NLP
   - Standard for sequence modeling

2. Why autoregressive generation?
   - Natural for sequential molecules
   - Can enforce grammar constraints
   - Flexible generation control
   - Standard in language modeling

3. Why two-phase training?
   - Phase 1: Learn molecular syntax (large dataset)
   - Phase 2: Learn protein conditioning (smaller dataset)
   - Prevents overfitting on limited protein-ligand data
   - Enables transfer learning

4. Why PPO instead of other RL algorithms?
   - Stable (clipped objective)
   - Sample efficient (compared to A3C)
   - Standard for policy gradient methods
   - Proven in language generation tasks

5. Why hybrid MLE + RL?
   - MLE alone: good syntax, poor validity
   - RL alone: unstable, high variance
   - Hybrid: stable + optimized
   - Gradual transition prevents collapse

6. How do you ensure validity?
   - RL with validity reward
   - Grammar constraints
   - Atomwise tokenization (simpler syntax)
   - SELFIES option (100% valid)

7. How do you measure success?
   - Validity rate (can RDKit parse?)
   - Uniqueness (diverse molecules?)
   - Drug-likeness (QED score)
   - Lipinski compliance (drug-like properties)

8. What if protein conditioning doesn't work?
   - Check weight transfer (decoder loaded?)
   - Check cross-attention (activated?)
   - Check protein tokenization (valid sequences?)
   - May need more Phase 2 epochs

9. Computational requirements?
   - GPU: 24GB VRAM (standard model)
   - Training time: ~3-5 hours (5+5 epochs)
   - Generation: <1 second per molecule
   - Can use smaller model or CPU for inference

10. Ethical considerations?
    - Generated molecules not guaranteed safe
    - Requires experimental validation
    - Potential dual-use concerns (toxins)
    - Should emphasize research use only

## COMMAND REFERENCE

### Training

Basic SMILES training:
```bash
python train.py --data_path data/smiles.csv --num_epochs 50
```

Multi-phase with RL:
```bash
python train.py \
  --data_path data/smiles_100k.csv \
  --enable_multiphase \
  --phase1_epochs 5 \
  --phase2_epochs 5 \
  --protein_ligand_data_path data/output/protein_ligand_training.csv \
  --use_rl_training \
  --use_strong_grammar
```

With augmentation:
```bash
python train.py \
  --data_path data/smiles.csv \
  --use_smiles_augmentation \
  --augmentation_factor 2
```

SELFIES mode (100% valid):
```bash
python train.py \
  --data_path data/smiles.csv \
  --use_selfies
```

### Generation

Protein-conditioned:
```bash
python generate_molecules.py \
  --protein_sequence "MKTAYIAKQRQISFVKSHFSRQLE" \
  --checkpoint checkpoints/run_XXX/phase2_final.pt \
  --vocab checkpoints/vocab.json \
  --num_samples 20 \
  --temperature 0.85
```

### Data Preparation

Clean SMILES:
```bash
python data/clean_smiles.py --input raw.csv --output clean.csv
```

Prepare CrossDocked data:
```bash
python data/prepare_crossdock.py \
  --crossdocked_dir data/crossdocked \
  --output data/output/protein_ligand_training.csv
```

## DEPENDENCIES

Core:
- torch >= 2.0
- transformers (Hugging Face)
- numpy
- pandas

Chemistry:
- rdkit >= 2022.09
- SmilesPE (for SPE tokenization)
- selfies (for SELFIES mode, optional)

RL:
- All built-in (no external RL library)

Visualization (optional):
- matplotlib
- seaborn

Install:
```bash
pip install -r requirements.txt
```

RDKit (recommended via conda):
```bash
conda install -c conda-forge rdkit
```

## FINAL NOTES

This project demonstrates:
- Strong understanding of transformers
- Knowledge of molecular representation
- Practical RL implementation
- End-to-end ML pipeline
- Research best practices (checkpointing, logging, evaluation)

Key strengths:
- Well-structured code
- Comprehensive documentation
- Multiple training modes
- Robust evaluation
- Production-ready features

For UG capstone:
- Scope is appropriate (challenging but achievable)
- Demonstrates technical depth
- Has practical application
- Includes research elements (hybrid training, grammar constraints)
- Shows software engineering skills

Good luck with your viva!

