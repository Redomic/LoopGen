"""
Reinforcement Learning components for molecular generation.

This module implements PPO (Proximal Policy Optimization) training with
comprehensive reward functions including chemical validity, QED drug-likeness,
and synthetic accessibility scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

# RDKit imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. RL training will be limited.")

# SA Score import
try:
    from rdkit.Contrib.SA_Score import sascorer
    SA_SCORE_AVAILABLE = True
except ImportError:
    SA_SCORE_AVAILABLE = False
    print("Warning: SA Score not available. Install RDKit contrib.")


class MolecularRewardCalculator:
    """
    Calculate comprehensive rewards for generated molecules.
    
    Combines three key metrics:
    1. Chemical Validity (0 or 1) - Can RDKit parse it?
    2. QED Score (0-1) - Quantitative Estimate of Drug-likeness
    3. SA Score (0-1 normalized) - Synthetic Accessibility
    
    Args:
        validity_weight: Weight for validity component (default: 1.0)
        qed_weight: Weight for QED component (default: 0.3)
        sa_weight: Weight for SA score component (default: 0.3)
    """
    
    def __init__(
        self,
        validity_weight: float = 1.0,
        qed_weight: float = 0.3,
        sa_weight: float = 0.3
    ):
        self.validity_weight = validity_weight
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight
        
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for reward calculation. Install with: conda install -c conda-forge rdkit")
    
    def calculate_validity(self, smiles: str) -> float:
        """
        Check if SMILES string is chemically valid.
        
        Returns:
            1.0 if valid, 0.0 if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return 1.0 if mol is not None else 0.0
        except:
            return 0.0
    
    def calculate_qed(self, smiles: str) -> float:
        """
        Calculate QED (Quantitative Estimate of Drug-likeness) score.
        
        QED ranges from 0 (non-drug-like) to 1 (drug-like).
        Based on molecular properties like MW, LogP, PSA, etc.
        
        Returns:
            QED score (0-1) or 0.0 if invalid molecule
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            return QED.qed(mol)
        except:
            return 0.0
    
    def calculate_sa_score(self, smiles: str) -> float:
        """
        Calculate Synthetic Accessibility score.
        
        SA Score ranges from 1 (easy to synthesize) to 10 (difficult).
        We normalize to 0-1 and invert so higher is better.
        
        Returns:
            Normalized SA score (0-1) where 1 is easy to synthesize, or 0.0 if invalid
        """
        if not SA_SCORE_AVAILABLE:
            return 0.5  # Neutral score if SA not available
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # Get raw SA score (1-10, lower is better)
            sa_raw = sascorer.calculateScore(mol)
            
            # Normalize to 0-1 and invert (higher is better)
            sa_normalized = 1.0 - (sa_raw - 1.0) / 9.0
            
            return max(0.0, min(1.0, sa_normalized))
        except:
            return 0.0
    
    def calculate_composite_reward(
        self,
        smiles: str,
        return_components: bool = False
    ) -> float:
        """
        Calculate weighted composite reward.
        
        Args:
            smiles: SMILES string to evaluate
            return_components: If True, return dict with component scores
        
        Returns:
            Composite reward score or dict if return_components=True
        """
        validity = self.calculate_validity(smiles)
        
        # Only calculate expensive metrics if molecule is valid AND their weights are non-zero
        if validity > 0:
            # Skip QED calculation if weight is 0 (saves 15-20% time)
            qed = self.calculate_qed(smiles) if self.qed_weight > 0 else 0.0
            # Skip SA calculation if weight is 0 (saves 30-40% time)
            sa = self.calculate_sa_score(smiles) if self.sa_weight > 0 else 0.0
        else:
            qed = 0.0
            sa = 0.0
        
        # Weighted sum
        composite = (
            self.validity_weight * validity +
            self.qed_weight * qed +
            self.sa_weight * sa
        )
        
        if return_components:
            return {
                'composite': composite,
                'validity': validity,
                'qed': qed,
                'sa': sa
            }
        
        return composite
    
    def batch_calculate_rewards(
        self,
        smiles_list: List[str],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Calculate rewards for a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            return_components: If True, return dict with tensors
        
        Returns:
            Tensor of rewards [batch_size] or dict of tensors
        """
        rewards = []
        validities = []
        qeds = []
        sas = []
        
        for smiles in smiles_list:
            if return_components:
                reward_dict = self.calculate_composite_reward(smiles, return_components=True)
                rewards.append(reward_dict['composite'])
                validities.append(reward_dict['validity'])
                qeds.append(reward_dict['qed'])
                sas.append(reward_dict['sa'])
            else:
                reward = self.calculate_composite_reward(smiles)
                rewards.append(reward)
        
        if return_components:
            return {
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'validity': torch.tensor(validities, dtype=torch.float32),
                'qed': torch.tensor(qeds, dtype=torch.float32),
                'sa': torch.tensor(sas, dtype=torch.float32)
            }
        
        return torch.tensor(rewards, dtype=torch.float32)


class ValueNetwork(nn.Module):
    """
    Critic network for PPO that estimates state values.
    
    Takes hidden states from the decoder and predicts expected cumulative reward.
    """
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of states.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
        
        Returns:
            values: [batch_size, seq_len]
        """
        values = self.network(hidden_states).squeeze(-1)
        return values


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for molecular generation.
    
    Implements PPO algorithm with:
    - Clipped surrogate objective
    - Value function learning
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization
    
    Args:
        model: SMILESGPTDecoder model (actor/policy)
        tokenizer: SMILES tokenizer
        reward_calculator: MolecularRewardCalculator instance
        clip_epsilon: PPO clipping parameter (default: 0.2)
        value_coef: Coefficient for value loss (default: 0.5)
        entropy_coef: Coefficient for entropy bonus (default: 0.01)
        gamma: Discount factor (default: 0.99)
        gae_lambda: GAE lambda parameter (default: 0.95)
        max_rollout_length: Maximum sequence length for rollouts (default: 100)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_calculator: MolecularRewardCalculator,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_rollout_length: int = 100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_rollout_length = max_rollout_length
        
        # Create value network
        self.value_network = ValueNetwork(d_model=model.config.d_model).to(
            next(model.parameters()).device
        )
        
        # Separate optimizer for value network
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=1e-4
        )
    
    def collect_rollouts(
        self,
        batch_size: int,
        protein_ids: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        num_rollouts: int = 4
    ) -> Dict:
        """
        Generate molecules and collect rollout data for PPO update.
        
        Memory-efficient: Only stores necessary data, clears intermediate tensors.
        
        Args:
            batch_size: Number of sequences in original batch
            protein_ids: Optional protein conditioning
            protein_mask: Optional protein mask
            num_rollouts: Number of sequences to generate per batch item
        
        Returns:
            Dictionary with rollout data including sequences, log_probs, values, rewards
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        self.value_network.eval()
        
        # Expand protein conditioning for multiple rollouts
        if protein_ids is not None:
            protein_ids_expanded = protein_ids.repeat_interleave(num_rollouts, dim=0)
            protein_mask_expanded = protein_mask.repeat_interleave(num_rollouts, dim=0)
        else:
            protein_ids_expanded = None
            protein_mask_expanded = None
        
        total_sequences = batch_size * num_rollouts
        
        # Start with BOS token
        generated = torch.tensor(
            [[self.model.bos_token_id]] * total_sequences,
            dtype=torch.long,
            device=device
        )
        
        all_log_probs = []
        all_values = []
        all_entropies = []
        
        with torch.no_grad():
            for step in range(self.max_rollout_length):
                # Forward pass
                outputs = self.model.forward(
                    generated,
                    protein_ids=protein_ids_expanded,
                    protein_mask=protein_mask_expanded
                )
                
                logits = outputs['logits'][:, -1, :].detach()  # Detach to free graph
                hidden_states = outputs['hidden_states'][:, -1, :].detach()
                
                # Clear outputs dict to free memory
                del outputs
                
                # Get value estimates
                values = self.value_network(hidden_states.unsqueeze(1)).squeeze(1).detach()
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_token = dist.sample()
                
                # Calculate log probability and entropy
                log_prob = dist.log_prob(next_token).detach()
                entropy = dist.entropy().detach()
                
                # Store (detached from computation graph)
                all_log_probs.append(log_prob.cpu())  # Move to CPU to free GPU memory
                all_values.append(values.cpu())
                all_entropies.append(entropy.cpu())
                
                # Clear intermediate tensors
                del logits, hidden_states, probs, dist, values
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                
                # Check for EOS
                if (next_token == self.model.eos_token_id).all():
                    break
        
        # Decode sequences to SMILES
        smiles_list = []
        for seq in generated:
            smiles = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            smiles_list.append(smiles)
        
        # Calculate rewards (CPU to avoid GPU memory)
        reward_dict = self.reward_calculator.batch_calculate_rewards(
            smiles_list,
            return_components=True
        )
        rewards = reward_dict['rewards']
        
        # Stack temporal data and move back to device only when needed
        log_probs = torch.stack(all_log_probs, dim=1)  # [batch, seq_len] on CPU
        values = torch.stack(all_values, dim=1)  # [batch, seq_len] on CPU
        entropies = torch.stack(all_entropies, dim=1)  # [batch, seq_len] on CPU
        
        # Clear intermediate lists
        del all_log_probs, all_values, all_entropies
        
        return {
            'sequences': generated.cpu(),  # Move to CPU to free GPU
            'log_probs': log_probs,
            'values': values,
            'entropies': entropies,
            'rewards': rewards,
            'reward_components': reward_dict,
            'smiles': smiles_list
        }
    
    def compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        seq_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            values: Value estimates [batch, seq_len]
            rewards: Terminal rewards [batch]
            seq_lengths: Actual sequence lengths [batch]
        
        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len]
        """
        batch_size, max_len = values.shape
        device = values.device
        
        # Terminal reward is only at the end
        # Create reward tensor with reward only at last step
        step_rewards = torch.zeros_like(values)
        for i, length in enumerate(seq_lengths):
            if length > 0 and length <= max_len:
                step_rewards[i, length - 1] = rewards[i]
        
        # Compute returns and advantages using GAE
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        gae = 0
        for t in reversed(range(max_len)):
            if t == max_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = step_rewards[:, t] + self.gamma * next_value - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_ppo_loss(
        self,
        batch_size: int,
        protein_ids: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        num_rollouts: int = 4
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss for a batch.
        
        Memory-efficient: Properly manages GPU memory, moves data strategically.
        
        Args:
            batch_size: Size of training batch
            protein_ids: Optional protein conditioning
            protein_mask: Optional protein mask
            num_rollouts: Number of sequences per batch item
        
        Returns:
            loss: PPO loss tensor
            metrics: Dictionary of metrics for logging
        """
        device = next(self.model.parameters()).device
        
        # Collect rollouts (data is on CPU to save GPU memory)
        rollout_data = self.collect_rollouts(
            batch_size, protein_ids, protein_mask, num_rollouts
        )
        
        # Move data to device only when needed
        sequences = rollout_data['sequences'].to(device)
        old_log_probs = rollout_data['log_probs'].to(device)
        old_values = rollout_data['values'].to(device)
        old_entropies = rollout_data['entropies'].to(device)
        rewards = rollout_data['rewards'].to(device)
        
        # Calculate sequence lengths (excluding padding)
        seq_lengths = (sequences != self.model.pad_token_id).sum(dim=1)
        
        # Compute advantages
        advantages, returns = self.compute_advantages(
            old_values, rewards, seq_lengths
        )
        
        # Clear old rollout data from GPU
        del old_values, old_entropies, rewards
        torch.cuda.empty_cache()
        
        # Now compute new log probs and values with gradients
        self.model.train()
        self.value_network.train()
        
        # Forward pass through model
        outputs = self.model.forward(
            sequences[:, :-1],  # Exclude last token
            protein_ids=protein_ids.repeat_interleave(num_rollouts, dim=0) if protein_ids is not None else None,
            protein_mask=protein_mask.repeat_interleave(num_rollouts, dim=0) if protein_mask is not None else None
        )
        
        logits = outputs['logits']
        hidden_states = outputs['hidden_states']
        
        # Get new log probs for actions taken
        log_probs_all = F.log_softmax(logits, dim=-1)
        new_log_probs = log_probs_all.gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Get new values
        new_values = self.value_network(hidden_states)
        
        # Calculate new entropies
        probs = F.softmax(logits, dim=-1)
        new_entropies = -(probs * log_probs_all).sum(dim=-1)
        
        # Clear intermediate tensors
        del logits, log_probs_all, probs
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, returns)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -new_entropies.mean()
        
        # Combined loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Update value network separately
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()
        
        # Clear computation graph
        del ratio, clipped_ratio, advantages, returns, new_log_probs, old_log_probs
        
        # Metrics for logging (convert to Python scalars to free tensors)
        metrics = {
            'ppo_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': new_entropies.mean().item(),
            'avg_reward': rollout_data['rewards'].mean().item(),
            'validity_rate': rollout_data['reward_components']['validity'].mean().item(),
            'avg_qed': rollout_data['reward_components']['qed'].mean().item(),
            'avg_sa': rollout_data['reward_components']['sa'].mean().item()
        }
        
        # Clean up
        del sequences, new_values, new_entropies, hidden_states, outputs
        torch.cuda.empty_cache()
        
        return total_loss, metrics

