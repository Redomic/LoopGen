"""
Scheduled Sampling for gradual transition from teacher forcing to autonomous generation.

Implements multiple scheduling strategies to progressively reduce reliance on
ground truth tokens during training.
"""

import math
from typing import Optional


class ScheduledSamplingScheduler:
    """
    Scheduler that determines probability of using teacher forcing vs model predictions.
    
    Supports multiple decay strategies:
    - Linear: Simple linear decay from 1.0 to 0.0
    - Exponential: Exponential decay with configurable base
    - Inverse Sigmoid: Smooth S-curve transition (recommended)
    
    Args:
        total_epochs: Total number of training epochs
        schedule_type: Type of schedule ('linear', 'exponential', 'inverse_sigmoid')
        k: Schedule-specific parameter (auto-computed if None)
            - Exponential: decay base (default: 0.95)
            - Inverse Sigmoid: steepness parameter (default: total_epochs/5)
        min_prob: Minimum teacher forcing probability (default: 0.0)
        warmup_epochs: Number of epochs to keep prob=1.0 (default: 0)
    """
    
    def __init__(
        self,
        total_epochs: int,
        schedule_type: str = 'inverse_sigmoid',
        k: Optional[float] = None,
        min_prob: float = 0.0,
        warmup_epochs: int = 0
    ):
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type.lower()
        self.min_prob = min_prob
        self.warmup_epochs = warmup_epochs
        
        # Set k parameter based on schedule type
        if k is None:
            if self.schedule_type == 'exponential':
                self.k = 0.95
            elif self.schedule_type == 'inverse_sigmoid':
                self.k = max(1, total_epochs / 5.0)
            else:  # linear
                self.k = 1.0
        else:
            self.k = k
        
        # Validate schedule type
        valid_types = ['linear', 'exponential', 'inverse_sigmoid']
        if self.schedule_type not in valid_types:
            raise ValueError(
                f"Invalid schedule_type: {schedule_type}. "
                f"Must be one of {valid_types}"
            )
    
    def get_probability(self, epoch: int) -> float:
        """
        Get teacher forcing probability for given epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
        
        Returns:
            Probability of using teacher forcing (0.0 to 1.0)
        """
        # Warmup period: always use teacher forcing
        if epoch < self.warmup_epochs:
            return 1.0
        
        # Adjust epoch for warmup
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        
        if adjusted_total <= 0:
            return self.min_prob
        
        # Compute probability based on schedule type
        if self.schedule_type == 'linear':
            prob = self._linear_schedule(adjusted_epoch, adjusted_total)
        elif self.schedule_type == 'exponential':
            prob = self._exponential_schedule(adjusted_epoch)
        elif self.schedule_type == 'inverse_sigmoid':
            prob = self._inverse_sigmoid_schedule(adjusted_epoch)
        else:
            prob = 1.0
        
        # Ensure within bounds
        prob = max(self.min_prob, min(1.0, prob))
        
        return prob
    
    def _linear_schedule(self, epoch: int, total: int) -> float:
        """
        Linear decay: p(t) = max(0, 1 - t/T)
        
        Simple linear decrease from 1.0 to 0.0.
        """
        return 1.0 - (epoch / total)
    
    def _exponential_schedule(self, epoch: int) -> float:
        """
        Exponential decay: p(t) = k^t
        
        Faster initial decay, slower later.
        Typical k values: 0.90-0.99
        """
        return self.k ** epoch
    
    def _inverse_sigmoid_schedule(self, epoch: int) -> float:
        """
        Inverse sigmoid: p(t) = k / (k + exp(t/k))
        
        Provides smooth S-curve transition:
        - Starts near 1.0 (almost full teacher forcing)
        - Smooth transition in middle
        - Ends near 0.0 (almost full autonomous)
        
        This is the recommended schedule as it provides:
        - Stable early training (high teacher forcing)
        - Gradual transition (not too abrupt)
        - Eventual independence (low teacher forcing)
        """
        return self.k / (self.k + math.exp(epoch / self.k))
    
    def get_schedule_info(self) -> dict:
        """
        Get information about the current schedule configuration.
        
        Returns:
            Dictionary with schedule parameters and sample probabilities
        """
        # Sample probabilities at key points
        sample_epochs = [
            0,
            self.total_epochs // 4,
            self.total_epochs // 2,
            3 * self.total_epochs // 4,
            self.total_epochs - 1
        ]
        
        sample_probs = {
            f"epoch_{epoch}": self.get_probability(epoch)
            for epoch in sample_epochs
        }
        
        return {
            'schedule_type': self.schedule_type,
            'total_epochs': self.total_epochs,
            'k_parameter': self.k,
            'min_prob': self.min_prob,
            'warmup_epochs': self.warmup_epochs,
            'sample_probabilities': sample_probs
        }
    
    def __repr__(self) -> str:
        return (
            f"ScheduledSamplingScheduler("
            f"type={self.schedule_type}, "
            f"epochs={self.total_epochs}, "
            f"k={self.k:.3f}, "
            f"warmup={self.warmup_epochs})"
        )


def visualize_schedule(
    scheduler: ScheduledSamplingScheduler,
    num_points: int = 50
) -> None:
    """
    Print a simple ASCII visualization of the schedule.
    
    Useful for debugging and understanding the schedule curve.
    
    Args:
        scheduler: ScheduledSamplingScheduler instance
        num_points: Number of points to plot
    """
    print(f"\n{scheduler}")
    print("="*60)
    print("Teacher Forcing Probability Schedule:")
    print("="*60)
    
    epochs = [
        int(i * scheduler.total_epochs / num_points)
        for i in range(num_points + 1)
    ]
    
    for epoch in epochs:
        prob = scheduler.get_probability(epoch)
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"Epoch {epoch:4d}: {bar} {prob:.3f}")
    
    print("="*60)

