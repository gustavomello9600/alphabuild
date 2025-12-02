"""
Training Loop for AlphaBuilder v3.1.

Implements weighted loss as per spec:
- Policy Loss with pos_weight to handle class imbalance
- Value Loss with negative sample weighting (w_neg = 5.0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from torch.utils.data import DataLoader


# Negative sample weight for value loss
W_NEG = 5.0

# Policy weight in total loss
LAMBDA_POLICY = 1.0

# Class imbalance weights for policy (positive:negative ratio compensation)
# These are CRITICAL to prevent "predict all zeros" collapse
# Based on class distribution: GROWTH ~1:10, REFINEMENT ADD ~1:6, REFINEMENT REM ~1:4
POS_WEIGHT_ADD = 8.0   # Compensate for ~9-16% positive class
POS_WEIGHT_REM = 5.0   # Compensate for ~20% positive class


def weighted_value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE loss for value prediction.
    
    Applies higher weight (w_neg) to negative targets to combat
    imbalance where most samples are "good".
    
    Args:
        pred: (B, 1) predicted values
        target: (B, 1) target values
        
    Returns:
        Weighted MSE loss
    """
    mse = (pred - target) ** 2
    
    # Weight negative targets more heavily
    weights = torch.where(target <= 0, W_NEG, 1.0)
    
    return (weights * mse).mean()


def policy_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    phase: str = None
) -> torch.Tensor:
    """
    Binary cross-entropy loss for policy prediction with class imbalance handling.
    
    CRITICAL: Uses pos_weight to prevent "predict all zeros" collapse.
    Without this, the network would learn to predict 0 everywhere
    and still achieve ~90% accuracy due to class imbalance.
    
    Phase-aware:
    - GROWTH: Focus on Add channel, mask Remove
    - REFINEMENT: Both channels active
    
    Args:
        pred: (B, 2, D, H, W) policy logits
        target: (B, 2, D, H, W) target policy
        phase: 'GROWTH' or 'REFINEMENT' (optional)
        
    Returns:
        Policy loss
    """
    B, C, D, H, W = pred.shape
    
    # Create per-channel pos_weight tensor
    # Channel 0: ADD, Channel 1: REMOVE
    pos_weight = torch.tensor([POS_WEIGHT_ADD, POS_WEIGHT_REM], device=pred.device)
    pos_weight = pos_weight.view(1, 2, 1, 1, 1).expand(B, 2, D, H, W)
    
    # Binary cross-entropy with logits and pos_weight
    # pos_weight increases loss for false negatives (missing positives)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, 
        pos_weight=pos_weight,
        reduction='none'
    )
    
    # Phase-aware masking
    if phase == 'GROWTH':
        # Focus on Add channel (index 0), reduce weight on Remove (index 1)
        mask = torch.ones_like(loss)
        mask[:, 1, :, :, :] = 0.1  # Low weight for Remove in Growth phase
        loss = loss * mask
    
    return loss.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: AlphaBuilderV31 model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Dict with loss metrics
    """
    model.train()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        state = batch['state'].to(device)
        target_policy = batch['policy'].to(device)
        target_value = batch['value'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_policy, pred_value = model(state)
        
        # Compute losses
        v_loss = weighted_value_loss(pred_value, target_value)
        p_loss = policy_loss(pred_policy, target_policy)
        
        loss = v_loss + LAMBDA_POLICY * p_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_policy_loss += p_loss.item()
        total_value_loss += v_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'policy_loss': total_policy_loss / max(num_batches, 1),
        'value_loss': total_value_loss / max(num_batches, 1)
    }

