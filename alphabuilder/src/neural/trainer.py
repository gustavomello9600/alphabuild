"""
Loss functions and single-epoch training helper for AlphaBuilder v3.3.

Implements:
- Masked joint Cross-Entropy / KL-divergence policy loss consistent with MCTSNode.expand
- SmoothL1 / Huber value loss with moderate negative weighting (W_NEG = 1.5, beta = 0.2)
- Auxiliary Blueprint Head loss (0.5 * BCE + 0.5 * Dice, Wu 2019)
- Backbone gradient norm ratio diagnostic (||grad(lambda_bp * L_bp)|| / ||grad(L_pol)||)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


LAMBDA_POLICY = 1.0
LAMBDA_VALUE = 2.0
LAMBDA_BLUEPRINT_WARMUP = 0.50
LAMBDA_BLUEPRINT_SELFPLAY = 0.25

W_NEG = 1.5
HUBER_BETA = 0.2


def weighted_value_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_neg: float = W_NEG,
    beta: float = HUBER_BETA,
) -> torch.Tensor:
    """
    SmoothL1 (Huber) loss for value prediction with moderate negative weighting.
    Computed in float32 for numerical stability under AMP.
    """
    pred_f = pred.float()
    target_f = target.float()
    huber = F.smooth_l1_loss(pred_f, target_f, reduction="none", beta=beta)
    weights = torch.where(target_f <= 0.0, float(w_neg), 1.0)
    return (weights * huber).mean()


def policy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    policy_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    phase: Optional[str] = None,
) -> torch.Tensor:
    """
    Masked joint KL-divergence / Cross-Entropy loss over legal actions (Add + Remove).

    Args:
        pred: (B, 2, D, H, W) policy logits
        target: (B, 2, D, H, W) normalized target distribution (or nonnegative scores)
        legal_mask: Optional (B, 2, D, H, W) binary mask of legal moves
        policy_mask: Optional (B,) mask (0 on stationary final frames with no action mass)
        temperature: Softmax temperature matching MCTS expansion
        phase: Optional phase hint used only when legal_mask is None
    """
    logits_f = pred.float()
    target_f = target.float()
    B = logits_f.shape[0]

    if legal_mask is None:
        valid_mask = torch.ones_like(logits_f, dtype=torch.bool)
        if isinstance(phase, str) and phase == "GROWTH":
            valid_mask[:, 1] = False
    else:
        valid_mask = legal_mask > 0.5

    temp = max(float(temperature), 1e-6)
    logits_flat = (logits_f / temp).reshape(B, -1)
    target_flat = target_f.reshape(B, -1)
    valid_move = valid_mask.reshape(B, -1)

    t_masked = (target_flat * valid_move.float()).clamp(min=0.0)
    t_sum = t_masked.sum(dim=-1, keepdim=True)
    t_dist = t_masked / t_sum.clamp(min=1e-8)

    masked_logits = logits_flat.masked_fill(~valid_move, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1).masked_fill(~valid_move, 0.0)

    kl_per_sample = torch.where(
        t_dist > 0.0,
        t_dist * (torch.log(t_dist.clamp(min=1e-12)) - log_probs),
        torch.zeros_like(t_dist),
    ).sum(dim=-1)

    valid_sample = (t_sum.squeeze(-1) > 1e-6) & (valid_move.sum(dim=-1) > 0)
    if policy_mask is not None:
        valid_sample = valid_sample & (policy_mask.reshape(-1) > 0.5)

    if not valid_sample.any():
        return pred.sum() * 0.0

    return kl_per_sample[valid_sample].mean()


def blueprint_loss(
    pred_bp: torch.Tensor,
    target_bp: torch.Tensor,
    blueprint_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Auxiliary Blueprint Head loss: 0.5 * BCEWithLogits + 0.5 * SoftDice on unpadded domain.
    """
    logits_f = pred_bp.float()
    target_f = target_bp.float()
    if logits_f.ndim == 4:
        logits_f = logits_f.unsqueeze(1)
    if target_f.ndim == 4:
        target_f = target_f.unsqueeze(1)

    bce = F.binary_cross_entropy_with_logits(logits_f, target_f, reduction="none").mean(dim=(1, 2, 3, 4))

    probs = torch.sigmoid(logits_f)
    dims = (1, 2, 3, 4)
    intersection = (probs * target_f).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_f.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)

    sample_loss = 0.5 * bce + 0.5 * dice

    if blueprint_mask is not None:
        valid = blueprint_mask.reshape(-1) > 0.5
        if not valid.any():
            return pred_bp.sum() * 0.0
        return sample_loss[valid].mean()

    return sample_loss.mean()


def compute_blueprint_dice(
    pred_bp: torch.Tensor,
    target_bp: torch.Tensor,
    blueprint_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Optional[float]:
    """Compute binary Dice score for auxiliary blueprint predictions on valid samples."""
    with torch.no_grad():
        preds = (torch.sigmoid(pred_bp.float()) > threshold).float()
        targets = (target_bp.float() > 0.5).float()
        if blueprint_mask is not None:
            valid = blueprint_mask.reshape(-1) > 0.5
            if not valid.any():
                return None
            preds = preds[valid]
            targets = targets[valid]
        dims = tuple(range(1, preds.ndim))
        inter = (preds * targets).sum(dim=dims)
        denom = preds.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        return float(dice.mean().item())


def compute_policy_top1_accuracy(
    pred_policy: torch.Tensor,
    target_policy: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    policy_mask: Optional[torch.Tensor] = None,
) -> Optional[float]:
    """Compute fraction of non-stationary samples where argmax legal logit hits argmax target."""
    with torch.no_grad():
        B = pred_policy.shape[0]
        logits_flat = pred_policy.float().reshape(B, -1)
        target_flat = target_policy.float().reshape(B, -1)
        if legal_mask is not None:
            valid_move = legal_mask.reshape(B, -1) > 0.5
            logits_flat = logits_flat.masked_fill(~valid_move, -1e9)
            target_flat = target_flat * valid_move.float()

        t_sum = target_flat.sum(dim=-1)
        valid_sample = t_sum > 1e-6
        if policy_mask is not None:
            valid_sample = valid_sample & (policy_mask.reshape(-1) > 0.5)
        if not valid_sample.any():
            return None

        pred_idx = logits_flat[valid_sample].argmax(dim=-1)
        target_idx = target_flat[valid_sample].argmax(dim=-1)
        return float((pred_idx == target_idx).float().mean().item())


def compute_total_loss(
    pred_policy: torch.Tensor,
    pred_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    policy_mask: Optional[torch.Tensor] = None,
    pred_blueprint: Optional[torch.Tensor] = None,
    target_blueprint: Optional[torch.Tensor] = None,
    blueprint_mask: Optional[torch.Tensor] = None,
    lambda_policy: float = LAMBDA_POLICY,
    lambda_value: float = LAMBDA_VALUE,
    lambda_bp: float = LAMBDA_BLUEPRINT_WARMUP,
    w_neg: float = W_NEG,
    huber_beta: float = HUBER_BETA,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-head loss:
        L_total = lambda_policy * L_pol + lambda_value * L_val + lambda_bp * L_bp
    """
    p_loss = policy_loss(
        pred_policy,
        target_policy,
        legal_mask=legal_mask,
        policy_mask=policy_mask,
    )
    v_loss = weighted_value_loss(pred_value, target_value, w_neg=w_neg, beta=huber_beta)
    total = lambda_policy * p_loss + lambda_value * v_loss

    bp_val = 0.0
    if pred_blueprint is not None and target_blueprint is not None and lambda_bp > 0.0:
        bp_loss_tensor = blueprint_loss(pred_blueprint, target_blueprint, blueprint_mask=blueprint_mask)
        total = total + lambda_bp * bp_loss_tensor
        bp_val = float(bp_loss_tensor.detach().item())

    return total, {
        "loss": float(total.detach().item()),
        "policy_loss": float(p_loss.detach().item()),
        "value_loss": float(v_loss.detach().item()),
        "blueprint_loss": bp_val,
    }


def compute_backbone_grad_ratio(
    model: nn.Module,
    p_loss: torch.Tensor,
    bp_loss: torch.Tensor,
    lambda_bp: float = LAMBDA_BLUEPRINT_WARMUP,
) -> Optional[float]:
    """
    Measure ||grad_encoder(lambda_bp * L_bp)||_2 / ||grad_encoder(L_pol)||_2
    before backward() to verify blueprint gradients remain within [0.15, 0.40].
    """
    base_model = model.module if hasattr(model, "module") else model
    encoder = getattr(base_model, "encoder", None)
    if encoder is None:
        return None
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    if not enc_params or not p_loss.requires_grad or not bp_loss.requires_grad:
        return None

    g_pol = torch.autograd.grad(p_loss, enc_params, retain_graph=True, allow_unused=True)
    g_bp = torch.autograd.grad(lambda_bp * bp_loss, enc_params, retain_graph=True, allow_unused=True)

    pol_sq = sum((g.detach().float().pow(2).sum().item() for g in g_pol if g is not None), 0.0)
    bp_sq = sum((g.detach().float().pow(2).sum().item() for g in g_bp if g is not None), 0.0)
    pol_norm = pol_sq ** 0.5
    if pol_norm <= 1e-12:
        return None
    return float((bp_sq ** 0.5) / pol_norm)


def _forward_with_optional_blueprint(
    model: nn.Module,
    state: torch.Tensor,
    want_blueprint: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if want_blueprint:
        try:
            outputs = model(state, return_blueprint=True)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                return outputs[0], outputs[1], outputs[2]
            return outputs[0], outputs[1], None
        except TypeError:
            pass
    outputs = model(state)
    if isinstance(outputs, tuple) and len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]
    return outputs[0], outputs[1], None


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_bp: float = LAMBDA_BLUEPRINT_WARMUP,
) -> Dict[str, float]:
    """Train model for one epoch cleanly (compatible with 2-head and 3-head models)."""
    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_bp_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        state = batch["state"].to(device)
        target_policy = batch["policy"].to(device)
        target_value = batch["value"].to(device)
        legal_mask = batch["legal_mask"].to(device) if "legal_mask" in batch else None
        policy_mask = batch["policy_mask"].to(device) if "policy_mask" in batch else None
        target_bp = batch["blueprint"].to(device) if "blueprint" in batch else None
        bp_mask = batch["blueprint_mask"].to(device) if "blueprint_mask" in batch else None

        optimizer.zero_grad(set_to_none=True)
        pred_policy, pred_value, pred_bp = _forward_with_optional_blueprint(
            model, state, want_blueprint=(target_bp is not None and lambda_bp > 0.0)
        )

        loss, metrics = compute_total_loss(
            pred_policy=pred_policy,
            pred_value=pred_value,
            target_policy=target_policy,
            target_value=target_value,
            legal_mask=legal_mask,
            policy_mask=policy_mask,
            pred_blueprint=pred_bp,
            target_blueprint=target_bp,
            blueprint_mask=bp_mask,
            lambda_bp=lambda_bp,
        )

        loss.backward()
        optimizer.step()

        total_loss += metrics["loss"]
        total_policy_loss += metrics["policy_loss"]
        total_value_loss += metrics["value_loss"]
        total_bp_loss += metrics["blueprint_loss"]
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "blueprint_loss": total_bp_loss / denom,
    }
