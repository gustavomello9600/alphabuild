"""
Inference utilities for AlphaBuilder neural network (v3.3).

This module provides:
- Checkpoint loading (single-GPU, DataParallel, DDP, and optional OpenVINO NPU)
- Standardized unpadded preprocessing matching training (`AlphaBuilderV31` pads/crops internally)
- Batch inference and optional auxiliary Blueprint Head prediction
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .model import AlphaBuilderV31, build_model, resolve_architecture


def pad_to_target_centered(
    state: np.ndarray,
    divisor: int = 32,
) -> np.ndarray:
    """
    Apply centered padding so spatial dimensions are divisible by divisor.
    Kept for backward compatibility with external utilities.
    """
    _, D, H, W = state.shape
    target_D = ((D + divisor - 1) // divisor) * divisor
    target_H = ((H + divisor - 1) // divisor) * divisor
    target_W = ((W + divisor - 1) // divisor) * divisor

    pad_D = target_D - D
    pad_H = target_H - H
    pad_W = target_W - W

    pad_before = (pad_D // 2, pad_H // 2, pad_W // 2)
    pad_after = (pad_D - pad_before[0], pad_H - pad_before[1], pad_W - pad_before[2])

    return np.pad(
        state,
        (
            (0, 0),
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ),
        mode="constant",
        constant_values=0,
    )


def unpad_from_target(
    tensor: np.ndarray,
    original_shape: Tuple[int, int, int, int],
) -> np.ndarray:
    """Center-crop tensor back to original (C, D, H, W) shape."""
    _, D, H, W = original_shape
    _, Dp, Hp, Wp = tensor.shape
    if (Dp, Hp, Wp) == (D, H, W):
        return tensor

    start_d = (Dp - D) // 2
    start_h = (Hp - H) // 2
    start_w = (Wp - W) // 2
    return tensor[:, start_d : start_d + D, start_h : start_h + H, start_w : start_w + W]


class AlphaBuilderInference:
    """
    Unified inference wrapper for AlphaBuilderV31.

    Note: `AlphaBuilderV31.forward()` encapsulates internal centered padding and cropping,
    so inference passes unpadded `(B, 7, D, H, W)` tensors identically to training.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        model_config: Optional[dict] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"

        self.device_str = device.lower()

        if self.device_str == "npu":
            from .openvino_inference import AlphaBuilderOpenVINO

            cfg = dict(model_config or {})
            cfg.setdefault("performance_hint", "THROUGHPUT")
            self.backend = AlphaBuilderOpenVINO(str(checkpoint_path), device="AUTO", model_config=cfg)
            self.device = torch.device("cpu")
            self.is_npu = True
            self.config = getattr(self.backend, "config", {})
            self.epoch = getattr(self.backend, "epoch", -1)
            self.val_loss = getattr(self.backend, "val_loss", float("inf"))
            self.use_swin = getattr(self.backend, "use_swin", False)
            self.architecture = getattr(self.backend, "architecture", "swin" if self.use_swin else "simple")
            return

        self.is_npu = False
        self.device = torch.device(self.device_str)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        self.config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        if model_config:
            self.config = {**self.config, **model_config}
        self.epoch = checkpoint.get("epoch", -1) if isinstance(checkpoint, dict) else -1
        metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        self.val_loss = checkpoint.get("val_loss", metrics.get("loss", float("inf"))) if isinstance(checkpoint, dict) else float("inf")
        self.architecture = resolve_architecture(self.config)
        self.use_swin = self.architecture == "swin"

        default_feat = 32 if self.architecture == "phys_uvit" else 24
        self.model = build_model(
            architecture=self.architecture,
            in_channels=int(self.config.get("in_channels", 7)),
            out_channels=int(self.config.get("out_channels", 2)),
            feature_size=int(self.config.get("feature_size", default_feat)),
            use_swin=self.use_swin,
            embed_dim=int(self.config.get("embed_dim", 192)),
            transformer_depth=int(self.config.get("transformer_depth", 5)),
            num_heads=int(self.config.get("num_heads", 6)),
        )
        self.feature_size = self.model.feature_size
        self.in_channels = self.model.in_channels

        raw_sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in raw_sd.items()
        }
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._optimize_for_device()
        self._warmup()

    def _optimize_for_device(self) -> None:
        if self.device.type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
            except ImportError:
                pass
        elif self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _warmup(self, num_iterations: int = 3) -> None:
        x = torch.randn(1, self.model.in_channels, 64, 32, 8, device=self.device)
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(x)

        if self.device.type == "xpu":
            torch.xpu.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def preprocess(
        self,
        state: np.ndarray,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Convert unpadded (C, D, H, W) state directly to batched tensor."""
        original_shape = tuple(int(s) for s in state.shape)
        tensor = torch.from_numpy(np.ascontiguousarray(state)).float().unsqueeze(0).to(self.device)
        return tensor, original_shape  # type: ignore[return-value]

    def postprocess(
        self,
        policy: np.ndarray,
        original_shape: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Ensure policy tensor matches original spatial shape (2, D, H, W)."""
        policy_shape = (policy.shape[0], original_shape[1], original_shape[2], original_shape[3])
        return unpad_from_target(policy, policy_shape)

    def _sync(self) -> None:
        if self.device.type == "xpu":
            torch.xpu.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def _prepare_domain_mask(
        self,
        domain_mask: Optional[np.ndarray],
        batch_size: int = 1,
    ) -> Optional[torch.Tensor]:
        if domain_mask is None:
            return None
        dm = np.ascontiguousarray(domain_mask, dtype=np.float32)
        if dm.ndim == 3:
            t = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)
        elif dm.ndim == 4:
            t = torch.from_numpy(dm).unsqueeze(0)
        else:
            t = torch.from_numpy(dm)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1, -1, -1, -1)
        return t.to(self.device)

    def predict(
        self,
        state: np.ndarray,
        return_raw_policy: bool = False,
        domain_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Predict (value, policy_add, policy_remove) for a single (7, D, H, W) state."""
        if self.is_npu:
            return self.backend.predict(state)

        tensor, original_shape = self.preprocess(state)
        dm = self._prepare_domain_mask(domain_mask, 1) if self.architecture == "phys_uvit" else None
        with torch.no_grad():
            if dm is not None:
                pred_policy, pred_value = self.model(tensor, domain_mask=dm)
            else:
                pred_policy, pred_value = self.model(tensor)
        self._sync()

        value = float(pred_value.cpu().item())
        policy = pred_policy[0].cpu().numpy()
        if not return_raw_policy:
            policy = self.postprocess(policy, original_shape)
        return value, policy[0], policy[1]

    def predict_with_blueprint(
        self,
        state: np.ndarray,
        domain_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Predict (value, policy_add, policy_remove, blueprint_probs) for visualization/analysis."""
        if self.is_npu:
            tensor = torch.from_numpy(np.ascontiguousarray(state)).float().unsqueeze(0)
            with torch.no_grad():
                pred_policy, pred_value, pred_bp = self.backend.pt_model(tensor, return_blueprint=True)
            bp_prob = torch.sigmoid(pred_bp)[0, 0].cpu().numpy()
            pol = pred_policy[0].cpu().numpy()
            return float(pred_value.cpu().item()), pol[0], pol[1], bp_prob

        tensor, original_shape = self.preprocess(state)
        dm = self._prepare_domain_mask(domain_mask, 1) if self.architecture == "phys_uvit" else None
        with torch.no_grad():
            if dm is not None:
                pred_policy, pred_value, pred_bp = self.model(tensor, domain_mask=dm, return_blueprint=True)
            else:
                pred_policy, pred_value, pred_bp = self.model(tensor, return_blueprint=True)
        self._sync()

        value = float(pred_value.cpu().item())
        policy = self.postprocess(pred_policy[0].cpu().numpy(), original_shape)
        bp_probs = torch.sigmoid(pred_bp)[0, 0].cpu().numpy()
        return value, policy[0], policy[1], bp_probs

    def predict_with_aux(
        self,
        state: np.ndarray,
        compute_sed: bool = False,
        domain_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, object]]:
        """Predict (value, policy_add, policy_remove, aux_dict) including factored physics heads."""
        if self.architecture != "phys_uvit" or self.is_npu:
            val, p_add, p_rem, bp_probs = self.predict_with_blueprint(state, domain_mask=domain_mask)
            return val, p_add, p_rem, {"blueprint_probs": bp_probs}

        tensor, original_shape = self.preprocess(state)
        dm = self._prepare_domain_mask(domain_mask, 1)
        with torch.no_grad():
            pred_policy, pred_value, aux = self.model(
                tensor, domain_mask=dm, return_aux=True, compute_sed=compute_sed
            )
        self._sync()

        value = float(pred_value.cpu().item())
        policy = self.postprocess(pred_policy[0].cpu().numpy(), original_shape)
        out_aux: Dict[str, object] = {
            "blueprint_probs": torch.sigmoid(aux["blueprint"])[0, 0].cpu().numpy(),
            "log_compliance": float(aux["log_compliance"][0, 0].cpu().item()),
            "conn_prob": float(torch.sigmoid(aux["conn_logit"])[0, 0].cpu().item()),
        }
        if aux.get("sed_u") is not None:
            out_aux["sed_u"] = aux["sed_u"][0].cpu().numpy()
        return value, policy[0], policy[1], out_aux

    def predict_batch(
        self,
        states: List[np.ndarray],
        domain_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """Predict (values, policies_add, policies_remove) for a batch of states."""
        if not states:
            return [], [], []
        if self.is_npu:
            return self.backend.predict_batch(states)

        tensors = []
        original_shapes = []
        for state in states:
            tensor, orig_shape = self.preprocess(state)
            tensors.append(tensor)
            original_shapes.append(orig_shape)

        batch_tensor = torch.cat(tensors, dim=0)
        dm = self._prepare_domain_mask(domain_mask, len(states)) if self.architecture == "phys_uvit" else None
        with torch.no_grad():
            if dm is not None:
                pred_policies, pred_values = self.model(batch_tensor, domain_mask=dm)
            else:
                pred_policies, pred_values = self.model(batch_tensor)
        self._sync()

        values = [float(v) for v in pred_values.cpu().numpy().flatten().tolist()]
        policies = pred_policies.cpu().numpy()

        policies_add: List[np.ndarray] = []
        policies_remove: List[np.ndarray] = []
        for i, orig_shape in enumerate(original_shapes):
            policy = self.postprocess(policies[i], orig_shape)
            policies_add.append(policy[0])
            policies_remove.append(policy[1])

        return values, policies_add, policies_remove

    def __repr__(self) -> str:
        return (
            f"AlphaBuilderInference(\n"
            f"  checkpoint='{self.checkpoint_path}',\n"
            f"  device={self.device},\n"
            f"  epoch={self.epoch},\n"
            f"  val_loss={self.val_loss:.4f},\n"
            f"  architecture={self.architecture},\n"
            f"  use_swin={self.use_swin}\n"
            f")"
        )
