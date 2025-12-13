"""
Inference utilities for AlphaBuilder neural network.

This module provides:
- Model loading from checkpoints
- Input preprocessing (centralized padding)
- Batch inference support

Usage:
    from alphabuilder.src.neural.inference import AlphaBuilderInference
    
    # Initialize
    model = AlphaBuilderInference('checkpoints/best_model.pt', device='xpu')
    
    # Single inference
    value, policy_add, policy_remove = model.predict(state)
    
    # Batch inference
    values, policies = model.predict_batch(states)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple, List, Optional

from .model import AlphaBuilderV31


def pad_to_target_centered(
    state: np.ndarray, 
    divisor: int = 32
) -> np.ndarray:
    """
    Apply centralized padding to ensure dimensions are divisible by divisor.
    
    This is the CORRECT preprocessing for inference - it places the structure
    in the center of the padded volume, unlike training which uses random positioning.
    
    Args:
        state: Input tensor of shape (C, D, H, W)
        divisor: Dimensions will be padded to be divisible by this value
        
    Returns:
        Padded tensor of shape (C, D', H', W') where D', H', W' are divisible by divisor
    """
    C, D, H, W = state.shape
    
    # Calculate target dimensions (next multiple of divisor)
    target_D = ((D + divisor - 1) // divisor) * divisor
    target_H = ((H + divisor - 1) // divisor) * divisor
    target_W = ((W + divisor - 1) // divisor) * divisor
    
    # Calculate padding amounts
    pad_D = target_D - D
    pad_H = target_H - H
    pad_W = target_W - W
    
    # Center the structure
    pad_before = (pad_D // 2, pad_H // 2, pad_W // 2)
    pad_after = (pad_D - pad_before[0], pad_H - pad_before[1], pad_W - pad_before[2])
    
    # Apply padding
    padded = np.pad(
        state,
        ((0, 0),  # No padding on channel dimension
         (pad_before[0], pad_after[0]),
         (pad_before[1], pad_after[1]),
         (pad_before[2], pad_after[2])),
        mode='constant',
        constant_values=0
    )
    
    return padded


def unpad_from_target(
    tensor: np.ndarray, 
    original_shape: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Remove padding to restore original shape.
    
    Args:
        tensor: Padded tensor of shape (C, D', H', W')
        original_shape: Original shape (C, D, H, W) before padding
        
    Returns:
        Unpadded tensor of shape (C, D, H, W)
    """
    C, D, H, W = original_shape
    _, Dp, Hp, Wp = tensor.shape
    
    # Calculate padding that was applied
    pad_D = Dp - D
    pad_H = Hp - H
    pad_W = Wp - W
    
    # Calculate start indices (center crop)
    start_d = pad_D // 2
    start_h = pad_H // 2
    start_w = pad_W // 2
    
    return tensor[:, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]


class AlphaBuilderInference:
    """
    Inference wrapper for AlphaBuilder model.
    
    Handles:
    - Model loading from checkpoint
    - Input preprocessing (padding)
    - Output postprocessing (unpadding)
    - GPU optimization (IPEX for XPU, CUDA for GPU)
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = 'auto',
        model_config: dict = None
    ):
        """
        Initialize inference model.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to use ('auto', 'xpu', 'cuda', 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = 'xpu'
            else:
                device = 'cpu'
        
        self.device_str = device.lower()
        
        if self.device_str == 'npu' or self.device_str == 'auto':
            from .openvino_inference import AlphaBuilderOpenVINO
            
            # Default to AUTO/GPU with THROUGHPUT hint if not specified
            model_config = model_config or {}
            if 'performance_hint' not in model_config:
                model_config['performance_hint'] = 'THROUGHPUT'

            self.backend = AlphaBuilderOpenVINO(checkpoint_path, device='AUTO', model_config=model_config)
            # We can't really set self.device to torch.device object here easily as it's not a torch device
            # But let's keep it for compatibility if needed, though NPU backend handles everything
            self.device = torch.device('cpu') 
            self.is_npu = True
            
            # Mimic attributes expected by repr or inspection
            self.config = getattr(self.backend, 'config', {})
            self.epoch = getattr(self.backend, 'epoch', -1)
            self.val_loss = getattr(self.backend, 'val_loss', 0.0)
            self.use_swin = getattr(self.backend, 'use_swin', False)
            
            # Assuming logger is defined elsewhere or will be added
            # logger.info(f"Initialized AlphaBuilderOpenVINO backend on device {self.backend.device}")
            return
        else:
            self.is_npu = False
            self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, 
            map_location='cpu',
            weights_only=False
        )
        
        # Extract config
        self.config = checkpoint.get('config', {})
        self.epoch = checkpoint.get('epoch', -1)
        self.val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Create model
        self.model = AlphaBuilderV31(
            in_channels=7,
            out_channels=2,
            feature_size=self.config.get('feature_size', 24),
            use_swin=self.config.get('use_swin', False)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Apply device-specific optimizations
        self._optimize_for_device()
        
        # Pre-warm the model
        self._warmup()
    
    def _optimize_for_device(self):
        """Apply device-specific optimizations."""
        if self.device.type == 'xpu':
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
            except ImportError:
                pass
        elif self.device.type == 'cuda':
            # Enable TensorFloat32 for faster computation on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _warmup(self, num_iterations: int = 5):
        """Warmup the model to avoid cold-start latency."""
        x = torch.randn(1, 7, 64, 32, 32).to(self.device)
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(x)
        
        if self.device.type == 'xpu':
            torch.xpu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    def preprocess(
        self, 
        state: np.ndarray, 
        divisor: int = 32
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Preprocess state for inference.
        
        Args:
            state: Input state of shape (C, D, H, W) or (7, D, H, W)
            divisor: Padding divisor (default 32)
            
        Returns:
            Tuple of (tensor ready for inference, original shape)
        """
        original_shape = state.shape
        
        # Apply centered padding
        padded = pad_to_target_centered(state, divisor=divisor)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(padded).float().unsqueeze(0).to(self.device)
        
        return tensor, original_shape
    
    def postprocess(
        self,
        policy: np.ndarray,
        original_shape: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Postprocess policy to remove padding.
        
        Args:
            policy: Padded policy of shape (2, D', H', W')
            original_shape: Original input shape (C, D, H, W)
            
        Returns:
            Unpadded policy of shape (2, D, H, W)
        """
        # Policy has shape (2, D', H', W'), original is (7, D, H, W)
        policy_shape = (2, original_shape[1], original_shape[2], original_shape[3])
        return unpad_from_target(policy, policy_shape)
    
    def predict(
        self,
        state: np.ndarray,
        return_raw_policy: bool = False
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Predict value and policy for a single state.
        
        Args:
            state: Input state of shape (7, D, H, W)
                   - Channel 0: density
                   - Channels 1-3: boundary condition masks
                   - Channels 4-6: force vectors
            return_raw_policy: If True, return padded policy without unpadding
            
        Returns:
            Tuple of (value, policy_add, policy_remove)
            - value: Predicted fitness score in [-1, 1]
            - policy_add: Heatmap of where to add material
            - policy_remove: Heatmap of where to remove material
        """

        if self.is_npu:
            # NPU Backend handles its own preprocessing/inference/postprocessing logic internally for simplicity, 
            # OR we should delegate to it.
            # The OpenVINO class we wrote implements predict(state) -> (val, padd, prem)
            # The signature here matches.
            return self.backend.predict(state)

        # Preprocess
        tensor, original_shape = self.preprocess(state)
        
        # Inference
        with torch.no_grad():
            pred_policy, pred_value = self.model(tensor)
        
        # Synchronize if needed
        if self.device.type == 'xpu':
            torch.xpu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Extract results
        value = pred_value.cpu().item()
        policy = pred_policy[0].cpu().numpy()  # (2, D', H', W')
        
        # Postprocess policy
        if not return_raw_policy:
            policy = self.postprocess(policy, original_shape)
        
        policy_add = policy[0]
        policy_remove = policy[1]
        
        return value, policy_add, policy_remove
    
    def predict_batch(
        self,
        states: List[np.ndarray]
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """
        Predict value and policy for a batch of states.
        
        Note: All states should have the same shape for efficient batching.
        
        Args:
            states: List of input states, each of shape (7, D, H, W)
            
        Returns:
            Tuple of (values, policies_add, policies_remove)
        """
        if not states:
            return [], [], []
        
        # Preprocess all states
        tensors = []
        original_shapes = []
        
        for state in states:
            tensor, orig_shape = self.preprocess(state)
            tensors.append(tensor)
            original_shapes.append(orig_shape)
        
        # Stack into batch
        batch_tensor = torch.cat(tensors, dim=0)
        
        # Inference
        with torch.no_grad():
            pred_policies, pred_values = self.model(batch_tensor)
        
        # Synchronize
        if self.device.type == 'xpu':
            torch.xpu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Extract results
        values = pred_values.cpu().numpy().flatten().tolist()
        policies = pred_policies.cpu().numpy()
        
        # Postprocess each policy
        policies_add = []
        policies_remove = []
        
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
            f"  use_swin={self.config.get('use_swin', False)}\n"
            f")"
        )
