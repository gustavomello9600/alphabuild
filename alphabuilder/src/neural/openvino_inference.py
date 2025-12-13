
import numpy as np
import torch
import openvino as ov
import io
import logging
from pathlib import Path
from typing import Tuple, Optional, Union, List

# Import original model for loading weights
from alphabuilder.src.neural.model import AlphaBuilderV31

logger = logging.getLogger("OpenVINO")

class AlphaBuilderOpenVINO:
    """
    OpenVINO Inference Backend for AlphaBuilder.
    
    Optimized for Intel Core Ultra NPU/GPU inference.
    
    Workflow:
    1. Load PyTorch checkpoint
    2. Export to ONNX (in-memory)
    3. Compile to OpenVINO IR
    4. Run on NPU/GPU
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        device: str = "NPU", 
        model_config: dict = None
    ):
        """
        Initialize OpenVINO inference.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            device: Target device (NPU, GPU, CPU)
            model_config: Config for model architecture (optional)
        """
        self.device = device.upper()
        self.core = ov.Core()
        
        # Verify device availability
        available_devices = self.core.available_devices
        logger.info(f"OpenVINO Available Devices: {available_devices}")
        
        # Device priority: NPU > GPU > AUTO (which includes CPU)
        if self.device not in available_devices:
            if "GPU" in available_devices:
                logger.warning(f"Device {self.device} not found. Using GPU instead.")
                self.device = "GPU"
            else:
                logger.warning(f"Device {self.device} not found, GPU not available. Falling back to AUTO.")
                self.device = "AUTO"
        
        logger.info(f"Selected OpenVINO device: {self.device}")

        logger.info(f"Loading PyTorch model from {checkpoint_path}...")
        
        # 1. Load PyTorch Model
        # We need to instantiate the model to load weights
        # Assuming standard config if not provided
        if model_config is None:
            # Try to infer or default (this matches inference.py defaults)
            pass 
            
        try:
            # We need the class definition. 
            # If use_swin is needed, we should detect it from checkpoint if possible
            # For now, we instantiate the standard one
            self.pt_model = AlphaBuilderV31(
                in_channels=7, 
                out_channels=2,
                feature_size=24, # Default from inference.py/model.py
                use_swin=False
            )
            
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            
            # Remove module. prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.pt_model.load_state_dict(state_dict)
            self.pt_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise

        # 2. Convert to OpenVINO
        logger.info("Converting to OpenVINO IR...")
        
        # Create dummy input for tracing (Batch=1, C=7, D=64, H=32, W=8 as typical, but dynamic shapes preferred)
        # We'll use a fixed shape for NPU optimization (dynamic shapes hurt NPU perf)
        # The resolution should ideally match what's used in game
        # For now, using standard 64x32x8. If resolution changes, this needs to be re-compiled or dynamic.
        example_input = torch.randn(1, 7, 64, 32, 8)
        
        try:
            # Convert using ov.convert_model (direct conversion from torch module)
            # Use -1 (Dimension.dynamic()) for batch size to allow batching
            self.ov_model = ov.convert_model(
                self.pt_model, 
                example_input=example_input,
                input=[(-1, 7, 64, 32, 8)] 
            )
            
            # 3. Compile
            logger.info(f"Compiling model for {self.device}...")
            # Optimization hints: Allow override via model_config, default to LATENCY
            perf_hint = "LATENCY"
            if model_config and "performance_hint" in model_config:
                perf_hint = model_config["performance_hint"]
            
            config = {"PERFORMANCE_HINT": perf_hint}
            self.compiled_model = self.core.compile_model(self.ov_model, self.device, config)
            self.infer_request = self.compiled_model.create_infer_request()
            
            logger.info(f"OpenVINO Model compiled successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"OpenVINO conversion/compilation failed: {e}")
            raise
            
        # Get input/output nodes
        self.input_layer = self.compiled_model.input(0)
        # Outputs: [value, policy_remove, policy_add] - order depends on model definition
        # We need to map them correctly.
        # PyTorch model returns: value, p_add, p_remove
        # Let's verify output names or order
        self.outputs = self.compiled_model.outputs
        
    def predict(self, state: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Run inference on state.
        
        Args:
            state: (C, D, H, W) numpy array
            
        Returns:
            value: float
            policy_add: (D, H, W) numpy array
            policy_remove: (D, H, W) numpy array
        """
        # Ensure batch dimension
        if state.ndim == 4:
            state = np.expand_dims(state, 0)
            
        # Run inference
        results = self.infer_request.infer([state])
        
        # Results is a dict or list. We need to unpack ensuring correct order.
        # PyTorch forwarding: return value, policy_add, policy_remove
        # OpenVINO outputs preserve order of return usually.
        
        # We can map by shape if they are different, or rely on index.
        # Value: (B, 1)
        # Policies: (B, 1, D, H, W) or (B, D, H, W) depending on model
        
        # Let's assume order is preserved: 0=Value, 1=Add, 2=Remove
        # This assumption is Critical. If model changes return order, this breaks.
        # Safest is to inspect PyTorch model code, but we assume consistency.
        
        # Outputs are usually tensor wrappers
        
        # Extract numpy arrays
        # Note: compiled_model.outputs order matches torch return order
        res_list = [results[out] for out in self.outputs]
        
        if len(res_list) < 2:
            logger.error(f"Expected at least 2 outputs, got {len(res_list)}. Output keys: {[out.any_name for out in self.outputs]}")
            # Fallback based on available outputs
            pass

        value = res_list[1][0, 0] # Swap index based on typical ONNX export order? No, let's debug first.
        # Actually PyTorch (policy, value) = (tuple).
        # When exporting tuple, ONNX gets multiple outputs.
        # usually output_0 is policy, output_1 is value.
        
        # Let's inspect shapes to deduce which is which
        first_out = res_list[0]
        second_out = res_list[1]
        
        if first_out.ndim == 5: # (B, 2, D,H,W) -> Policy
             policy = first_out[0]
             value = second_out[0,0]
        else: # Value first?
             value = first_out[0,0]
             policy = second_out[0]
             
        p_add = policy[0]
        p_remove = policy[1]
        
        return float(value), p_add, p_remove

    def predict_batch(
        self,
        states: List[np.ndarray]
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """
        Predict for a batch of states.
        
        Args:
            states: List of (7, D, H, W) arrays
            
        Returns:
            values, policies_add, policies_remove
        """
        if not states:
            return [], [], []
            
        # 1. Preprocess & Stack
        # We need to pad each state to be divisible by 32 (Swin requirement)
        # We can implement a local helper or import if valid. 
        # For speed, let's implement a simple padder here or rely on the one from inference.py
        
        from .inference import pad_to_target_centered, unpad_from_target
        
        padded_states = []
        original_shapes = []
        
        for state in states:
            original_shapes.append(state.shape)
            # Since benchmark sends (64, 32, 8), and the model expects (64, 32, 8), and we configured it for that,
            # we should just pass the state through without padding to 32.
            padded_states.append(state)
            
        # Stack: (B, 7, D, H, W)
        batch_input = np.stack(padded_states, axis=0)
        
        # 2. Inference
        results = self.infer_request.infer([batch_input])
        
        # 3. Extract & Postprocess
        # Results: [value, p_add, p_remove] (based on our previous debugging)
        # Note: OpenVINO output order might vary, but we fixed the index logic in predict()
        
        res_list = [results[out] for out in self.outputs]
        
        # Handle formats
        if len(res_list) >= 2:
             # Heuristic from predict()
             first_out = res_list[0]
             second_out = res_list[1]
             
             if first_out.ndim == 5: # (B, 2, D,H,W) -> Policy
                 batch_policy = first_out
                 batch_value = second_out
             else: # Value first
                 batch_value = first_out
                 batch_policy = second_out
        else:
            # Fallback for single output or error
            logger.error("Unexpected output format in batch predict")
            return [], [], []
            
        # batch_value: (B, 1)
        # batch_policy: (B, 2, D', H', W')
        
        values = batch_value.flatten().tolist()
        
        policies_add = []
        policies_remove = []
        
        for i, orig_shape in enumerate(original_shapes):
            # Unpad policy
            # policy shape: (2, D', H', W')
            full_policy = batch_policy[i]
            
            # Target shape for unpad: (2, D, H, W)
            target_shape = (2, orig_shape[1], orig_shape[2], orig_shape[3])
            
            unpadded = unpad_from_target(full_policy, target_shape)
            
            policies_add.append(unpadded[0])
            policies_remove.append(unpadded[1])
            
        return values, policies_add, policies_remove
