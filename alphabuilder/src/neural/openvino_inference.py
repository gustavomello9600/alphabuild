"""
OpenVINO Inference Backend for AlphaBuilder v3.3.

Optimized for Intel Core Ultra NPU / GPU / CPU inference.
Reads model architecture metadata directly from checkpoint config.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import openvino as ov

from alphabuilder.src.neural.model import AlphaBuilderV31

logger = logging.getLogger("OpenVINO")


class AlphaBuilderOpenVINO:
    """
    OpenVINO Inference Backend for AlphaBuilder.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "NPU",
        model_config: Optional[dict] = None,
    ):
        self.device = device.upper()
        self.core = ov.Core()

        available_devices = self.core.available_devices
        logger.info(f"OpenVINO Available Devices: {available_devices}")

        if self.device not in available_devices:
            if "GPU" in available_devices:
                logger.warning(f"Device {self.device} not found. Using GPU instead.")
                self.device = "GPU"
            else:
                logger.warning(f"Device {self.device} not found, GPU not available. Falling back to AUTO.")
                self.device = "AUTO"

        logger.info(f"Selected OpenVINO device: {self.device}")
        logger.info(f"Loading PyTorch model from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        merged_cfg: Dict[str, object] = {**ckpt_config, **(model_config or {})}

        self.config = merged_cfg
        self.epoch = checkpoint.get("epoch", -1) if isinstance(checkpoint, dict) else -1
        metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        self.val_loss = checkpoint.get("val_loss", metrics.get("loss", float("inf"))) if isinstance(checkpoint, dict) else float("inf")

        in_channels = int(merged_cfg.get("in_channels", 7))
        out_channels = int(merged_cfg.get("out_channels", 2))
        feature_size = int(merged_cfg.get("feature_size", 24))
        self.use_swin = bool(merged_cfg.get("use_swin", False))

        self.pt_model = AlphaBuilderV31(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_swin=self.use_swin,
        )

        raw_sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in raw_sd.items()
        }
        self.pt_model.load_state_dict(state_dict)
        self.pt_model.eval()

        logger.info("Converting to OpenVINO IR...")
        example_input = torch.randn(1, in_channels, 64, 32, 8)
        try:
            self.ov_model = ov.convert_model(
                self.pt_model,
                example_input=example_input,
                input=[(-1, in_channels, 64, 32, 8)],
            )

            logger.info(f"Compiling model for {self.device}...")
            perf_hint = str(merged_cfg.get("performance_hint", "LATENCY"))
            compile_config = {"PERFORMANCE_HINT": perf_hint}
            self.compiled_model = self.core.compile_model(self.ov_model, self.device, compile_config)
            self.infer_request = self.compiled_model.create_infer_request()
            logger.info(f"OpenVINO Model compiled successfully on {self.device}")
        except Exception as e:
            logger.error(f"OpenVINO conversion/compilation failed: {e}")
            raise

        self.input_layer = self.compiled_model.input(0)
        self.outputs = self.compiled_model.outputs

    def _unpack_outputs(self, results: dict) -> Tuple[np.ndarray, np.ndarray]:
        res_list = [results[out] for out in self.outputs]
        if len(res_list) < 2:
            raise RuntimeError(f"Expected at least 2 outputs from OpenVINO IR, got {len(res_list)}")
        first_out, second_out = res_list[0], res_list[1]
        if first_out.ndim == 5:
            return first_out, second_out
        return second_out, first_out

    def predict(self, state: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run inference on a single (C, D, H, W) numpy state."""
        batch_input = np.expand_dims(state, 0) if state.ndim == 4 else state
        batch_input = np.ascontiguousarray(batch_input, dtype=np.float32)

        results = self.infer_request.infer([batch_input])
        batch_policy, batch_value = self._unpack_outputs(results)

        policy = batch_policy[0]
        value = float(batch_value.reshape(-1)[0])
        return value, policy[0], policy[1]

    def predict_batch(
        self,
        states: List[np.ndarray],
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """Run batched inference on a list of (7, D, H, W) states."""
        if not states:
            return [], [], []

        from .inference import unpad_from_target

        original_shapes = [s.shape for s in states]
        batch_input = np.ascontiguousarray(np.stack(states, axis=0), dtype=np.float32)

        results = self.infer_request.infer([batch_input])
        batch_policy, batch_value = self._unpack_outputs(results)

        values = [float(v) for v in batch_value.reshape(-1).tolist()]
        policies_add: List[np.ndarray] = []
        policies_remove: List[np.ndarray] = []
        for i, orig_shape in enumerate(original_shapes):
            pol = batch_policy[i]
            target_shape = (pol.shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
            pol = unpad_from_target(pol, target_shape)
            policies_add.append(pol[0])
            policies_remove.append(pol[1])

        return values, policies_add, policies_remove
