"""
Background DeepLIFT worker thread.
Consumes (frame_tensor, output_index) jobs, produces AttributionResult.
Non-blocking: main loop always gets the latest completed result.
"""
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

OUTPUT_NAMES = ["steering", "gas", "brake", "gear"]


@dataclass
class AttributionResult:
    original: np.ndarray    # (224, 224, 3) uint8 RGB
    heatmap: np.ndarray     # (224, 224, 3) uint8 hot colormap
    overlay: np.ndarray     # (224, 224, 3) uint8 blended
    output_name: str        # "steering" | "gas" | "brake" | "gear"
    haze_label: str         # "HAZE" | "NON-HAZE"


class DeepLIFTWorker(threading.Thread):
    def __init__(self, model: nn.Module):
        super().__init__(daemon=True, name="DeepLIFTWorker")
        self.model = model
        self._input_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_result: Optional[AttributionResult] = None
        self._running = True

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        from captum.attr import DeepLift
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_tensor, output_index, haze_active = item
            try:
                result = self._compute(frame_tensor, output_index, haze_active)
                with self._result_lock:
                    self._latest_result = result
            except Exception as e:
                log.warning("DeepLIFT computation failed: %s", e)
            finally:
                self._input_queue.task_done()

    def stop(self) -> None:
        self._running = False
        self.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, frame_tensor: torch.Tensor, output_index: int,
               haze_active: bool = False) -> None:
        """Non-blocking submit. Drops job if queue is full (avoids backlog)."""
        try:
            self._input_queue.put_nowait((frame_tensor, output_index, haze_active))
        except queue.Full:
            pass  # drop — main loop won't stall

    def get_latest(self) -> Optional[AttributionResult]:
        with self._result_lock:
            return self._latest_result

    def switch_model(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute(self, frame_tensor: torch.Tensor, output_index: int,
                 haze_active: bool) -> AttributionResult:
        from captum.attr import DeepLift

        self.model.eval()
        inp = frame_tensor.clone().requires_grad_(True)
        baseline = torch.zeros_like(inp)

        dl = DeepLift(self.model)
        attributions = dl.attribute(inp, baseline, target=output_index)

        # (1, 3, 224, 224) -> (3, 224, 224)
        attr_np = attributions.squeeze(0).detach().cpu().numpy()
        attr_sum = np.sum(np.abs(attr_np), axis=0)   # (224, 224)

        # Normalize to [0, 255]
        attr_min, attr_max = attr_sum.min(), attr_sum.max()
        attr_norm = (attr_sum - attr_min) / (attr_max - attr_min + 1e-8)
        heatmap_gray = (attr_norm * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

        # Original frame: denormalize tensor -> uint8 RGB
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        orig = frame_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        orig = (orig * std + mean) * 255
        orig = np.clip(orig, 0, 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_bgr, 0.4, 0)

        return AttributionResult(
            original=cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB),
            heatmap=cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB),
            overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            output_name=OUTPUT_NAMES[output_index],
            haze_label="HAZE" if haze_active else "NON-HAZE",
        )
