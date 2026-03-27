"""
Inference pipeline: frame preprocessing, model forward pass, output clamping.
Handles model loading errors gracefully and logs inference timing.
"""
import sys
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from config import Config
from models.cnn_model import CNNModel
from models.ltc_model import LTCModel, NCPS_AVAILABLE

log = logging.getLogger(__name__)

# ImageNet normalization (matches training preprocessing)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


@dataclass
class ModelOutput:
    steering_degrees: float   # clamped [-122.0, 115.0]
    gas_pedal: float          # clamped [0.0, 1.0]
    brake_pedal: float        # clamped [0.0, 1.0]
    gear: int                 # clamped [0, 5]


class InferencePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            log.warning("No CUDA GPU found — running inference on CPU. FPS may be below target.")

        self._models: Dict[str, Optional[nn.Module]] = {"cnn": None, "ltc": None}
        self._active_key: str = config.active_model
        self._inference_times: deque = deque(maxlen=config.log_inference_every)
        self._frame_count: int = 0

        self.load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        self._models["cnn"] = self._load_one("cnn", self.config.cnn_model_path, CNNModel)
        if NCPS_AVAILABLE:
            self._models["ltc"] = self._load_one("ltc", self.config.ltc_model_path, LTCModel)
        else:
            log.warning("ncps library not available — LTC model disabled.")

        available = [k for k, v in self._models.items() if v is not None]
        if not available:
            log.critical("No models could be loaded. Exiting.")
            sys.exit(1)

        if self._active_key not in available:
            self._active_key = available[0]
            log.warning("Configured active model unavailable; falling back to '%s'.", self._active_key)

    def _load_one(self, key: str, path: str, cls) -> Optional[nn.Module]:
        try:
            model = cls()
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            log.info("Loaded %s model from %s", key.upper(), path)
            return model
        except FileNotFoundError:
            log.error("%s model weights not found at '%s' — model disabled.", key.upper(), path)
            return None
        except Exception as exc:
            log.error("Failed to load %s model: %s", key.upper(), exc)
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray, haze_active: bool) -> ModelOutput:
        """Run inference on a raw BGR/RGB uint8 frame."""
        t0 = time.perf_counter()

        tensor = self._preprocess(frame, haze_active)
        model = self._models[self._active_key]

        with torch.no_grad():
            raw = model(tensor)   # (1, 4)

        output = self._clamp_output(raw)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._inference_times.append(elapsed_ms)
        self._frame_count += 1

        if self._frame_count % self.config.log_inference_every == 0:
            avg = sum(self._inference_times) / len(self._inference_times)
            log.info("Frame %d | avg inference: %.1f ms | model: %s",
                     self._frame_count, avg, self._active_key.upper())

        return output

    def _preprocess(self, frame: np.ndarray, haze_active: bool) -> torch.Tensor:
        """Apply optional haze, resize to 224x224, normalize. Returns (1,3,224,224)."""
        if haze_active:
            intensity = self.config.haze_intensity
            white = np.full_like(frame, 255)
            frame = cv2.addWeighted(frame, 1.0 - intensity, white, intensity, 0)

        # frame is BGR from OpenCV / Panda3D — convert to RGB for torchvision
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tensor = _TRANSFORM(frame)          # (3, 224, 224)
        return tensor.unsqueeze(0).to(self.device)   # (1, 3, 224, 224)

    def _clamp_output(self, raw: torch.Tensor) -> ModelOutput:
        """Clamp raw model output to valid ranges. Replaces NaN with 0."""
        raw = raw.squeeze(0).cpu().float()
        # Replace NaN
        raw = torch.nan_to_num(raw, nan=0.0)

        steering = float(torch.clamp(raw[0], -122.0, 115.0))
        gas      = float(torch.clamp(raw[1], 0.0, 1.0))
        brake    = float(torch.clamp(raw[2], 0.0, 1.0))
        gear     = int(torch.clamp(raw[3].round(), 0, 5))

        return ModelOutput(steering, gas, brake, gear)

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    def switch_model(self, key: str) -> bool:
        """Switch active model. Returns False if model is unavailable."""
        if self._models.get(key) is None:
            log.warning("Cannot switch to '%s' — model not loaded.", key)
            return False
        self._active_key = key
        # Reset LTC hidden state on switch
        m = self._models["ltc"]
        if m is not None and hasattr(m, "reset_hidden"):
            m.reset_hidden()
        log.info("Switched active model to %s.", key.upper())
        return True

    @property
    def active_key(self) -> str:
        return self._active_key

    def get_active_model(self) -> nn.Module:
        return self._models[self._active_key]
