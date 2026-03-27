"""
Unit tests for inference pipeline, vehicle, haze, and XAI worker lifecycle.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import threading
import time
import numpy as np
import pytest
import torch

from config import Config
from engine.haze import HazeEffect
from engine.vehicle import VehicleController, VehicleState
from models.inference import ModelOutput


# ---------------------------------------------------------------------------
# HazeEffect unit tests
# ---------------------------------------------------------------------------

class TestHazeEffect:
    def test_output_shape_preserved(self):
        haze = HazeEffect(0.6)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = haze.apply_to_frame(frame)
        assert result.shape == frame.shape

    def test_output_dtype_uint8(self):
        haze = HazeEffect(0.6)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = haze.apply_to_frame(frame)
        assert result.dtype == np.uint8

    def test_full_haze_produces_white(self):
        haze = HazeEffect(intensity=1.0)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = haze.apply_to_frame(frame)
        assert np.all(result == 255)

    def test_zero_haze_preserves_frame(self):
        haze = HazeEffect(intensity=0.0)
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = haze.apply_to_frame(frame)
        assert np.allclose(result.astype(int), frame.astype(int), atol=1)

    def test_boundary_pixel_values(self):
        haze = HazeEffect(0.6)
        frame = np.array([[[0, 0, 0]], [[255, 255, 255]]], dtype=np.uint8)
        result = haze.apply_to_frame(frame)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# VehicleController unit tests
# ---------------------------------------------------------------------------

class TestVehicleController:
    def test_speed_increases_with_gas(self):
        ctrl = VehicleController()
        output = ModelOutput(0.0, 1.0, 0.0, 1)
        ctrl.update(output, dt=1.0)
        assert ctrl.state.speed > 0

    def test_speed_capped_at_max(self):
        ctrl = VehicleController(max_speed=120.0)
        output = ModelOutput(0.0, 1.0, 0.0, 5)
        for _ in range(1000):
            ctrl.update(output, dt=0.1)
        assert ctrl.state.speed <= 120.0

    def test_speed_never_negative(self):
        ctrl = VehicleController()
        ctrl.state.speed = 0.0
        output = ModelOutput(0.0, 0.0, 1.0, 0)
        ctrl.update(output, dt=1.0)
        assert ctrl.state.speed >= 0.0

    def test_pause_freezes_state(self):
        ctrl = VehicleController()
        ctrl.state.speed = 50.0
        ctrl.state.x = 10.0
        output = ModelOutput(30.0, 1.0, 0.0, 2)
        ctrl.update(output, dt=0.1, paused=True)
        assert ctrl.state.speed == 50.0
        assert ctrl.state.x == 10.0

    def test_reset_returns_to_origin(self):
        ctrl = VehicleController()
        ctrl.state.x = 100.0
        ctrl.state.speed = 80.0
        ctrl.reset()
        assert ctrl.state.x == 0.0
        assert ctrl.state.speed == 0.0

    def test_heading_wraps_360(self):
        ctrl = VehicleController()
        ctrl.state.speed = 120.0
        ctrl.state.heading = 350.0
        output = ModelOutput(115.0, 0.5, 0.0, 2)
        for _ in range(100):
            ctrl.update(output, dt=0.1)
        assert 0.0 <= ctrl.state.heading < 360.0

    def test_gear_updated_from_output(self):
        ctrl = VehicleController()
        output = ModelOutput(0.0, 0.5, 0.0, 4)
        ctrl.update(output, dt=0.1)
        assert ctrl.state.gear == 4


# ---------------------------------------------------------------------------
# Output clamping unit tests
# ---------------------------------------------------------------------------

class TestOutputClamping:
    def _clamp(self, s, g, b, gear_f):
        raw = torch.tensor([[s, g, b, gear_f]])
        raw = raw.squeeze(0).float()
        raw = torch.nan_to_num(raw, nan=0.0)
        steering = float(torch.clamp(raw[0], -122.0, 115.0))
        gas      = float(torch.clamp(raw[1], 0.0, 1.0))
        brake    = float(torch.clamp(raw[2], 0.0, 1.0))
        gear     = int(torch.clamp(raw[3].round(), 0, 5))
        return ModelOutput(steering, gas, brake, gear)

    def test_steering_lower_bound(self):
        out = self._clamp(-999, 0.5, 0.0, 2)
        assert out.steering_degrees == -122.0

    def test_steering_upper_bound(self):
        out = self._clamp(999, 0.5, 0.0, 2)
        assert out.steering_degrees == 115.0

    def test_gas_clamped_to_one(self):
        out = self._clamp(0, 5.0, 0.0, 1)
        assert out.gas_pedal == 1.0

    def test_brake_clamped_to_zero(self):
        out = self._clamp(0, 0.5, -3.0, 1)
        assert out.brake_pedal == 0.0

    def test_gear_clamped_to_five(self):
        out = self._clamp(0, 0.5, 0.0, 99)
        assert out.gear == 5

    def test_gear_clamped_to_zero(self):
        out = self._clamp(0, 0.5, 0.0, -5)
        assert out.gear == 0

    def test_nan_replaced_with_zero(self):
        raw = torch.tensor([[float("nan"), float("nan"), float("nan"), float("nan")]])
        raw = raw.squeeze(0).float()
        raw = torch.nan_to_num(raw, nan=0.0)
        assert not torch.any(torch.isnan(raw))


# ---------------------------------------------------------------------------
# Config unit tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_instantiation(self):
        cfg = Config()
        assert cfg.target_fps == 30
        assert cfg.haze_intensity == 0.6
        assert cfg.max_speed_kmh == 120.0

    def test_override_values(self):
        cfg = Config(target_fps=60, haze_intensity=0.3)
        assert cfg.target_fps == 60
        assert cfg.haze_intensity == 0.3

    def test_all_keys_present(self):
        cfg = Config()
        required = [
            "cnn_model_path", "ltc_model_path", "xai_update_interval",
            "target_fps", "haze_intensity", "manual_override",
            "window_width", "window_height", "max_speed_kmh",
            "road_tile_length", "road_tiles_ahead", "active_model",
        ]
        for key in required:
            assert hasattr(cfg, key), f"Missing: {key}"


# ---------------------------------------------------------------------------
# DeepLIFT worker lifecycle (no GPU required — uses mock model)
# ---------------------------------------------------------------------------

class _TinyModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=[1, 2, 3]).unsqueeze(1).expand(-1, 4)


class TestDeepLIFTWorker:
    def test_worker_starts_and_stops(self):
        from xai.deeplift import DeepLIFTWorker
        model = _TinyModel()
        worker = DeepLIFTWorker(model)
        worker.start()
        assert worker.is_alive()
        worker.stop()
        worker.join(timeout=3.0)
        assert not worker.is_alive()

    def test_get_latest_returns_none_initially(self):
        from xai.deeplift import DeepLIFTWorker
        model = _TinyModel()
        worker = DeepLIFTWorker(model)
        worker.start()
        result = worker.get_latest()
        assert result is None
        worker.stop()

    def test_submit_does_not_block(self):
        from xai.deeplift import DeepLIFTWorker
        model = _TinyModel()
        worker = DeepLIFTWorker(model)
        worker.start()
        frame = torch.zeros(1, 3, 224, 224)
        t0 = time.time()
        for _ in range(10):
            worker.submit(frame, 0)
        elapsed = time.time() - t0
        assert elapsed < 0.5, "submit() should be non-blocking"
        worker.stop()
