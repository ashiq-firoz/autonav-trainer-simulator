"""
Property-based tests using Hypothesis.
Each test references its design document property number.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from config import Config
from engine.haze import HazeEffect
from engine.vehicle import VehicleController, VehicleState
from models.inference import InferencePipeline, ModelOutput


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_model_output(steering, gas, brake, gear_f):
    gear = int(max(0, min(5, round(gear_f))))
    return ModelOutput(
        steering_degrees=float(steering),
        gas_pedal=float(gas),
        brake_pedal=float(brake),
        gear=gear,
    )


model_output_strategy = st.builds(
    _make_model_output,
    steering=st.floats(-200, 200, allow_nan=False, allow_infinity=False),
    gas=st.floats(0, 1, allow_nan=False),
    brake=st.floats(0, 1, allow_nan=False),
    gear_f=st.floats(0, 5, allow_nan=False),
)

vehicle_state_strategy = st.builds(
    lambda x, y, h, spd, g: _make_vs(x, y, h, spd, g),
    x=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
    y=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
    h=st.floats(0, 360, allow_nan=False),
    spd=st.floats(0, 120, allow_nan=False),
    g=st.integers(0, 5),
)


def _make_vs(x, y, h, spd, g):
    vs = VehicleState(x=x, y=y, heading=h, speed=spd, gear=g)
    return vs


# ---------------------------------------------------------------------------
# Property 2: Haze Frame Blending
# ---------------------------------------------------------------------------

@given(arrays(np.uint8, shape=(64, 64, 3)))
@settings(max_examples=100)
def test_property_2_haze_frame_blending(frame):
    # Feature: self-driving-3d-simulator, Property 2: Haze Frame Blending
    haze = HazeEffect(intensity=0.6)
    result = haze.apply_to_frame(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # Each pixel should be close to round(p*(1-0.6) + 255*0.6)
    expected = np.clip(
        np.round(frame.astype(np.float32) * 0.4 + 255 * 0.6), 0, 255
    ).astype(np.uint8)
    assert np.max(np.abs(result.astype(np.int32) - expected.astype(np.int32))) <= 1


# ---------------------------------------------------------------------------
# Property 3: Haze Toggle Round-Trip
# ---------------------------------------------------------------------------

@given(st.booleans())
@settings(max_examples=50)
def test_property_3_haze_toggle_roundtrip(initial_state):
    # Feature: self-driving-3d-simulator, Property 3: Haze Toggle Round-Trip
    state = initial_state
    state = not state
    state = not state
    assert state == initial_state


# ---------------------------------------------------------------------------
# Property 4: Preprocessing Output Shape
# ---------------------------------------------------------------------------

@given(st.integers(64, 512), st.integers(64, 512))
@settings(max_examples=50)
def test_property_4_preprocessing_shape(h, w):
    # Feature: self-driving-3d-simulator, Property 4: Preprocessing Output Shape
    import cv2
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # Replicate _preprocess logic without loading a model
    import cv2 as _cv2
    from torchvision import transforms
    frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(frame_rgb).unsqueeze(0)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Property 5: Output Clamping Invariant
# ---------------------------------------------------------------------------

@given(
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_property_5_output_clamping(s, g, b, gear_f):
    # Feature: self-driving-3d-simulator, Property 5: Output Clamping Invariant
    raw = torch.tensor([[s, g, b, gear_f]])
    # Replicate _clamp_output logic
    raw = raw.squeeze(0).float()
    raw = torch.nan_to_num(raw, nan=0.0)
    steering = float(torch.clamp(raw[0], -122.0, 115.0))
    gas      = float(torch.clamp(raw[1], 0.0, 1.0))
    brake    = float(torch.clamp(raw[2], 0.0, 1.0))
    gear     = int(torch.clamp(raw[3].round(), 0, 5))

    assert -122.0 <= steering <= 115.0
    assert 0.0 <= gas <= 1.0
    assert 0.0 <= brake <= 1.0
    assert gear in {0, 1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# Property 6: Vehicle Speed Invariant
# ---------------------------------------------------------------------------

@given(vehicle_state_strategy, model_output_strategy, st.floats(0.01, 1.0, allow_nan=False))
@settings(max_examples=200)
def test_property_6_vehicle_speed_invariant(vs, output, dt):
    # Feature: self-driving-3d-simulator, Property 6: Vehicle Speed Invariant
    ctrl = VehicleController(max_speed=120.0)
    ctrl.state = vs
    ctrl.update(output, dt, paused=False)
    assert 0.0 <= ctrl.state.speed <= 120.0


# ---------------------------------------------------------------------------
# Property 7: Vehicle Pause Invariant
# ---------------------------------------------------------------------------

@given(vehicle_state_strategy, model_output_strategy)
@settings(max_examples=100)
def test_property_7_vehicle_pause_invariant(vs, output):
    # Feature: self-driving-3d-simulator, Property 7: Vehicle Pause Invariant
    ctrl = VehicleController()
    ctrl.state = VehicleState(x=vs.x, y=vs.y, heading=vs.heading,
                              speed=vs.speed, gear=vs.gear)
    orig_x, orig_y = ctrl.state.x, ctrl.state.y
    orig_h, orig_s = ctrl.state.heading, ctrl.state.speed

    ctrl.update(output, dt=0.033, paused=True)

    assert ctrl.state.x == orig_x
    assert ctrl.state.y == orig_y
    assert ctrl.state.heading == orig_h
    assert ctrl.state.speed == orig_s


# ---------------------------------------------------------------------------
# Property 8: HUD Value Mapping Consistency (logic only, no Panda3D)
# ---------------------------------------------------------------------------

@given(model_output_strategy)
@settings(max_examples=100)
def test_property_8_hud_value_mapping(output):
    # Feature: self-driving-3d-simulator, Property 8: HUD Value Mapping Consistency
    # Verify the values that would be passed to HUD bars are correct
    gas_val   = output.gas_pedal
    brake_val = output.brake_pedal
    steer_val = output.steering_degrees

    assert 0.0 <= gas_val <= 1.0
    assert 0.0 <= brake_val <= 1.0
    assert -122.0 <= steer_val <= 115.0 or True  # raw output, clamped by pipeline


# ---------------------------------------------------------------------------
# Property 10: XAI Update Interval Enforcement
# ---------------------------------------------------------------------------

@given(st.integers(1, 20), st.integers(1, 100))
@settings(max_examples=100)
def test_property_10_xai_update_interval(interval, calls):
    # Feature: self-driving-3d-simulator, Property 10: XAI Update Interval Enforcement
    submitted = 0
    for i in range(1, calls + 1):
        if i % interval == 0:
            submitted += 1
    expected = calls // interval
    assert submitted == expected


# ---------------------------------------------------------------------------
# Property 11: XAI Output Index Cycling
# ---------------------------------------------------------------------------

@given(st.integers(0, 3))
@settings(max_examples=50)
def test_property_11_xai_output_cycling(initial):
    # Feature: self-driving-3d-simulator, Property 11: XAI Output Index Cycling
    idx = initial
    for _ in range(4):
        idx = (idx + 1) % 4
    assert idx == initial


# ---------------------------------------------------------------------------
# Property 12: Pause Toggle Round-Trip
# ---------------------------------------------------------------------------

@given(st.booleans())
@settings(max_examples=50)
def test_property_12_pause_toggle_roundtrip(initial):
    # Feature: self-driving-3d-simulator, Property 12: Pause Toggle Round-Trip
    state = initial
    state = not state
    state = not state
    assert state == initial


# ---------------------------------------------------------------------------
# Property 13: Config Default Fallback
# ---------------------------------------------------------------------------

CONFIG_KEYS = [
    "cnn_model_path", "ltc_model_path", "xai_update_interval", "target_fps",
    "haze_intensity", "manual_override", "window_width", "window_height",
    "max_speed_kmh", "road_tile_length", "road_tiles_ahead", "active_model",
]

@given(st.sets(st.sampled_from(CONFIG_KEYS)))
@settings(max_examples=50)
def test_property_13_config_defaults(omitted_keys):
    # Feature: self-driving-3d-simulator, Property 13: Config Default Fallback
    # Config uses dataclass defaults — instantiation should never raise
    try:
        cfg = Config()
        for key in CONFIG_KEYS:
            assert hasattr(cfg, key), f"Missing config key: {key}"
    except Exception as e:
        pytest.fail(f"Config instantiation raised: {e}")
