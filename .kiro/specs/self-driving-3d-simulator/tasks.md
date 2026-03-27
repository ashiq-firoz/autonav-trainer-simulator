# Implementation Plan: Self-Driving 3D Simulator

## Overview

Incremental implementation starting from project scaffolding and config, through the 3D scene and vehicle physics, then model inference, HUD, and XAI panel. Each task builds on the previous. Property-based tests (Hypothesis) are placed close to the component they validate.

## Tasks

- [ ] 1. Project scaffolding and configuration
  - Create the directory structure: `simulator/engine/`, `simulator/models/`, `simulator/xai/`, `simulator/hud/`, `simulator/assets/textures/`, `simulator/assets/models/`
  - Write `config.py` with the `Config` dataclass covering all parameters (model paths, XAI interval, FPS target, haze intensity, manual override, window resolution, max speed, tile length, tiles ahead, log interval) with inline comments and documented defaults
  - Write `requirements.txt` with pinned dependencies: panda3d, torch, torchvision, timm, captum, ncps, opencv-python, numpy, hypothesis
  - Write `__init__.py` files for each package
  - _Requirements: 9.1, 9.2, 9.3_

  - [ ] 1.1 Write property test for Config default fallback
    - **Property 13: Config Default Fallback**
    - **Validates: Requirements 9.2**
    - Use `@given(st.sets(st.sampled_from(CONFIG_KEYS)))` to omit random subsets of keys and assert all missing keys resolve to their documented defaults with no exceptions

- [ ] 2. Model definitions
  - [ ] 2.1 Implement `CNNModel` in `models/cnn_model.py`
    - EfficientNetV2-S backbone via timm (`create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0)`)
    - FC head: Linear(1280→512)+ReLU+Dropout(0.3), Linear(512→256)+ReLU+Dropout(0.2), Linear(256→4)
    - `forward(x)` returns raw tensor of shape (B, 4)
    - _Requirements: 3.1, 3.7_

  - [ ] 2.2 Implement `LTCModel` in `models/ltc_model.py`
    - EfficientNetV2-S backbone (same as CNN, features only)
    - Feature adapter: Linear(1280→128)
    - LTC/CfC layer: `AutoNCP` wiring from ncps, 128 hidden units, 4 outputs
    - `forward(x)` returns raw tensor of shape (B, 4)
    - `reset_hidden()` clears the LTC hidden state
    - _Requirements: 3.2, 3.7_

- [ ] 3. Inference pipeline
  - [ ] 3.1 Implement `InferencePipeline` in `models/inference.py`
    - `load_models(config)`: load CNN and LTC from .pth paths; handle missing files per error handling table (log + continue / sys.exit)
    - `_preprocess(frame, haze_active)`: apply haze if active, resize to 224×224, ToTensor, ImageNet normalize, add batch dim, move to device
    - `_clamp_output(raw_tensor)`: clamp steering to [-122, 115], gas/brake to [0, 1], gear to nearest int in [0, 5]
    - `run(frame, haze_active)`: preprocess → forward pass → clamp → return `ModelOutput`
    - `switch_model(key)`: set active model, return False if model unavailable
    - Rolling inference time logging every `config.log_inference_every` frames
    - _Requirements: 3.1–3.10_

  - [ ] 3.2 Write property test for preprocessing output shape
    - **Property 4: Preprocessing Output Shape**
    - **Validates: Requirements 3.5**
    - `@given(st.integers(64,1024), st.integers(64,1024))` → create random uint8 frame, assert output tensor shape is (1, 3, 224, 224)

  - [ ] 3.3 Write property test for output clamping invariant
    - **Property 5: Output Clamping Invariant**
    - **Validates: Requirements 3.8, 3.9, 3.10**
    - `@given(st.floats(-1e6,1e6), st.floats(-1e6,1e6), st.floats(-1e6,1e6), st.floats(-1e6,1e6))` → assert all output fields within valid ranges

  - [ ] 3.4 Write unit tests for model loading edge cases
    - Test: valid path loads model successfully
    - Test: missing CNN path logs error, LTC still available
    - Test: both paths missing triggers sys.exit(1)
    - Test: NaN in raw output is replaced with 0.0
    - _Requirements: 3.3, 3.4_

- [ ] 4. Checkpoint — inference pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Haze effect
  - Implement `HazeEffect` in `engine/haze.py`
  - `apply_to_frame(frame)`: `cv2.addWeighted(frame, 1-intensity, white_frame, intensity, 0)` where white_frame is `np.full_like(frame, 255)`
  - `apply_to_scene(render)`: create `Fog(Fog.MExponential)`, set color to white, set density, attach to render node
  - `remove_from_scene(render)`: clear fog from render node
  - _Requirements: 2.1, 2.2_

  - [ ] 5.1 Write property test for haze frame blending
    - **Property 2: Haze Frame Blending**
    - **Validates: Requirements 2.1, 2.5**
    - `@given(arrays(np.uint8, shape=(H, W, 3)))` → verify each output pixel equals `round(p * (1-intensity) + 255 * intensity)` within tolerance 1

  - [ ] 5.2 Write property test for haze toggle round-trip
    - **Property 3: Haze Toggle Round-Trip**
    - **Validates: Requirements 2.3**
    - `@given(st.booleans())` → set initial state, toggle twice, assert state equals original

- [ ] 6. Vehicle controller
  - Implement `VehicleState` dataclass and `VehicleController` in `engine/vehicle.py`
  - `update(output, dt)`: apply kinematic model (speed update with gas/brake, heading update with steering, position update with forward vector)
  - Clamp speed to [0.0, max_speed] after each update
  - Skip update entirely when `sim_state.paused == True`
  - `apply_manual(steering, throttle, dt)`: manual override path
  - `reset()`: return vehicle to origin with zero speed
  - _Requirements: 4.1–4.5_

  - [ ] 6.1 Write property test for vehicle speed invariant
    - **Property 6: Vehicle Speed Invariant**
    - **Validates: Requirements 4.2, 4.3**
    - `@given(vehicle_state_strategy, model_output_strategy, st.floats(0.01, 1.0))` → assert speed ∈ [0.0, 120.0] after update

  - [ ] 6.2 Write property test for vehicle pause invariant
    - **Property 7: Vehicle Pause Invariant**
    - **Validates: Requirements 4.5**
    - `@given(vehicle_state_strategy, model_output_strategy)` → call update with paused=True, assert position/heading/speed unchanged

- [ ] 7. 3D scene manager
  - [ ] 7.1 Implement `SceneManager` in `engine/scene.py`
    - Initialize Panda3D scene: directional light, ambient light, sky dome (SkyBox or simple colored background)
    - Road tile pool: pre-create 10 `CardMaker` quads with asphalt texture tiling; place tiles in a line ahead of origin
    - `update(vehicle_pos, vehicle_heading)`: recycle tiles behind vehicle, spawn new tiles ahead to maintain `road_tiles_ahead` count
    - `capture_frame()`: use Panda3D `GraphicsOutput.getScreenshot()` or offscreen buffer to capture 224×224 RGB numpy array
    - `set_haze(active)`: delegate to `HazeEffect`
    - _Requirements: 1.1–1.6_

  - [ ] 7.2 Add lane markings and roadside elements
    - Lane marking decal texture overlaid on road tiles using a second `TextureStage`
    - Simple billboard vegetation (flat quads with tree/bush textures) placed at random offsets along road edges
    - _Requirements: 1.2, 1.3_

  - [ ] 7.3 Write property test for road tile continuity
    - **Property 1: Road Tile Continuity**
    - **Validates: Requirements 1.5**
    - `@given(st.floats(0, 10000))` → simulate vehicle advancing to random position, assert active tiles ahead >= `road_tiles_ahead`

- [ ] 8. HUD dashboard
  - Implement `HUDRenderer` in `hud/dashboard.py`
  - Steering arc: `LineSegs` arc from -122° to +115° with a needle `LineSegs` at current angle, attached to `aspect2d`
  - Gas bar: `DirectWaitBar` (green, vertical, bottom-left area)
  - Brake bar: `DirectWaitBar` (red, vertical, next to gas bar)
  - Gear, speed, frame counter, haze label, model label: `OnscreenText` nodes
  - Pause indicator: `OnscreenText` shown/hidden based on `sim_state.paused`
  - `update(state, output, sim)`: update all display values each frame
  - _Requirements: 5.1–5.10_

  - [ ] 8.1 Write property test for HUD value mapping consistency
    - **Property 8: HUD Value Mapping Consistency**
    - **Validates: Requirements 5.1, 5.2, 5.3**
    - `@given(model_output_strategy)` → call update(), assert gas_bar.value == output.gas_pedal, brake_bar.value == output.brake_pedal, steering needle angle proportional to steering_degrees

- [ ] 9. DeepLIFT worker
  - Implement `DeepLIFTWorker` and `AttributionResult` in `xai/deeplift.py`
  - Worker thread: consume `(frame_tensor, output_index)` from `input_queue`, run `DeepLift(model).attribute(input, target=output_index)`, compute heatmap and overlay, store in `latest_result` under `result_lock`
  - `submit(frame, output_index)`: non-blocking put to queue (drop if queue full to avoid backlog)
  - `get_latest()`: acquire lock, return copy of `latest_result`
  - `stop()`: set `running=False`, join thread
  - _Requirements: 6.1, 6.2, 6.7, 6.8_

  - [ ] 9.1 Write property test for attribution result completeness
    - **Property 9: Attribution Result Completeness**
    - **Validates: Requirements 6.1, 6.2**
    - `@given(st.integers(0, 3))` → submit a fixed frame with each output index, assert result shapes are (224, 224, 3) uint8 and output_name is valid

  - [ ] 9.2 Write unit tests for DeepLIFT worker lifecycle
    - Test: worker thread starts and stops cleanly
    - Test: get_latest() returns None before first result
    - Test: get_latest() returns last result while new computation is in progress
    - _Requirements: 6.7, 6.8_

- [ ] 10. XAI panel
  - Implement `XAIPanel` in `xai/visualizer.py`
  - Three `OnscreenImage` nodes on the right side of the screen: original, heatmap, overlay
  - `maybe_update(frame, haze_active)`: increment counter; every N frames, preprocess frame and submit to worker; call `_refresh_display` if new result available
  - `cycle_output()`: `xai_output_index = (xai_output_index + 1) % 4`; update output label
  - `_refresh_display(result)`: update `OnscreenImage` textures from numpy arrays
  - `switch_model(model)`: update worker's model reference
  - _Requirements: 6.1–6.8_

  - [ ] 10.1 Write property test for XAI update interval enforcement
    - **Property 10: XAI Update Interval Enforcement**
    - **Validates: Requirements 6.3**
    - `@given(st.integers(1,20), st.integers(1,100))` → call maybe_update M times with interval N, assert worker received exactly floor(M/N) submissions

  - [ ] 10.2 Write property test for XAI output index cycling
    - **Property 11: XAI Output Index Cycling**
    - **Validates: Requirements 6.4**
    - `@given(st.integers(0,3))` → cycle 4 times from any starting index, assert index equals original

- [ ] 11. Checkpoint — all subsystems implemented
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. SimulatorApp wiring (`main.py`)
  - Subclass `ShowBase`, instantiate all subsystems with config
  - Register keyboard bindings: Space→toggle_pause, H→toggle_haze, 1→switch_model("cnn"), 2→switch_model("ltc"), X→cycle_xai_output, ESC→quit, WASD/arrows→manual override (if enabled)
  - `main_loop(task)`: if not paused → capture_frame → run inference → update vehicle → update scene; always → update HUD → maybe update XAI; return `Task.cont`
  - `switch_model(key)`: call `inference.switch_model(key)`, update `sim_state.active_model`, call `xai_panel.switch_model()`, show HUD error if unavailable
  - `quit()`: stop XAI worker, cleanly exit
  - _Requirements: 1.1, 2.3, 3.1–3.3, 7.1–7.5, 8.1–8.6_

  - [ ] 12.1 Write property test for pause toggle round-trip
    - **Property 12: Pause Toggle Round-Trip**
    - **Validates: Requirements 8.1**
    - `@given(st.booleans())` → set initial paused state, toggle twice, assert equals original

- [ ] 13. Final checkpoint — full integration
  - Ensure all tests pass with `pytest --hypothesis-seed=0`
  - Verify simulator launches with `python simulator/main.py` and renders at target FPS
  - Ask the user if questions arise.

## Notes

- All tasks including property-based and unit tests are required
- Property tests use Hypothesis with `@settings(max_examples=100)` minimum
- Each property test references its design document property number in a comment
- Checkpoints ensure incremental validation before proceeding
- The LTC model requires `reset_hidden()` to be called between simulation episodes to clear recurrent state
