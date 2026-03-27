# Requirements Document

## Introduction

A 3D visual simulator for testing and comparing two trained self-driving car models (EfficientNetV2-based CNN and LTC/CfC Liquid Neural Network) on procedurally generated Indian road environments. The simulator renders a real-time 3D scene using Panda3D, feeds camera frames through the selected model, applies vehicle physics based on model outputs, and displays a HUD dashboard alongside an XAI explainability panel showing DeepLIFT attributions. The primary focus (90%) is on the simulator itself.

## Glossary

- **Simulator**: The Panda3D-based 3D visual simulation application
- **CNN_Model**: The EfficientNetV2-S backbone with fully-connected head outputting [steering_degrees, gas_pedal, brake_pedal, gear]
- **LTC_Model**: The EfficientNetV2-S visual encoder with LTC/CfC liquid neural network head outputting the same four outputs
- **Active_Model**: The currently selected model driving the vehicle (CNN or LTC)
- **Vehicle**: The simulated car entity with position, heading, speed, and gear state
- **HUD**: The heads-up display overlay rendered on top of the 3D scene
- **XAI_Panel**: The side panel displaying DeepLIFT attribution heatmaps for the current frame
- **Attribution**: A per-pixel importance score computed by DeepLIFT indicating which image regions influenced the model output
- **Haze_Effect**: A visual fog/overlay applied to the 3D scene and camera frame to simulate the training data augmentation (cv2.addWeighted, intensity 0.6)
- **Frame**: A single rendered 224x224 RGB image captured from the in-scene camera and passed to the Active_Model
- **Inference_Pipeline**: The preprocessing, model forward pass, and output extraction sequence run each simulation tick
- **Scene**: The Panda3D 3D environment including road segments, sky, vegetation, and lighting
- **Road_Segment**: A procedurally placed tile of road geometry with real-world asphalt/lane-marking textures
- **Steering_Degrees**: Model output in the range [-122, +115] degrees mapped to vehicle wheel turn
- **Gas_Pedal**: Model output in [0, 1] controlling vehicle acceleration
- **Brake_Pedal**: Model output in [0, 1] controlling vehicle deceleration
- **Gear**: Model output integer in [0, 5] representing current transmission gear
- **DeepLIFT**: The Captum-based attribution method used to compute per-pixel importance scores
- **XAI_Output_Index**: The currently selected model output (0=steering, 1=gas, 2=brake, 3=gear) for which attributions are displayed

---

## Requirements

### Requirement 1: 3D Scene Rendering

**User Story:** As a researcher, I want a 3D road environment rendered in real time, so that I can visually evaluate how the self-driving model perceives and navigates realistic road conditions.

#### Acceptance Criteria

1. THE Simulator SHALL render a 3D scene at a minimum of 30 frames per second during normal operation.
2. THE Scene SHALL include procedurally generated road segments using real-world asphalt and lane-marking textures sourced from CC0 libraries (ambientCG or Poly Haven).
3. THE Scene SHALL include sky, vegetation, and roadside environment elements with real-world textures.
4. WHEN the simulation starts, THE Scene SHALL initialize with a straight road segment of at least 500 meters of drivable surface.
5. WHEN the Vehicle reaches the end of a road segment, THE Scene SHALL extend the road procedurally to maintain continuous drivable surface ahead.
6. THE Scene SHALL use Panda3D as the 3D rendering engine.

---

### Requirement 2: Haze Weather Effect

**User Story:** As a researcher, I want the simulator to support haze/fog conditions matching the training data augmentation, so that I can test model robustness under the same visual conditions it was trained on.

#### Acceptance Criteria

1. THE Simulator SHALL support a haze mode that applies a white overlay to the rendered camera frame using the same parameters as training augmentation (cv2.addWeighted, intensity 0.6).
2. WHEN haze mode is active, THE Scene SHALL also apply a Panda3D exponential fog effect to the 3D environment to visually match the haze appearance.
3. WHEN the user presses H, THE Simulator SHALL toggle haze mode between active and inactive.
4. THE HUD SHALL display "HAZY" when haze mode is active and "CLEAR" when haze mode is inactive.
5. WHEN haze mode is toggled, THE Inference_Pipeline SHALL apply or remove the haze preprocessing step on the next frame.

---

### Requirement 3: Model Loading and Inference

**User Story:** As a researcher, I want to load trained model weights at startup and run real-time inference, so that I can observe model behavior on the simulated road.

#### Acceptance Criteria

1. WHEN the Simulator starts, THE Inference_Pipeline SHALL load the CNN_Model weights from a .pth file path specified in config.py.
2. WHEN the Simulator starts, THE Inference_Pipeline SHALL load the LTC_Model weights from a .pth file path specified in config.py.
3. WHEN a model .pth file is not found at the configured path, THE Inference_Pipeline SHALL log a descriptive error and continue with the remaining model available.
4. WHEN both model .pth files are not found, THE Inference_Pipeline SHALL log a descriptive error and exit with a non-zero status code.
5. EACH simulation frame, THE Inference_Pipeline SHALL capture the current camera view, resize it to 224x224 pixels, normalize it using ImageNet mean and standard deviation, and pass it to the Active_Model.
6. THE Inference_Pipeline SHALL complete model inference within 50 milliseconds per frame on a system with a CUDA-capable GPU.
7. THE Inference_Pipeline SHALL output four values per frame: steering_degrees, gas_pedal, brake_pedal, and gear.
8. WHEN the Active_Model outputs a gear value, THE Inference_Pipeline SHALL clamp it to the nearest integer in [0, 5].
9. WHEN the Active_Model outputs a gas_pedal or brake_pedal value, THE Inference_Pipeline SHALL clamp it to [0.0, 1.0].
10. WHEN the Active_Model outputs a steering_degrees value, THE Inference_Pipeline SHALL clamp it to [-122.0, 115.0].

---

### Requirement 4: Vehicle Physics

**User Story:** As a researcher, I want the simulated vehicle to respond to model outputs with realistic motion, so that I can observe how the model's decisions translate to driving behavior.

#### Acceptance Criteria

1. EACH simulation frame, THE Vehicle SHALL update its heading by mapping the steering_degrees output linearly to a wheel turn rate proportional to current speed.
2. EACH simulation frame, THE Vehicle SHALL update its speed by applying gas_pedal as acceleration and brake_pedal as deceleration, subject to a maximum speed of 120 km/h.
3. THE Vehicle SHALL maintain a minimum speed of 0 km/h (no reverse).
4. THE Vehicle SHALL display the current gear value from the model output on the HUD.
5. WHEN the simulation is paused, THE Vehicle SHALL hold its current position, heading, and speed without updating.

---

### Requirement 5: HUD Dashboard

**User Story:** As a researcher, I want a real-time HUD overlay showing all model outputs and vehicle state, so that I can monitor the model's decisions at a glance during simulation.

#### Acceptance Criteria

1. THE HUD SHALL display a steering wheel arc indicator showing the current steering_degrees value mapped to a visual arc within the [-122, +115] degree range.
2. THE HUD SHALL display a green vertical bar representing gas_pedal value (0.0 to 1.0, bottom to top).
3. THE HUD SHALL display a red vertical bar representing brake_pedal value (0.0 to 1.0, bottom to top).
4. THE HUD SHALL display the current gear as a numeric label (0–5).
5. THE HUD SHALL display the current vehicle speed in km/h.
6. THE HUD SHALL display the current simulation frame counter.
7. THE HUD SHALL display the haze status label ("HAZY" or "CLEAR").
8. THE HUD SHALL display the active model type label ("CNN" or "LTC").
9. THE HUD SHALL update all displayed values every simulation frame.
10. WHEN the simulation is paused, THE HUD SHALL display a "PAUSED" indicator.

---

### Requirement 6: XAI Explainability Panel

**User Story:** As a researcher, I want to see DeepLIFT attribution heatmaps alongside the simulation, so that I can understand which parts of the road scene the model is focusing on when making decisions.

#### Acceptance Criteria

1. THE XAI_Panel SHALL display the original camera frame, the DeepLIFT attribution heatmap using the "hot" colormap, and an overlay of the heatmap on the original frame.
2. THE XAI_Panel SHALL compute attributions using Captum's DeepLIFT for the currently selected XAI_Output_Index.
3. THE XAI_Panel SHALL update its displayed attribution every N frames, where N is configurable in config.py with a default value of 5.
4. WHEN the user presses X, THE XAI_Panel SHALL cycle the XAI_Output_Index through [0, 1, 2, 3] and display the output name (steering, gas, brake, gear).
5. THE XAI_Panel SHALL display a legend indicating positive and negative attribution regions.
6. THE XAI_Panel SHALL display the haze status label ("HAZE" or "NON-HAZE") on the attribution visualization.
7. THE XAI_Panel SHALL run DeepLIFT computation in a background thread to avoid blocking the main simulation loop.
8. WHEN a background XAI computation is in progress, THE XAI_Panel SHALL continue displaying the most recently completed attribution result.

---

### Requirement 7: Model Switching

**User Story:** As a researcher, I want to switch between the CNN and LTC models at runtime, so that I can compare their driving behavior and attributions side by side in the same environment.

#### Acceptance Criteria

1. WHEN the user presses 1, THE Simulator SHALL set the Active_Model to CNN_Model.
2. WHEN the user presses 2, THE Simulator SHALL set the Active_Model to LTC_Model.
3. WHEN the Active_Model is switched, THE HUD SHALL update the model type label on the next frame.
4. WHEN the Active_Model is switched, THE XAI_Panel SHALL recompute attributions using the new Active_Model on the next XAI update cycle.
5. WHEN the Active_Model is switched to a model whose weights failed to load, THE Simulator SHALL display an error message on the HUD and retain the previous Active_Model.

---

### Requirement 8: Simulation Controls

**User Story:** As a researcher, I want keyboard controls to manage the simulation state, so that I can pause, inspect, and interact with the simulation during a run.

#### Acceptance Criteria

1. WHEN the user presses Space, THE Simulator SHALL toggle the simulation between paused and running states.
2. WHEN the user presses ESC, THE Simulator SHALL cleanly shut down all threads and exit.
3. WHERE manual override is enabled in config.py, WHEN the user presses arrow keys or WASD, THE Vehicle SHALL respond to manual steering and throttle inputs instead of model outputs.
4. WHEN manual override is active, THE HUD SHALL display a "MANUAL" indicator replacing the model type label.
5. WHEN the user presses H, THE Simulator SHALL toggle haze mode as specified in Requirement 2.3.
6. WHEN the user presses X, THE XAI_Panel SHALL cycle the XAI_Output_Index as specified in Requirement 6.4.

---

### Requirement 9: Configuration

**User Story:** As a researcher, I want all key parameters in a single config file, so that I can adjust model paths, performance settings, and simulation parameters without modifying source code.

#### Acceptance Criteria

1. THE Simulator SHALL read CNN_Model weight path, LTC_Model weight path, XAI update interval, target FPS, haze intensity, manual override flag, and window resolution from config.py at startup.
2. WHEN a configuration value is missing from config.py, THE Simulator SHALL use a documented default value and log a warning.
3. THE config.py file SHALL document each configuration parameter with an inline comment.

---

### Requirement 10: Performance

**User Story:** As a researcher, I want the simulator to run smoothly at 30 FPS, so that the simulation is usable for real-time evaluation without lag.

#### Acceptance Criteria

1. THE Simulator SHALL maintain a minimum of 30 FPS during normal operation with model inference and HUD rendering active.
2. THE XAI_Panel SHALL run DeepLIFT computation in a background thread so that XAI updates do not reduce the main loop below 30 FPS.
3. WHEN the system does not have a CUDA-capable GPU, THE Inference_Pipeline SHALL fall back to CPU inference and log a warning that performance may be below the 30 FPS target.
4. THE Simulator SHALL log the average inference time per frame to the console every 100 frames.
