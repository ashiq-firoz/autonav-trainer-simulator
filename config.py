"""
Simulator configuration. All parameters have documented defaults.
Override by editing this file — no source code changes needed.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Model weights ---
    cnn_model_path: str = "weights/cnn_model.pth"   # Path to EfficientNetV2 CNN weights
    ltc_model_path: str = "assets/weights/best_ltc.pth" #"weights/ltc_model.pth" #   # Path to LTC/CfC liquid network weights

    # --- XAI ---
    xai_update_interval: int = 5        # Frames between DeepLIFT attribution updates
    xai_output_index: int = 0           # Default output to explain: 0=steering,1=gas,2=brake,3=gear

    # --- Rendering ---
    target_fps: int = 30                # Target simulation frames per second
    window_width: int = 1280            # Window width in pixels
    window_height: int = 720            # Window height in pixels

    # --- Haze ---
    haze_intensity: float = 0.6         # Haze blend intensity (matches training augmentation)
    haze_active: bool = False           # Start with haze disabled

    # --- Vehicle ---
    max_speed_kmh: float = 120.0        # Maximum vehicle speed in km/h
    accel_rate: float = 50.0            # Acceleration rate (km/h per second at full gas)
    decel_rate: float = 30.0            # Deceleration rate (km/h per second at full brake)
    turn_rate: float = 60.0             # Max heading change rate in degrees/second at full steer

    # --- Road / Scene ---
    road_tile_length: float = 50.0      # Length of each road tile in meters
    road_tiles_ahead: int = 10          # Number of tiles to keep ahead of vehicle
    road_width: float = 8.0             # Road width in meters

    # --- Controls ---
    manual_override: bool = False       # Enable WASD/arrow key manual driving

    # --- Logging ---
    log_inference_every: int = 100      # Log average inference time every N frames

    # --- Active model ---
    active_model: str = "cnn"           # Starting model: "cnn" or "ltc"
