# AutoNav Trainer and Simulator

A Panda3D-based 3D simulator for testing CNN and LTC (Liquid Time-Constant) models for autonomous driving. The simulator features real-world textures, weather effects (haze), real-time HUD dashboard, and explainable AI (XAI) visualization using DeepLIFT.

## Features

- **Two Model Architectures**: CNN (EfficientNetV2-S) and LTC (MobileNetV2 + Liquid Network)
- **Real-World Textures**: High-quality textures from Poly Haven for realistic road environments
- **Haze Effects**: Weather simulation matching training augmentation
- **Real-Time HUD**: Displays steering angle, speed, gear, gas/brake status
- **XAI Visualization**: DeepLIFT attribution showing model focus areas
- **Manual Override**: WASD/Arrow key controls for testing

## Installation

```bash
git clone 
cd simulator

# Install dependencies using uv
uv sync
```

## Running the Simulator

```bash
uv run python main.py
```

## Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| H | Toggle haze effect |
| 1 | Switch to CNN model |
| 2 | Switch to LTC model |
| X | Cycle XAI output (steering/gas/brake/gear) |
| WASD / Arrows | Manual override (if enabled in config) |
| ESC | Quit |

## Configuration

Edit `config.py` to customize:

- **Model paths**: `cnn_model_path`, `ltc_model_path`
- **Rendering**: `target_fps`, `window_width`, `window_height`
- **Haze**: `haze_intensity`, `haze_active`
- **Vehicle physics**: `max_speed_kmh`, `accel_rate`, `decel_rate`, `turn_rate`
- **Road**: `road_tile_length`, `road_width`, `road_tiles_ahead`
- **Manual override**: Set `manual_override = True` to enable keyboard controls
- **Active model**: `active_model` - "cnn" or "ltc"

## Model Weights

Place your trained model weights in the `weights/` directory:

- `weights/cnn_model.pth` - EfficientNetV2 CNN model
- `weights/ltc_model.pth` - LTC liquid network model

Download weights using these links:

- `cnn_model.pth` - https://drive.google.com/file/d/1MVSSfLaWVyOjy7CHZZRy5ADgnYdG71ME/view?usp=sharing
- `ltc_model.pth` - https://drive.google.com/file/d/1y00Qip8d9FAtmaVst1wBae1TiY-yjrW9/view?usp=drive_link

or 
```bash
sh download_weights.sh
or
./download_weights.sh
```

The simulator expects models with the following architecture:

- **CNN**: EfficientNetV2-S backbone + FC regression head
- **LTC**: MobileNetV2 features + feature adapter + liquid layer

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_units.py
```

## Project Structure

```
simulator/
├── main.py              # Entry point
├── config.py            # Configuration
├── engine/
│   ├── scene.py         # 3D scene with textures
│   ├── vehicle.py       # Vehicle physics
│   └── haze.py          # Haze effects
├── models/
│   ├── cnn_model.py     # CNN model definition
│   ├── ltc_model.py     # LTC model definition
│   └── inference.py     # Inference pipeline
├── hud/
│   └── dashboard.py     # HUD renderer
├── xai/
│   ├── deeplift.py      # DeepLIFT worker
│   └── visualizer.py    # XAI panel
├── weights/             # Model weights
└── assets/textures/     # Texture files
```

## Requirements

- Python 3.11 - 3.12
- Panda3D 1.10+
- PyTorch 2.0 - 2.3
- OpenCV 4.8+
- Captum 0.6+ (for XAI)
- NCPS 0.0.7+ (for LTC models)