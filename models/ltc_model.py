"""
LTC/CfC Liquid Neural Network model.
EfficientNetV2-S visual encoder → Linear adapter → AutoNCP wiring.
Outputs: [steering_degrees, gas_pedal, brake_pedal, gear]
"""
import torch
import torch.nn as nn
import timm

try:
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


class LTCModel(nn.Module):
    def __init__(self, pretrained: bool = False, use_ltc: bool = False):
        super().__init__()
        if not NCPS_AVAILABLE:
            raise ImportError("ncps library required. Install: pip install ncps")

        # Shared visual encoder
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0
        )
        # Adapter: 1280 -> 128
        self.feature_adapter = nn.Linear(1280, 128)

        # Neural Circuit Policy wiring
        self.wiring = AutoNCP(128, 4)

        if use_ltc:
            self.liquid_layer = LTC(128, self.wiring, batch_first=True)
        else:
            self.liquid_layer = CfC(128, self.wiring, batch_first=True)

        self._hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 4)"""
        features = self.backbone(x)                    # (B, 1280)
        features = self.feature_adapter(features)      # (B, 128)
        features = features.unsqueeze(1)               # (B, 1, 128) — seq_len=1

        output, self._hidden = self.liquid_layer(features, self._hidden)
        return output.squeeze(1)                       # (B, 4)

    def reset_hidden(self) -> None:
        """Reset recurrent hidden state between episodes."""
        self._hidden = None
