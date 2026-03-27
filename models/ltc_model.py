"""
LTC model matching the notebook's saved architecture.
Keys:
  cnn.*            - MobileNetV2 backbone (torchvision), outputs 1280-dim features
  feature_adapter.* - Sequential: Linear(1280->256) at idx 0, Linear(256->128) at idx 3
  liquid_layer.*   - LTC cell from ncps with 128 inputs, 128 hidden, 4 outputs
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

try:
    from ncps.torch import LTC
    from ncps.wirings import NCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


class LTCModel(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if not NCPS_AVAILABLE:
            raise ImportError("ncps library required. Install: pip install ncps")

        # MobileNetV2 — saved as cnn.features.* so we assign the full model to self.cnn
        # The notebook did: self.cnn = mobilenet_v2(...) then used self.cnn.features in forward
        mv2 = mobilenet_v2(weights=None)
        self.cnn = mv2   # keys will be cnn.features.*, cnn.classifier.*

        # feature_adapter: indices 0 and 3 are Linear layers
        self.feature_adapter = nn.Sequential(
            nn.Linear(1280, 256),   # index 0
            nn.ReLU(),              # index 1
            nn.Dropout(0.3),        # index 2
            nn.Linear(256, 128),    # index 3
        )

        # FullyConnected(128, 4): 128 neurons, 4 motor outputs, 128 sensory inputs
        # Matches saved weight shapes: [128,128] for w/erev, [4] for output_w/b
        from ncps.wirings import FullyConnected
        wiring = FullyConnected(128, 4)
        self.liquid_layer = LTC(128, wiring, batch_first=True)

        self._hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 4)"""
        # CNN feature extraction — use features only, global avg pool
        feat = self.cnn.features(x)             # (B, 1280, 7, 7)
        feat = feat.mean(dim=[2, 3])                # global avg pool -> (B, 1280)
        feat = self.feature_adapter(feat)           # (B, 128)
        feat = feat.unsqueeze(1)                    # (B, 1, 128) seq_len=1

        out, self._hidden = self.liquid_layer(feat, self._hidden)
        return out.squeeze(1)                       # (B, 4)

    def reset_hidden(self) -> None:
        self._hidden = None
