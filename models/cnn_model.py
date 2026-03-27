"""
EfficientNetV2-S backbone with fully-connected head.
Outputs: [steering_degrees, gas_pedal, brake_pedal, gear]
XAI: compatible with captum DeepLIFT (no in-place ops in head).
"""
import torch
import torch.nn as nn
import timm


class CNNModel(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # EfficientNetV2-S feature extractor (1280-dim output)
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 4)"""
        features = self.backbone(x)   # (B, 1280)
        return self.head(features)    # (B, 4)
