"""
CNN model matching the notebook's saved architecture.
Keys: backbone.* (EfficientNetV2-S via timm) + regressor.0/3/6 (Linear layers)
"""
import torch
import torch.nn as nn
import timm


class CNNModel(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # backbone keys match timm EfficientNetV2-S with num_classes=0
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0
        )
        # regressor indices 0, 3, 6 → Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),   # index 0
            nn.ReLU(),              # index 1
            nn.Dropout(0.3),        # index 2
            nn.Linear(512, 256),    # index 3
            nn.ReLU(),              # index 4
            nn.Dropout(0.2),        # index 5
            nn.Linear(256, 4),      # index 6
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 4)"""
        features = self.backbone(x)      # (B, 1280)
        return self.regressor(features)  # (B, 4)
