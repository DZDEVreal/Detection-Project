import torch.nn as nn
import torchvision.models as models

class VisionModel(nn.Module):
    def __init__(self, output_shape: int, freeze_base: bool = False):
        super().__init__()
        self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        nfc = self.base.classifier[1].in_features

        if freeze_base:
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(nfc, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, output_shape)
            )
        else:
            for i in range(6):
                for param in self.base.features[i].parameters():
                    param.requires_grad = False
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(nfc, output_shape)
            )

    def forward(self, x):
        return self.base(x)
