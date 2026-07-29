import torch.nn as nn
import torchvision.models as models

class VisionModel(nn.Module):
    def __init__(self, output_shape: int, freeze_base: bool = False):
        super().__init__()
        self.base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        nfc = self.base.head.in_features

        if freeze_base:
            self.base.head = nn.Sequential(
                nn.Linear(nfc, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, output_shape)
            )
        else:
            for param in self.base.features[0].parameters():
                param.requires_grad = False
            for param in self.base.features[1].parameters():
                param.requires_grad = False
            for param in self.base.features[2].parameters():
                param.requires_grad = False
            self.base.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(nfc, output_shape)
            )

    def forward(self, x):
        return self.base(x)
