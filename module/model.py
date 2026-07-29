import torch.nn as nn
import torchvision.models as models

class VisionModel(nn.Module):
    def __init__(self, output_shape: int, freeze_base: bool = False):
        super().__init__()
        self.base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        nfc = self.base.heads[0].in_features

        if freeze_base:
            self.base.heads = nn.Sequential(
                nn.Linear(nfc, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, output_shape)
            )
        else:
            self.base.heads = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(nfc, output_shape)
            )

    def forward(self, x):
        return self.base(x)
