import torch.nn as nn
import torchvision.models as models


class VisionModel(nn.Module):
    def __init__(self, output_shape: int, freeze_base: bool = False):
        super().__init__()
        self.base = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        nfc = self.base.fc.in_features
        self.base.fc = nn.Linear(nfc, output_shape)

    def forward(self, x):
        return self.base(x)
