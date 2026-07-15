import torchvision.models as models
import torch.nn as nn


class DetectionModel(nn.Module):
    def __init__(self, output_shape: int, freeze_base: bool = False):
        super().__init__()
        self.base = models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            num_classes=output_shape,
        )

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    @staticmethod
    def transforms():
        return models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()

    def forward(self, x):
        return self.base(x)
