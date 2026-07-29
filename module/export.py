import torch
from torch import nn
from torch import Tensor

def export_to_onnx(
    model: nn.Module,
    dummy_input: Tensor,
    output_path: str = "tumor_model.onnx"
):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            f=output_path,
            opset_version=18,
            input_names=["image"],
            output_names=["class_logits"],
            dynamic_axes={"image": {0: "batch_size"}}
        )
    print(f"Model exported to {output_path}")
