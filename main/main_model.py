import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch
torch.set_float32_matmul_precision("high")

import torch.nn as nn
from module.model import VisionModel
from module.data import BrainTumorDataset, train_transform, test_transform
from module.export import export_to_onnx
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from module.train import train
from module.plot import plot_curves, plot_predictions


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dir = "dataset/Training"
    test_dir = "dataset/Testing"

    print("Loading datasets...")
    train_data = BrainTumorDataset(image_dir=train_dir, transform=train_transform)
    test_data = BrainTumorDataset(image_dir=test_dir, transform=test_transform)

    NUM_WORKERS = min(4, os.cpu_count())

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_data.classes
    print(f"Classes found: {class_names}")
    print(f"Train size: {len(train_data)} images | Test size: {len(test_data)} images\n")

    model = VisionModel(output_shape=len(class_names), freeze_base=False).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=train_data.class_weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6
    )

    acc_metric = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

    print("Starting training...")
    history = train(
        epochs=30,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        acc_metric=acc_metric,
        scheduler=scheduler,
        is_compiled=True,
        device=device,
        print_per_epoch=1,
        patience=7,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
    )

    plot_curves(history, save_dir="outputs")
    plot_predictions(model, test_dataloader, class_names, device, save_dir="outputs")

    torch.save(model.state_dict(), "tumor_classifier.pth")
    print("\nPyTorch weights saved to 'tumor_classifier.pth'")

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path="tumor_classifier.onnx"
    )
