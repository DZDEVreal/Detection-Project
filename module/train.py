import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import Metric
from torch.utils.data import DataLoader
from engine import train_step, test_step
from tqdm.auto import tqdm


def train(
    epochs: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    acc_metric: Metric,
    is_compiled: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_per_epoch: int = 1,
):
    if is_compiled:
        model = torch.compile(model)
        print("Model has been Compiled!")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            acc_metric=acc_metric,
            train_dataloader=train_dataloader,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            loss_fn=loss_fn,
            acc_metric=acc_metric,
            test_dataloader=test_dataloader,
            device=device,
        )
        if epoch % print_per_epoch == 0:
            print(
                f"\nEpoch: {epoch} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f} | Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}\n"
            )
