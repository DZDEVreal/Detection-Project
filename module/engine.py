import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Tuple


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    acc_metric: Metric,
    train_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float]:

    acc_metric.reset()
    train_loss, train_acc = 0.0, 0.0
    model.train()

    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item()
        acc_metric.update(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc = acc_metric.compute().item()

    return train_loss, train_acc


def test_step(
    model: nn.Module,
    loss_fn: nn.Module,
    acc_metric: Metric,
    test_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float]:

    acc_metric.reset()
    test_loss, test_acc = 0.0, 0.0
    model.eval()

    with torch.inference_mode():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item()
            acc_metric.update(y_pred, y)

    test_loss /= len(test_dataloader)
    test_acc = acc_metric.compute().item()

    return test_loss, test_acc
