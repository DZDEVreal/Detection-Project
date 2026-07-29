import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Tuple


def mixup_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    H, W = x.size(2), x.size(3)
    cut_w = int(W * (1 - lam) ** 0.5)
    cut_h = int(H * (1 - lam) ** 0.5)
    cx = torch.randint(0, W, (1,), device=x.device)
    cy = torch.randint(0, H, (1,), device=x.device)
    x1 = int(torch.clamp(cx - cut_w // 2, 0, W).item())
    x2 = int(torch.clamp(cx + cut_w // 2, 0, W).item())
    y1 = int(torch.clamp(cy - cut_h // 2, 0, H).item())
    y2 = int(torch.clamp(cy + cut_h // 2, 0, H).item())
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)
    return mixed_x, y, y[idx], lam


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    acc_metric: Metric,
    train_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> Tuple[float, float]:

    use_mixup = mixup_alpha > 0 or cutmix_alpha > 0
    if not use_mixup:
        acc_metric.reset()
    train_loss, train_acc = 0.0, 0.0
    model.train()

    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        if use_mixup:
            if cutmix_alpha > 0 and (mixup_alpha == 0 or torch.rand(1).item() < 0.5):
                x, y_a, y_b, lam = cutmix_data(x, y, cutmix_alpha)
            else:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

        y_pred = model(x)

        if use_mixup:
            loss = lam * loss_fn(y_pred, y_a) + (1 - lam) * loss_fn(y_pred, y_b)
        else:
            loss = loss_fn(y_pred, y)
            acc_metric.update(y_pred, y)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc = acc_metric.compute().item() if not use_mixup else 0.0

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
