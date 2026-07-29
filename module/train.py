from torchmetrics import Metric
from torch.utils.data import DataLoader
from module.engine import train_step, test_step
from tqdm.auto import tqdm
import torch


def train(
    epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    acc_metric: Metric,
    scheduler: object = None,
    is_compiled: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_per_epoch: int = 1,
    patience: int = 5,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
):
    if is_compiled:
        model = torch.compile(model)

    best_test_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    history_dict = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            acc_metric=acc_metric,
            train_dataloader=train_dataloader,
            device=device,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )
        test_loss, test_acc = test_step(
            model=model,
            loss_fn=loss_fn,
            acc_metric=acc_metric,
            test_dataloader=test_dataloader,
            device=device,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        history_dict["train_loss"].append(train_loss)
        history_dict["train_acc"].append(train_acc * 100)
        history_dict["test_loss"].append(test_loss)
        history_dict["test_acc"].append(test_acc * 100)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % print_per_epoch == 0:
            print(
                f"\nEpoch: {epoch} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f} | Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}\n"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history_dict
