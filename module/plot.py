import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

denormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

def plot_curves(history_dict: dict, save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(len(history_dict["train_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history_dict["train_loss"], label="Train", color="#4C72B0", linewidth=2)
    ax1.plot(epochs, history_dict["test_loss"], label="Test", color="#DD8452", linewidth=2)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history_dict["train_acc"], label="Train", color="#4C72B0", linewidth=2)
    ax2.plot(epochs, history_dict["test_acc"], label="Test", color="#DD8452", linewidth=2)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    best_test_loss = min(history_dict["test_loss"])
    best_test_acc = max(history_dict["test_acc"])
    ax1.axhline(best_test_loss, color="#DD8452", ls=":", alpha=0.5)
    ax2.axhline(best_test_acc, color="#DD8452", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    device: str,
    save_dir: str = "outputs",
    num_samples: int = 12
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    images, labels = next(iter(dataloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    with torch.inference_mode():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    probs = probs.cpu()

    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
    axes = axes.flatten()

    correct = 0
    for i in range(num_samples):
        img = denormalize(images[i]).permute(1, 2, 0).clamp(0, 1)
        axes[i].imshow(img)

        is_correct = labels[i] == preds[i]
        correct += int(is_correct)
        color = "#2E8B57" if is_correct else "#DC143C"
        title = f"{class_names[preds[i]]} ({probs[i][preds[i]]:.0%})\ntrue: {class_names[labels[i]]}"

        axes[i].set_title(title, fontsize=9, color=color)
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Accuracy: {correct}/{num_samples} ({correct / num_samples:.0%})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "predictions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
