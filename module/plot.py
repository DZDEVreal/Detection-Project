import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

denormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

def plot_curves(history_dict: dict, save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(len(history_dict["train_loss"]))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    plt.plot(epochs, history_dict["test_loss"], label="Test Loss", color="darkorange", linewidth=2)
    plt.title("Loss Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["train_acc"], label="Train Acc", color="royalblue", linewidth=2)
    plt.plot(epochs, history_dict["test_acc"], label="Test Acc", color="darkorange", linewidth=2)
    plt.title("Accuracy Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()

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
        preds = logits.argmax(dim=1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    cols = 4
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(cols * 3.5, rows * 3.5))

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        img = images[i]
        img = denormalize(img)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.imshow(img)

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        correct = labels[i] == preds[i]
        color = "green" if correct else "red"

        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=150)
    plt.close()
