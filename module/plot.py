import matplotlib.pyplot as plt


def plot_curves(history_dict: dict):
    """
    Plots training and testing loss and accuracy curves side-by-side.
    """
    epochs = range(len(history_dict["train_loss"]))

    plt.figure(figsize=(14, 5))

    # Left subplot: Loss
    plt.subplot(1, 2, 1)
    plt.plot(
        epochs,
        history_dict["train_loss"],
        label="Train Loss",
        color="royalblue",
        linewidth=2,
    )
    plt.plot(
        epochs,
        history_dict["test_loss"],
        label="Test Loss",
        color="darkorange",
        linewidth=2,
    )
    plt.title("Loss Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    # Right subplot: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        history_dict["train_acc"],
        label="Train Acc",
        color="royalblue",
        linewidth=2,
    )
    plt.plot(
        epochs,
        history_dict["test_acc"],
        label="Test Acc",
        color="darkorange",
        linewidth=2,
    )
    plt.title("Accuracy Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()
