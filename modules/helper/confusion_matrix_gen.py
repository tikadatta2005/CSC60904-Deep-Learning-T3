# modules/helper/plot_confusion_matrix.py

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_pred,
    y_true,
    save_path,
    class_names=None,
    figsize=(7, 6),
    cmap="Blues"
):


    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap=cmap,
        values_format="d",
        colorbar=True
    )

    ax.set_title("Confusion Matrix")

    # Prevent label overlap
    plt.tight_layout()

    # Create parent directories if they do not exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # Show figure
    plt.show()

    # Close figure to free memory
    plt.close(fig)

    return cm