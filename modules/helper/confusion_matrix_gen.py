# modules/helper/plot_confusion_matrix.py

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Configure Devanagari font
def set_devanagari_font():
    preferred_fonts = [
        "Noto Sans Devanagari",
        "Nirmala UI",
        "Mangal"
    ]

    available_fonts = {
        font.name: font.fname
        for font in fm.fontManager.ttflist
    }

    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            return

    # fallback search
    for name in available_fonts.keys():
        if "devanagari" in name.lower() or "noto" in name.lower():
            plt.rcParams["font.family"] = name
            return

    print(
        "Warning: Devanagari font not found. "
        "Install Noto Sans Devanagari."
    )


set_devanagari_font()


# Plot confusion matrix
def plot_confusion_matrix(
    y_pred,
    y_true,
    save_path,
    class_names=None,
    figsize=(16, 14),
    cmap="Blues"
):

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)


    # Display confusion matrix
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


    # Title
    ax.set_title(
        "Confusion Matrix",
        fontsize=18,
        pad=20
    )


    # Label formatting
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=11
    )

    plt.yticks(
        rotation=0,
        fontsize=11
    )


    ax.tick_params(
        axis="both",
        which="major",
        labelsize=11
    )


    # Reduce confusion matrix number size
    if display.text_ is not None:
        for text in display.text_.ravel():
            text.set_fontsize(7)


    # Layout adjustment
    plt.tight_layout()


    # Create directory if missing
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save image
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close(fig)


    return cm