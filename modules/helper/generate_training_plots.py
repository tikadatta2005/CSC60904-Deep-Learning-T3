import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_report_training_plot(input_csv, output_path):

    # Read CSV
    data = pd.read_csv(input_csv)

    # Create parent directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Figure
    fig = plt.figure(figsize=(8, 9), facecolor="white")

    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 1],
        hspace=0.45,
        wspace=0.3
    )


    def plot_metric(position, metric, title):

        ax = fig.add_subplot(position)

        for col in [f"train_{metric}", f"val_{metric}"]:
            if col in data.columns:
                sns.lineplot(
                    data=data,
                    x="epoch",
                    y=col,
                    linewidth=1.5,
                    ax=ax,
                    label=col.replace("_", " ").title()
                )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(title, fontsize=12)

        ax.tick_params(
            axis="both",
            labelsize=9
        )

        ax.grid(alpha=0.3)

        ax.legend(
            fontsize=8
        )


    # Layout
    plot_metric(grid[0, :], "loss", "Loss")

    plot_metric(grid[1, 0], "accuracy", "Accuracy")
    plot_metric(grid[1, 1], "f1_score", "F1 Score")

    plot_metric(grid[2, 0], "precision", "Precision")
    plot_metric(grid[2, 1], "recall", "Recall")


    plt.savefig(
        output_path,
        dpi=100,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {output_path}")