import os
import math

import matplotlib.pyplot as plt
import seaborn as sns


# Global plotting style
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
})


def sub_plots(
    rows,
    columns,
    position,
    data,
    x,
    y,
    x_label,
    y_label,
    title
):
    """
    Create one subplot containing one or more line charts.
    """

    plt.subplot(rows, columns, position)

    for y_name in y:
        if y_name in data.columns:
            sns.lineplot(
                data=data,
                x=x,
                y=y_name,
                label=y_name,
                linewidth=3,
                marker="o",
                markersize=5
            )
        else:
            print(f"Warning: Column '{y_name}' was not found. Skipping.")

    plt.title(
        title,
        fontweight="bold",
        fontsize=18
    )

    plt.xlabel(
        x_label,
        fontweight="bold",
        fontsize=15
    )

    plt.ylabel(
        y_label,
        fontweight="bold",
        fontsize=15
    )

    plt.grid(
        True,
        alpha=0.3
    )

    if len(plt.gca().lines) > 0:
        plt.legend(
            frameon=True
        )


def save_plots(
    data,
    x,
    y,
    x_label,
    y_label,
    title,
    save_path
):
    """
    Save an individual line chart.
    """

    plt.figure(
        figsize=(12, 8),
        facecolor="white"
    )

    for y_name in y:
        if y_name in data.columns:
            sns.lineplot(
                data=data,
                x=x,
                y=y_name,
                label=y_name,
                linewidth=3,
                marker="o",
                markersize=6
            )
        else:
            print(f"Warning: Column '{y_name}' was not found. Skipping.")

    plt.title(
        title,
        fontsize=20,
        fontweight="bold"
    )

    plt.xlabel(
        x_label,
        fontsize=16,
        fontweight="bold"
    )

    plt.ylabel(
        y_label,
        fontsize=16,
        fontweight="bold"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    if len(plt.gca().lines) > 0:
        plt.legend()

    filename = title.replace(" ", "_").lower() + ".png"

    plt.savefig(
        os.path.join(save_path, filename),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()


def gen_line_charts(
    data,
    save_path,
    plot_name,
    prefix=None
):

    if prefix is None:
        prefix = []

    os.makedirs(save_path, exist_ok=True)

    all_y_names = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    paired = []

    for y_name in all_y_names:

        if len(prefix) == 0:
            paired.append([y_name])

        else:
            paired.append(
                [
                    f"{pref}{y_name}"
                    for pref in prefix
                ]
            )


    # Fixed 2 columns
    columns = 2

    # Dynamic rows
    rows = math.ceil(
        len(paired) / columns
    )


    # Larger figure for readability
    plt.figure(
        figsize=(
            18,
            rows * 6
        ),
        facecolor="white"
    )


    for index, pair in enumerate(paired):

        metric_name = (
            all_y_names[index]
            .replace("_", " ")
            .title()
        )

        sub_plots(
            rows=rows,
            columns=columns,
            position=index + 1,
            data=data,
            x="epoch",
            y=pair,
            title=f"{metric_name} Curves",
            x_label="Epochs",
            y_label=metric_name
        )


    plt.tight_layout(
        pad=3
    )


    plt.savefig(
        os.path.join(
            save_path,
            plot_name
        ),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )


    plt.show()
    plt.close()


    for index, pair in enumerate(paired):

        metric_name = (
            all_y_names[index]
            .replace("_", " ")
            .title()
        )

        save_plots(
            data=data,
            x="epoch",
            y=pair,
            x_label="Epochs",
            y_label=metric_name,
            title=f"{metric_name} Curves",
            save_path=save_path
        )


    plt.close("all")