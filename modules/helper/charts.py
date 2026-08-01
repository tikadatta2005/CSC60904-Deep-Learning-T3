import os
import math

import matplotlib.pyplot as plt
import seaborn as sns


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

    # Create subplot
    plt.subplot(rows, columns, position)

    # Plot each metric
    for y_name in y:
        if y_name in data.columns:
            sns.lineplot(
                data=data,
                x=x,
                y=y_name,
                label=y_name
            )
        else:
            print(f"Warning: Column '{y_name}' was not found. Skipping.")

    # Add labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add legend only when lines exist
    if len(plt.gca().lines) > 0:
        plt.legend()


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

    # Create figure
    plt.figure(
        figsize=(16, 10),
        facecolor="white"
    )

    # Plot each metric
    for y_name in y:
        if y_name in data.columns:
            sns.lineplot(
                data=data,
                x=x,
                y=y_name,
                label=y_name
            )
        else:
            print(f"Warning: Column '{y_name}' was not found. Skipping.")

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add legend only when lines exist
    if len(plt.gca().lines) > 0:
        plt.legend()

    # Generate filename
    filename = title.replace(" ", "_").lower() + ".png"

    # Save figure
    plt.savefig(
        os.path.join(save_path, filename),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    # Close figure
    plt.close()


def gen_line_charts(
    data,
    save_path,
    plot_name,
    prefix=None
):


    # Prevent mutable default argument
    if prefix is None:
        prefix = []

    # Create save directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # Metric names
    all_y_names = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    # Create metric groups
    paired = []

    for y_name in all_y_names:

        # No prefix
        if len(prefix) == 0:
            paired.append([y_name])

        # Add prefixes
        else:
            pair = [
                f"{pref}{y_name}"
                for pref in prefix
            ]

            paired.append(pair)

    # Number of columns in subplot grid
    columns = 3

    # Calculate rows using the SAME column count
    rows = math.ceil(
        len(paired) / columns
    )

    # Create combined figure
    plt.figure(
        figsize=(24, 10),
        facecolor="white"
    )

    # Create all subplots
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

    # Improve spacing
    plt.tight_layout()

    # Save combined figure
    plt.savefig(
        os.path.join(
            save_path,
            plot_name
        ),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    # Display combined figure
    plt.show()

    # Close combined figure
    plt.close()

    # Save individual metric figures
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

    # Clear all figures
    plt.close("all")