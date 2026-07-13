import os
import torch
import pandas as pd


import os


def tester(checkpoint_dir, test_dir, batch_size=32):
    """
    Evaluate every checkpoint inside checkpoint_dir
    using images from test_dir.

    Parameters
    ----------
    checkpoint_dir : str
        Folder containing model checkpoints.

    test_dir : str
        Folder containing test images.

    batch_size : int, default=32
        Batch size for DataLoader.

    Returns
    -------
    pandas.DataFrame
        Evaluation results for every checkpoint.
    """

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )
    # --------------------------
    # Find checkpoint files
    # --------------------------

    checkpoint_files = []

    for file in os.listdir(checkpoint_dir):

        if file.endswith(".pt") or file.endswith(".pth"):
            checkpoint_files.append(file)

    checkpoint_files.sort()

    if len(checkpoint_files) == 0:
        raise FileNotFoundError(
            "No checkpoint files found."
        )