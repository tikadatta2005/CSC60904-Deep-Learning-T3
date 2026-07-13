import os
import torch
import pandas as pd


def tester(model, folder_path):
    """
    Evaluate every checkpoint (.pt file) in folder_path.

    Parameters
    ----------
    model : torch.nn.Module
        Model architecture used during training.

    folder_path : str
        Folder containing saved checkpoints.

    Returns
    -------
    pandas.DataFrame
        Metrics for each checkpoint.
    """
    pass