import pandas as pd
import numpy as np

def calculate_metrics(actual_value, predicted_value):
    """
    Calculates Accuracy, Precision, Recall,
    F1-Score and Confusion Matrix.

    Returns:
        - Accuracy
        - Precision (Macro)
        - Recall (Macro)
        - F1-Score (Macro)
        - Per Class Metrics
        - Confusion Matrix
    """

    df = pd.DataFrame({
        "actual": actual_value,
        "predicted": predicted_value
    })

    classes = sorted(
        set(df["actual"]).union(set(df["predicted"]))
    )

    # Confusion Matrix
    confusion_matrix = pd.DataFrame(
        0,
        index=classes,
        columns=classes
    )

    for actual, predicted in zip(df["actual"], df["predicted"]):
        confusion_matrix.loc[actual, predicted] += 1

    total = len(df)

    accuracy = (
        (df["actual"] == df["predicted"]).sum() / total
        if total > 0 else 0
    )

    per_class = {}

    precisions = []
    recalls = []
    f1_scores = []

    for cls in classes:

        TP = confusion_matrix.loc[cls, cls]
        FP = confusion_matrix[cls].sum() - TP
        FN = confusion_matrix.loc[cls].sum() - TP

        precision = (
            TP / (TP + FP)
            if (TP + FP) > 0 else 0
        )

        recall = (
            TP / (TP + FN)
            if (TP + FN) > 0 else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0 else 0
        )

        per_class[cls] = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        }

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(np.mean(precisions), 4),
        "Recall": round(np.mean(recalls), 4),
        "F1-Score": round(np.mean(f1_scores), 4),
        "Per Class Metrics": per_class,
        "Confusion Matrix": confusion_matrix
    }