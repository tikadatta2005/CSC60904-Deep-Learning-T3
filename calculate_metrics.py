import pandas as pd
import numpy as np


def calculate_metrics(actual_value, predicted_value):
    """
    Calculates Accuracy, Precision, Recall, F1-Score,
    and Confusion Matrix for any number of classes.

    Parameters:
        actual_value: list/array of ground truth labels
        predicted_value: list/array of predicted labels

    Returns:
        dict containing:
        - Accuracy
        - Macro Precision
        - Macro Recall
        - Macro F1
        - Weighted Precision
        - Weighted Recall
        - Weighted F1
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

    # Dynamic confusion matrix
    confusion_matrix = pd.DataFrame(
        0,
        index=classes,
        columns=classes
    )

    for actual, predicted in zip(
        df["actual"],
        df["predicted"]
    ):
        confusion_matrix.loc[actual, predicted] += 1


    total = len(df)

    accuracy = (
        sum(df["actual"] == df["predicted"]) / total
        if total > 0 else 0
    )


    per_class = {}

    precisions = []
    recalls = []
    f1_scores = []
    supports = []


    for cls in classes:

        TP = confusion_matrix.loc[cls, cls]

        FP = confusion_matrix[cls].sum() - TP

        FN = confusion_matrix.loc[cls].sum() - TP

        support = confusion_matrix.loc[cls].sum()


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
            "F1-Score": round(f1, 4),
            "Support": int(support)
        }


        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)



    # Macro average
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)


    # Weighted average
    weighted_precision = np.average(
        precisions,
        weights=supports
    )

    weighted_recall = np.average(
        recalls,
        weights=supports
    )

    weighted_f1 = np.average(
        f1_scores,
        weights=supports
    )


    return {
        "Accuracy": round(accuracy, 4),

        "Macro Precision": round(macro_precision, 4),
        "Macro Recall": round(macro_recall, 4),
        "Macro F1": round(macro_f1, 4),

        "Weighted Precision": round(weighted_precision, 4),
        "Weighted Recall": round(weighted_recall, 4),
        "Weighted F1": round(weighted_f1, 4),

        "Per Class Metrics": per_class,

        "Confusion Matrix": confusion_matrix
    }