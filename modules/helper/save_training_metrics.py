import pandas as pd


def extract_confusion_values(confusion_matrix):
    """
    Aggregate multiclass TP, FP, TN, FN
    using one-vs-rest calculation.
    """

    total = confusion_matrix.values.sum()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for cls in confusion_matrix.index:

        tp = confusion_matrix.loc[cls, cls]

        fp = confusion_matrix[cls].sum() - tp

        fn = confusion_matrix.loc[cls].sum() - tp

        tn = total - (tp + fp + fn)

        TP += tp
        FP += fp
        TN += tn
        FN += fn

    return int(TP), int(FP), int(TN), int(FN)



def save_training_metrics(train_test_data, save_location):

    records = []

    for epoch, data in enumerate(train_test_data, start=1):

        train = data["train"]
        test = data["test"]


        train_tp, train_fp, train_tn, train_fn = extract_confusion_values(
            train["Confusion Matrix"]
        )

        test_tp, test_fp, test_tn, test_fn = extract_confusion_values(
            test["Confusion Matrix"]
        )


        records.append({

            "epoch": epoch,

            "train_loss": train["Loss"],
            "test_loss": test["Loss"],

            "train_accuracy": train["Accuracy"],
            "test_accuracy": test["Accuracy"],

            "train_precision": train["Macro Precision"],
            "test_precision": test["Macro Precision"],

            "train_recall": train["Macro Recall"],
            "test_recall": test["Macro Recall"],

            "train_f1": train["Macro F1"],
            "test_f1": test["Macro F1"],

            "train_tp": train_tp,
            "test_tp": test_tp,

            "train_fp": train_fp,
            "test_fp": test_fp,

            "train_tn": train_tn,
            "test_tn": test_tn,

            "train_fn": train_fn,
            "test_fn": test_fn
        })


    df = pd.DataFrame(records)

    df.to_csv(
        save_location,
        index=False
    )

    return df