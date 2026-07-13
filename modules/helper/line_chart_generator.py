import os
import matplotlib.pyplot as plt


def line_chart_generator(data, folder_path):
    """
    Generates line charts from training/testing metrics dataframe.

    Parameters:
        data:
            pandas DataFrame containing epoch-wise metrics

        folder_path:
            Folder where charts will be saved
    """

    os.makedirs(folder_path, exist_ok=True)


    def save_chart(filename, title, columns):

        plt.figure(figsize=(10, 6))

        for column in columns:
            plt.plot(
                data["epoch"],
                data[column],
                label=column
            )

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(folder_path, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()



    # 1. Loss
    save_chart(
        "loss_train_vs_test.png",
        "Training vs Testing Loss",
        [
            "train_loss",
            "test_loss"
        ]
    )


    # 2. Accuracy
    save_chart(
        "accuracy_train_vs_test.png",
        "Training vs Testing Accuracy",
        [
            "train_accuracy",
            "test_accuracy"
        ]
    )


    # 3. Precision
    save_chart(
        "precision_train_vs_test.png",
        "Training vs Testing Precision",
        [
            "train_precision",
            "test_precision"
        ]
    )


    # 4. Recall
    save_chart(
        "recall_train_vs_test.png",
        "Training vs Testing Recall",
        [
            "train_recall",
            "test_recall"
        ]
    )


    # 5. F1
    save_chart(
        "f1_train_vs_test.png",
        "Training vs Testing F1 Score",
        [
            "train_f1",
            "test_f1"
        ]
    )


    # 6. Train TP TN FP FN
    save_chart(
        "train_confusion_metrics.png",
        "Training TP TN FP FN",
        [
            "train_tp",
            "train_tn",
            "train_fp",
            "train_fn"
        ]
    )


    # 7. Test TP TN FP FN
    save_chart(
        "test_confusion_metrics.png",
        "Testing TP TN FP FN",
        [
            "test_tp",
            "test_tn",
            "test_fp",
            "test_fn"
        ]
    )


    # 8. TP Train vs Test
    save_chart(
        "tp_train_vs_test.png",
        "TP Training vs Testing",
        [
            "train_tp",
            "test_tp"
        ]
    )


    # 9. TN Train vs Test
    save_chart(
        "tn_train_vs_test.png",
        "TN Training vs Testing",
        [
            "train_tn",
            "test_tn"
        ]
    )


    # 10. FP Train vs Test
    save_chart(
        "fp_train_vs_test.png",
        "FP Training vs Testing",
        [
            "train_fp",
            "test_fp"
        ]
    )


    # 11. FN Train vs Test
    save_chart(
        "fn_train_vs_test.png",
        "FN Training vs Testing",
        [
            "train_fn",
            "test_fn"
        ]
    )


    # 12. All metric comparison
    save_chart(
        "all_performance_metrics_train_vs_test.png",
        "All Performance Metrics Train vs Test",
        [
            "train_accuracy",
            "test_accuracy",
            "train_precision",
            "test_precision",
            "train_recall",
            "test_recall",
            "train_f1",
            "test_f1"
        ]
    )


    # 13. Loss + Accuracy comparison
    save_chart(
        "loss_accuracy_train_vs_test.png",
        "Loss and Accuracy Train vs Test",
        [
            "train_loss",
            "test_loss",
            "train_accuracy",
            "test_accuracy"
        ]
    )


    print(f"Generated 13 charts in: {folder_path}")