import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class NLUEvaluator:
    """
    Evaluates intent classification performance.
    Works for both CLI pipeline and Streamlit UI.
    """

    def evaluate(self, df):
        """
        Console-based evaluation (used in main_pipeline.py)
        """
        y_true = df["intent"]
        y_pred = df["predicted_intent"]

        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    def evaluate_with_results(self, df):
        """
        UI-friendly evaluation (used in Streamlit)
        Returns metrics dictionary + confusion matrix DataFrame
        """
        y_true = df["intent"]
        y_pred = df["predicted_intent"]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        labels = sorted(y_true.unique())

        cm = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=labels,
            columns=labels
        )

        return metrics, cm
