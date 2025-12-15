from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NLUEvaluator:

    def evaluate(self, df):
        y_true = df["true_intent"]
        y_pred = df["predicted_intent"]

        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
