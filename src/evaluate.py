import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f"  {model_name} - Degerlendirme Raporu")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"\n  Siniflandirma Raporu:\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    return metrics


def compare_models(results: list[dict]) -> dict:
    best = max(results, key=lambda r: r["f1_score"])
    print(f"\n>>> Kazanan Model: {best['model']} (F1: {best['f1_score']:.4f}) <<<\n")
    return best


if __name__ == "__main__":
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)

    with open("models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    X = df[feature_columns]
    y = df["Churn"]

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    evaluate_model(model, X_test, y_test, model_name="Kayitli Model")