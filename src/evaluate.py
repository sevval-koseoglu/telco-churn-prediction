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
    """
    Verilen modeli test verisi üzerinde değerlendirir ve metrikleri raporlar.

    Args:
        model: Eğitilmiş sklearn modeli.
        X_test: Test özellik matrisi.
        y_test: Gerçek etiketler.
        model_name: Raporda görünecek model adı.

    Returns:
        dict: Hesaplanan metrikler (accuracy, precision, recall, f1_score).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Değerlendirme Raporu")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Skoru  : {metrics['f1_score']:.4f}")
    print(f"\n  Sınıflandırma Raporu:\n{classification_report(y_test, y_pred, target_names=['Kayıp Yok', 'Churn'])}")
    print(f"  Karışıklık Matrisi:\n{confusion_matrix(y_test, y_pred)}\n")

    return metrics


def compare_models(results: list[dict]) -> dict:
    """
    Birden fazla modelin metriklerini karşılaştırır ve F1 Skoru'na göre kazananı seçer.

    Args:
        results: Her biri evaluate_model() çıktısı olan dict listesi.

    Returns:
        dict: En yüksek F1 Skoru'na sahip modelin metrikleri.
    """
    best = max(results, key=lambda r: r["f1_score"])
    print(f"\n>>> Kazanan Model: {best['model']} (F1: {best['f1_score']:.4f}) <<<\n")
    return best


if __name__ == "__main__":
    # Bu blok yalnızca doğrudan çalıştırıldığında devreye girer.
    # Kullanım: python src/evaluate.py
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

    evaluate_model(model, X_test, y_test, model_name="Kayıtlı Model")
