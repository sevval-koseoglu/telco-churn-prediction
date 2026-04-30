import pandas as pd


def preprocess_input(data: dict, feature_columns: list) -> pd.DataFrame:
    """
    Tek bir müşteri girdisini (dict) modele hazır bir DataFrame'e dönüştürür.

    Adımlar:
    1. Sözlüğü tek satırlık bir DataFrame'e çevirir.
    2. TotalCharges sütununu sayısala dönüştürür.
    3. customerID sütununu (varsa) çıkarır — model bu bilgiyi kullanmaz.
    4. Kategorik değişkenleri one-hot encode eder.
    5. Eğitimde kullanılan kolon sırasıyla hizalar.

    Args:
        data: /predict endpoint'inden gelen ham JSON verisi.
        feature_columns: Eğitim sırasında kaydedilmiş kolon listesi.

    Returns:
        pd.DataFrame: Tahmin için hazır, feature_columns ile hizalanmış DataFrame.
    """
    df = pd.DataFrame([data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    # Eğitimde görülmeyen sütunlar sıfırla doldurulur;
    # ardından modelin beklediği kolon sırasına göre hizalanır.
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df
