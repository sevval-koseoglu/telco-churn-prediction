import pandas as pd


def preprocess_input(data: dict, feature_columns: list) -> pd.DataFrame:
    """
    Tek bir musteri girdisini (dict) modele verilecek hazir DataFrame'e cevirir.

    Adimlar:
    1. Dict'i DataFrame'e cevirir.
    2. TotalCharges kolonunu numerige donusturur.
    3. customerID kolonunu (varsa) dusurir.
    4. Kategorik degiskenleri one-hot encode eder.
    5. Modelin egitimde gordugu feature_columns listesiyle hizalar.

    Args:
        data: /predict endpoint'inden gelen ham JSON verisi.
        feature_columns: Egitim sirasinda kaydedilmis kolon listesi.

    Returns:
        pd.DataFrame: Tahmin icin hazir, feature_columns ile hizalanmis DataFrame.
    """
    df = pd.DataFrame([data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    # Eksik kolonlari sifir ile doldur, modelin bekledigi kolonlara hizala
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df
