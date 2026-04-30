import pandas as pd


def preprocess_input(data: dict, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df