import pandas as pd

from src.config import feature_columns


def preprocess_results(result: pd.DataFrame) -> pd.DataFrame:
    """Clean race result DataFrame into training features."""
    df = result.copy()

    df = df[~(df["着 順"].astype(str).str.contains(r"\D"))]
    df["着順"] = df["着 順"].astype(int)

    if {"馬の体重", "体重変化"} & set(feature_columns):
        weight_split = df["馬体重"].str.split("(", expand=True)
        if "馬の体重" in feature_columns:
            df["馬の体重"] = weight_split[0].astype(int)
        if "体重変化" in feature_columns:
            df["体重変化"] = weight_split[1].str[:-1].astype(int)

    df["単勝"] = df["単勝"].astype(float)
    df["馬番"] = df["馬 番"].astype(object)

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in preprocessing: {missing_columns}")

    return df[feature_columns]
