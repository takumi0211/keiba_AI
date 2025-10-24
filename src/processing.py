import pandas as pd


def preprocess_results(result: pd.DataFrame) -> pd.DataFrame:
    """Clean race result DataFrame without applying feature selection.

    役割:
    - 列名の正規化（例: `着 順`→`着順`, `馬 番`→`馬番`）
    - 文字列数値の型変換（`単勝` など）
    - 体重/体重変化の分解（`馬体重` → `馬の体重`, `体重変化`）

    注意: ここでは列の絞り込みは行わない。学習時に `config.feature_columns` で絞る。
    """
    df = result.copy()

    # 着順: 数値行のみを対象にし、整数化
    if "着 順" in df.columns:
        df = df[~(df["着 順"].astype(str).str.contains(r"\D"))]
        df["着順"] = df["着 順"].astype(int)

    # 馬番: 全角スペース列を正規化
    if "馬 番" in df.columns:
        df["馬番"] = df["馬 番"].astype(object)

    # 単勝: 数値化（欠損や非数値は NaN）
    if "単勝" in df.columns:
        df["単勝"] = pd.to_numeric(df["単勝"], errors="coerce")

    # 体重/体重変化: `馬体重` 列があれば分解して追加
    if "馬体重" in df.columns:
        # 例: "480(+6)" → weight="480", delta="+6"
        weight_split = df["馬体重"].astype(str).str.extract(r"(?P<weight>\d+)(?:\((?P<delta>[-+−]?\d+)\))?")
        df["馬の体重"] = pd.to_numeric(weight_split["weight"], errors="coerce")
        df["体重変化"] = (
            weight_split["delta"]
            .str.replace("−", "-", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # ここでは列の削除・フィルタはしない（学習時に選択）
    return df
