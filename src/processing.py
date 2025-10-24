import pandas as pd


def preprocess_results(result: pd.DataFrame) -> pd.DataFrame:
  df = result.copy()

  #着順を整数型にする
  df = df[~(df['着 順'].astype(str).str.contains('\D'))]
  df['着順'] = df['着 順'].astype(int)

  #馬体重を体重と変化量に分解
  df['馬の体重'] = df['馬体重'].str.split('(', expand=True)[0].astype(int)
  df['体重変化'] = df['馬体重'].str.split('(', expand=True)[1].str[:-1].astype(int)

  #単勝をfloat型に変更
  df['単勝'] = df['単勝'].astype(float)

  #馬番をobject型に変更
  df['馬番'] = df['馬 番'].astype(object)

  #必要な列のみを抽出
  new_order = ['着順', '馬番', '単勝', '人 気', '馬の体重', '体重変化', '平均賞金', '平均着順']
  df = df[new_order]
  return df
