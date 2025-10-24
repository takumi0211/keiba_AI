from joblib import load
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
import time
from tqdm import tqdm
import re
import numpy as np
import pandas as pd

from src.config import (
    race_name,
    feature_columns,
    race_id,
    model_path,
    prediction_strategy,
    prediction_top_k,
    prediction_threshold,
)
from src.scraping import scrape_horse_metrics

# 定数
MAX_HORSE_NUMBER = 18
# 互換用（旧定数）。configの設定を優先し、未設定時のみ使う。
PREDICTION_THRESHOLD = 0.5


class ShutubaTable:
    """出馬表をスクレイピングするクラス"""
    
    def __init__(self):
        self.col = ['枠番', '馬番', '3', '馬名', 'horse_id', '性齢', '斤量', 
                    '騎手', 'jockey_id', '調教師', '馬体重', '単勝', '人気', '12', '13']
        self.shutuba_table = pd.DataFrame(columns=self.col)

    def scrape_shutuba_table(self, race_id_list):
        """出馬表をスクレイピング"""
        options = ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless")
        options.add_argument("start-maximized")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        driver = Chrome(options=options)

        for race_id in race_id_list:
            url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
            driver.get(url)
            time.sleep(2)
            elements = driver.find_elements(By.CLASS_NAME, 'HorseList')

            for element in tqdm(elements):
                tds = element.find_elements(By.TAG_NAME, 'td')
                row = []
                for td in tds:
                    row.append(td.text)
                    if td.get_attribute('class') in ['HorseInfo', 'Jockey']:
                        href = td.find_element(By.TAG_NAME, 'a').get_attribute('href')
                        row.append(re.findall(r'\d+', href)[0])
                
                # ウェブサイトの構造変更に対応: 17要素の場合は最後の2つを削除
                if len(row) == 17:
                    row = row[:15]
                
                # 出馬取り消しの馬を取り除く
                if len(row) != 15:
                    continue
                    
                temp_df = pd.DataFrame([row], columns=self.col)
                self.shutuba_table = pd.concat([self.shutuba_table, temp_df], ignore_index=True)

        driver.quit()
        
        # 不要な列を削除
        self.shutuba_table = self.shutuba_table.drop(['3', '調教師', '12', '13'], axis=1)


def normalize_popularity(popularity_series, default_value):
    """人気を正規化する"""
    return (
        popularity_series
        .replace({'**': None, '': None})
        .pipe(pd.to_numeric, errors='coerce')
        .fillna(default_value)
        .astype(int)
    )


def preprocess_shutuba_data(df):
    """出馬表データを前処理"""
    df = df.copy()
    
    # 馬体重の解析
    weight_info = df['馬体重'].str.extract(r'(?P<weight>\d+)(?:\((?P<delta>[-+−]?\d+)\))?')
    df['馬の体重'] = pd.to_numeric(weight_info['weight'], errors='coerce')
    df['体重変化'] = (
        weight_info['delta']
        .str.replace('−', '-', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
        .fillna(0)
    )
    
    # 単勝オッズの正規化
    df['単勝'] = (
        df['単勝']
        .replace({'---.-': None, '---': None, '': None})
        .pipe(pd.to_numeric, errors='coerce')
    )
    
    # 人気の正規化
    df['人 気'] = normalize_popularity(df['人気'], len(df) + 1)
    df['馬番'] = df['馬番'].astype(str)
    
    # 不要な列を削除
    columns_to_drop = ['性齢', '枠番', '斤量', '馬体重', '馬名', 'horse_id', 'jockey_id', '人気']
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    return df


def prepare_features_for_model(df):
    """モデル用の特徴量を準備"""
    # config.pyのfeature_columnsから"着順"を除外して特徴量の順序を取得
    feature_order = [col for col in feature_columns if col != "着順"]
    
    # データフレームから必要な特徴量のみを選択
    available_features = [col for col in feature_order if col in df.columns]
    df = df[available_features]
    
    # ダミー変数を作成
    df_dummies = pd.get_dummies(df)
    
    # モデルが期待する列を動的に生成
    # カテゴリ変数（ダミー変数化される列）とそうでない列を分離
    categorical_features = ['馬番']
    numerical_features = [col for col in feature_order if col not in categorical_features]
    
    # expected_columnsを動的に生成
    expected_columns = numerical_features.copy()
    
    # 馬番のダミー変数を追加
    if '馬番' in feature_order:
        expected_columns += [f'馬番_{i}' for i in range(1, MAX_HORSE_NUMBER + 1)]
    
    # 存在しない列は0で埋める
    for column in expected_columns:
        if column not in df_dummies.columns:
            df_dummies[column] = 0
    
    return df_dummies[expected_columns]


def create_prediction_result(df_original, y_proba):
    """予測結果のデータフレームを作成"""
    # 予想順位を算出
    sorted_indices = np.argsort(y_proba)[::-1]
    rankings = np.argsort(sorted_indices) + 1
    
    # 馬番を予想順位で並び替え
    horse_numbers = [0] * len(rankings)
    for i, ranking in enumerate(rankings):
        horse_numbers[ranking - 1] = i + 1
    
    # 判定情報の追加（設定で切り替え）
    sorted_proba_desc = sorted(y_proba, reverse=True)
    if prediction_strategy == 'top_k':
        k = max(1, int(prediction_top_k))
        judgments = ['買うべき' if i < k else '買うべきでない' for i in range(len(sorted_proba_desc))]
    else:
        thr = prediction_threshold if prediction_threshold is not None else PREDICTION_THRESHOLD
        judgments = ['買うべき' if prob >= thr else '買うべきでない' for prob in sorted_proba_desc]
    
    # 予測結果のデータフレーム作成
    df_pred = pd.DataFrame({
        '予想順位': range(1, len(rankings) + 1),
        '馬番': horse_numbers,
        '三着以内にくる確率': sorted_proba_desc,
        '判定': judgments
    })
    
    # 馬番と馬名を紐づけ
    df_horse_info = df_original[['馬番', '馬名']].copy()
    df_horse_info['馬番'] = df_horse_info['馬番'].astype(int)
    df_pred = pd.merge(df_pred, df_horse_info, on='馬番', how='inner')
    
    # オッズ情報を追加
    df_odds = df_original[['人気', '単勝', '馬番']].copy()
    df_odds['実際の人気'] = normalize_popularity(df_odds['人気'], len(df_odds) + 1)
    df_odds = df_odds.drop(['人気'], axis=1).rename(columns={'単勝': '単勝オッズ'})
    df_odds['馬番'] = df_odds['馬番'].astype(int)
    
    # 最終結果を結合
    df_result = pd.merge(df_pred, df_odds, on='馬番', how='inner')
    df_result = df_result[['予想順位', '三着以内にくる確率', '馬番', '馬名', '判定', '単勝オッズ', '実際の人気']]
    
    return df_result


def main():
    """メイン処理"""
    # 出馬表をスクレイピング
    print("出馬表をスクレイピング中...")
    st = ShutubaTable()
    st.scrape_shutuba_table([race_id])
    
    df = st.shutuba_table
    print(f"\n取得した出馬表:\n{df}\n")
    
    # データが空の場合はエラー
    if df.empty:
        raise ValueError(
            "スクレイピングに失敗しました。\n"
            "race_idが正しいか、ウェブサイトの構造が変わっていないか確認してください。"
        )
    
    # 馬の過去成績を取得
    print("馬の過去成績を取得中...")
    horse_id_list = list(df['horse_id'].unique())
    horse_prize, horse_rank = scrape_horse_metrics(horse_id_list, race_name)
    
    df['平均賞金'] = df['horse_id'].map(horse_prize)
    df['平均着順'] = df['horse_id'].map(horse_rank)
    
    # データの前処理
    print("データを前処理中...")
    df_processed = preprocess_shutuba_data(df)
    df_features = prepare_features_for_model(df_processed)
    
    # モデルを読み込んで予測
    print("予測を実行中...")
    model = load(model_path)
    y_proba = model.predict_proba(df_features)[:, 1]
    
    # 予測結果を作成
    df_result = create_prediction_result(df, y_proba)
    
    print("\n" + "="*80)
    print("予測結果:")
    print("="*80)
    print(df_result.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
