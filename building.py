"""
main.py
========

レース結果のスクレイピング → 特徴量加工 → LightGBMモデル学習までを
一度の実行で完了させるパイプラインスクリプト。
既存の `src/` モジュールと `light_gbm.train_model` を連携させている。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import race_id_list, race_name, model_path
from src.modeling.light_gbm import TrainingArtifacts, train_model
from src.processing import preprocess_results
from src.scraping import scrape_horse_metrics, scrape_race_results

RAW_DATA_PATH = Path("data/raw/race_results_with_metrics.pickle")
PROCESSED_DATA_PATH = Path("data/processed/前処理後のデータ.csv")


def collect_race_data() -> pd.DataFrame:
    """レース結果＆馬体情報をスクレイピングし、馬ごとの平均賞金/平均着順を付与する。"""
    print("=== Step 1/3: レース結果をスクレイピング中 ===")
    race_result_dict = scrape_race_results(race_id_list)
    if not race_result_dict:
        raise RuntimeError("レース結果の取得に失敗しました。")

    for race_id, df in race_result_dict.items():
        df.index = [race_id] * len(df)
        race_result_dict[race_id] = df

    race_results = pd.concat(race_result_dict.values())
    print(f"  -> {len(race_results)} 行のレース結果を取得しました。")

    print("=== Step 1/3: 馬ごとの平均賞金/平均着順を算出中 ===")
    horse_ids = race_results["horse_id"].unique()
    horse_prize, horse_rank = scrape_horse_metrics(horse_ids, race_name)

    race_results["平均賞金"] = race_results["horse_id"].map(horse_prize)
    race_results["平均着順"] = race_results["horse_id"].map(horse_rank)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    race_results.to_pickle(RAW_DATA_PATH)
    print(f"  -> 生データを {RAW_DATA_PATH.resolve()} に保存しました。")

    return race_results


def preprocess_data(race_results: pd.DataFrame) -> pd.DataFrame:
    """前処理関数を呼び出し、学習用の特徴量テーブルを生成・保存する。"""
    print("=== Step 2/3: 前処理を実行中 ===")
    processed = preprocess_results(race_results)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=True)
    print(f"  -> 前処理済みデータを {PROCESSED_DATA_PATH.resolve()} に保存しました。")
    return processed


def train_lightgbm() -> TrainingArtifacts:
    """LightGBMでモデリングを実行し、成果物を返す。"""
    print("=== Step 3/3: LightGBMで学習中 ===")
    artifacts = train_model(
        processed_csv_path=str(PROCESSED_DATA_PATH),
        model_output_path=str(model_path),
        save_model=True,
    )
    print(f"  -> Train AUC: {artifacts.train_auc:.4f}")
    print(f"  -> Test AUC : {artifacts.test_auc:.4f}")
    print(f"  -> モデルを {model_path.resolve()} に保存しました。")

    return artifacts


def main() -> Dict[str, float]:
    # レース結果をスクレイピング
    race_results = collect_race_data()
    # 前処理
    preprocess_data(race_results)
    # 学習
    artifacts = train_lightgbm()

    return {
        "train_auc": artifacts.train_auc,
        "test_auc": artifacts.test_auc,
        "best_test_auc": artifacts.best_test_auc,
    }

if __name__ == "__main__":
    main()
