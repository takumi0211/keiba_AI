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
    # configの model_path は文字列のため、表示用に Path へ変換
    print(f"  -> モデルを {Path(model_path).resolve()} に保存しました。")

    return artifacts

def main(force: bool = False) -> Dict[str, float]:
    """
    パイプライン本体。既存データがあれば該当ステップをスキップする。

    - 前処理済みCSVが存在: Step1(スクレイピング)・Step2(前処理)をスキップし学習のみ
    - 生データPickleのみ存在: Step1(スクレイピング)をスキップし前処理→学習
    - どちらも無い: 全ステップ実行

    force=True の場合はスキップせず全ステップを実行する。
    """

    if not force and PROCESSED_DATA_PATH.exists():
        print("検出: 前処理済みデータが存在するため Step1/2 をスキップします → 学習のみ実行")
        artifacts = train_lightgbm()
        return {
            "train_auc": artifacts.train_auc,
            "test_auc": artifacts.test_auc,
            "best_test_auc": artifacts.best_test_auc,
        }

    if not force and RAW_DATA_PATH.exists():
        print("検出: 生データPickleが存在するため Step1(スクレイピング) をスキップします → 前処理から実行")
        race_results = pd.read_pickle(RAW_DATA_PATH)
        preprocess_data(race_results)
        artifacts = train_lightgbm()
        return {
            "train_auc": artifacts.train_auc,
            "test_auc": artifacts.test_auc,
            "best_test_auc": artifacts.best_test_auc,
        }

    # 上記に該当しなければフルパイプライン
    race_results = collect_race_data()
    preprocess_data(race_results)
    artifacts = train_lightgbm()
    return {
        "train_auc": artifacts.train_auc,
        "test_auc": artifacts.test_auc,
        "best_test_auc": artifacts.best_test_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬パイプライン: スクレイピング→前処理→学習")
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存データがあっても全ステップを実行する",
    )
    args = parser.parse_args()
    main(force=args.force)
