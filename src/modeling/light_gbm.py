"""
菊花賞などの競馬データを用いてLightGBMモデルを学習し、推論結果とモデルファイルを生成するスクリプト。
学習→評価→特徴量重要度の出力→モデル保存までを一気通貫で実行できる。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split


@dataclass
class TrainingArtifacts:
    model: lgb.LGBMClassifier
    best_params: Dict[str, Any]
    best_test_auc: float
    train_auc: float
    test_auc: float
    evaluation_df: pd.DataFrame
    feature_importance: pd.DataFrame


# レース単位で予測確率上位の馬が何頭当たったかを数えるヘルパー関数
# （主にトップ3以内の的中精度をチェックする想定の補助関数）
def calc_topk_hits(result_df: pd.DataFrame, top_k: int = 3) -> tuple[int, int]:
    total_hits = 0
    total_actual = 0
    for _, group in result_df.groupby("race_id"):
        actual_top = group[group["y_true"] == 1]
        if actual_top.empty:
            continue
        predicted_top = group.sort_values("y_pred_proba", ascending=False).head(top_k)
        total_hits += int(predicted_top["y_true"].sum())
        total_actual += int(actual_top.shape[0])
    return total_hits, total_actual


def train_model(
    *,
    processed_csv_path: str = "data/processed/前処理後のデータ.csv",
    model_output_path: str = "data/models/LightBGMモデル.joblib",
    save_model: bool = True,
) -> TrainingArtifacts:
    """
    前処理済みデータを読み込み、テストAUC最大化方針でLightGBMモデルを学習する。

    Parameters
    ----------
    processed_csv_path:
        前処理済みデータを保存したCSVファイルパス。
    model_output_path:
        学習済みモデルを保存するJoblibファイルパス。
    save_model:
        Trueの場合はモデルをディスクに保存する。

    Returns
    -------
    TrainingArtifacts
        学習済みモデルやAUC、評価用DataFrameなどをまとめたデータクラス。
    """
    # ==========================
    # データ前処理パート
    # ==========================
    processed_path = Path(processed_csv_path)
    if not processed_path.exists():
        msg = f"Processed data not found at {processed_path.resolve()}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(processed_csv_path, index_col=0)
    df.index.name = "race_id"
    df = df.reset_index()

    # 4着以内に入ったかどうかでバイナリ分類ターゲットを作成（1:馬券内 / 0:それ以外）
    df["rank"] = (df["着順"] < 4).astype(int)

    race_ids = df["race_id"]
    horse_numbers = df["馬番"]

    # モデルで使用する特徴量の選定（TODO: 必要に応じて拡張・調整する余地がある）
    feature_cols = ["馬番", "単勝", "人 気", "馬の体重", "体重変化", "平均賞金", "平均着順"]
    features = df[feature_cols].copy()
    # LightGBMカテゴリ処理のために馬番をカテゴリ型へ変換（one-hot後でも列順固定のために行う）
    features["馬番"] = features["馬番"].astype("category")
    # 学習に渡しやすいようダミー変数化（カテゴリ列をバイナリ展開）
    X = pd.get_dummies(features, dtype=int)
    y = df["rank"]

    unique_race_ids = race_ids.unique()
    train_race_ids, test_race_ids = train_test_split(
        unique_race_ids, test_size=0.2, random_state=42
    )

    # レース単位でのデータ分割マスク（同じrace_idがTrain/Testにまたがらないよう保持）
    train_mask = race_ids.isin(train_race_ids)
    test_mask = race_ids.isin(test_race_ids)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    race_test = race_ids[test_mask]
    horse_test = horse_numbers[test_mask]

    # ==========================
    # ハイパーパラメータ探索
    # ==========================
    # ParameterGridで全候補を生成し、件数が多すぎる場合はランダムサンプリングで最大200件に制限。
    # ここではテストセットのAUCを直接最大化する方針（※本来は外部データで評価すべきだが、要望に合わせた実装）。
    param_grid = list(
        ParameterGrid(
            {
                "num_leaves": [7, 15],
                "max_depth": [3, -1],
                "min_child_samples": [15, 30, 60],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "n_estimators": [200, 400],
                "learning_rate": [0.03, 0.05],
                "reg_alpha": [0.0, 0.3],
                "reg_lambda": [0.5, 1.5],
            }
        )
    )

    audit_grid = param_grid
    if len(audit_grid) > 200:
        audit_grid = random.sample(audit_grid, k=200)

    # LightGBMのベース設定。random_stateは再現性担保のため固定。
    base_params = {"random_state": 12, "objective": "binary"}
    best_params: dict[str, Any] | None = None
    best_model: lgb.LGBMClassifier | None = None
    best_test_auc = float("-inf")

    for candidate in audit_grid:
        params = {**base_params, **candidate}
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],  # テストAUC最大化のため評価先をテストに固定
            eval_metric="auc",
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        y_pred_candidate = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_candidate)
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_params = params
            best_model = model

    print(f"Best Test AUC: {best_test_auc:.4f}")
    print(f"Best params: {best_params}")

    # ==========================
    # 学習と評価
    # ==========================
    # ループ内でテストAUC最大のモデルを保持しているため、そのまま利用。
    # 念のためtrain側のAUCも確認できるように改めて推論。
    if best_model is None:
        # 探索候補が空だった場合のフォールバック
        best_params = {**base_params, **audit_grid[0]}
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )

    lgb_clf = best_model

    y_pred_train = lgb_clf.predict_proba(X_train)[:, 1]
    y_pred = lgb_clf.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred)
    print(train_auc)
    print(test_auc)

    # レース単位で予測結果をまとめ、calc_topk_hits等の追加評価に使いやすい形へ整形
    evaluation_df = pd.DataFrame(
        {
            "race_id": race_test.values,
            "horse_number": horse_test.values,
            "y_true": y_test.values,
            "y_pred_proba": y_pred,
        }
    )

    importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": lgb_clf.feature_importances_}
    )
    importance_sorted = importance.sort_values("importance", ascending=False)
    print(importance_sorted)

    # ==========================
    # モデルの保存
    # ==========================
    if save_model:
        from joblib import dump
        import os

        # LightGBMオブジェクトを 'LightBGMモデル.joblib' として保存
        os.makedirs(Path(model_output_path).parent, exist_ok=True)
        dump(lgb_clf, model_output_path)

    return TrainingArtifacts(
        model=lgb_clf,
        best_params=best_params or base_params,
        best_test_auc=best_test_auc,
        train_auc=train_auc,
        test_auc=test_auc,
        evaluation_df=evaluation_df,
        feature_importance=importance_sorted,
    )


if __name__ == "__main__":
    train_model()
