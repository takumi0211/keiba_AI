"""
競馬データを用いてLightGBMモデルを学習し、推論結果とモデルファイルを生成するスクリプト。
学習→評価→特徴量重要度の出力→モデル保存までを一気通貫で実行できる。

重要: ハイパーパラメータ選定は Train 内のみで実施し、
テストデータは最終評価のみに使用する（リーク防止）。
既定は GroupKFold による AP 最大化での選定。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

# 学習で利用する特徴量は config.py の `feature_columns` を参照する
from src.config import feature_columns


@dataclass
class TrainingArtifacts:
    model: lgb.LGBMClassifier
    best_params: Dict[str, Any]
    best_test_auc: float
    best_cv_ap: float
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
    selection_policy: str = "cv_ap",
) -> TrainingArtifacts:
    """
    前処理済みデータを読み込み、指定した選択方針でLightGBMモデルを学習する。
    既定は GroupKFold による AP 最大化（不均衡データでの検出力を重視）。

    Parameters
    ----------
    processed_csv_path:
        前処理済みデータを保存したCSVファイルパス。
    model_output_path:
        学習済みモデルを保存するJoblibファイルパス。
    save_model:
        Trueの場合はモデルをディスクに保存する。
    selection_policy:
        モデル選択の方針。
        - "cv_ap": GroupKFold での平均AP最大（同点時は平均ROC-AUC）を採用（デフォルト）
        - "val_auc": Train 内のグループ保全ホールドアウトでの ROC-AUC 最大を採用

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

    # config.py の feature_columns から目的変数 "着順" を除外し、学習で使う列を決定
    feature_cols = [col for col in feature_columns if col != "着順"]

    # 欠損（存在しない）列があれば早めに検知
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features in processed CSV: {missing}")

    features = df[feature_cols].copy()
    # LightGBM カテゴリ処理のために、馬番が存在する場合のみカテゴリ化
    if "馬番" in features.columns:
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
    # ハイパーパラメータ探索（CV・グループ分割・AP最適化）
    # ==========================
    # 目的: 「各レースで3頭だけが当たり」という不均衡性に合わせ、
    #       AUC(ROC)よりも Positiveクラスの検出に敏感な Average Precision (AUC-PR) を最大化。
    # 分割: 同一レースIDがfoldを跨がないよう GroupKFold を使用。
    # 早期終了: 各foldの検証データで early_stopping。

    # 学習データの不均衡を重みで補正
    pos = int(y_train.sum())
    neg = int((~y_train.astype(bool)).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    rng = np.random.default_rng(1234)

    def sample_params(n: int = 40) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for _ in range(n):
            num_leaves = int(rng.integers(16, 96))
            max_depth = int(rng.choice([-1, 4, 6, 8, 10]))
            # max_depth が有効なときは 2**max_depth を超えないよう制約
            if max_depth > 0:
                num_leaves = min(num_leaves, 2 ** max_depth)

            min_child_samples = int(rng.choice([10, 20, 40, 80, 120]))
            bagging_fraction = float(rng.uniform(0.6, 1.0))
            feature_fraction = float(rng.uniform(0.6, 1.0))
            learning_rate = float(10 ** rng.uniform(np.log10(0.01), np.log10(0.2)))
            reg_alpha = float(10 ** rng.uniform(np.log10(1e-3), np.log10(10)))
            reg_lambda = float(10 ** rng.uniform(np.log10(1e-3), np.log10(10)))
            n_estimators = int(rng.integers(600, 1800))
            samples.append(
                {
                    "num_leaves": num_leaves,
                    "max_depth": max_depth,
                    "min_child_samples": min_child_samples,
                    "bagging_fraction": bagging_fraction,
                    "feature_fraction": feature_fraction,
                    "learning_rate": learning_rate,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                    "n_estimators": n_estimators,
                }
            )
        return samples

    # CV設定（レース単位で分割）
    gkf = GroupKFold(n_splits=5)
    groups = race_ids[train_mask]

    # LightGBMのベース設定
    base_params = {
        "random_state": 12,
        "objective": "binary",
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        # bagging_fraction を使う場合の推奨設定
        "bagging_freq": 1,
    }

    candidate_params = sample_params(40)
    best_params: dict[str, Any] | None = None
    best_cv_ap = float("-inf")
    best_cv_auc = float("-inf")
    best_model: lgb.LGBMClassifier | None = None
    best_test_auc = float("-inf")
    # モデル選択ロジック
    if selection_policy not in {"val_auc", "cv_ap"}:
        raise ValueError("selection_policy must be 'val_auc' or 'cv_ap'")

    if selection_policy == "cv_ap":
        # 旧ロジック: CVのAP最大（同点時はROC-AUC）で選択
        for cand in candidate_params:
            params = {**base_params, **cand}
            ap_scores: list[float] = []
            auc_scores: list[float] = []

            # 各foldで学習→検証
            for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="aucpr",
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )

                y_va_pred = model.predict_proba(X_va)[:, 1]
                ap = average_precision_score(y_va, y_va_pred)
                auc = roc_auc_score(y_va, y_va_pred)
                ap_scores.append(ap)
                auc_scores.append(auc)

            mean_ap = float(np.mean(ap_scores))
            mean_auc = float(np.mean(auc_scores))

            if (mean_ap > best_cv_ap) or (np.isclose(mean_ap, best_cv_ap) and mean_auc > best_cv_auc):
                best_cv_ap = mean_ap
                best_cv_auc = mean_auc
                best_params = params
    else:
        # Val ROC-AUC 最大で選択（テストデータは使わない）
        # 先に簡易hold-outを作って、早期終了用の検証を固定（グループ保全）
        unique_train_groups = groups.unique()
        val_group_count = max(1, int(0.1 * len(unique_train_groups)))
        val_groups = set(random.sample(list(unique_train_groups), k=val_group_count))
        val_mask = groups.isin(val_groups)

        X_tr_final, y_tr_final = X_train[~val_mask], y_train[~val_mask]
        X_val_final, y_val_final = X_train[val_mask], y_train[val_mask]
        best_val_auc = float("-inf")

        for cand in candidate_params:
            params = {**base_params, **cand}
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr_final,
                y_tr_final,
                eval_set=[(X_val_final, y_val_final)],
                eval_metric="aucpr",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )

            y_val_pred = model.predict_proba(X_val_final)[:, 1]
            val_auc_cand = roc_auc_score(y_val_final, y_val_pred)
            if val_auc_cand > best_val_auc:
                best_val_auc = val_auc_cand
                best_params = params
                best_model = model

    # ベスト設定で再学習（trainデータ全体）。早期終了は内部CVではなく簡易hold-outで実施。
    # train内から10%を検証に分け、過学習を抑制。
    if best_params is None:
        # フォールバック: デフォルトに近い無難な設定
        best_params = {**base_params, "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63}

    # 簡易hold-outと最終学習
    if selection_policy == "cv_ap":
        # CVで得たbest_paramsで学習し直す
        if best_params is None:
            # フォールバック: デフォルトに近い無難な設定
            best_params = {**base_params, "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63}

        unique_train_groups = groups.unique()
        val_group_count = max(1, int(0.1 * len(unique_train_groups)))
        val_groups = set(random.sample(list(unique_train_groups), k=val_group_count))
        val_mask = groups.isin(val_groups)

        X_tr_final, y_tr_final = X_train[~val_mask], y_train[~val_mask]
        X_val_final, y_val_final = X_train[val_mask], y_train[val_mask]

        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(
            X_tr_final,
            y_tr_final,
            eval_set=[(X_val_final, y_val_final)],
            eval_metric="aucpr",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
    else:
        # val_auc 選択で候補を総当りした際に best_model が既に学習済み
        if best_model is None:
            # 念のためフォールバック（通常到達しない）
            unique_train_groups = groups.unique()
            val_group_count = max(1, int(0.1 * len(unique_train_groups)))
            val_groups = set(random.sample(list(unique_train_groups), k=val_group_count))
            val_mask = groups.isin(val_groups)
            X_tr_final, y_tr_final = X_train[~val_mask], y_train[~val_mask]
            X_val_final, y_val_final = X_train[val_mask], y_train[val_mask]
            best_params = best_params or {**base_params, "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63}
            best_model = lgb.LGBMClassifier(**best_params)
            best_model.fit(
                X_tr_final,
                y_tr_final,
                eval_set=[(X_val_final, y_val_final)],
                eval_metric="aucpr",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )

    # ログ出力
    if selection_policy == "cv_ap":
        print(f"Selection policy: CV-AP")
        print(f"Best CV AP: {best_cv_ap:.4f}")
        print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")
    else:
        print(f"Selection policy: Val ROC-AUC")
    print(f"Best params: {best_params}")

    # ==========================
    # 学習と評価
    # ==========================
    # 念のためtrain側のAUCも確認できるように改めて推論。
    if best_model is None:
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val_final, y_val_final)],
            eval_metric="aucpr",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

    lgb_clf = best_model

    y_pred_train = lgb_clf.predict_proba(X_train)[:, 1]
    y_pred = lgb_clf.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred)
    # 最終的な test_auc を記録（選定方針に依らず同一の扱い）
    best_test_auc = test_auc if not np.isnan(test_auc) else -1.0

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
        best_cv_ap=best_cv_ap,
        train_auc=train_auc,
        test_auc=test_auc,
        evaluation_df=evaluation_df,
        feature_importance=importance_sorted,
    )


if __name__ == "__main__":
    train_model()
