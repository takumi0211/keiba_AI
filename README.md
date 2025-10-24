競馬AI
===========

競馬レースの過去データをスクレイピングし、特徴量を整形して LightGBM で学習・推論まで行うプロジェクトです。学習用の一括パイプラインは `building.py`、最新の出馬表からの推論は `inference.py` で実行できます。


**主な機能**
- 過去レース結果と馬ごとの成績をスクレイピング（netkeiba データベース）
- 前処理（列名統一、体重/体重変化の分解、型変換、特徴量抽出）
- LightGBM による学習・評価（AUC、特徴量重要度、トップK当たり数の補助関数）
- 出馬表ページをスクレイピングし、学習済みモデルで三着以内確率を推定


**プロジェクト構成**
- `building.py:1` – スクレイピング→前処理→学習をまとめて実行するエントリポイント
- `inference.py:1` – 出馬表をスクレイピングし、学習済みモデルで推論
- `src/config.py:1` – レース名・レースIDリスト・使用特徴量に加え、推論用の `model_path`/`race_id`/`shutuba_table_path` を設定
- `src/scraping.py:1` – レース結果・馬成績（平均賞金/平均着順）のスクレイピング
- `src/processing.py:1` – 生データから学習用特徴量へ整形
- `src/modeling/light_gbm.py:1` – LightGBM 学習・評価・モデル保存
- `scripts/scrape_data.py:1` – 生データのみ取得して保存
- `scripts/process_data.py:1` – 生データから前処理済みCSVを生成
- `data/` – 生データ・前処理済みデータ・モデルなどの出力先


セットアップ
-------------

- 推奨: Python 3.10 以上（3.11 動作確認）
- 依存関係をインストール
  
  ```bash
  pip install -r requirements.txt
  ```

- macOS で LightGBM を使う場合は OpenMP ランタイムが必要
  
  ```bash
  brew install libomp
  ```

- 推論（`inference.py`）には Google Chrome と ChromeDriver が必要（ヘッドレス動作）
  
  - macOS: `brew install --cask google-chrome`、`brew install chromedriver`
  - Chrome と ChromeDriver のバージョンを一致させ、`chromedriver` が `PATH` にあることを確認


使い方
------

**1) 学習パイプラインを一括実行**

```bash
python building.py
```

- 主要出力
  - `data/raw/race_results_with_metrics.pickle` – スクレイピング＋平均賞金/平均着順付与後の生データ
  - `data/processed/前処理後のデータ.csv` – 学習用前処理データ
  - `data/models/LightBGMモデル.joblib` – 学習済みモデル
  - 標準出力に Train/Test AUC と特徴量重要度

**2) 最新の出馬表から推論のみ実行**

```bash
python inference.py
```

- 主要挙動
  - 出馬表（`race_id` は `src/config.py`）を Selenium で取得
  - `src/scraping.py` の馬成績スクレイピングから「平均賞金」「平均着順」を付与
- `model_path`（`src/config.py`）で指定した学習済みモデルを読み込み、三着以内確率を推定
- 予想順位・確率・馬番・馬名・単勝オッズ・実際の人気を一覧表示

### 判定ロジックの切り替え

推論時の『買うべき/買うべきでない』は `src/config.py` で選択できます。

```
# 'top_k' なら確率上位k頭を『買うべき』にする。'threshold' なら確率が閾値以上を選定。
prediction_strategy = 'top_k'  # 'top_k' | 'threshold'
prediction_top_k = 3           # 上位何頭を買うか（'top_k' のとき有効）
prediction_threshold = 0.5     # 閾値（'threshold' のとき有効）
```

レースは常に「3頭が三着以内に入る」前提のため、初期値は `top_k=3` を採用しています。固定閾値0.5で全頭『買うべきでない』が並ぶ状況を避け、来る可能性が高い上位馬を確実に拾います。
  - 取得した出馬表は `shutuba_table_path`（`src/config.py`）に保存

**3) ステップを個別に実行**

- 生データ取得のみ: `python scripts/scrape_data.py`
- 前処理のみ: `python scripts/process_data.py`
- 学習のみ: `python -m src.modeling.light_gbm`


設定（重要）
------------

`src/config.py:1` を編集して対象レースと特徴量を制御します。

- `race_name` – レース名（例: 菊花賞）。馬の過去成績テーブルから、このレース“以降”のデータを抽出するためのキーワードにも使用
- `race_id_list` – 取得対象レースIDの配列。netkeiba のレース詳細URLの `race_id` を指定（例: `202408050611`）
- `feature_columns` – 学習用に使用する列の順序。既定: `着順, 馬番, 単勝, 人 気, 馬の体重, 体重変化, 平均賞金, 平均着順`

推論関連の設定（`inference.py` が参照）

- `model_path` – 学習済みモデル（Joblib）の読み込み先
- `race_id` – 出馬表スクレイピング対象のレースID
- `shutuba_table_path` – 取得した出馬表CSVの保存先

メモ: `building.py` は既定で `data/models/LightBGMモデル.joblib` に保存します。`model_path` を変更する場合は、保存先と読み込み先が一致するように調整してください。

注意: 生データには全角スペースを含む列（例: `着 順`, `馬 番`, `人 気`）があり、`src/processing.py:1` 側で `着順/馬番` などに正規化しています。


モデルと評価の概要
------------------

- 目的変数: `着順 < 4` を 1（馬券内）、それ以外を 0 とする二値分類
- 特徴量例: `馬番, 単勝, 人 気, 馬の体重, 体重変化, 平均賞金, 平均着順`
- 分割方法: レースID単位で Train/Test を 8:2 に分割（リーク防止）
- 指標: ROC-AUC（ハイパーパラメータ探索もテストAUC最大化方針）
- 追加評価: `calc_topk_hits` でレース単位のトップK的中数を集計（実装は `src/modeling/light_gbm.py:1`）


出力物と保存先
--------------

- 生データ: `data/raw/race_results_with_metrics.pickle`
- 前処理: `data/processed/前処理後のデータ.csv`
- モデル: `data/models/LightBGMモデル.joblib`
- 推論ログ/結果: 標準出力（必要に応じてCSV化など拡張可能）


トラブルシューティング
----------------------

- Selenium のエラー「Unable to obtain driver for chrome」
  - Chrome と ChromeDriver のバージョン一致、`chromedriver` が `PATH` にあるか確認
- LightGBM のビルド/実行で OpenMP 関連のエラー
  - macOS: `brew install libomp` を実行
- スクレイピングでデータが空/列が見つからない
  - サイト構造変更の可能性。`src/scraping.py:1` のセレクタや `pd.read_html` の取り込みを確認


注意事項
--------

- 本プロジェクトのスクレイピングは netkeiba の公開ページを対象とします。利用規約・robots.txt を確認し、アクセス頻度（`sleep_seconds` 既定 1.0 秒）を守ってください。
- ハイパーパラメータ探索はテストAUC最大化の簡易実装です。外部検証や時系列分割、リーク点検などは各自の責任で実施してください。


ライセンス
----------

現時点では未定義です。公開配布や商用利用を行う場合は、適切なライセンスを設定してください。
