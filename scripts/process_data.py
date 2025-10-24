import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing import preprocess_results  # noqa: E402

RAW_DATA_PATH = Path("data/raw/race_results_with_metrics.pickle")
PROCESSED_DATA_PATH = Path("data/processed/前処理後のデータ.csv")


def main() -> None:
    df = pd.read_pickle(RAW_DATA_PATH)
    processed = preprocess_results(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=True)


if __name__ == "__main__":
    main()
