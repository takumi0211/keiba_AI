import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import race_id_list, race_name  # noqa: E402
from src.scraping import scrape_horse_metrics, scrape_race_results  # noqa: E402

RAW_DATA_DIR = Path("data/raw")


def main() -> None:
    race_result = scrape_race_results(race_id_list)

    for key in race_result:
        race_result[key].index = [key] * len(race_result[key])
    race_results = pd.concat(race_result.values())

    horse_prize, horse_rank = scrape_horse_metrics(race_results["horse_id"].unique(), race_name)

    race_results["平均賞金"] = race_results["horse_id"].map(horse_prize)
    race_results["平均着順"] = race_results["horse_id"].map(horse_rank)

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    race_results.to_pickle(RAW_DATA_DIR / "race_results_with_metrics.pickle")


if __name__ == "__main__":
    main()
