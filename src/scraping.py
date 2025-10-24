import re
import time
from io import StringIO
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}


def scrape_race_results(
    race_id_list: Iterable[str],
    *,
    sleep_seconds: float = 1.0,
    pre_race_result: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Scrape race result tables keyed by race_id."""
    race_results = pre_race_result or {}
    for race_id in tqdm(race_id_list):
        if race_id in race_results:
            continue
        try:
            url = f"https://db.netkeiba.com/race/{race_id}"
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.content, "html.parser")
            race_table = soup.find("table", attrs={"summary": "レース結果"})
            if race_table is None:
                print(f"Race result table not found for race_id {race_id}. Skipping...")
                continue

            dfs = pd.read_html(StringIO(str(race_table)))
            if not dfs:
                print(f"No data found in the race result table for race_id {race_id}. Skipping...")
                continue
            df = dfs[0]

            horse_links = race_table.find_all("a", attrs={"href": re.compile(r"^/horse")})
            horse_id_list = [re.findall(r"\d+", a["href"])[0] for a in horse_links]

            jockey_links = race_table.find_all("a", attrs={"href": re.compile(r"^/jockey")})
            jockey_id_list = [re.findall(r"\d+", a["href"])[0] for a in jockey_links]

            data_intro = soup.find("div", attrs={"class": "data_intro"})
            if data_intro is None:
                print(f"Data intro section not found for race_id {race_id}. Skipping...")
                continue
            text = data_intro.find_all("p")[1].text
            date = re.findall(r"\d+年\d+月\d+日", text)[0]

            df["horse_id"] = horse_id_list
            df["jockey_id"] = jockey_id_list
            df["日付"] = pd.to_datetime(date, format="%Y年%m月%d日")

            race_results[race_id] = df
            time.sleep(sleep_seconds)
        except IndexError:
            print(f"IndexError occurred for race_id {race_id}. Skipping...")
            continue
        except Exception as exc:
            print(f"Error processing race_id {race_id}: {exc}")
            break
    return race_results


def scrape_horse_metrics(
    horse_id_list: Iterable[str],
    race_name: str,
    *,
    sleep_seconds: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return per-horse average prize and rank after the specified race."""
    horse_prize: Dict[str, float] = {}
    horse_rank: Dict[str, float] = {}

    for horse_id in tqdm(horse_id_list):
        url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", class_="db_h_race_results")
        if table is None:
            print(f"成績テーブルが見つかりませんでした (horse_id={horse_id})。")
            continue

        column_headers = [th.get_text(strip=True) for th in table.find("tr").find_all("th")]

        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        df = pd.DataFrame(rows, columns=column_headers)

        target_mask = df["レース名"].str.contains(race_name, na=False)
        if not target_mask.any():
            print(f"{race_name} が見つからなかったため horse_id {horse_id} をスキップします。")
            continue

        prize_index = df.index[target_mask][0]
        filtered_df = df.loc[df.index > prize_index].copy()
        if filtered_df.empty:
            print(f"{race_name} 以降のレースが存在しないため horse_id {horse_id} をスキップします。")
            continue

        filtered_df.loc[:, "賞金"] = (
            filtered_df["賞金"]
            .astype(str)
            .replace("-", "0")
            .replace(",", "", regex=True)
        )
        filtered_df.loc[:, "賞金"] = pd.to_numeric(filtered_df["賞金"], errors="coerce").fillna(0)
        horse_prize[horse_id] = filtered_df["賞金"].mean()

        rank_mean = (
            pd.to_numeric(filtered_df["着順"], errors="coerce")
            .fillna(0)
            .astype(int)
            .mean()
        )
        horse_rank[horse_id] = rank_mean

        time.sleep(sleep_seconds)

    return horse_prize, horse_rank
