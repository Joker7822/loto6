"""Standalone Loto6 scraper, predictor and walk-forward backtester."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

MIN_NUMBER = 1
MAX_NUMBER = 43
PICK_SIZE = 6
NUMBER_COLUMNS = [f"n{i}" for i in range(1, 7)]
DRAW_COLUMNS = ["draw_no", "date", *NUMBER_COLUMNS, "bonus"]
TICKET_PRICE_YEN = 200
CURRENT_URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html"
DETAIL_URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/detail.html?fromto={start}_{end}&type=loto6"
OLD_URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto6{start:04d}.html"
REFERENCE_PAYOUT_YEN = {"1等": 200_000_000, "2等": 10_000_000, "3等": 300_000, "4等": 6_800, "5等": 1_000, "はずれ": 0}
DRAW_RE = re.compile(r"第?\s*(\d+)\s*回")
DATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
NUM_RE = re.compile(r"\d{1,2}")


@dataclass(frozen=True)
class MatchResult:
    main_matches: int
    bonus_match: bool
    grade: str


@dataclass(frozen=True)
class Prediction:
    rank: int
    numbers: tuple[int, ...]
    score: float
    source: str = "heuristic_ensemble"

    @property
    def numbers_text(self) -> str:
        return format_numbers(self.numbers)


def normalize_numbers(numbers: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted(int(n) for n in numbers))
    if len(values) != PICK_SIZE:
        raise ValueError(f"Loto6 requires exactly {PICK_SIZE} numbers: {values}")
    if len(set(values)) != PICK_SIZE:
        raise ValueError(f"Loto6 numbers must be unique: {values}")
    if min(values) < MIN_NUMBER or max(values) > MAX_NUMBER:
        raise ValueError(f"Loto6 numbers must be {MIN_NUMBER}..{MAX_NUMBER}: {values}")
    return values


def format_numbers(numbers: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in normalize_numbers(numbers))


def classify_loto6(prediction: Sequence[int], main_numbers: Sequence[int], bonus_number: int | None = None) -> MatchResult:
    pred = set(normalize_numbers(prediction))
    main = set(normalize_numbers(main_numbers))
    main_matches = len(pred & main)
    bonus_match = bonus_number is not None and int(bonus_number) in pred
    if main_matches == 6:
        grade = "1等"
    elif main_matches == 5 and bonus_match:
        grade = "2等"
    elif main_matches == 5:
        grade = "3等"
    elif main_matches == 4:
        grade = "4等"
    elif main_matches == 3:
        grade = "5等"
    else:
        grade = "はずれ"
    return MatchResult(main_matches, bonus_match, grade)


def normalize_draw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    aliases = {"回別": "draw_no", "回": "draw_no", "抽せん日": "date", "抽選日": "date", "日付": "date", "ボーナス数字": "bonus", "ボーナス": "bonus"}
    work = work.rename(columns={c: aliases.get(c, c) for c in work.columns})
    for i in range(1, 7):
        for c in (f"num{i}", f"number{i}", f"本数字{i}", f"数字{i}"):
            if c in work.columns:
                work = work.rename(columns={c: f"n{i}"})
    missing = [c for c in DRAW_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; actual={list(work.columns)}")
    work = work[DRAW_COLUMNS].copy()
    work["draw_no"] = work["draw_no"].astype(str).str.extract(r"(\d+)").astype(int)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for c in NUMBER_COLUMNS + ["bonus"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").astype("Int64")
    work = work.dropna(subset=["date", *NUMBER_COLUMNS, "bonus"]).copy()
    for c in NUMBER_COLUMNS + ["bonus"]:
        work[c] = work[c].astype(int)
    sorted_rows = [normalize_numbers(row) for row in work[NUMBER_COLUMNS].itertuples(index=False, name=None)]
    for i, c in enumerate(NUMBER_COLUMNS):
        work[c] = [row[i] for row in sorted_rows]
    return work.drop_duplicates("draw_no", keep="last").sort_values("draw_no").reset_index(drop=True)


def load_draws(path: str | Path) -> pd.DataFrame:
    try:
        return normalize_draw_dataframe(pd.read_csv(path))
    except UnicodeDecodeError:
        return normalize_draw_dataframe(pd.read_csv(path, encoding="cp932"))


def _scale(values: dict[int, float]) -> dict[int, float]:
    arr = np.array([values.get(n, 0.0) for n in range(1, 44)], dtype=float)
    if arr.max() == arr.min():
        return {n: 0.0 for n in range(1, 44)}
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return {n: float(arr[n - 1]) for n in range(1, 44)}


def number_scores(df: pd.DataFrame, recent_window: int = 80) -> dict[int, float]:
    draws = normalize_draw_dataframe(df)
    recent = draws.tail(min(recent_window, len(draws)))
    prior = draws.iloc[: max(0, len(draws) - len(recent))]
    all_cnt = Counter(int(v) for v in draws[NUMBER_COLUMNS].to_numpy().ravel())
    recent_cnt = Counter(int(v) for v in recent[NUMBER_COLUMNS].to_numpy().ravel())
    prior_cnt = Counter(int(v) for v in prior[NUMBER_COLUMNS].to_numpy().ravel()) if not prior.empty else Counter()
    gap = {n: len(draws) + 1 for n in range(1, 44)}
    for offset, row in enumerate(draws[NUMBER_COLUMNS].itertuples(index=False, name=None), start=1):
        for n in row:
            gap[int(n)] = len(draws) - offset
    f = _scale({n: all_cnt[n] for n in range(1, 44)})
    r = _scale({n: recent_cnt[n] for n in range(1, 44)})
    g = _scale(gap)
    t = _scale({n: recent_cnt[n] / max(1, len(recent)) - prior_cnt[n] / max(1, len(prior)) for n in range(1, 44)})
    return {n: 0.35 * f[n] + 0.35 * r[n] + 0.20 * g[n] + 0.10 * t[n] for n in range(1, 44)}


def pair_scores(df: pd.DataFrame, recent_window: int = 160) -> dict[tuple[int, int], float]:
    draws = normalize_draw_dataframe(df).tail(recent_window)
    cnt = Counter()
    for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None):
        for pair in combinations(sorted(map(int, row)), 2):
            cnt[pair] += 1
    if not cnt:
        return {}
    m = max(cnt.values())
    return {p: v / m for p, v in cnt.items()}


class Loto6Predictor:
    def __init__(self, recent_window: int = 80, seed: int = 42) -> None:
        self.recent_window = recent_window
        self.seed = seed
        self.scores: dict[int, float] = {}
        self.pairs: dict[tuple[int, int], float] = {}
        self.history: set[tuple[int, ...]] = set()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "Loto6Predictor":
        draws = normalize_draw_dataframe(df)
        self.scores = number_scores(draws, self.recent_window)
        self.pairs = pair_scores(draws, max(160, self.recent_window * 2))
        self.history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}
        self.fitted = True
        return self

    def _combo_score(self, combo: Sequence[int]) -> float:
        nums = normalize_numbers(combo)
        base = sum(self.scores.get(n, 0.0) for n in nums) / 6
        pair = np.mean([self.pairs.get(tuple(sorted(p)), 0.0) for p in combinations(nums, 2)])
        odd_balance = 1.0 - abs(sum(n % 2 for n in nums) - 3) / 3.0
        sum_balance = max(0.0, 1.0 - abs(sum(nums) - 132) / 90.0)
        zone_balance = len({(n - 1) // 10 for n in nums}) / 5.0
        consecutive_penalty = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1) * 0.025
        duplicate_penalty = 0.20 if nums in self.history else 0.0
        return 0.52 * base + 0.20 * pair + 0.10 * odd_balance + 0.10 * sum_balance + 0.08 * zone_balance - consecutive_penalty - duplicate_penalty

    def predict(self, n: int = 5, candidate_count: int = 5000) -> list[Prediction]:
        if not self.fitted:
            raise RuntimeError("Call fit(df) before predict().")
        rng = random.Random(self.seed)
        numbers = list(range(1, 44))
        weights = [self.scores.get(x, 0.0) + 0.05 for x in numbers]
        candidates: set[tuple[int, ...]] = set()
        top_pool = sorted(numbers, key=lambda x: self.scores.get(x, 0.0), reverse=True)[:18]
        for combo in combinations(top_pool, 6):
            candidates.add(normalize_numbers(combo))
            if len(candidates) >= candidate_count // 3:
                break
        while len(candidates) < candidate_count:
            available = numbers[:]
            available_weights = weights[:]
            pick = []
            for _ in range(6):
                total = sum(available_weights)
                r = rng.random() * total
                upto = 0.0
                idx = 0
                for i, w in enumerate(available_weights):
                    upto += w
                    if upto >= r:
                        idx = i
                        break
                pick.append(available.pop(idx))
                available_weights.pop(idx)
            candidates.add(normalize_numbers(pick))
        ranked = sorted(((self._combo_score(c), c) for c in candidates), reverse=True)
        return [Prediction(i + 1, combo, float(score)) for i, (score, combo) in enumerate(ranked[:n])]


def predictions_to_dataframe(predictions: Iterable[Prediction]) -> pd.DataFrame:
    return pd.DataFrame([{"rank": p.rank, "numbers": p.numbers_text, "score": round(p.score, 6), "source": p.source, **{f"n{i}": x for i, x in enumerate(p.numbers, 1)}} for p in predictions])


def parse_loto6_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        text = " ".join(cells)
        draw = DRAW_RE.search(text)
        date = DATE_RE.search(text)
        nums = [int(x) for x in NUM_RE.findall(text) if 1 <= int(x) <= 43]
        if not draw or not date or len(nums) < 7:
            continue
        y, m, d = map(int, date.groups())
        seven = nums[-7:]
        rows.append({"draw_no": int(draw.group(1)), "date": f"{y:04d}-{m:02d}-{d:02d}", **{f"n{i}": seven[i - 1] for i in range(1, 7)}, "bonus": seven[6]})
    return normalize_draw_dataframe(pd.DataFrame(rows)).to_dict("records") if rows else []


def fetch_page(url: str) -> list[dict]:
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0 loto6-scraper/1.0", "Accept-Language": "ja,en;q=0.8"}, timeout=20)
    res.raise_for_status()
    res.encoding = res.apparent_encoding or res.encoding
    return parse_loto6_html(res.text)


def update_csv(path: str = "data/loto6.csv", max_draw: int = 9999, block_size: int = 20, sleep_seconds: float = 0.6, max_empty_blocks: int = 8) -> pd.DataFrame:
    rows: list[dict] = []
    empty = 0
    for start in range(1, max_draw + 1, block_size):
        block = []
        for url in (DETAIL_URL.format(start=start, end=start + block_size - 1), OLD_URL.format(start=start)):
            try:
                block = fetch_page(url)
            except Exception:
                block = []
            if block:
                break
            time.sleep(sleep_seconds)
        if block:
            empty = 0
            rows.extend(block)
        else:
            empty += 1
            if empty >= max_empty_blocks:
                break
        time.sleep(sleep_seconds)
    try:
        rows.extend(fetch_page(CURRENT_URL))
    except Exception:
        pass
    if not rows:
        raise RuntimeError("No Loto6 rows scraped. Check Mizuho page structure or network access.")
    df = normalize_draw_dataframe(pd.DataFrame(rows))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    return df


def walk_forward_backtest(df: pd.DataFrame, start_after: int = 300, top_n: int = 5, candidate_count: int = 3000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    draws = normalize_draw_dataframe(df)
    if len(draws) <= start_after:
        raise ValueError(f"Need more than start_after={start_after} rows; got {len(draws)}")
    records = []
    for idx in range(start_after, len(draws)):
        actual = draws.iloc[idx]
        preds = Loto6Predictor(seed=seed + idx).fit(draws.iloc[:idx]).predict(top_n, candidate_count)
        main = [int(actual[c]) for c in NUMBER_COLUMNS]
        bonus = int(actual["bonus"])
        for pred in preds:
            r = classify_loto6(pred.numbers, main, bonus)
            records.append({"draw_no": int(actual["draw_no"]), "date": actual["date"].date().isoformat(), "rank": pred.rank, "prediction": pred.numbers_text, "actual": " ".join(f"{x:02d}" for x in main), "bonus": f"{bonus:02d}", "main_matches": r.main_matches, "bonus_match": r.bonus_match, "grade": r.grade, "score": round(pred.score, 6), "reference_payout_yen": REFERENCE_PAYOUT_YEN[r.grade]})
    result_df = pd.DataFrame(records)
    cost = (len(draws) - start_after) * top_n * TICKET_PRICE_YEN
    payout = int(result_df["reference_payout_yen"].sum()) if not result_df.empty else 0
    summary = {"tested_draws": len(draws) - start_after, "tickets": len(result_df), "top_n": top_n, "cost_yen": cost, "reference_payout_yen": payout, "reference_profit_yen": payout - cost, "reference_roi": round(payout / cost, 6) if cost else 0.0, "grade_counts": dict(Counter(result_df["grade"])), "max_main_matches": int(result_df["main_matches"].max()) if not result_df.empty else 0}
    return result_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Loto6 scraper / predictor / backtester")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("update")
    p.add_argument("--csv", default="data/loto6.csv")
    p.add_argument("--max-draw", type=int, default=9999)
    p = sub.add_parser("predict")
    p.add_argument("--csv", default="data/loto6.csv")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recent-window", type=int, default=80)
    p.add_argument("--candidates", type=int, default=5000)
    p.add_argument("--output", default="outputs/predictions.csv")
    p = sub.add_parser("backtest")
    p.add_argument("--csv", default="data/loto6.csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--start-after", type=int, default=300)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--candidates", type=int, default=3000)
    args = parser.parse_args()
    if args.__dict__.get("max_draw") is not None:
        df = update_csv(args.csv, max_draw=args.max_draw)
        print(f"updated: {args.csv} rows={len(df)} latest_draw={int(df['draw_no'].max())}")
    elif args.__dict__.get("recent_window") is not None:
        preds = Loto6Predictor(seed=args.seed, recent_window=args.recent_window).fit(load_draws(args.csv)).predict(args.n, args.candidates)
        out_df = predictions_to_dataframe(preds)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False, encoding="utf-8")
        print(out_df.to_string(index=False))
    else:
        result_df, summary = walk_forward_backtest(load_draws(args.csv), args.start_after, args.top_n, args.candidates)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out / "backtest_result.csv", index=False, encoding="utf-8")
        pd.DataFrame([summary | {"grade_counts": str(summary["grade_counts"])}]).to_csv(out / "backtest_summary.csv", index=False, encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
