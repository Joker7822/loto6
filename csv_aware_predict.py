# -*- coding: utf-8 -*-
"""
csv_aware_predict.py

リポジトリ内のすべての .csv ファイルを次回予測に活用します。

重複スキップ方針:
  - 実抽せんデータは draw_no 単位で1件に統合
  - backtest_result 系は draw_no + rank + prediction + actual で重複行をスキップ
  - latest_predictions 系は numbers 単位で重複予測をスキップ

利用するCSVの種類:
  1. 実抽せんデータCSV
     - draw_no,date,n1,n2,n3,n4,n5,n6,bonus
     - または 回別,抽せん日,本数字,ボーナス数字
  2. 検証結果CSV
     - prediction,main_matches,grade を含むCSV
  3. 過去予測CSV
     - numbers を含むCSV

出力:
  outputs/latest_predictions.csv
  outputs/latest_predictions_3rd_target.csv
  outputs/csv_aware_prediction_context.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from loto6 import NUMBER_COLUMNS, format_numbers, normalize_draw_dataframe, normalize_numbers

CSV_DRAW_COLUMNS = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
GRADE_WEIGHT = {"1等": 12.0, "2等": 9.0, "3等": 7.0, "4等": 3.0, "5等": 1.0, "はずれ": 0.0}


@dataclass(frozen=True)
class CsvFeatureSet:
    csv_paths: list[str]
    actual_draws: pd.DataFrame
    prediction_number_bonus: dict[int, float]
    prediction_pair_bonus: dict[tuple[int, int], float]
    prior_prediction_bonus: dict[int, float]
    file_summaries: list[dict]
    duplicate_summary: dict[str, int]


def _parse_numbers_text(value: object, expected_min: int = 6) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    nums = [int(x) for x in re.findall(r"\d+", str(value))]
    nums = [n for n in nums if 1 <= n <= 43]
    if len(nums) < expected_min:
        return []
    return nums[:expected_min]


def _read_csv_safely(path: Path) -> pd.DataFrame | None:
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def _extract_actual_draws(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CSV_DRAW_COLUMNS)

    if set(CSV_DRAW_COLUMNS).issubset(df.columns):
        work = df[CSV_DRAW_COLUMNS].copy()
        try:
            return normalize_draw_dataframe(work)
        except Exception:
            return pd.DataFrame(columns=CSV_DRAW_COLUMNS)

    jp_required = {"回別", "抽せん日", "本数字", "ボーナス数字"}
    if jp_required.issubset(df.columns):
        rows = []
        for _, row in df.iterrows():
            main = _parse_numbers_text(row.get("本数字"), expected_min=6)
            bonus_values = _parse_numbers_text(row.get("ボーナス数字"), expected_min=1)
            if len(main) < 6 or not bonus_values:
                continue
            try:
                main = list(normalize_numbers(main[:6]))
            except Exception:
                continue
            parsed_date = pd.to_datetime(row.get("抽せん日"), errors="coerce")
            if pd.isna(parsed_date):
                continue
            try:
                draw_no = int(row.get("回別"))
            except Exception:
                continue
            rows.append(
                {
                    "draw_no": draw_no,
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    **{f"n{i}": main[i - 1] for i in range(1, 7)},
                    "bonus": int(bonus_values[0]),
                }
            )
        if rows:
            try:
                return normalize_draw_dataframe(pd.DataFrame(rows))
            except Exception:
                return pd.DataFrame(columns=CSV_DRAW_COLUMNS)

    return pd.DataFrame(columns=CSV_DRAW_COLUMNS)


def _iter_repo_csvs(root: Path, include_outputs: bool = True) -> list[Path]:
    csvs = []
    for p in root.rglob("*.csv"):
        parts = set(p.parts)
        if ".git" in parts or ".venv" in parts or "venv" in parts:
            continue
        if not include_outputs and "outputs" in parts:
            continue
        csvs.append(p)
    return sorted(csvs)


def _actual_draw_key(row: pd.Series) -> int | None:
    try:
        return int(row["draw_no"])
    except Exception:
        return None


def _prediction_key(nums: Sequence[int]) -> tuple[int, ...]:
    return tuple(normalize_numbers(nums[:6]))


def _performance_key(row: pd.Series, nums: Sequence[int]) -> tuple:
    draw_no = str(row.get("draw_no", ""))
    rank = str(row.get("rank", ""))
    actual = str(row.get("actual", ""))
    return (draw_no, rank, _prediction_key(nums), actual)


def load_csv_features(root: str | Path = ".", include_outputs: bool = True) -> CsvFeatureSet:
    root_path = Path(root)
    csv_paths = _iter_repo_csvs(root_path, include_outputs=include_outputs)
    actual_frames: list[pd.DataFrame] = []
    prediction_number_weight: Counter[int] = Counter()
    prediction_pair_weight: Counter[tuple[int, int]] = Counter()
    prior_prediction_weight: Counter[int] = Counter()
    file_summaries: list[dict] = []

    seen_actual_draws: set[int] = set()
    seen_performance_rows: set[tuple] = set()
    seen_prior_predictions: set[tuple[int, ...]] = set()
    duplicate_summary = {
        "duplicate_actual_draw_rows_skipped": 0,
        "duplicate_performance_rows_skipped": 0,
        "duplicate_prior_prediction_rows_skipped": 0,
    }

    for path in csv_paths:
        df = _read_csv_safely(path)
        if df is None:
            file_summaries.append({"path": str(path), "status": "read_failed"})
            continue

        summary = {
            "path": str(path),
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "used_as": [],
            "duplicates_skipped": {},
        }

        actual = _extract_actual_draws(df)
        if not actual.empty:
            unique_rows = []
            skipped = 0
            for _, row in actual.iterrows():
                key = _actual_draw_key(row)
                if key is None:
                    continue
                if key in seen_actual_draws:
                    skipped += 1
                    continue
                seen_actual_draws.add(key)
                unique_rows.append(row.to_dict())
            if unique_rows:
                actual_frames.append(pd.DataFrame(unique_rows))
                summary["used_as"].append("actual_draws")
                summary["actual_draw_rows"] = int(len(unique_rows))
            if skipped:
                duplicate_summary["duplicate_actual_draw_rows_skipped"] += skipped
                summary["duplicates_skipped"]["actual_draws"] = skipped

        if "prediction" in df.columns and ("main_matches" in df.columns or "grade" in df.columns):
            used_rows = 0
            skipped = 0
            for _, row in df.iterrows():
                nums = _parse_numbers_text(row.get("prediction"), expected_min=6)
                if len(nums) < 6:
                    continue
                try:
                    nums = list(normalize_numbers(nums[:6]))
                except Exception:
                    continue
                key = _performance_key(row, nums)
                if key in seen_performance_rows:
                    skipped += 1
                    continue
                seen_performance_rows.add(key)

                grade = str(row.get("grade", ""))
                try:
                    matches = int(row.get("main_matches", 0))
                except Exception:
                    matches = 0
                weight = max(0.0, matches - 1) + GRADE_WEIGHT.get(grade, 0.0)
                if weight <= 0:
                    continue
                used_rows += 1
                for n in nums:
                    prediction_number_weight[n] += weight
                for pair in combinations(nums, 2):
                    prediction_pair_weight[tuple(sorted(pair))] += weight
            if used_rows:
                summary["used_as"].append("backtest_performance")
                summary["performance_rows"] = used_rows
            if skipped:
                duplicate_summary["duplicate_performance_rows_skipped"] += skipped
                summary["duplicates_skipped"]["backtest_performance"] = skipped

        if "numbers" in df.columns:
            used_rows = 0
            skipped = 0
            for _, row in df.iterrows():
                nums = _parse_numbers_text(row.get("numbers"), expected_min=6)
                if len(nums) < 6:
                    continue
                try:
                    nums_key = _prediction_key(nums[:6])
                except Exception:
                    continue
                if nums_key in seen_prior_predictions:
                    skipped += 1
                    continue
                seen_prior_predictions.add(nums_key)
                used_rows += 1
                for n in nums_key:
                    prior_prediction_weight[n] += 1.0
            if used_rows:
                summary["used_as"].append("prior_predictions")
                summary["prior_prediction_rows"] = used_rows
            if skipped:
                duplicate_summary["duplicate_prior_prediction_rows_skipped"] += skipped
                summary["duplicates_skipped"]["prior_predictions"] = skipped

        file_summaries.append(summary)

    if actual_frames:
        actual_draws = normalize_draw_dataframe(pd.concat(actual_frames, ignore_index=True))
    else:
        actual_draws = pd.DataFrame(columns=CSV_DRAW_COLUMNS)

    def normalize_counter(counter: Counter) -> dict:
        if not counter:
            return {}
        max_v = max(counter.values())
        if max_v <= 0:
            return {}
        return {k: float(v / max_v) for k, v in counter.items()}

    return CsvFeatureSet(
        csv_paths=[str(p) for p in csv_paths],
        actual_draws=actual_draws,
        prediction_number_bonus=normalize_counter(prediction_number_weight),
        prediction_pair_bonus=normalize_counter(prediction_pair_weight),
        prior_prediction_bonus=normalize_counter(prior_prediction_weight),
        file_summaries=file_summaries,
        duplicate_summary=duplicate_summary,
    )


def _scale(values: dict[int, float]) -> dict[int, float]:
    nums = range(1, 44)
    arr = np.array([values.get(n, 0.0) for n in nums], dtype=float)
    if len(arr) == 0 or arr.max() == arr.min():
        return {n: 0.0 for n in nums}
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return {n: float(arr[n - 1]) for n in nums}


def build_number_scores(draws: pd.DataFrame, features: CsvFeatureSet, recent_window: int = 120) -> dict[int, float]:
    draws = normalize_draw_dataframe(draws)
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
    return {
        n: 0.32 * f[n]
        + 0.32 * r[n]
        + 0.16 * g[n]
        + 0.08 * t[n]
        + 0.07 * features.prediction_number_bonus.get(n, 0.0)
        + 0.05 * features.prior_prediction_bonus.get(n, 0.0)
        for n in range(1, 44)
    }


def build_pair_scores(draws: pd.DataFrame, features: CsvFeatureSet, recent_window: int = 240) -> dict[tuple[int, int], float]:
    draws = normalize_draw_dataframe(draws).tail(recent_window)
    cnt: Counter[tuple[int, int]] = Counter()
    for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None):
        nums = list(map(int, row))
        for pair in combinations(nums, 2):
            cnt[tuple(sorted(pair))] += 1.0
    for pair, v in features.prediction_pair_bonus.items():
        cnt[tuple(sorted(pair))] += 0.6 * float(v)
    if not cnt:
        return {}
    max_v = max(cnt.values())
    return {p: float(v / max_v) for p, v in cnt.items()}


def combo_score(combo: Sequence[int], number_scores: dict[int, float], pair_scores: dict[tuple[int, int], float], history: set[tuple[int, ...]]) -> float:
    nums = normalize_numbers(combo)
    base = float(np.mean([number_scores.get(n, 0.0) for n in nums]))
    pair = float(np.mean([pair_scores.get(tuple(sorted(p)), 0.0) for p in combinations(nums, 2)]))
    odd_balance = 1.0 - abs(sum(n % 2 for n in nums) - 3) / 3.0
    sum_balance = max(0.0, 1.0 - abs(sum(nums) - 132) / 90.0)
    zone_balance = len({(n - 1) // 10 for n in nums}) / 5.0
    consecutive_penalty = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1) * 0.025
    duplicate_penalty = 0.20 if nums in history else 0.0
    return 0.50 * base + 0.23 * pair + 0.10 * odd_balance + 0.09 * sum_balance + 0.08 * zone_balance - consecutive_penalty - duplicate_penalty


def generate_standard_predictions(features: CsvFeatureSet, n: int = 5, candidates: int = 5000, seed: int = 42, recent_window: int = 120) -> pd.DataFrame:
    draws = normalize_draw_dataframe(features.actual_draws)
    if draws.empty:
        raise RuntimeError("実抽せんデータCSVを検出できませんでした。")
    number_scores = build_number_scores(draws, features, recent_window=recent_window)
    pair_scores = build_pair_scores(draws, features)
    history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}
    rng = np.random.default_rng(seed)
    numbers = np.array(list(range(1, 44)))
    weights = np.array([number_scores.get(int(x), 0.0) + 0.03 for x in numbers], dtype=float)
    weights = weights / weights.sum()
    ranked_numbers = sorted(range(1, 44), key=lambda x: number_scores.get(x, 0.0), reverse=True)
    candidate_set: set[tuple[int, ...]] = set()
    top_pool = ranked_numbers[:20]
    for combo in combinations(top_pool, 6):
        candidate_set.add(normalize_numbers(combo))
        if len(candidate_set) >= max(100, candidates // 3):
            break
    while len(candidate_set) < candidates:
        pick = rng.choice(numbers, size=6, replace=False, p=weights)
        candidate_set.add(normalize_numbers(map(int, pick)))
    ranked = sorted(((combo_score(c, number_scores, pair_scores, history), c) for c in candidate_set), reverse=True)
    latest = draws.sort_values("draw_no").iloc[-1]
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for rank, (score, combo) in enumerate(ranked[:n], 1):
        rows.append(
            {
                "generated_at_utc": generated_at,
                "strategy": "csv_aware_ensemble",
                "source_csv_files": len(features.csv_paths),
                "source_draws": int(len(draws)),
                "trained_through_draw_no": int(latest["draw_no"]),
                "trained_through_date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
                "target": f"after_draw_{int(latest['draw_no'])}",
                "rank": rank,
                "numbers": format_numbers(combo),
                "score": round(float(score), 6),
                **{f"n{i}": x for i, x in enumerate(combo, 1)},
            }
        )
    return pd.DataFrame(rows)


def _core_score(core: Sequence[int], number_scores: dict[int, float], pair_scores: dict[tuple[int, int], float]) -> float:
    core = tuple(sorted(int(n) for n in core))
    base = float(np.mean([number_scores.get(n, 0.0) for n in core]))
    pair = float(np.mean([pair_scores.get(tuple(sorted(p)), 0.0) for p in combinations(core, 2)]))
    odd_balance = 1.0 - abs(sum(n % 2 for n in core) - 2.5) / 2.5
    zone_balance = len({(n - 1) // 10 for n in core}) / 5.0
    return 0.62 * base + 0.25 * pair + 0.07 * odd_balance + 0.06 * zone_balance


def generate_third_prize_predictions(features: CsvFeatureSet, n: int = 5, seed: int = 42, recent_window: int = 120, core_pool_size: int = 18, cover_pool_size: int = 32) -> pd.DataFrame:
    draws = normalize_draw_dataframe(features.actual_draws)
    if draws.empty:
        raise RuntimeError("実抽せんデータCSVを検出できませんでした。")
    number_scores = build_number_scores(draws, features, recent_window=recent_window)
    pair_scores = build_pair_scores(draws, features)
    history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}
    ranked_numbers = sorted(range(1, 44), key=lambda x: number_scores.get(x, 0.0), reverse=True)
    core_pool = ranked_numbers[: max(5, core_pool_size)]
    cover_pool = ranked_numbers[: max(6, cover_pool_size)]
    cores = []
    for core in combinations(core_pool, 5):
        score = _core_score(core, number_scores, pair_scores)
        core_set = set(core)
        historical_core_hits = sum(1 for h in history if len(core_set & set(h)) >= 5)
        score -= min(historical_core_hits, 5) * 0.01
        cores.append((score, tuple(sorted(core))))
    cores.sort(reverse=True)
    latest = draws.sort_values("draw_no").iloc[-1]
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    used: set[tuple[int, ...]] = set()
    for core_score_value, core in cores:
        covers = [x for x in cover_pool if x not in core]
        cover_ranked = []
        for cover in covers:
            ticket = normalize_numbers([*core, cover])
            score = 0.74 * core_score_value + 0.16 * combo_score(ticket, number_scores, pair_scores, history) + 0.10 * number_scores.get(cover, 0.0)
            if ticket in history:
                score -= 0.12
            cover_ranked.append((score, cover, ticket))
        cover_ranked.sort(reverse=True)
        for score, cover, ticket in cover_ranked:
            if ticket in used:
                continue
            used.add(ticket)
            rows.append(
                {
                    "generated_at_utc": generated_at,
                    "strategy": "csv_aware_third_prize_core_cover",
                    "target_grade": "3等",
                    "source_csv_files": len(features.csv_paths),
                    "source_draws": int(len(draws)),
                    "trained_through_draw_no": int(latest["draw_no"]),
                    "trained_through_date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
                    "target": f"after_draw_{int(latest['draw_no'])}",
                    "rank": len(rows) + 1,
                    "numbers": format_numbers(ticket),
                    "core5": " ".join(f"{x:02d}" for x in core),
                    "cover_number": f"{int(cover):02d}",
                    "score": round(float(score), 6),
                    "core_score": round(float(core_score_value), 6),
                    **{f"n{i}": x for i, x in enumerate(ticket, 1)},
                }
            )
            if len(rows) >= n:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def write_outputs(repo_root: str, standard_output: str, third_output: str, context_output: str, n: int, candidates: int, seed: int, include_outputs: bool) -> None:
    features = load_csv_features(repo_root, include_outputs=include_outputs)
    standard = generate_standard_predictions(features, n=n, candidates=candidates, seed=seed)
    third = generate_third_prize_predictions(features, n=n, seed=seed)
    Path(standard_output).parent.mkdir(parents=True, exist_ok=True)
    standard.to_csv(standard_output, index=False, encoding="utf-8")
    third.to_csv(third_output, index=False, encoding="utf-8")
    context = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "csv_aware_prediction",
        "standard_output": standard_output,
        "third_prize_output": third_output,
        "source_csv_files": features.csv_paths,
        "source_csv_file_count": len(features.csv_paths),
        "actual_draws": int(len(features.actual_draws)),
        "number_bonus_count": len(features.prediction_number_bonus),
        "pair_bonus_count": len(features.prediction_pair_bonus),
        "prior_prediction_bonus_count": len(features.prior_prediction_bonus),
        "duplicate_summary": features.duplicate_summary,
        "file_summaries": features.file_summaries,
        "note": "All readable CSV files are inspected, but duplicated draw rows, duplicated backtest rows, and duplicated prior prediction rows are skipped before weighting.",
    }
    Path(context_output).write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[STANDARD]")
    print(standard.to_string(index=False))
    print("[THIRD_PRIZE_TARGET]")
    print(third.to_string(index=False))
    print("[DUPLICATES]")
    print(json.dumps(features.duplicate_summary, ensure_ascii=False, indent=2))
    print(f"[CONTEXT] {context_output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Loto6 predictions using all CSV files in the repository.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--standard-output", default="outputs/latest_predictions.csv")
    parser.add_argument("--third-output", default="outputs/latest_predictions_3rd_target.csv")
    parser.add_argument("--context-output", default="outputs/csv_aware_prediction_context.json")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-outputs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    write_outputs(
        repo_root=args.repo_root,
        standard_output=args.standard_output,
        third_output=args.third_output,
        context_output=args.context_output,
        n=args.n,
        candidates=args.candidates,
        seed=args.seed,
        include_outputs=args.include_outputs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
