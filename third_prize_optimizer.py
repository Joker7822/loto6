# -*- coding: utf-8 -*-
"""
third_prize_optimizer.py

Loto6の3等狙いを強化する専用モジュールです。

目的:
  - 3等 = 本数字5個一致を狙うため、6数字全体ではなく5数字コアを評価する
  - 5口出力時は Aコア2口 / Bコア2口 / Cコア1口 に分散する
  - walk-forward で未来リークなしの3等専用backtestを出力する

出力:
  outputs/latest_predictions_3rd_target.csv
  outputs/latest_predictions_3rd_target_context.json
  outputs/backtest_3rd_target_result.csv
  outputs/backtest_3rd_target_summary.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from csv_aware_predict import (
    CsvFeatureSet,
    build_number_scores,
    build_pair_scores,
    combo_score,
    load_csv_features,
)
from loto6 import NUMBER_COLUMNS, classify_loto6, format_numbers, load_draws, normalize_draw_dataframe, normalize_numbers


@dataclass(frozen=True)
class CoreCandidate:
    core5: tuple[int, ...]
    score: float
    pair_score: float
    number_score: float
    recent_hit_penalty: float


@dataclass(frozen=True)
class ThirdTicket:
    rank: int
    numbers: tuple[int, ...]
    core5: tuple[int, ...]
    cover_number: int
    core_label: str
    score: float
    core_score: float
    cover_score: float


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _history_sets(draws: pd.DataFrame) -> list[set[int]]:
    return [set(map(int, row)) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)]


def core5_score(
    core: Sequence[int],
    number_scores: dict[int, float],
    pair_scores: dict[tuple[int, int], float],
    history: list[set[int]],
    recent_history: list[set[int]],
) -> CoreCandidate:
    core5 = tuple(sorted(int(n) for n in core))
    number_part = _mean([number_scores.get(n, 0.0) for n in core5])
    pair_part = _mean([pair_scores.get(tuple(sorted(p)), 0.0) for p in combinations(core5, 2)])
    odd_balance = 1.0 - abs(sum(n % 2 for n in core5) - 2.5) / 2.5
    zone_balance = len({(n - 1) // 10 for n in core5}) / 5.0
    spread_balance = max(0.0, 1.0 - abs((max(core5) - min(core5)) - 28) / 30.0)

    # 過去・直近で5個一致相当の同型に寄りすぎるコアは少し減点。
    historical_5plus = sum(1 for h in history if len(set(core5) & h) >= 5)
    recent_4plus = sum(1 for h in recent_history if len(set(core5) & h) >= 4)
    penalty = min(historical_5plus, 5) * 0.012 + min(recent_4plus, 5) * 0.008

    score = 0.56 * number_part + 0.27 * pair_part + 0.07 * odd_balance + 0.06 * zone_balance + 0.04 * spread_balance - penalty
    return CoreCandidate(core5=core5, score=float(score), pair_score=float(pair_part), number_score=float(number_part), recent_hit_penalty=float(penalty))


def select_diverse_cores(
    draws: pd.DataFrame,
    features: CsvFeatureSet,
    recent_window: int = 120,
    core_pool_size: int = 20,
    selected_count: int = 3,
) -> list[CoreCandidate]:
    draws = normalize_draw_dataframe(draws)
    number_scores = build_number_scores(draws, features, recent_window=recent_window)
    pair_scores = build_pair_scores(draws, features, recent_window=max(240, recent_window * 2))
    history = _history_sets(draws)
    recent_history = _history_sets(draws.tail(min(recent_window, len(draws))))

    ranked_numbers = sorted(range(1, 44), key=lambda n: number_scores.get(n, 0.0), reverse=True)
    pool = ranked_numbers[: max(5, core_pool_size)]
    candidates = [core5_score(core, number_scores, pair_scores, history, recent_history) for core in combinations(pool, 5)]
    candidates.sort(key=lambda c: c.score, reverse=True)

    selected: list[CoreCandidate] = []
    for cand in candidates:
        if not selected:
            selected.append(cand)
        else:
            # A/B/Cコアを似せすぎない。5数字中4個以上重複するコアは原則避ける。
            max_overlap = max(len(set(cand.core5) & set(s.core5)) for s in selected)
            if max_overlap <= 3:
                selected.append(cand)
        if len(selected) >= selected_count:
            break

    # 候補が足りない場合だけ制約を緩める。
    if len(selected) < selected_count:
        for cand in candidates:
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= selected_count:
                break
    return selected[:selected_count]


def generate_third_prize_tickets(
    draws: pd.DataFrame,
    features: CsvFeatureSet,
    n: int = 5,
    seed: int = 42,
    recent_window: int = 120,
    core_pool_size: int = 20,
    cover_pool_size: int = 34,
) -> list[ThirdTicket]:
    draws = normalize_draw_dataframe(draws)
    if draws.empty:
        raise RuntimeError("No draw data available for third-prize prediction.")

    number_scores = build_number_scores(draws, features, recent_window=recent_window)
    pair_scores = build_pair_scores(draws, features, recent_window=max(240, recent_window * 2))
    history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}
    selected_cores = select_diverse_cores(draws, features, recent_window=recent_window, core_pool_size=core_pool_size, selected_count=3)

    ranked_numbers = sorted(range(1, 44), key=lambda x: number_scores.get(x, 0.0), reverse=True)
    cover_pool = ranked_numbers[: max(6, cover_pool_size)]
    allocation = [2, 2, 1] if n == 5 else []
    if not allocation:
        base = n // max(1, len(selected_cores))
        allocation = [base] * len(selected_cores)
        for i in range(n - sum(allocation)):
            allocation[i % len(allocation)] += 1

    labels = ["A", "B", "C", "D", "E"]
    tickets: list[ThirdTicket] = []
    used_tickets: set[tuple[int, ...]] = set()
    used_covers_by_core: dict[tuple[int, ...], set[int]] = {}

    for core_index, core in enumerate(selected_cores):
        label = labels[core_index] if core_index < len(labels) else f"C{core_index+1}"
        need = allocation[core_index] if core_index < len(allocation) else 1
        used_covers_by_core.setdefault(core.core5, set())
        cover_candidates = []
        for cover in cover_pool:
            if cover in core.core5 or cover in used_covers_by_core[core.core5]:
                continue
            ticket = normalize_numbers([*core.core5, cover])
            if ticket in used_tickets:
                continue
            cover_score = number_scores.get(cover, 0.0)
            full_score = combo_score(ticket, number_scores, pair_scores, history)
            duplicate_penalty = 0.10 if ticket in history else 0.0
            score = 0.70 * core.score + 0.18 * full_score + 0.12 * cover_score - duplicate_penalty
            cover_candidates.append((float(score), int(cover), ticket, float(cover_score)))
        cover_candidates.sort(reverse=True)

        for score, cover, ticket, cover_score in cover_candidates[:need]:
            tickets.append(
                ThirdTicket(
                    rank=len(tickets) + 1,
                    numbers=ticket,
                    core5=core.core5,
                    cover_number=cover,
                    core_label=label,
                    score=score,
                    core_score=core.score,
                    cover_score=cover_score,
                )
            )
            used_tickets.add(ticket)
            used_covers_by_core[core.core5].add(cover)
            if len(tickets) >= n:
                return tickets

    return tickets[:n]


def tickets_to_dataframe(tickets: Sequence[ThirdTicket], draws: pd.DataFrame, features: CsvFeatureSet) -> pd.DataFrame:
    latest = normalize_draw_dataframe(draws).sort_values("draw_no").iloc[-1]
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for t in tickets:
        rows.append(
            {
                "generated_at_utc": generated_at,
                "strategy": "third_prize_core_diversified_A2_B2_C1",
                "target_grade": "3等",
                "source_csv_files": len(features.csv_paths),
                "source_draws": int(len(draws)),
                "trained_through_draw_no": int(latest["draw_no"]),
                "trained_through_date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
                "target": f"after_draw_{int(latest['draw_no'])}",
                "rank": t.rank,
                "core_label": t.core_label,
                "numbers": format_numbers(t.numbers),
                "core5": " ".join(f"{x:02d}" for x in t.core5),
                "cover_number": f"{t.cover_number:02d}",
                "score": round(t.score, 6),
                "core_score": round(t.core_score, 6),
                "cover_score": round(t.cover_score, 6),
                **{f"n{i}": x for i, x in enumerate(t.numbers, 1)},
            }
        )
    return pd.DataFrame(rows)


def write_latest_predictions(
    repo_root: str = ".",
    output: str = "outputs/latest_predictions_3rd_target.csv",
    context: str = "outputs/latest_predictions_3rd_target_context.json",
    n: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    features = load_csv_features(repo_root, include_outputs=True)
    draws = normalize_draw_dataframe(features.actual_draws)
    tickets = generate_third_prize_tickets(draws, features, n=n, seed=seed)
    out_df = tickets_to_dataframe(tickets, draws, features)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False, encoding="utf-8")
    latest = draws.sort_values("draw_no").iloc[-1]
    ctx = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "third_prize_core_diversified_A2_B2_C1",
        "output": output,
        "source_draws": int(len(draws)),
        "source_csv_files": features.csv_paths,
        "trained_through_draw_no": int(latest["draw_no"]),
        "trained_through_date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
        "core_allocation": "A:2, B:2, C:1",
        "note": "3rd-prize-oriented output. It prioritizes diversified 5-number cores and does not guarantee winning.",
    }
    Path(context).write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_df.to_string(index=False))
    return out_df


def evaluate_ticket(ticket: Sequence[int], core5: Sequence[int], actual_main: Sequence[int], bonus: int) -> dict:
    result = classify_loto6(ticket, actual_main, bonus)
    core_matches = len(set(core5) & set(actual_main))
    return {
        "main_matches": int(result.main_matches),
        "bonus_match": bool(result.bonus_match),
        "grade": result.grade,
        "core5_matches": int(core_matches),
        "core5_5hit": bool(core_matches == 5),
        "core5_4plus": bool(core_matches >= 4),
    }


def backtest_third_prize(
    csv_path: str = "data/loto6.csv",
    output_dir: str = "outputs",
    n: int = 5,
    min_train_draws: int = 1,
    max_draws: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    draws = load_draws(csv_path)
    draws = normalize_draw_dataframe(draws)
    if len(draws) <= min_train_draws:
        raise ValueError(f"Need more than min_train_draws={min_train_draws}; got {len(draws)}")

    records = []
    target_indexes = list(range(min_train_draws, len(draws)))
    if max_draws is not None:
        target_indexes = target_indexes[:max_draws]

    for idx in target_indexes:
        train_df = draws.iloc[:idx].copy()
        actual = draws.iloc[idx]
        draw_no = int(actual["draw_no"])
        if draw_no in set(train_df["draw_no"].astype(int)):
            raise RuntimeError(f"Future leakage guard failed: draw_no={draw_no}")

        # backtest中は未来のoutputsを混ぜない。学習対象はtrain_dfだけに限定する。
        features = CsvFeatureSet(
            csv_paths=[csv_path],
            actual_draws=train_df,
            prediction_number_bonus={},
            prediction_pair_bonus={},
            prior_prediction_bonus={},
            file_summaries=[],
            duplicate_summary={},
        )
        tickets = generate_third_prize_tickets(train_df, features, n=n, seed=seed + draw_no)
        actual_main = [int(actual[c]) for c in NUMBER_COLUMNS]
        bonus = int(actual["bonus"])

        for t in tickets:
            ev = evaluate_ticket(t.numbers, t.core5, actual_main, bonus)
            records.append(
                {
                    "draw_no": draw_no,
                    "date": actual["date"].date().isoformat(),
                    "rank": t.rank,
                    "core_label": t.core_label,
                    "prediction": format_numbers(t.numbers),
                    "core5": " ".join(f"{x:02d}" for x in t.core5),
                    "cover_number": f"{t.cover_number:02d}",
                    "actual": " ".join(f"{x:02d}" for x in actual_main),
                    "bonus": f"{bonus:02d}",
                    "train_draws": int(len(train_df)),
                    "train_until_draw_no": int(train_df["draw_no"].max()),
                    "score": round(t.score, 6),
                    "core_score": round(t.core_score, 6),
                    **ev,
                }
            )
        print(f"third_backtest draw={draw_no} train={len(train_df)} records={len(records)}")

    result_df = pd.DataFrame(records)
    summary = summarize_third_backtest(result_df, total_draws=len(draws), n=n, min_train_draws=min_train_draws)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out / "backtest_3rd_target_result.csv", index=False, encoding="utf-8")
    pd.DataFrame([summary | {"grade_counts": json.dumps(summary.get("grade_counts", {}), ensure_ascii=False)}]).to_csv(
        out / "backtest_3rd_target_summary.csv", index=False, encoding="utf-8"
    )
    (out / "backtest_3rd_target_progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result_df, summary


def summarize_third_backtest(result_df: pd.DataFrame, total_draws: int, n: int, min_train_draws: int) -> dict:
    if result_df.empty:
        return {
            "mode": "third_prize_core_diversified_A2_B2_C1",
            "total_source_draws": int(total_draws),
            "min_train_draws": int(min_train_draws),
            "completed_draws": 0,
            "tickets": 0,
            "top_n": int(n),
            "max_main_matches": 0,
            "max_core5_matches": 0,
            "core5_5hit_count": 0,
            "core5_4plus_count": 0,
            "third_or_better_count": 0,
            "grade_counts": {},
        }
    grade_counts = dict(Counter(result_df["grade"]))
    third_or_better = int(result_df["grade"].isin(["1等", "2等", "3等"]).sum())
    return {
        "mode": "third_prize_core_diversified_A2_B2_C1",
        "total_source_draws": int(total_draws),
        "min_train_draws": int(min_train_draws),
        "completed_draws": int(result_df["draw_no"].nunique()),
        "tickets": int(len(result_df)),
        "top_n": int(n),
        "max_main_matches": int(result_df["main_matches"].max()),
        "max_core5_matches": int(result_df["core5_matches"].max()),
        "core5_5hit_count": int(result_df["core5_5hit"].sum()),
        "core5_4plus_count": int(result_df["core5_4plus"].sum()),
        "third_or_better_count": third_or_better,
        "grade_counts": grade_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Loto6 third-prize optimizer and backtester")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("predict")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output", default="outputs/latest_predictions_3rd_target.csv")
    p.add_argument("--context", default="outputs/latest_predictions_3rd_target_context.json")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("backtest")
    p.add_argument("--csv", default="data/loto6.csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--min-train-draws", type=int, default=1)
    p.add_argument("--max-draws", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if hasattr(args, "repo_root"):
        write_latest_predictions(args.repo_root, args.output, args.context, args.n, args.seed)
    else:
        backtest_third_prize(args.csv, args.output_dir, args.n, args.min_train_draws, args.max_draws, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
