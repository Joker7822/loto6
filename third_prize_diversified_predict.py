# -*- coding: utf-8 -*-
"""
third_prize_diversified_predict.py

3等狙いの5数字コアを A/B/C に分散して出力します。

既定の5口構成:
  - Aコア: 2口
  - Bコア: 2口
  - Cコア: 1口

目的:
  1つの5数字コアに寄せすぎて全滅するリスクを下げる。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from csv_aware_predict import (
    build_number_scores,
    build_pair_scores,
    combo_score,
    load_csv_features,
)
from loto6 import NUMBER_COLUMNS, format_numbers, normalize_draw_dataframe, normalize_numbers


def _core_score(core, number_scores, pair_scores) -> float:
    core = tuple(sorted(int(n) for n in core))
    base = float(np.mean([number_scores.get(n, 0.0) for n in core]))
    pair = float(np.mean([pair_scores.get(tuple(sorted(p)), 0.0) for p in combinations(core, 2)]))
    odd_balance = 1.0 - abs(sum(n % 2 for n in core) - 2.5) / 2.5
    zone_balance = len({(n - 1) // 10 for n in core}) / 5.0
    spread_balance = max(0.0, 1.0 - abs((max(core) - min(core)) - 28) / 30.0)
    return 0.58 * base + 0.24 * pair + 0.07 * odd_balance + 0.06 * zone_balance + 0.05 * spread_balance


def _overlap(a, b) -> int:
    return len(set(a) & set(b))


def _select_diverse_cores(core_ranked: list[tuple[float, tuple[int, ...]]], core_count: int = 3) -> list[tuple[float, tuple[int, ...]]]:
    selected: list[tuple[float, tuple[int, ...]]] = []
    for score, core in core_ranked:
        if not selected:
            selected.append((score, core))
            continue
        # A/B/Cを別物にするため、5数字コア同士の重複は最大3個までを優先。
        if all(_overlap(core, chosen) <= 3 for _, chosen in selected):
            selected.append((score, core))
        if len(selected) >= core_count:
            return selected

    # 条件が厳しすぎて足りない場合のみ、重複4個まで許容する。
    for score, core in core_ranked:
        if any(core == chosen for _, chosen in selected):
            continue
        if all(_overlap(core, chosen) <= 4 for _, chosen in selected):
            selected.append((score, core))
        if len(selected) >= core_count:
            return selected
    return selected


def generate_diversified_third_predictions(
    repo_root: str = ".",
    n: int = 5,
    seed: int = 42,
    include_outputs: bool = True,
    recent_window: int = 120,
    core_pool_size: int = 22,
    cover_pool_size: int = 34,
) -> tuple[pd.DataFrame, dict]:
    features = load_csv_features(repo_root, include_outputs=include_outputs)
    draws = normalize_draw_dataframe(features.actual_draws)
    if draws.empty:
        raise RuntimeError("実抽せんデータCSVを検出できませんでした。")

    number_scores = build_number_scores(draws, features, recent_window=recent_window)
    pair_scores = build_pair_scores(draws, features)
    history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}

    ranked_numbers = sorted(range(1, 44), key=lambda x: number_scores.get(x, 0.0), reverse=True)
    core_pool = ranked_numbers[: max(5, core_pool_size)]
    cover_pool = ranked_numbers[: max(6, cover_pool_size)]

    core_ranked: list[tuple[float, tuple[int, ...]]] = []
    for core in combinations(core_pool, 5):
        core = tuple(sorted(core))
        score = _core_score(core, number_scores, pair_scores)
        historical_core_hits = sum(1 for h in history if len(set(core) & set(h)) >= 5)
        score -= min(historical_core_hits, 5) * 0.01
        core_ranked.append((float(score), core))
    core_ranked.sort(reverse=True)
    selected_cores = _select_diverse_cores(core_ranked, core_count=3)

    allocation = [2, 2, 1]
    if n != 5:
        allocation = [max(1, n // 3), max(1, n // 3), max(1, n - 2 * max(1, n // 3))]

    rows = []
    used_tickets: set[tuple[int, ...]] = set()
    latest = draws.sort_values("draw_no").iloc[-1]
    generated_at = datetime.now(timezone.utc).isoformat()

    for core_index, (core_score_value, core) in enumerate(selected_cores):
        quota = allocation[core_index] if core_index < len(allocation) else 1
        cover_ranked = []
        for cover in cover_pool:
            if cover in core:
                continue
            ticket = normalize_numbers([*core, cover])
            score = (
                0.74 * core_score_value
                + 0.16 * combo_score(ticket, number_scores, pair_scores, history)
                + 0.10 * number_scores.get(int(cover), 0.0)
            )
            if ticket in history:
                score -= 0.12
            cover_ranked.append((float(score), int(cover), ticket))
        cover_ranked.sort(reverse=True)

        emitted_for_core = 0
        for score, cover, ticket in cover_ranked:
            if ticket in used_tickets:
                continue
            used_tickets.add(ticket)
            emitted_for_core += 1
            rows.append(
                {
                    "generated_at_utc": generated_at,
                    "strategy": "csv_aware_third_prize_core_cover_diversified",
                    "target_grade": "3等",
                    "core_group": chr(ord("A") + core_index),
                    "source_csv_files": len(features.csv_paths),
                    "source_draws": int(len(draws)),
                    "trained_through_draw_no": int(latest["draw_no"]),
                    "trained_through_date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
                    "target": f"after_draw_{int(latest['draw_no'])}",
                    "rank": len(rows) + 1,
                    "numbers": format_numbers(ticket),
                    "core5": " ".join(f"{x:02d}" for x in core),
                    "cover_number": f"{cover:02d}",
                    "score": round(score, 6),
                    "core_score": round(core_score_value, 6),
                    **{f"n{i}": x for i, x in enumerate(ticket, 1)},
                }
            )
            if emitted_for_core >= quota or len(rows) >= n:
                break
        if len(rows) >= n:
            break

    out = pd.DataFrame(rows)
    context = {
        "generated_at_utc": generated_at,
        "strategy": "csv_aware_third_prize_core_cover_diversified",
        "allocation": allocation,
        "selected_cores": [
            {"core_group": chr(ord("A") + i), "core5": " ".join(f"{x:02d}" for x in core), "core_score": round(score, 6)}
            for i, (score, core) in enumerate(selected_cores)
        ],
        "source_csv_files": features.csv_paths,
        "source_csv_file_count": len(features.csv_paths),
        "source_draws": int(len(draws)),
        "trained_through_draw_no": int(latest["draw_no"]),
        "trained_through_date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
        "duplicate_summary": features.duplicate_summary,
        "note": "3等狙いの5数字コアをA/B/Cへ分散。既定はAコア2口、Bコア2口、Cコア1口。",
    }
    return out, context


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate diversified Loto6 third-prize-target predictions.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="outputs/latest_predictions_3rd_target.csv")
    parser.add_argument("--context", default="outputs/latest_predictions_3rd_target_context.json")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-outputs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    df, context = generate_diversified_third_predictions(
        repo_root=args.repo_root,
        n=args.n,
        seed=args.seed,
        include_outputs=args.include_outputs,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    Path(args.context).write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(context, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
