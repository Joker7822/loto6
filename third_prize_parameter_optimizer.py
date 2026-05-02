# -*- coding: utf-8 -*-
"""
third_prize_parameter_optimizer.py

3等狙いのbacktest結果を読み、次回予測用パラメータを自動最適化します。

入力:
  outputs/backtest_3rd_target_result.csv
  data/loto6.csv

出力:
  outputs/third_prize_optimized_config.json
  outputs/third_prize_optimization_report.csv

最適化対象:
  - recent_window
  - core_pool_size
  - cover_pool_size
  - core_count
  - core_overlap_limit
  - allocation
  - history_penalty
  - cover_reuse_penalty
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = {
    "recent_window": 120,
    "core_pool_size": 22,
    "cover_pool_size": 34,
    "core_count": 3,
    "core_overlap_limit": 3,
    "allocation": [2, 2, 1],
    "history_penalty": 0.12,
    "cover_reuse_penalty": 0.0,
    "seed": 42,
}


def _safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def _score_window(df: pd.DataFrame, window: int | None) -> dict:
    work = df.copy()
    if work.empty:
        return {
            "window": window or 0,
            "rows": 0,
            "draws": 0,
            "score": 0.0,
            "third_or_better": 0,
            "core5_4plus": 0,
            "max_core5": 0,
            "max_main": 0,
            "near_third_draws": 0,
            "fourth_count": 0,
            "fifth_count": 0,
        }
    work = work.sort_values(["draw_no", "rank"])
    if window is not None and window > 0:
        latest_draws = sorted(work["draw_no"].dropna().astype(int).unique())[-window:]
        work = work[work["draw_no"].astype(int).isin(latest_draws)]

    grade = work.get("grade", pd.Series(dtype=str)).astype(str)
    core5 = pd.to_numeric(work.get("core5_matches", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)
    main = pd.to_numeric(work.get("main_matches", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)
    per_draw = work.assign(core5_matches=core5, main_matches=main).groupby("draw_no").agg(
        max_core5=("core5_matches", "max"),
        max_main=("main_matches", "max"),
    )

    third_or_better = int(grade.isin(["1等", "2等", "3等"]).sum())
    fourth_count = int((grade == "4等").sum())
    fifth_count = int((grade == "5等").sum())
    core5_4plus = int((core5 >= 4).sum())
    near_third_draws = int((per_draw["max_core5"] >= 4).sum()) if not per_draw.empty else 0
    max_core5 = int(core5.max()) if len(core5) else 0
    max_main = int(main.max()) if len(main) else 0

    # 3等到達を最重視し、core5 4個一致と4等を次点評価する。
    score = (
        third_or_better * 120.0
        + near_third_draws * 35.0
        + core5_4plus * 18.0
        + fourth_count * 10.0
        + fifth_count * 1.5
        + max_core5 * 8.0
        + max_main * 5.0
    )
    draws = int(work["draw_no"].nunique()) if not work.empty else 0
    score_per_draw = score / max(1, draws)
    return {
        "window": window or 0,
        "rows": int(len(work)),
        "draws": draws,
        "score": round(float(score), 6),
        "score_per_draw": round(float(score_per_draw), 6),
        "third_or_better": third_or_better,
        "core5_4plus": core5_4plus,
        "max_core5": max_core5,
        "max_main": max_main,
        "near_third_draws": near_third_draws,
        "fourth_count": fourth_count,
        "fifth_count": fifth_count,
    }


def _derive_config(result_df: pd.DataFrame, draw_df: pd.DataFrame, report_df: pd.DataFrame) -> dict:
    config = dict(DEFAULT_CONFIG)
    if result_df.empty:
        config["reason"] = "no_backtest_result; using default config"
        return config

    # 直近の安定性を見て recent_window を選ぶ。
    candidates = report_df[report_df["window"] > 0].copy()
    candidates = candidates[candidates["draws"] >= 60]
    if not candidates.empty:
        best = candidates.sort_values(["score_per_draw", "near_third_draws", "third_or_better"], ascending=False).iloc[0]
        config["recent_window"] = int(best["window"])
        best_window_score = float(best["score_per_draw"])
    else:
        best_window_score = 0.0

    total = _score_window(result_df, None)
    recent120 = _score_window(result_df, 120)
    recent240 = _score_window(result_df, 240)

    # core5 4一致が少ない場合は、より広い候補プールで探索する。
    if total["core5_4plus"] < 15 or total["near_third_draws"] < 5:
        config["core_pool_size"] = 28
        config["cover_pool_size"] = 40
        config["core_overlap_limit"] = 3
        config["reason"] = "core5_4plus_sparse; increase core and cover diversity"
    else:
        config["core_pool_size"] = 24
        config["cover_pool_size"] = 36
        config["core_overlap_limit"] = 3
        config["reason"] = "core5_4plus_detected; keep balanced diversity"

    # 直近で5等/4等が弱い場合はCコア比率を少し増やす。ただし5口では基本2/2/1。
    if recent120["fourth_count"] == 0 and recent120["fifth_count"] < max(1, recent120["draws"] // 12):
        config["allocation"] = [2, 1, 2]
        config["reason"] += "; recent_weakness_detected; increase C-core allocation"
    else:
        config["allocation"] = [2, 2, 1]

    # max_mainが5に到達済みなら、完全に分散しすぎず既存強パターンを残す。
    if total["max_main"] >= 5:
        config["history_penalty"] = 0.08
        config["cover_reuse_penalty"] = 0.01
        config["reason"] += "; main5_found; reduce history penalty slightly"
    else:
        config["history_penalty"] = 0.12
        config["cover_reuse_penalty"] = 0.02

    config["optimized_metrics"] = {
        "total": total,
        "recent120": recent120,
        "recent240": recent240,
        "best_window_score_per_draw": best_window_score,
        "source_draws": int(len(draw_df)) if not draw_df.empty else None,
        "latest_draw_no": int(draw_df["draw_no"].max()) if not draw_df.empty and "draw_no" in draw_df.columns else None,
    }
    return config


def optimize(
    result_csv: str = "outputs/backtest_3rd_target_result.csv",
    draw_csv: str = "data/loto6.csv",
    config_output: str = "outputs/third_prize_optimized_config.json",
    report_output: str = "outputs/third_prize_optimization_report.csv",
) -> tuple[dict, pd.DataFrame]:
    result_df = _safe_read_csv(result_csv)
    draw_df = _safe_read_csv(draw_csv)

    windows = [80, 120, 180, 240, 360, 520, None]
    report = pd.DataFrame([_score_window(result_df, w) for w in windows])
    config = _derive_config(result_df, draw_df, report)
    config["optimized_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["result_csv"] = result_csv
    config["draw_csv"] = draw_csv
    config["report_csv"] = report_output

    Path(config_output).parent.mkdir(parents=True, exist_ok=True)
    Path(config_output).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    report.to_csv(report_output, index=False, encoding="utf-8")
    return config, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize Loto6 third-prize prediction parameters from backtest results.")
    parser.add_argument("--result-csv", default="outputs/backtest_3rd_target_result.csv")
    parser.add_argument("--draw-csv", default="data/loto6.csv")
    parser.add_argument("--config-output", default="outputs/third_prize_optimized_config.json")
    parser.add_argument("--report-output", default="outputs/third_prize_optimization_report.csv")
    args = parser.parse_args()
    config, report = optimize(
        result_csv=args.result_csv,
        draw_csv=args.draw_csv,
        config_output=args.config_output,
        report_output=args.report_output,
    )
    print("[OPTIMIZED_CONFIG]")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("[OPTIMIZATION_REPORT]")
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
