# -*- coding: utf-8 -*-
"""
core5_performance.py

3等狙いbacktest結果から、5数字コア(core5)単位の成績を集計します。

入力:
  outputs/backtest_3rd_target_result.csv

出力:
  outputs/core5_performance.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_core5_performance(result_csv: str = "outputs/backtest_3rd_target_result.csv") -> pd.DataFrame:
    path = Path(result_csv)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {"core5", "core5_matches", "main_matches", "grade", "draw_no"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {result_csv}: {sorted(missing)}")

    df = df.copy()
    df["core5_matches"] = pd.to_numeric(df["core5_matches"], errors="coerce").fillna(0).astype(int)
    df["main_matches"] = pd.to_numeric(df["main_matches"], errors="coerce").fillna(0).astype(int)
    df["is_3rd_or_better"] = df["grade"].astype(str).isin(["1等", "2等", "3等"]).astype(int)
    df["is_4th"] = (df["grade"].astype(str) == "4等").astype(int)
    df["is_5th"] = (df["grade"].astype(str) == "5等").astype(int)
    df["core5_4plus"] = (df["core5_matches"] >= 4).astype(int)
    df["core5_5hit"] = (df["core5_matches"] >= 5).astype(int)

    rows = []
    for core5, g in df.groupby("core5", dropna=False):
        total = len(g)
        draws = g["draw_no"].nunique()
        score = (
            int(g["is_3rd_or_better"].sum()) * 120
            + int(g["core5_5hit"].sum()) * 80
            + int(g["core5_4plus"].sum()) * 30
            + int(g["is_4th"].sum()) * 10
            + int(g["is_5th"].sum()) * 2
            + int(g["core5_matches"].max()) * 8
            + int(g["main_matches"].max()) * 5
        )
        rows.append(
            {
                "core5": core5,
                "tickets": int(total),
                "draws": int(draws),
                "score": float(score),
                "score_per_ticket": round(float(score) / max(1, total), 8),
                "score_per_draw": round(float(score) / max(1, draws), 8),
                "third_or_better_count": int(g["is_3rd_or_better"].sum()),
                "fourth_count": int(g["is_4th"].sum()),
                "fifth_count": int(g["is_5th"].sum()),
                "core5_5hit_count": int(g["core5_5hit"].sum()),
                "core5_4plus_count": int(g["core5_4plus"].sum()),
                "max_core5_matches": int(g["core5_matches"].max()),
                "max_main_matches": int(g["main_matches"].max()),
                "latest_draw_no": int(g["draw_no"].max()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["third_or_better_count", "core5_4plus_count", "score_per_draw", "score"],
        ascending=False,
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build core5 performance summary from third-prize backtest result.")
    parser.add_argument("--result-csv", default="outputs/backtest_3rd_target_result.csv")
    parser.add_argument("--output", default="outputs/core5_performance.csv")
    args = parser.parse_args()

    df = build_core5_performance(args.result_csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(df.head(30).to_string(index=False) if not df.empty else "[core5_performance] no rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
