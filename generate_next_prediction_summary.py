# -*- coding: utf-8 -*-
"""
generate_next_prediction_summary.py

次回抽せん日の最新予測を分かりやすくまとめます。

入力:
  data/loto6.csv
  outputs/latest_predictions.csv
  outputs/latest_predictions_3rd_target.csv
  outputs/latest_predictions_3rd_target_context.json

出力:
  outputs/NEXT_DRAW_PREDICTION.md
  outputs/next_draw_prediction_summary.csv
  outputs/next_draw_prediction_context.json

目的:
  - 次回抽せん回
  - 次回抽せん予定日
  - どのCSVが最新予測なのか
  - 通常予測と3等狙い予測
  を1か所で確認できるようにする。
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

LATO6_DRAW_WEEKDAYS = {0, 3}  # Monday, Thursday
JST = timezone(timedelta(hours=9))


def _read_csv_optional(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_json_optional(path: str) -> dict:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _next_loto6_draw_date(latest_date: date) -> date:
    """Return the next nominal Loto6 draw date after latest_date.

    Loto6 is normally drawn on Monday and Thursday. If a holiday or official schedule
    exception exists, the next scrape will correct this when the actual draw is added.
    """
    d = latest_date + timedelta(days=1)
    for _ in range(14):
        if d.weekday() in LATO6_DRAW_WEEKDAYS:
            return d
        d += timedelta(days=1)
    return latest_date + timedelta(days=3)


def _format_numbers(row: pd.Series) -> str:
    if "numbers" in row and pd.notna(row["numbers"]):
        return str(row["numbers"])
    cols = [f"n{i}" for i in range(1, 7)]
    if all(c in row for c in cols):
        return " ".join(f"{int(row[c]):02d}" for c in cols)
    return ""


def _build_prediction_rows(df: pd.DataFrame, strategy_label: str, next_draw_no: int, next_draw_date: str) -> list[dict]:
    rows: list[dict] = []
    if df.empty:
        return rows
    for _, r in df.iterrows():
        rows.append(
            {
                "next_draw_no": next_draw_no,
                "next_draw_date": next_draw_date,
                "prediction_type": strategy_label,
                "rank": int(r.get("rank", len(rows) + 1)),
                "numbers": _format_numbers(r),
                "core_group": r.get("core_group", ""),
                "core5": r.get("core5", ""),
                "cover_number": r.get("cover_number", ""),
                "score": r.get("score", ""),
                "core_score": r.get("core_score", ""),
                "core5_performance_bonus": r.get("core5_performance_bonus", ""),
                "strategy": r.get("strategy", ""),
                "source_draws": r.get("source_draws", ""),
                "trained_through_draw_no": r.get("trained_through_draw_no", ""),
                "trained_through_date": r.get("trained_through_date", ""),
            }
        )
    return rows


def generate_summary(
    draw_csv: str = "data/loto6.csv",
    standard_csv: str = "outputs/latest_predictions.csv",
    third_csv: str = "outputs/latest_predictions_3rd_target.csv",
    third_context_json: str = "outputs/latest_predictions_3rd_target_context.json",
    markdown_output: str = "outputs/NEXT_DRAW_PREDICTION.md",
    summary_csv_output: str = "outputs/next_draw_prediction_summary.csv",
    context_output: str = "outputs/next_draw_prediction_context.json",
) -> dict:
    draws = pd.read_csv(draw_csv)
    draws = draws.sort_values("draw_no").copy()
    latest = draws.iloc[-1]
    latest_draw_no = int(latest["draw_no"])
    latest_draw_date = pd.to_datetime(latest["date"]).date()
    next_draw_no = latest_draw_no + 1
    next_draw_date = _next_loto6_draw_date(latest_draw_date)
    next_draw_date_str = next_draw_date.isoformat()

    standard = _read_csv_optional(standard_csv)
    third = _read_csv_optional(third_csv)
    third_context = _read_json_optional(third_context_json)

    rows = []
    rows.extend(_build_prediction_rows(standard, "通常予測", next_draw_no, next_draw_date_str))
    rows.extend(_build_prediction_rows(third, "3等狙い", next_draw_no, next_draw_date_str))
    summary_df = pd.DataFrame(rows)

    out_csv = Path(summary_csv_output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False, encoding="utf-8")

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    generated_at_jst = datetime.now(JST).isoformat()
    context = {
        "generated_at_utc": generated_at_utc,
        "generated_at_jst": generated_at_jst,
        "source_draw_csv": draw_csv,
        "latest_confirmed_draw_no": latest_draw_no,
        "latest_confirmed_draw_date": latest_draw_date.isoformat(),
        "next_draw_no": next_draw_no,
        "next_draw_date": next_draw_date_str,
        "next_draw_date_rule": "Nominal next Monday/Thursday after latest confirmed draw date. Actual official schedule is corrected after scraping.",
        "standard_prediction_csv": standard_csv if not standard.empty else None,
        "third_target_prediction_csv": third_csv if not third.empty else None,
        "summary_csv": summary_csv_output,
        "summary_markdown": markdown_output,
        "third_target_context": third_context,
    }
    Path(context_output).write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Loto6 次回抽せん 最新予測",
        "",
        f"- 生成日時(JST): `{generated_at_jst}`",
        f"- 最新反映済み抽せん: `第{latest_draw_no}回 / {latest_draw_date.isoformat()}`",
        f"- 次回予測対象: `第{next_draw_no}回 / {next_draw_date_str}`",
        f"- まとめCSV: `{summary_csv_output}`",
        "",
        "## どれを見ればいいか",
        "",
        "| 用途 | ファイル |",
        "|---|---|",
        f"| 次回予測の一覧 | `{summary_csv_output}` |",
        f"| 3等狙いの最新予測 | `{third_csv}` |",
        f"| 通常予測 | `{standard_csv}` |",
        f"| このサマリー | `{markdown_output}` |",
        "",
    ]

    if not third.empty:
        md_lines.extend(["## 3等狙い 最新予測", "", "| rank | core | numbers | core5 | bonus | score |", "|---:|---|---|---|---:|---:|"])
        for _, r in third.iterrows():
            md_lines.append(
                f"| {int(r.get('rank', 0))} | {r.get('core_group', '')} | `{_format_numbers(r)}` | `{r.get('core5', '')}` | {r.get('core5_performance_bonus', '')} | {r.get('score', '')} |"
            )
        md_lines.append("")
    else:
        md_lines.extend(["## 3等狙い 最新予測", "", "確認できません。`outputs/latest_predictions_3rd_target.csv` が未生成です。", ""])

    if not standard.empty:
        md_lines.extend(["## 通常予測", "", "| rank | numbers | score |", "|---:|---|---:|"])
        for _, r in standard.iterrows():
            md_lines.append(f"| {int(r.get('rank', 0))} | `{_format_numbers(r)}` | {r.get('score', '')} |")
        md_lines.append("")
    else:
        md_lines.extend(["## 通常予測", "", "確認できません。`outputs/latest_predictions.csv` が未生成です。", ""])

    md_lines.extend(
        [
            "## 注意",
            "",
            "次回抽せん日は、最新CSVの最終抽せん日の次の月曜または木曜として算出しています。公式日程の例外がある場合は、次回スクレイピングで実績日付に更新されます。",
            "",
        ]
    )
    Path(markdown_output).write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a clear next-draw prediction summary for Loto6.")
    parser.add_argument("--draw-csv", default="data/loto6.csv")
    parser.add_argument("--standard-csv", default="outputs/latest_predictions.csv")
    parser.add_argument("--third-csv", default="outputs/latest_predictions_3rd_target.csv")
    parser.add_argument("--third-context-json", default="outputs/latest_predictions_3rd_target_context.json")
    parser.add_argument("--markdown-output", default="outputs/NEXT_DRAW_PREDICTION.md")
    parser.add_argument("--summary-csv-output", default="outputs/next_draw_prediction_summary.csv")
    parser.add_argument("--context-output", default="outputs/next_draw_prediction_context.json")
    args = parser.parse_args()
    generate_summary(
        draw_csv=args.draw_csv,
        standard_csv=args.standard_csv,
        third_csv=args.third_csv,
        third_context_json=args.third_context_json,
        markdown_output=args.markdown_output,
        summary_csv_output=args.summary_csv_output,
        context_output=args.context_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
