# -*- coding: utf-8 -*-
"""
third_prize_backtest.py

Loto6の3等狙い専用 walk-forward 検証です。

目的:
  - 3等条件である「本数字5個一致」にどれだけ近づいているかを測る
  - 5数字コア(core5)の一致数を抽せん回ごとに記録する
  - 未来リークなしで、対象回NはNより前の抽せんだけで予測する

出力:
  outputs/backtest_3rd_target_result.csv
  outputs/backtest_3rd_target_summary.csv
  outputs/backtest_3rd_target_progress.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from csv_aware_predict import CsvFeatureSet, generate_third_prize_predictions
from loto6 import NUMBER_COLUMNS, classify_loto6, load_draws, normalize_draw_dataframe

TRACKED_OUTPUTS = [
    "backtest_3rd_target_result.csv",
    "backtest_3rd_target_summary.csv",
    "backtest_3rd_target_progress.json",
]

REQUIRED_RESULT_COLUMNS = {
    "draw_no",
    "rank",
    "numbers",
    "core5",
    "core5_matches",
    "main_matches",
    "grade",
}


def _empty_features(train_df: pd.DataFrame) -> CsvFeatureSet:
    return CsvFeatureSet(
        csv_paths=["walk_forward_train_df"],
        actual_draws=normalize_draw_dataframe(train_df),
        prediction_number_bonus={},
        prediction_pair_bonus={},
        prior_prediction_bonus={},
        file_summaries=[{"path": "walk_forward_train_df", "used_as": ["actual_draws"], "rows": int(len(train_df))}],
        duplicate_summary={
            "duplicate_actual_draw_rows_skipped": 0,
            "duplicate_performance_rows_skipped": 0,
            "duplicate_prior_prediction_rows_skipped": 0,
        },
    )


def _core5_matches(core5_text: str, actual_main: Sequence[int]) -> int:
    core_nums = {int(x) for x in str(core5_text).split() if str(x).isdigit()}
    return len(core_nums & set(map(int, actual_main)))


def _backup_incompatible_result(path: Path, reason: str) -> None:
    """Move incompatible resume file out of the way and restart safely.

    Older experimental versions produced outputs/backtest_3rd_target_result.csv with
    different columns. Failing hard blocks Actions forever, so keep a timestamped backup
    and regenerate the new schema from scratch.
    """
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}.incompatible.{stamp}.bak.csv")
    path.replace(backup_path)
    print(f"[RESET_RESUME] moved incompatible file: {path} -> {backup_path}; reason={reason}")


def _read_existing_result(path: Path, top_n: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _backup_incompatible_result(path, f"read_failed:{exc}")
        return pd.DataFrame()

    missing = REQUIRED_RESULT_COLUMNS - set(df.columns)
    if missing:
        _backup_incompatible_result(path, f"missing_columns:{sorted(missing)}")
        return pd.DataFrame()

    try:
        counts = df.groupby("draw_no")["rank"].nunique()
        complete = set(counts[counts >= top_n].index.astype(int))
        filtered = df[df["draw_no"].astype(int).isin(complete)].copy()
    except Exception as exc:
        _backup_incompatible_result(path, f"invalid_resume_rows:{exc}")
        return pd.DataFrame()

    print(f"[RESUME] loaded completed 3rd-target draws={filtered['draw_no'].nunique() if not filtered.empty else 0}")
    return filtered


def _summarize(result_df: pd.DataFrame, total_source_draws: int, min_train_draws: int, top_n: int) -> dict:
    if result_df.empty:
        return {
            "mode": "third_prize_walk_forward_no_future_leakage",
            "total_source_draws": int(total_source_draws),
            "min_train_draws": int(min_train_draws),
            "completed_draws": 0,
            "latest_completed_draw_no": None,
            "tickets": 0,
            "top_n": int(top_n),
            "max_main_matches": 0,
            "max_core5_matches": 0,
            "core5_5hit_count": 0,
            "core5_4hit_count": 0,
            "near_third_count": 0,
            "third_or_better_count": 0,
            "grade_counts": {},
        }

    per_draw = result_df.groupby("draw_no").agg(
        max_core5_matches=("core5_matches", "max"),
        max_main_matches=("main_matches", "max"),
    )
    grade_counts = dict(Counter(result_df["grade"]))
    return {
        "mode": "third_prize_walk_forward_no_future_leakage",
        "total_source_draws": int(total_source_draws),
        "min_train_draws": int(min_train_draws),
        "completed_draws": int(result_df["draw_no"].nunique()),
        "latest_completed_draw_no": int(result_df["draw_no"].max()),
        "tickets": int(len(result_df)),
        "top_n": int(top_n),
        "max_main_matches": int(result_df["main_matches"].max()),
        "max_core5_matches": int(result_df["core5_matches"].max()),
        "core5_5hit_count": int((result_df["core5_matches"] >= 5).sum()),
        "core5_4hit_count": int((result_df["core5_matches"] >= 4).sum()),
        "near_third_count": int((per_draw["max_core5_matches"] >= 4).sum()),
        "third_or_better_count": int(result_df["grade"].isin(["1等", "2等", "3等"]).sum()),
        "grade_counts": grade_counts,
    }


def _write_outputs(result_df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "backtest_3rd_target_result.csv"
    summary_path = output_dir / "backtest_3rd_target_summary.csv"
    progress_path = output_dir / "backtest_3rd_target_progress.json"
    result_df.sort_values(["draw_no", "rank"]).to_csv(result_path, index=False, encoding="utf-8")
    summary_row = summary | {"grade_counts": json.dumps(summary.get("grade_counts", {}), ensure_ascii=False)}
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8")
    progress_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def _current_branch() -> str:
    ref = os.environ.get("GITHUB_REF_NAME")
    if ref:
        return ref
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def git_commit_and_push(paths: Sequence[Path | str], message: str, max_attempts: int = 3) -> bool:
    branch = _current_branch()
    _run_git(["config", "user.name", os.environ.get("GIT_COMMITTER_NAME", "github-actions")], check=False)
    _run_git(["config", "user.email", os.environ.get("GIT_COMMITTER_EMAIL", "github-actions@github.com")], check=False)
    _run_git(["add", *[str(p) for p in paths]])
    if _run_git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
        return False
    _run_git(["commit", "-m", message])
    for attempt in range(1, max_attempts + 1):
        _run_git(["fetch", "origin", branch], check=False)
        _run_git(["rebase", f"origin/{branch}"], check=False)
        pushed = _run_git(["push", "origin", f"HEAD:{branch}"], check=False)
        if pushed.returncode == 0:
            return True
        if attempt == max_attempts:
            print(pushed.stdout)
            raise RuntimeError(f"git push failed after {max_attempts} attempts")
        time.sleep(2 * attempt)
    return False


def run_backtest(
    csv_path: str,
    output_dir: str | Path = "outputs",
    min_train_draws: int = 1,
    top_n: int = 5,
    seed: int = 42,
    resume: bool = True,
    push_every: int = 0,
    push_final: bool = True,
    max_draws: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    draws = normalize_draw_dataframe(load_draws(csv_path))
    if len(draws) <= min_train_draws:
        raise ValueError(f"Need more than min_train_draws={min_train_draws} rows; got {len(draws)}")

    output_path = Path(output_dir)
    result_path = output_path / "backtest_3rd_target_result.csv"
    tracked_paths = [output_path / name for name in TRACKED_OUTPUTS]
    result_df = _read_existing_result(result_path, top_n) if resume else pd.DataFrame()
    completed = set(result_df["draw_no"].astype(int)) if not result_df.empty else set()
    target_indexes = list(range(min_train_draws, len(draws)))
    if max_draws is not None:
        target_indexes = target_indexes[:max_draws]

    new_since_push = 0
    for idx in target_indexes:
        actual = draws.iloc[idx]
        draw_no = int(actual["draw_no"])
        if draw_no in completed:
            continue

        train_df = draws.iloc[:idx].copy()
        if draw_no in set(train_df["draw_no"].astype(int)):
            raise RuntimeError(f"Future leakage guard failed for draw_no={draw_no}")

        features = _empty_features(train_df)
        pred_df = generate_third_prize_predictions(features, n=top_n, seed=seed + draw_no)
        main = [int(actual[c]) for c in NUMBER_COLUMNS]
        bonus = int(actual["bonus"])
        rows = []
        for _, pred in pred_df.iterrows():
            nums = [int(pred[f"n{i}"]) for i in range(1, 7)]
            match = classify_loto6(nums, main, bonus)
            c5 = _core5_matches(str(pred.get("core5", "")), main)
            rows.append(
                {
                    "draw_no": draw_no,
                    "date": actual["date"].date().isoformat(),
                    "rank": int(pred["rank"]),
                    "numbers": pred["numbers"],
                    "core5": pred.get("core5", ""),
                    "cover_number": pred.get("cover_number", ""),
                    "core5_matches": int(c5),
                    "main_matches": int(match.main_matches),
                    "bonus_match": bool(match.bonus_match),
                    "grade": match.grade,
                    "score": float(pred["score"]),
                    "core_score": float(pred.get("core_score", 0.0)),
                    "actual": " ".join(f"{x:02d}" for x in main),
                    "bonus": f"{bonus:02d}",
                    "train_draws": int(len(train_df)),
                    "train_until_draw_no": int(train_df["draw_no"].max()),
                }
            )

        result_df = pd.concat([result_df, pd.DataFrame(rows)], ignore_index=True)
        completed.add(draw_no)
        new_since_push += 1
        summary = _summarize(result_df, len(draws), min_train_draws, top_n)
        _write_outputs(result_df, summary, output_path)
        print(
            f"verified_3rd_target draw={draw_no} completed={summary['completed_draws']} "
            f"max_core5={summary['max_core5_matches']} max_main={summary['max_main_matches']}"
        )

        if push_every > 0 and new_since_push >= push_every:
            git_commit_and_push(tracked_paths, f"3rd target backtest progress up to draw {draw_no} [skip ci]")
            new_since_push = 0

    summary = _summarize(result_df, len(draws), min_train_draws, top_n)
    _write_outputs(result_df, summary, output_path)
    if push_every > 0 and push_final and new_since_push > 0:
        suffix = summary.get("latest_completed_draw_no") or "none"
        git_commit_and_push(tracked_paths, f"3rd target backtest final progress up to draw {suffix} [skip ci]")
    return result_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Loto6 third-prize-oriented no-leakage backtest.")
    parser.add_argument("--csv", default="data/loto6.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-every", type=int, default=0)
    parser.add_argument("--push-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-draws", type=int, default=None)
    args = parser.parse_args()
    _, summary = run_backtest(
        csv_path=args.csv,
        output_dir=args.output_dir,
        min_train_draws=args.min_train_draws,
        top_n=args.top_n,
        seed=args.seed,
        resume=args.resume,
        push_every=args.push_every,
        push_final=args.push_final,
        max_draws=args.max_draws,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
