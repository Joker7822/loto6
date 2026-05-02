# -*- coding: utf-8 -*-
"""
nested_third_prize_backtest.py

3等狙いの nested walk-forward 検証です。

第N回を予測する時点で使える情報:
  - 第N-1回までの抽せん結果
  - 第N-1回までのnested検証結果

これにより、最適化パラメータが未来の検証結果を見ない形で評価できます。

逐次push:
  --push-every 100 を指定すると、100抽せん回ごとに
  nested_3rd_target_result.csv / summary / progress をcommit/pushします。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from csv_aware_predict import CsvFeatureSet, build_number_scores, build_pair_scores, combo_score
from loto6 import NUMBER_COLUMNS, classify_loto6, format_numbers, load_draws, normalize_draw_dataframe, normalize_numbers

TRACKED_OUTPUTS = [
    "nested_3rd_target_result.csv",
    "nested_3rd_target_summary.csv",
    "nested_3rd_target_progress.json",
]


def _empty_features(train_df: pd.DataFrame) -> CsvFeatureSet:
    return CsvFeatureSet(
        csv_paths=["nested_train_df"],
        actual_draws=normalize_draw_dataframe(train_df),
        prediction_number_bonus={},
        prediction_pair_bonus={},
        prior_prediction_bonus={},
        file_summaries=[{"path": "nested_train_df", "used_as": ["actual_draws"], "rows": int(len(train_df))}],
        duplicate_summary={
            "duplicate_actual_draw_rows_skipped": 0,
            "duplicate_performance_rows_skipped": 0,
            "duplicate_prior_prediction_rows_skipped": 0,
            "backup_or_noise_csv_files_skipped": 0,
        },
    )


def _derive_nested_config(prior_result: pd.DataFrame) -> dict:
    config = {
        "recent_window": 120,
        "core_pool_size": 22,
        "cover_pool_size": 34,
        "core_count": 3,
        "core_overlap_limit": 3,
        "allocation": [2, 2, 1],
        "history_penalty": 0.12,
        "cover_reuse_penalty": 0.0,
    }
    if prior_result.empty or "draw_no" not in prior_result.columns:
        config["reason"] = "no_prior_nested_result"
        return config

    work = prior_result.copy()
    work["core5_matches"] = pd.to_numeric(work.get("core5_matches", 0), errors="coerce").fillna(0).astype(int)
    work["main_matches"] = pd.to_numeric(work.get("main_matches", 0), errors="coerce").fillna(0).astype(int)
    recent_draws = sorted(work["draw_no"].astype(int).unique())[-120:]
    recent = work[work["draw_no"].astype(int).isin(recent_draws)]

    total_core4 = int((work["core5_matches"] >= 4).sum())
    recent_core4 = int((recent["core5_matches"] >= 4).sum())
    max_main = int(work["main_matches"].max()) if not work.empty else 0

    if total_core4 == 0:
        config.update({"core_pool_size": 30, "cover_pool_size": 40, "core_overlap_limit": 3})
        config["reason"] = "no_core4_seen_expand_pool"
    elif recent_core4 == 0:
        config.update({"recent_window": 180, "core_pool_size": 28, "cover_pool_size": 38, "allocation": [2, 1, 2]})
        config["reason"] = "recent_core4_weak_expand_c_core"
    else:
        config.update({"recent_window": 120, "core_pool_size": 24, "cover_pool_size": 36, "allocation": [2, 2, 1]})
        config["reason"] = "core4_seen_balanced"

    if max_main >= 5:
        config["history_penalty"] = 0.08
        config["cover_reuse_penalty"] = 0.01
        config["reason"] += "; main5_seen_reduce_history_penalty"
    else:
        config["history_penalty"] = 0.12
        config["cover_reuse_penalty"] = 0.02
    return config


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


def _select_diverse_cores(core_ranked, core_count: int, overlap_limit: int):
    selected = []
    for score, core in core_ranked:
        if not selected or all(_overlap(core, chosen) <= overlap_limit for _, chosen in selected):
            selected.append((score, core))
        if len(selected) >= core_count:
            return selected
    for score, core in core_ranked:
        if any(core == chosen for _, chosen in selected):
            continue
        if all(_overlap(core, chosen) <= min(4, overlap_limit + 1) for _, chosen in selected):
            selected.append((score, core))
        if len(selected) >= core_count:
            return selected
    return selected


def _predict_nested(train_df: pd.DataFrame, prior_result: pd.DataFrame, top_n: int, seed: int) -> tuple[pd.DataFrame, dict]:
    config = _derive_nested_config(prior_result)
    features = _empty_features(train_df)
    draws = normalize_draw_dataframe(train_df)
    number_scores = build_number_scores(draws, features, recent_window=int(config["recent_window"]))
    pair_scores = build_pair_scores(draws, features)
    history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}
    ranked_numbers = sorted(range(1, 44), key=lambda x: number_scores.get(x, 0.0), reverse=True)
    core_pool = ranked_numbers[: int(config["core_pool_size"])]
    cover_pool = ranked_numbers[: int(config["cover_pool_size"])]

    core_ranked = []
    for core in combinations(core_pool, 5):
        core = tuple(sorted(core))
        score = _core_score(core, number_scores, pair_scores)
        historical_core_hits = sum(1 for h in history if len(set(core) & set(h)) >= 5)
        score -= min(historical_core_hits, 5) * 0.01
        core_ranked.append((float(score), core))
    core_ranked.sort(reverse=True)
    selected = _select_diverse_cores(core_ranked, int(config["core_count"]), int(config["core_overlap_limit"]))

    allocation = [int(x) for x in config["allocation"]]
    if sum(allocation) != top_n:
        allocation = [2, 2, 1] if top_n == 5 else [max(1, top_n // 3), max(1, top_n // 3), max(1, top_n - 2 * max(1, top_n // 3))]

    rows = []
    used_tickets = set()
    used_covers: dict[int, int] = {}
    for core_index, (core_score_value, core) in enumerate(selected):
        quota = allocation[core_index] if core_index < len(allocation) else 1
        ranked_covers = []
        for cover in cover_pool:
            if cover in core:
                continue
            ticket = normalize_numbers([*core, cover])
            score = 0.74 * core_score_value + 0.16 * combo_score(ticket, number_scores, pair_scores, history) + 0.10 * number_scores.get(int(cover), 0.0)
            if ticket in history:
                score -= float(config["history_penalty"])
            score -= float(config["cover_reuse_penalty"]) * used_covers.get(int(cover), 0)
            ranked_covers.append((float(score), int(cover), ticket))
        ranked_covers.sort(reverse=True)
        emitted = 0
        for score, cover, ticket in ranked_covers:
            if ticket in used_tickets:
                continue
            used_tickets.add(ticket)
            used_covers[cover] = used_covers.get(cover, 0) + 1
            emitted += 1
            rows.append({
                "rank": len(rows) + 1,
                "numbers": format_numbers(ticket),
                "core5": " ".join(f"{x:02d}" for x in core),
                "cover_number": f"{cover:02d}",
                "score": round(score, 6),
                "core_score": round(core_score_value, 6),
                "core_group": chr(ord("A") + core_index),
                **{f"n{i}": x for i, x in enumerate(ticket, 1)},
            })
            if emitted >= quota or len(rows) >= top_n:
                break
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows), config


def _core5_matches(core5_text: str, actual_main) -> int:
    core_nums = {int(x) for x in str(core5_text).split() if str(x).isdigit()}
    return len(core_nums & set(map(int, actual_main)))


def _summarize(df: pd.DataFrame, total_draws: int, min_train_draws: int, top_n: int) -> dict:
    if df.empty:
        return {"mode": "nested_third_prize_walk_forward", "total_source_draws": total_draws, "min_train_draws": min_train_draws, "completed_draws": 0, "latest_completed_draw_no": None, "tickets": 0, "top_n": top_n}
    grade_counts = dict(Counter(df["grade"]))
    per_draw = df.groupby("draw_no").agg(max_core5=("core5_matches", "max"), max_main=("main_matches", "max"))
    return {
        "mode": "nested_third_prize_walk_forward",
        "total_source_draws": int(total_draws),
        "min_train_draws": int(min_train_draws),
        "completed_draws": int(df["draw_no"].nunique()),
        "latest_completed_draw_no": int(df["draw_no"].max()),
        "tickets": int(len(df)),
        "top_n": int(top_n),
        "max_main_matches": int(df["main_matches"].max()),
        "max_core5_matches": int(df["core5_matches"].max()),
        "core5_5hit_count": int((df["core5_matches"] >= 5).sum()),
        "core5_4hit_count": int((df["core5_matches"] >= 4).sum()),
        "near_third_count": int((per_draw["max_core5"] >= 4).sum()),
        "third_or_better_count": int(df["grade"].isin(["1等", "2等", "3等"]).sum()),
        "grade_counts": grade_counts,
    }


def _write_outputs(result_df: pd.DataFrame, summary: dict, out_dir: Path) -> tuple[Path, Path, Path]:
    result_path = out_dir / "nested_3rd_target_result.csv"
    summary_path = out_dir / "nested_3rd_target_summary.csv"
    progress_path = out_dir / "nested_3rd_target_progress.json"
    result_df.sort_values(["draw_no", "rank"]).to_csv(result_path, index=False, encoding="utf-8")
    pd.DataFrame([summary | {"grade_counts": json.dumps(summary.get("grade_counts", {}), ensure_ascii=False)}]).to_csv(summary_path, index=False, encoding="utf-8")
    progress_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path, summary_path, progress_path


def _run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def _current_branch() -> str:
    ref = os.environ.get("GITHUB_REF_NAME")
    if ref:
        return ref
    result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def git_commit_and_push(paths: Sequence[Path | str], message: str, max_attempts: int = 3) -> bool:
    branch = _current_branch()
    _run_git(["config", "user.name", os.environ.get("GIT_COMMITTER_NAME", "github-actions")], check=False)
    _run_git(["config", "user.email", os.environ.get("GIT_COMMITTER_EMAIL", "github-actions@github.com")], check=False)
    _run_git(["add", *[str(p) for p in paths]])
    if _run_git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("[PUSH] no nested changes to commit")
        return False
    _run_git(["commit", "-m", message])
    for attempt in range(1, max_attempts + 1):
        _run_git(["fetch", "origin", branch], check=False)
        _run_git(["rebase", f"origin/{branch}"], check=False)
        pushed = _run_git(["push", "origin", f"HEAD:{branch}"], check=False)
        if pushed.returncode == 0:
            print(f"[PUSH] pushed nested progress: {message}")
            return True
        print(f"[PUSH] failed attempt {attempt}/{max_attempts}: {pushed.stdout}")
        if attempt == max_attempts:
            raise RuntimeError(f"git push failed after {max_attempts} attempts")
        time.sleep(2 * attempt)
    return False


def run_nested(
    csv_path: str,
    output_dir: str,
    min_train_draws: int,
    top_n: int,
    seed: int,
    max_draws: int | None,
    push_every: int = 0,
    push_final: bool = True,
) -> tuple[pd.DataFrame, dict]:
    draws = normalize_draw_dataframe(load_draws(csv_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "nested_3rd_target_result.csv"
    tracked_paths = [out_dir / name for name in TRACKED_OUTPUTS]

    result_df = pd.read_csv(result_path) if result_path.exists() and result_path.stat().st_size else pd.DataFrame()
    completed = set(result_df["draw_no"].astype(int)) if not result_df.empty and "draw_no" in result_df.columns else set()
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
        prior = result_df.copy() if not result_df.empty else pd.DataFrame()
        pred_df, config = _predict_nested(train_df, prior, top_n, seed + draw_no)
        main = [int(actual[c]) for c in NUMBER_COLUMNS]
        bonus = int(actual["bonus"])
        rows = []
        for _, p in pred_df.iterrows():
            nums = [int(p[f"n{i}"]) for i in range(1, 7)]
            match = classify_loto6(nums, main, bonus)
            c5 = _core5_matches(p["core5"], main)
            rows.append({
                "draw_no": draw_no,
                "date": actual["date"].date().isoformat(),
                "rank": int(p["rank"]),
                "numbers": p["numbers"],
                "core_group": p["core_group"],
                "core5": p["core5"],
                "cover_number": p["cover_number"],
                "core5_matches": c5,
                "main_matches": int(match.main_matches),
                "bonus_match": bool(match.bonus_match),
                "grade": match.grade,
                "score": float(p["score"]),
                "core_score": float(p["core_score"]),
                "config_reason": config.get("reason", ""),
                "recent_window": int(config["recent_window"]),
                "core_pool_size": int(config["core_pool_size"]),
                "cover_pool_size": int(config["cover_pool_size"]),
                "actual": " ".join(f"{x:02d}" for x in main),
                "bonus": f"{bonus:02d}",
                "train_draws": int(len(train_df)),
                "train_until_draw_no": int(train_df["draw_no"].max()),
            })
        result_df = pd.concat([result_df, pd.DataFrame(rows)], ignore_index=True)
        completed.add(draw_no)
        new_since_push += 1
        summary = _summarize(result_df, len(draws), min_train_draws, top_n)
        _write_outputs(result_df, summary, out_dir)
        print(f"nested_verified draw={draw_no} completed={summary['completed_draws']} max_core5={summary.get('max_core5_matches')} max_main={summary.get('max_main_matches')}")

        if push_every > 0 and new_since_push >= push_every:
            git_commit_and_push(tracked_paths, f"Nested 3rd target progress up to draw {draw_no} [skip ci]")
            new_since_push = 0

    summary = _summarize(result_df, len(draws), min_train_draws, top_n)
    _write_outputs(result_df, summary, out_dir)
    if push_every > 0 and push_final and new_since_push > 0:
        suffix = summary.get("latest_completed_draw_no") or "none"
        git_commit_and_push(tracked_paths, f"Nested 3rd target final progress up to draw {suffix} [skip ci]")
    return result_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nested optimized Loto6 third-prize backtest.")
    parser.add_argument("--csv", default="data/loto6.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-draws", type=int, default=None)
    parser.add_argument("--push-every", type=int, default=0)
    parser.add_argument("--push-final", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    _, summary = run_nested(
        args.csv,
        args.output_dir,
        args.min_train_draws,
        args.top_n,
        args.seed,
        args.max_draws,
        push_every=args.push_every,
        push_final=args.push_final,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
