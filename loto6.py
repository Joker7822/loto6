"""Loto6-only scraper, predictor, resumable verifier and walk-forward backtester.

The verifier is intentionally anti-leakage:
for target draw N, it trains only with draws earlier than N.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
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

REFERENCE_PAYOUT_YEN = {
    "1等": 200_000_000,
    "2等": 10_000_000,
    "3等": 300_000,
    "4等": 6_800,
    "5等": 1_000,
    "はずれ": 0,
}
DRAW_RE = re.compile(r"第?\s*(\d+)\s*回")
DATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
NUM_RE = re.compile(r"\d{1,2}")
ROW_RE = re.compile(
    r"第\s*(?P<draw>\d{1,5})\s*回\s+"
    r"(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日\s+"
    r"(?P<num1>\d{1,2})\s+(?P<num2>\d{1,2})\s+(?P<num3>\d{1,2})\s+"
    r"(?P<num4>\d{1,2})\s+(?P<num5>\d{1,2})\s+(?P<num6>\d{1,2})\s+"
    r"(?P<bonus>\d{1,2})"
)
MOBILE_ROW_RE = re.compile(
    r"第\s*(?P<draw>\d{1,5})\s*回.*?"
    r"(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日.*?"
    r"本数字\s*(?P<main>(?:\d{1,2}\s*){6}).*?"
    r"ボーナス数字\s*(?P<bonus>\d{1,2})",
    re.DOTALL,
)


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
    source: str = "loto6_heuristic_ensemble"

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
    """Classify one Loto6 prediction against one draw result."""
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
    if df.empty:
        return pd.DataFrame(columns=DRAW_COLUMNS)

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    aliases = {
        "回別": "draw_no",
        "回": "draw_no",
        "抽せん日": "date",
        "抽選日": "date",
        "日付": "date",
        "ボーナス数字": "bonus",
        "ボーナス": "bonus",
    }
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
    arr = np.array([values.get(n, 0.0) for n in range(MIN_NUMBER, MAX_NUMBER + 1)], dtype=float)
    if arr.max() == arr.min():
        return {n: 0.0 for n in range(MIN_NUMBER, MAX_NUMBER + 1)}
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return {n: float(arr[n - 1]) for n in range(MIN_NUMBER, MAX_NUMBER + 1)}


def number_scores(df: pd.DataFrame, recent_window: int = 80) -> dict[int, float]:
    draws = normalize_draw_dataframe(df)
    recent = draws.tail(min(recent_window, len(draws)))
    prior = draws.iloc[: max(0, len(draws) - len(recent))]

    all_cnt = Counter(int(v) for v in draws[NUMBER_COLUMNS].to_numpy().ravel())
    recent_cnt = Counter(int(v) for v in recent[NUMBER_COLUMNS].to_numpy().ravel())
    prior_cnt = Counter(int(v) for v in prior[NUMBER_COLUMNS].to_numpy().ravel()) if not prior.empty else Counter()

    gap = {n: len(draws) + 1 for n in range(MIN_NUMBER, MAX_NUMBER + 1)}
    for offset, row in enumerate(draws[NUMBER_COLUMNS].itertuples(index=False, name=None), start=1):
        for n in row:
            gap[int(n)] = len(draws) - offset

    f = _scale({n: all_cnt[n] for n in range(MIN_NUMBER, MAX_NUMBER + 1)})
    r = _scale({n: recent_cnt[n] for n in range(MIN_NUMBER, MAX_NUMBER + 1)})
    g = _scale(gap)
    t = _scale({
        n: recent_cnt[n] / max(1, len(recent)) - prior_cnt[n] / max(1, len(prior))
        for n in range(MIN_NUMBER, MAX_NUMBER + 1)
    })
    return {n: 0.35 * f[n] + 0.35 * r[n] + 0.20 * g[n] + 0.10 * t[n] for n in range(MIN_NUMBER, MAX_NUMBER + 1)}


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
    """Small deterministic Loto6 predictor.

    It is designed for walk-forward verification, not for guaranteed winnings.
    """

    def __init__(self, recent_window: int = 80, seed: int = 42) -> None:
        self.recent_window = recent_window
        self.seed = seed
        self.scores: dict[int, float] = {}
        self.pairs: dict[tuple[int, int], float] = {}
        self.history: set[tuple[int, ...]] = set()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "Loto6Predictor":
        draws = normalize_draw_dataframe(df)
        if draws.empty:
            raise ValueError("At least one past draw is required to fit without future leakage.")
        self.scores = number_scores(draws, self.recent_window)
        self.pairs = pair_scores(draws, max(160, self.recent_window * 2))
        self.history = {tuple(row) for row in draws[NUMBER_COLUMNS].itertuples(index=False, name=None)}
        self.fitted = True
        return self

    def _combo_score(self, combo: Sequence[int]) -> float:
        nums = normalize_numbers(combo)
        base = sum(self.scores.get(n, 0.0) for n in nums) / PICK_SIZE
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
        numbers = list(range(MIN_NUMBER, MAX_NUMBER + 1))
        weights = [self.scores.get(x, 0.0) + 0.05 for x in numbers]
        candidates: set[tuple[int, ...]] = set()

        top_pool = sorted(numbers, key=lambda x: self.scores.get(x, 0.0), reverse=True)[:18]
        for combo in combinations(top_pool, PICK_SIZE):
            candidates.add(normalize_numbers(combo))
            if len(candidates) >= max(1, candidate_count // 3):
                break

        while len(candidates) < candidate_count:
            available = numbers[:]
            available_weights = weights[:]
            pick = []
            for _ in range(PICK_SIZE):
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
    return pd.DataFrame([
        {"rank": p.rank, "numbers": p.numbers_text, "score": round(p.score, 6), "source": p.source, **{f"n{i}": x for i, x in enumerate(p.numbers, 1)}}
        for p in predictions
    ])


def _valid_draw_numbers(nums: Sequence[int]) -> bool:
    return len(nums) == 7 and all(MIN_NUMBER <= int(n) <= MAX_NUMBER for n in nums[:6]) and MIN_NUMBER <= int(nums[6]) <= MAX_NUMBER


def _row_dict(draw_no: int, year: int, month: int, day: int, nums: Sequence[int]) -> dict | None:
    nums = [int(n) for n in nums]
    if not _valid_draw_numbers(nums):
        return None
    try:
        main = normalize_numbers(nums[:6])
    except ValueError:
        return None
    bonus = int(nums[6])
    return {
        "draw_no": int(draw_no),
        "date": f"{int(year):04d}-{int(month):02d}-{int(day):02d}",
        **{f"n{i}": main[i - 1] for i in range(1, 7)},
        "bonus": bonus,
    }


def parse_loto6_html(html: str) -> list[dict]:
    """Parse old/current Mizuho Loto6 pages.

    The old backnumber pages are static tables such as loto60001.html.
    The current page and mobile fragments can use different row structures, so this parser
    first reads real table rows, then falls back to whole-page regex parsing.
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    seen: set[int] = set()

    # 1) Table-row parser. This avoids mixing draw/date numbers into the seven result numbers.
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        joined = " ".join(cells)
        draw = DRAW_RE.search(joined)
        date = DATE_RE.search(joined)
        if not draw or not date:
            continue

        numeric_cells: list[int] = []
        for cell in cells:
            if DRAW_RE.search(cell) or DATE_RE.search(cell):
                continue
            for token in NUM_RE.findall(cell):
                value = int(token)
                if MIN_NUMBER <= value <= MAX_NUMBER:
                    numeric_cells.append(value)

        if len(numeric_cells) >= 7:
            y, m, d = map(int, date.groups())
            row = _row_dict(int(draw.group(1)), y, m, d, numeric_cells[:7])
            if row and row["draw_no"] not in seen:
                rows.append(row)
                seen.add(row["draw_no"])

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\s+", " ", text)

    # 2) Desktop table text fallback.
    for match in ROW_RE.finditer(text):
        nums = [int(match.group(f"num{i}")) for i in range(1, 7)] + [int(match.group("bonus"))]
        row = _row_dict(
            int(match.group("draw")),
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            nums,
        )
        if row and row["draw_no"] not in seen:
            rows.append(row)
            seen.add(row["draw_no"])

    # 3) Mobile/definition-list style fallback.
    for match in MOBILE_ROW_RE.finditer(text):
        nums = [int(x) for x in NUM_RE.findall(match.group("main"))][:6] + [int(match.group("bonus"))]
        row = _row_dict(
            int(match.group("draw")),
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            nums,
        )
        if row and row["draw_no"] not in seen:
            rows.append(row)
            seen.add(row["draw_no"])

    if not rows:
        return []
    return normalize_draw_dataframe(pd.DataFrame(rows)).to_dict("records")


def fetch_page(url: str, retries: int = 2, sleep_seconds: float = 0.8) -> list[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en;q=0.8",
        "Cache-Control": "no-cache",
        "Referer": "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            res = requests.get(url, headers=headers, timeout=30)
            res.raise_for_status()
            res.encoding = res.apparent_encoding or res.encoding
            rows = parse_loto6_html(res.text)
            if rows:
                return rows
            raise RuntimeError(f"parsed 0 rows from {url}; status={res.status_code}; bytes={len(res.text)}")
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(str(last_error))


def update_csv(path: str = "data/loto6.csv", max_draw: int = 9999, block_size: int = 20, sleep_seconds: float = 0.6, max_empty_blocks: int = 8) -> pd.DataFrame:
    rows: list[dict] = []
    empty = 0
    failures: list[str] = []

    for start in range(1, max_draw + 1, block_size):
        block: list[dict] = []
        urls = [
            OLD_URL.format(start=start),
            DETAIL_URL.format(start=start, end=start + block_size - 1),
        ]
        for url in urls:
            try:
                block = fetch_page(url)
            except Exception as exc:
                failures.append(f"{url} -> {exc}")
                block = []
            if block:
                print(f"scraped start={start} rows={len(block)} url={url}")
                break
            time.sleep(sleep_seconds)

        if block:
            empty = 0
            rows.extend(block)
        else:
            empty += 1
            print(f"empty block start={start} empty_count={empty}")
            if empty >= max_empty_blocks:
                break
        time.sleep(sleep_seconds)

    try:
        current_rows = fetch_page(CURRENT_URL)
        if current_rows:
            print(f"scraped current rows={len(current_rows)}")
            rows.extend(current_rows)
    except Exception as exc:
        failures.append(f"{CURRENT_URL} -> {exc}")

    if not rows:
        diagnostic = "\n".join(failures[-20:]) if failures else "no fetch attempts recorded"
        raise RuntimeError(f"No Loto6 rows scraped. Last failures:\n{diagnostic}")

    df = normalize_draw_dataframe(pd.DataFrame(rows))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} rows={len(df)} first_draw={int(df['draw_no'].min())} latest_draw={int(df['draw_no'].max())}")
    return df


def _read_existing_result(path: Path, top_n: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"draw_no", "rank", "prediction", "actual", "grade"}
    if not required.issubset(df.columns):
        raise ValueError(f"Existing result file is incompatible: {path}")
    counts = df.groupby("draw_no")["rank"].nunique()
    complete = set(counts[counts >= top_n].index.astype(int))
    return df[df["draw_no"].astype(int).isin(complete)].copy()


def _write_outputs(result_df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "backtest_result.csv"
    summary_path = output_dir / "backtest_summary.csv"
    progress_path = output_dir / "backtest_progress.json"
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
    result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def git_commit_and_push(paths: Sequence[Path | str], message: str, max_attempts: int = 3) -> bool:
    """Commit selected paths and push. Returns True when a commit was pushed."""
    branch = _current_branch()
    _run_git(["config", "user.name", os.environ.get("GIT_COMMITTER_NAME", "github-actions")], check=False)
    _run_git(["config", "user.email", os.environ.get("GIT_COMMITTER_EMAIL", "github-actions@github.com")], check=False)
    _run_git(["add", *[str(p) for p in paths]])
    diff = _run_git(["diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
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


def _summarize_result(result_df: pd.DataFrame, total_draws: int, top_n: int, min_train_draws: int, latest_draw_no: int | None) -> dict:
    tickets = int(len(result_df))
    cost = tickets * TICKET_PRICE_YEN
    payout = int(result_df["reference_payout_yen"].sum()) if not result_df.empty else 0
    completed_draws = int(result_df["draw_no"].nunique()) if not result_df.empty else 0
    max_main_matches = int(result_df["main_matches"].max()) if not result_df.empty else 0
    grade_counts = dict(Counter(result_df["grade"])) if not result_df.empty else {}
    return {
        "mode": "walk_forward_no_future_leakage",
        "total_source_draws": int(total_draws),
        "min_train_draws": int(min_train_draws),
        "completed_draws": completed_draws,
        "latest_completed_draw_no": int(latest_draw_no) if latest_draw_no is not None else None,
        "tickets": tickets,
        "top_n": int(top_n),
        "cost_yen": cost,
        "reference_payout_yen": payout,
        "reference_profit_yen": payout - cost,
        "reference_roi": round(payout / cost, 6) if cost else 0.0,
        "grade_counts": grade_counts,
        "max_main_matches": max_main_matches,
    }


def resumable_walk_forward_backtest(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs",
    top_n: int = 5,
    candidate_count: int = 3000,
    seed: int = 42,
    min_train_draws: int = 1,
    resume: bool = True,
    push_every: int = 0,
    push_final: bool = True,
    max_draws: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Verify predictions draw-by-draw with no future leakage.

    Draw index i is predicted from draws [:i] only.
    Therefore the first verifiable target is the second historical draw when min_train_draws=1.
    """
    draws = normalize_draw_dataframe(df)
    if len(draws) <= min_train_draws:
        raise ValueError(f"Need more than min_train_draws={min_train_draws} rows; got {len(draws)}")

    output_path = Path(output_dir)
    result_path = output_path / "backtest_result.csv"
    tracked_paths = [result_path, output_path / "backtest_summary.csv", output_path / "backtest_progress.json"]
    result_df = _read_existing_result(result_path, top_n) if resume else pd.DataFrame()
    completed = set(result_df["draw_no"].astype(int)) if not result_df.empty else set()
    new_since_push = 0
    target_indexes = list(range(min_train_draws, len(draws)))
    if max_draws is not None:
        target_indexes = target_indexes[:max_draws]

    for idx in target_indexes:
        actual = draws.iloc[idx]
        draw_no = int(actual["draw_no"])
        if draw_no in completed:
            continue

        train_df = draws.iloc[:idx].copy()
        if draw_no in set(train_df["draw_no"].astype(int)):
            raise RuntimeError(f"Future leakage guard failed for draw_no={draw_no}")

        predictions = Loto6Predictor(seed=seed + draw_no).fit(train_df).predict(top_n, candidate_count)
        main = [int(actual[c]) for c in NUMBER_COLUMNS]
        bonus = int(actual["bonus"])
        records = []

        for pred in predictions:
            match = classify_loto6(pred.numbers, main, bonus)
            records.append({
                "draw_no": draw_no,
                "date": actual["date"].date().isoformat(),
                "rank": pred.rank,
                "prediction": pred.numbers_text,
                "actual": " ".join(f"{x:02d}" for x in main),
                "bonus": f"{bonus:02d}",
                "main_matches": match.main_matches,
                "bonus_match": bool(match.bonus_match),
                "grade": match.grade,
                "score": round(pred.score, 6),
                "source": pred.source,
                "train_draws": int(len(train_df)),
                "train_until_draw_no": int(train_df["draw_no"].max()),
                "reference_payout_yen": REFERENCE_PAYOUT_YEN[match.grade],
            })

        result_df = pd.concat([result_df, pd.DataFrame(records)], ignore_index=True)
        completed.add(draw_no)
        new_since_push += 1
        latest = int(result_df["draw_no"].max()) if not result_df.empty else None
        summary = _summarize_result(result_df, len(draws), top_n, min_train_draws, latest)
        _write_outputs(result_df, summary, output_path)
        print(f"verified draw={draw_no} train_draws={len(train_df)} completed={summary['completed_draws']} max_match={summary['max_main_matches']}")

        if push_every > 0 and new_since_push >= push_every:
            git_commit_and_push(tracked_paths, f"Backtest progress up to draw {draw_no} [skip ci]")
            new_since_push = 0

    latest = int(result_df["draw_no"].max()) if not result_df.empty else None
    summary = _summarize_result(result_df, len(draws), top_n, min_train_draws, latest)
    _write_outputs(result_df, summary, output_path)
    if push_every > 0 and push_final and new_since_push > 0:
        suffix = latest if latest is not None else "none"
        git_commit_and_push(tracked_paths, f"Backtest final progress up to draw {suffix} [skip ci]")
    return result_df, summary


def walk_forward_backtest(df: pd.DataFrame, start_after: int = 300, top_n: int = 5, candidate_count: int = 3000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Compatibility wrapper. Prefer resumable_walk_forward_backtest()."""
    tmp_dir = Path("outputs")
    return resumable_walk_forward_backtest(
        df,
        output_dir=tmp_dir,
        top_n=top_n,
        candidate_count=candidate_count,
        seed=seed,
        min_train_draws=start_after,
        resume=False,
        push_every=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Loto6 scraper / predictor / resumable verifier")
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
    p.add_argument("--min-train-draws", type=int, default=1, help="1 means the first target is draw 2, trained only on draw 1.")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--candidates", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--push-every", type=int, default=0, help="Commit and push every N newly verified draws. Use 100 in GitHub Actions.")
    p.add_argument("--push-final", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-draws", type=int, default=None, help="Debug limit for number of target draws to verify.")

    args = parser.parse_args()
    if hasattr(args, "max_draw"):
        df = update_csv(args.csv, max_draw=args.max_draw)
        print(f"updated: {args.csv} rows={len(df)} latest_draw={int(df['draw_no'].max())}")
    elif hasattr(args, "recent_window"):
        preds = Loto6Predictor(seed=args.seed, recent_window=args.recent_window).fit(load_draws(args.csv)).predict(args.n, args.candidates)
        out_df = predictions_to_dataframe(preds)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False, encoding="utf-8")
        print(out_df.to_string(index=False))
    else:
        result_df, summary = resumable_walk_forward_backtest(
            load_draws(args.csv),
            output_dir=args.output_dir,
            top_n=args.top_n,
            candidate_count=args.candidates,
            seed=args.seed,
            min_train_draws=args.min_train_draws,
            resume=args.resume,
            push_every=args.push_every,
            push_final=args.push_final,
            max_draws=args.max_draws,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
