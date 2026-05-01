#!/usr/bin/env python3
"""Robust Loto6 scraper for GitHub Actions.

Why this exists:
- Mizuho's plain current URL can return HTTP 403 from GitHub Actions.
- Month URLs such as index.html?year=2026&month=3 may be available when the plain URL is blocked.
- Older static backnumber pages and a fallback Loto6 result table are tried as backup sources.

Output format:
    draw_no,date,n1,n2,n3,n4,n5,n6,bonus
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

MIN_NUMBER = 1
MAX_NUMBER = 43
PICK_SIZE = 6
NUMBER_COLUMNS = [f"n{i}" for i in range(1, 7)]
DRAW_COLUMNS = ["draw_no", "date", *NUMBER_COLUMNS, "bonus"]

MIZUHO_MONTH_URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year={year}&month={month}"
MIZUHO_OLD_URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto6{start:04d}.html"
MIZUHO_DETAIL_URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/detail.html?fromto={start}_{end}&type=loto6"
TAKARAKUJI_LEGACY_URL = "https://www.takarakuji.co.jp/loto6/lt6ts/lt6ts.htm"

DRAW_RE = re.compile(r"第?\s*(\d{1,5})\s*回")
DATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
NUM_RE = re.compile(r"\d{1,2}")
ERA_DATE_RE = re.compile(r"([HRS])\s*(\d{1,2})\s*[./年]\s*(\d{1,2})\s*[./月]\s*(\d{1,2})")


def headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Referer": "https://www.mizuhobank.co.jp/takarakuji/check/loto/index.html",
    }


def normalize_numbers(numbers: Iterable[int]) -> tuple[int, ...]:
    nums = tuple(sorted(int(n) for n in numbers))
    if len(nums) != PICK_SIZE:
        raise ValueError(f"need 6 numbers: {nums}")
    if len(set(nums)) != PICK_SIZE:
        raise ValueError(f"duplicated numbers: {nums}")
    if min(nums) < MIN_NUMBER or max(nums) > MAX_NUMBER:
        raise ValueError(f"out of range: {nums}")
    return nums


def row_dict(draw_no: int, y: int, m: int, d: int, main: Iterable[int], bonus: int) -> dict | None:
    try:
        nums = normalize_numbers(main)
        bonus = int(bonus)
        if not (MIN_NUMBER <= bonus <= MAX_NUMBER):
            return None
    except Exception:
        return None
    return {
        "draw_no": int(draw_no),
        "date": f"{int(y):04d}-{int(m):02d}-{int(d):02d}",
        **{f"n{i}": nums[i - 1] for i in range(1, 7)},
        "bonus": bonus,
    }


def normalize_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DRAW_COLUMNS)
    df = pd.DataFrame(rows)[DRAW_COLUMNS].copy()
    df["draw_no"] = pd.to_numeric(df["draw_no"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in NUMBER_COLUMNS + ["bonus"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=DRAW_COLUMNS).copy()
    df["draw_no"] = df["draw_no"].astype(int)
    for c in NUMBER_COLUMNS + ["bonus"]:
        df[c] = df[c].astype(int)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.drop_duplicates("draw_no", keep="last").sort_values("draw_no").reset_index(drop=True)


def fetch(url: str, timeout: int = 30) -> str:
    res = requests.get(url, headers=headers(), timeout=timeout)
    res.raise_for_status()
    res.encoding = res.apparent_encoding or res.encoding
    return res.text


def parse_mizuho_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    seen: set[int] = set()

    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        joined = " ".join(cells)
        draw = DRAW_RE.search(joined)
        dt = DATE_RE.search(joined)
        if not draw or not dt:
            continue

        nums: list[int] = []
        for cell in cells:
            if DRAW_RE.search(cell) or DATE_RE.search(cell):
                continue
            for token in NUM_RE.findall(cell):
                v = int(token)
                if MIN_NUMBER <= v <= MAX_NUMBER:
                    nums.append(v)

        if len(nums) < 7:
            continue
        y, m, d = map(int, dt.groups())
        row = row_dict(int(draw.group(1)), y, m, d, nums[:6], nums[6])
        if row and row["draw_no"] not in seen:
            rows.append(row)
            seen.add(row["draw_no"])

    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    pattern = re.compile(
        r"第\s*(\d{1,5})\s*回\s+"
        r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s+"
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})"
    )
    for m in pattern.finditer(text):
        draw_no = int(m.group(1))
        row = row_dict(draw_no, int(m.group(2)), int(m.group(3)), int(m.group(4)), [int(m.group(i)) for i in range(5, 11)], int(m.group(11)))
        if row and row["draw_no"] not in seen:
            rows.append(row)
            seen.add(row["draw_no"])

    return rows


def era_to_year(era: str, year: int) -> int:
    era = era.upper()
    if era == "H":
        return 1988 + year
    if era == "R":
        return 2018 + year
    if era == "S":
        return 1925 + year
    raise ValueError(f"unsupported era: {era}")


def parse_era_date(value: str) -> str | None:
    m = ERA_DATE_RE.search(str(value).replace(" ", ""))
    if not m:
        return None
    y = era_to_year(m.group(1), int(m.group(2)))
    return f"{y:04d}-{int(m.group(3)):02d}-{int(m.group(4)):02d}"


def parse_takarakuji_legacy(html: str) -> list[dict]:
    rows: list[dict] = []
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return rows

    for table in tables:
        if table.empty:
            continue
        table.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip() for c in table.columns]
        cols = list(table.columns)
        draw_col = next((c for c in cols if c == "回" or "回" in c), None)
        date_col = next((c for c in cols if "抽" in c and "日" in c), None)
        main_col = next((c for c in cols if "数字" in c and "ボ" not in c), None)
        bonus_col = next((c for c in cols if "ボ" in c), None)
        if not all([draw_col, date_col, main_col, bonus_col]):
            continue

        for _, r in table.iterrows():
            draw_match = re.search(r"\d+", str(r[draw_col]))
            draw_no = int(draw_match.group()) if draw_match else None
            dt = parse_era_date(str(r[date_col]))
            nums = [int(x) for x in NUM_RE.findall(str(r[main_col])) if MIN_NUMBER <= int(x) <= MAX_NUMBER]
            bonus_match = re.search(r"\d{1,2}", str(r[bonus_col]))
            bonus = int(bonus_match.group()) if bonus_match else None
            if draw_no and dt and len(nums) >= 6 and bonus:
                y, m, d = map(int, dt.split("-"))
                row = row_dict(draw_no, y, m, d, nums[:6], bonus)
                if row:
                    rows.append(row)
    return rows


def month_sequence(start_year: int, start_month: int, months_back: int) -> list[tuple[int, int]]:
    result = []
    y, m = start_year, start_month
    for _ in range(months_back):
        result.append((y, m))
        m -= 1
        if m == 0:
            y -= 1
            m = 12
    return result


def scrape_month_urls(start_year: int, start_month: int, months_back: int) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    failures: list[str] = []
    for y, m in month_sequence(start_year, start_month, months_back):
        url = MIZUHO_MONTH_URL.format(year=y, month=m)
        try:
            parsed = parse_mizuho_html(fetch(url))
            print(f"month url rows={len(parsed)} url={url}")
            rows.extend(parsed)
        except Exception as exc:
            failures.append(f"{url} -> {exc}")
        time.sleep(0.3)
    return rows, failures


def scrape_mizuho_backnumbers(max_draw: int, block_size: int) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    failures: list[str] = []
    consecutive_empty = 0
    for start in range(1, max_draw + 1, block_size):
        urls = [MIZUHO_OLD_URL.format(start=start), MIZUHO_DETAIL_URL.format(start=start, end=start + block_size - 1)]
        parsed: list[dict] = []
        for url in urls:
            try:
                parsed = parse_mizuho_html(fetch(url))
                print(f"backnumber rows={len(parsed)} start={start} url={url}")
                if parsed:
                    break
            except Exception as exc:
                failures.append(f"{url} -> {exc}")
        if parsed:
            rows.extend(parsed)
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty >= 8:
                break
        time.sleep(0.3)
    return rows, failures


def scrape_takarakuji_legacy() -> tuple[list[dict], list[str]]:
    try:
        parsed = parse_takarakuji_legacy(fetch(TAKARAKUJI_LEGACY_URL))
        print(f"legacy rows={len(parsed)} url={TAKARAKUJI_LEGACY_URL}")
        return parsed, []
    except Exception as exc:
        return [], [f"{TAKARAKUJI_LEGACY_URL} -> {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust Loto6 scraper")
    parser.add_argument("--csv", default="data/loto6.csv")
    parser.add_argument("--max-draw", type=int, default=9999)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--month-year", type=int, default=date.today().year)
    parser.add_argument("--month", type=int, default=date.today().month)
    parser.add_argument("--months-back", type=int, default=18)
    args = parser.parse_args()

    all_rows: list[dict] = []
    failures: list[str] = []

    # User-provided style URL first: index.html?year=YYYY&month=M
    rows, errs = scrape_month_urls(args.month_year, args.month, args.months_back)
    all_rows.extend(rows)
    failures.extend(errs)

    # Full official-ish backnumber pages.
    rows, errs = scrape_mizuho_backnumbers(args.max_draw, args.block_size)
    all_rows.extend(rows)
    failures.extend(errs)

    # Fallback table. This is used when Mizuho blocks GitHub-hosted HTTP requests.
    if len(all_rows) < 100:
        rows, errs = scrape_takarakuji_legacy()
        all_rows.extend(rows)
        failures.extend(errs)

    df = normalize_df(all_rows)
    if df.empty:
        print("No Loto6 rows scraped. Last failures:", file=sys.stderr)
        for msg in failures[-40:]:
            print(msg, file=sys.stderr)
        return 1

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} rows={len(df)} first_draw={df['draw_no'].min()} latest_draw={df['draw_no'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
