# -*- coding: utf-8 -*-
"""
scrapingloto6.py

loto6.csv を最新化します。

みずほ銀行ページは CI 環境で 403 になるケースがあるため、
静的HTMLで当せん番号を取得しやすい「楽天×宝くじ」のページから取得します。

重複スキップ方針:
  - 既存CSVにある過去月は、原則として月ページ取得自体をスキップ
  - 取得した中に既存 draw_no が含まれる場合は追加しない
  - 保存前にも draw_no / date で重複除去

出力形式:
  draw_no,date,n1,n2,n3,n4,n5,n6,bonus

使い方:
  python scrapingloto6.py --csv data/loto6.csv --all --sleep 2
  python scrapingloto6.py --csv data/loto6.csv --months 6 --sleep 2
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin

import pandas as pd


RAKUTEN_PAST_INDEX = "https://takarakuji.rakuten.co.jp/backnumber/loto6_past/"
RAKUTEN_MONTH_URL = "https://takarakuji.rakuten.co.jp/backnumber/loto6/{yyyymm}/"
CSV_COLUMNS = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]
DEFAULT_FIRST_YEAR = 2000
DEFAULT_FIRST_MONTH = 10
DEFAULT_SLEEP_SECONDS = 2.0


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "ja,en;q=0.9",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return raw.decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_month_urls_from_past_index(past_index_html: str) -> list[str]:
    rels = set(re.findall(r"/backnumber/loto6/\d{6}/", past_index_html))
    month_keys = sorted({re.search(r"(\d{6})", r).group(1) for r in rels}, reverse=True)
    return [urljoin(RAKUTEN_PAST_INDEX, f"/backnumber/loto6/{k}/") for k in month_keys]


def _month_keys_from_range(start_year: int, start_month: int, end_year: int | None = None, end_month: int | None = None) -> list[str]:
    today = dt.date.today()
    ey = end_year or today.year
    em = end_month or today.month
    keys: list[str] = []
    y, m = int(start_year), int(start_month)
    while (y, m) <= (ey, em):
        keys.append(f"{y:04d}{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return keys


def _build_month_urls(months: int, all_history: bool, start_year: int, start_month: int) -> list[str]:
    if all_history:
        keys = _month_keys_from_range(start_year, start_month)
        return [RAKUTEN_MONTH_URL.format(yyyymm=k) for k in keys]

    idx_html = _http_get(RAKUTEN_PAST_INDEX)
    month_urls = _parse_month_urls_from_past_index(idx_html)
    if not month_urls:
        raise RuntimeError("月別ページURLの抽出に失敗しました。")
    return month_urls[: max(1, months)]


def _month_key_from_url(url: str) -> str | None:
    m = re.search(r"/backnumber/loto6/(\d{6})/", url)
    return m.group(1) if m else None


def _normalize_main_numbers(nums: list[int]) -> list[int]:
    if len(nums) != 6:
        raise ValueError(f"Loto6本数字は6個必要です: {nums}")
    if len(set(nums)) != 6:
        raise ValueError(f"Loto6本数字に重複があります: {nums}")
    if min(nums) < 1 or max(nums) > 43:
        raise ValueError(f"Loto6本数字は1〜43です: {nums}")
    return sorted(nums)


def _parse_draws_from_month_page(month_html: str) -> list[dict]:
    text = _strip_html(month_html)
    parts = re.split(r"回号\s*第", text)
    out: list[dict] = []

    for seg in parts[1:]:
        m_draw = re.match(r"(\d{1,6})回\b", seg)
        if not m_draw:
            continue
        draw = int(m_draw.group(1))

        m_date = re.search(r"抽せん日\s*(\d{4}/\d{2}/\d{2})", seg)
        if not m_date:
            continue
        date_str = m_date.group(1)

        m_main = re.search(r"本数字\s*([0-9 ]+?)\s*ボーナス数字", seg)
        if not m_main:
            continue
        main_nums = [int(x) for x in m_main.group(1).split() if x.isdigit()]

        m_bonus = re.search(r"ボーナス数字\s*[\(\（]?(\d+)[\)\）]?", seg)
        if not m_bonus:
            continue
        bonus = int(m_bonus.group(1))

        try:
            main_nums = _normalize_main_numbers(main_nums)
        except ValueError:
            continue
        if not (1 <= bonus <= 43):
            continue

        out.append(
            {
                "draw_no": draw,
                "date": date_str,
                "n1": main_nums[0],
                "n2": main_nums[1],
                "n3": main_nums[2],
                "n4": main_nums[3],
                "n5": main_nums[4],
                "n6": main_nums[5],
                "bonus": bonus,
            }
        )

    return out


def _normalize_output_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)

    work = df.copy()
    work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["_date"]).copy()

    for col in ["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
    work = work.dropna(subset=["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]).copy()
    for col in ["draw_no", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
        work[col] = work[col].astype(int)

    # 同じ抽せん回は後勝ち、同じ日付も後勝ち。
    before = len(work)
    work = work.sort_values(["draw_no", "_date"]).drop_duplicates(subset=["draw_no"], keep="last")
    work = work.sort_values("_date").drop_duplicates(subset=["date"], keep="last")
    dropped = before - len(work)
    if dropped:
        print(f"[DEDUP] dropped_duplicate_rows={dropped}")

    work["date"] = work["_date"].dt.strftime("%Y-%m-%d")
    return work.drop(columns=["_date"])[CSV_COLUMNS]


def _load_existing(csv_path: str) -> pd.DataFrame:
    try:
        existing = pd.read_csv(csv_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=CSV_COLUMNS)

    if set(CSV_COLUMNS).issubset(existing.columns):
        return _normalize_output_df(existing[CSV_COLUMNS].copy())

    required = {"回別", "抽せん日", "本数字", "ボーナス数字"}
    if required.issubset(existing.columns):
        rows = []
        for _, r in existing.iterrows():
            nums = [int(x) for x in str(r["本数字"]).split() if x.isdigit()]
            bonus_values = [int(x) for x in re.findall(r"\d+", str(r["ボーナス数字"]))]
            if len(nums) != 6 or not bonus_values:
                continue
            try:
                nums = _normalize_main_numbers(nums)
            except ValueError:
                continue
            parsed_date = pd.to_datetime(r["抽せん日"], errors="coerce")
            if pd.isna(parsed_date):
                continue
            rows.append(
                {
                    "draw_no": int(r["回別"]),
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "n1": nums[0],
                    "n2": nums[1],
                    "n3": nums[2],
                    "n4": nums[3],
                    "n5": nums[4],
                    "n6": nums[5],
                    "bonus": bonus_values[0],
                }
            )
        return _normalize_output_df(pd.DataFrame(rows, columns=CSV_COLUMNS))

    return pd.DataFrame(columns=CSV_COLUMNS)


def fetch_latest_draws(
    months: int = 6,
    all_history: bool = False,
    start_year: int = DEFAULT_FIRST_YEAR,
    start_month: int = DEFAULT_FIRST_MONTH,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    existing_draws: set[int] | None = None,
    existing_months: set[str] | None = None,
    skip_existing_months: bool = True,
) -> pd.DataFrame:
    month_urls = _build_month_urls(months=months, all_history=all_history, start_year=start_year, start_month=start_month)
    if not month_urls:
        raise RuntimeError("取得対象の月別ページURLがありません。")

    existing_draws = existing_draws or set()
    existing_months = existing_months or set()
    current_month_key = dt.date.today().strftime("%Y%m")

    rows: list[dict] = []
    missing = 0
    skipped_months = 0
    skipped_existing_draws = 0

    for url in month_urls:
        month_key = _month_key_from_url(url)
        if skip_existing_months and month_key and month_key in existing_months and month_key != current_month_key:
            print(f"[SKIP_MONTH] {url} reason=already_in_existing_csv")
            skipped_months += 1
            continue

        try:
            html = _http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"[SKIP] {url} status=404")
                missing += 1
                time.sleep(max(0.0, sleep_seconds))
                continue
            raise

        parsed = _parse_draws_from_month_page(html)
        new_rows = []
        for row in parsed:
            draw_no = int(row["draw_no"])
            if draw_no in existing_draws:
                skipped_existing_draws += 1
                continue
            new_rows.append(row)

        print(
            f"[SCRAPE] {url} rows={len(parsed)} new_rows={len(new_rows)} "
            f"skipped_existing_draws={len(parsed) - len(new_rows)} sleep={sleep_seconds}"
        )
        rows.extend(new_rows)
        time.sleep(max(0.0, sleep_seconds))

    if not rows:
        print(
            f"[SUMMARY] no_new_rows missing_months={missing} "
            f"skipped_months={skipped_months} skipped_existing_draws={skipped_existing_draws}"
        )
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = _normalize_output_df(pd.DataFrame(rows))
    print(
        f"[SUMMARY] fetched_new_rows={len(rows)} unique_new_draws={len(df)} "
        f"missing_months={missing} skipped_months={skipped_months} "
        f"skipped_existing_draws={skipped_existing_draws}"
    )
    return df[CSV_COLUMNS]


def update_loto6_csv(
    csv_path: str = "data/loto6.csv",
    months: int = 6,
    all_history: bool = False,
    start_year: int = DEFAULT_FIRST_YEAR,
    start_month: int = DEFAULT_FIRST_MONTH,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    skip_existing_months: bool = True,
) -> pd.DataFrame:
    existing = _load_existing(csv_path)
    existing_draws = set(existing["draw_no"].astype(int).tolist()) if not existing.empty else set()
    existing_months = set(pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y%m").dropna().tolist()) if not existing.empty else set()

    latest = fetch_latest_draws(
        months=months,
        all_history=all_history,
        start_year=start_year,
        start_month=start_month,
        sleep_seconds=sleep_seconds,
        existing_draws=existing_draws,
        existing_months=existing_months,
        skip_existing_months=skip_existing_months,
    )

    merged = _normalize_output_df(pd.concat([existing, latest], ignore_index=True))

    parent = os.path.dirname(pd.io.common.stringify_path(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    merged.to_csv(csv_path, index=False, encoding="utf-8")

    if latest.empty:
        print(f"[OK] {csv_path}: 追加はありません（重複取得データはスキップ） total={len(merged)}")
    else:
        print(f"[OK] {csv_path}: {len(latest)}件 追加/更新しました total={len(merged)}")
        show = latest.sort_values("draw_no", ascending=False).head(20)
        for _, r in show.iterrows():
            main = " ".join(str(int(r[f"n{i}"])) for i in range(1, 7))
            print(f'  第{int(r["draw_no"]):04d}回 {r["date"]}  本: {main}  B: {int(r["bonus"])}')

    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/loto6.csv", help="出力/更新するCSVパス")
    ap.add_argument("--months", type=int, default=6, help="取得する直近月数")
    ap.add_argument("--all", action="store_true", help="第1回相当の開始月から現在月まで全期間取得")
    ap.add_argument("--start-year", type=int, default=DEFAULT_FIRST_YEAR, help="--all 時の開始年")
    ap.add_argument("--start-month", type=int, default=DEFAULT_FIRST_MONTH, help="--all 時の開始月")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="月別ページ取得ごとの待機秒数")
    ap.add_argument("--skip-existing-months", action=argparse.BooleanOptionalAction, default=True, help="既存CSVにある過去月の取得をスキップ")
    args = ap.parse_args(argv)
    update_loto6_csv(
        args.csv,
        months=max(1, args.months),
        all_history=bool(args.all),
        start_year=args.start_year,
        start_month=args.start_month,
        sleep_seconds=max(0.0, float(args.sleep)),
        skip_existing_months=bool(args.skip_existing_months),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
