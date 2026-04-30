# loto6

Loto6 専用のスクレイピング・予測・未来リークなし検証リポジトリです。

## 結論

このリポジトリは Loto6 のみに対応します。別宝くじ用のロジック・説明・成果物は含めません。

## 実装方針

- Loto6専用ルールに対応
  - 1〜43から異なる6個を選択
  - 抽せん結果は本数字6個 + ボーナス数字1個
  - ボーナス数字は2等判定のみで使用
- 未来リークなし検証
  - 対象抽せん回 `N` の予測には、`N` より前の抽せん結果だけを使用
  - デフォルトでは第1回だけで第2回を予測し、その後1回ずつ進める
- 抽せん回ごとに照合
  - 各対象回ごとに予測を生成
  - 実当せん結果と照合
  - 等級・一致数・ボーナス一致をCSVへ保存
- 中断再開対応
  - `outputs/backtest_result.csv` を読み、完了済み抽せん回をスキップ
  - `outputs/backtest_progress.json` に進捗を保存
- 100抽せん回ごとにpush可能
  - `--push-every 100` を指定すると、100回分の新規検証ごとにcommit/push

## ファイル構成

```text
.
├── loto6.py                         # Loto6専用ロジック本体
├── requirements.txt                 # 依存ライブラリ
├── tests/test_loto6.py              # ルール・予測・再開検証テスト
├── .github/workflows/ci.yml         # 通常CI
├── .github/workflows/backtest.yml   # 手動実行用バックテスト
├── data/.gitkeep                    # CSV配置用
└── .gitignore
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## データ更新

```bash
python loto6.py update --csv data/loto6.csv
```

## 予測

```bash
python loto6.py predict --csv data/loto6.csv -n 5 --output outputs/predictions.csv
```

## 未来リークなし検証

最初から最後まで検証します。

```bash
python loto6.py backtest --csv data/loto6.csv --min-train-draws 1 --top-n 5 --resume --push-every 100
```

### 動作

| 項目 | 内容 |
|---|---|
| 第2回の予測 | 第1回だけで学習 |
| 第3回の予測 | 第1回〜第2回だけで学習 |
| 第100回の予測 | 第1回〜第99回だけで学習 |
| 中断後 | 完了済み抽せん回をスキップして続きから再開 |
| push | 新規検証100抽せん回ごとにcommit/push |

## 出力

```text
outputs/backtest_result.csv
outputs/backtest_summary.csv
outputs/backtest_progress.json
```

### `backtest_result.csv`

抽せん回ごと、予測順位ごとに以下を保存します。

| 列 | 内容 |
|---|---|
| draw_no | 検証対象の抽せん回 |
| date | 抽せん日 |
| rank | 予測順位 |
| prediction | 予測番号 |
| actual | 実際の本数字 |
| bonus | 実際のボーナス数字 |
| main_matches | 本数字一致数 |
| bonus_match | ボーナス一致 |
| grade | 等級 |
| train_draws | 学習に使った過去抽せん回数 |
| train_until_draw_no | 学習に使った最終抽せん回 |

## GitHub Actionsで実行

Actions の `Loto6 Backtest` から手動実行します。

推奨設定:

| 入力 | 値 |
|---|---|
| csv_path | `data/loto6.csv` |
| top_n | `5` |
| candidates | `3000` |
| min_train_draws | `1` |
| push_every | `100` |

## 注意

このリポジトリは予測・検証ロジックを提供しますが、当せんを保証するものではありません。宝くじは確率事象であり、過去データから将来の当せん番号を確定することはできません。
