# loto6

`Joker7822/loto7` を参考に、Loto6 専用として作り直した軽量リポジトリです。

## 結論

Loto7 の巨大な学習済みモデル・ログ・画像・生成CSVをコピーせず、Loto6 専用の最小構成で新規作成しています。

## 実装方針

- Loto6専用ルールに対応
  - 1〜43から異なる6個を選択
  - 抽せん結果は本数字6個 + ボーナス数字1個
  - ボーナス数字は2等判定のみで使用
- スクレイピング、CSV正規化、予測、walk-forward検証を1ファイルに集約
- Loto7由来の不要成果物は持ち込まない
- GitHub Actionsで最低限のCIを実行

## ファイル構成

```text
.
├── loto6.py                 # Loto6専用ロジック本体
├── requirements.txt         # 依存ライブラリ
├── tests/test_loto6.py      # ルール・予測の基本テスト
├── .github/workflows/ci.yml # CI
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

## 検証

```bash
python loto6.py backtest --csv data/loto6.csv --start-after 300 --top-n 5
```

出力:

- `outputs/backtest_result.csv`
- `outputs/backtest_summary.csv`

## 注意

このリポジトリは予測ロジックを提供しますが、当せんを保証するものではありません。宝くじは確率事象であり、過去データから将来の当せん番号を確定することはできません。
