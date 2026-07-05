# ============================================================
# analyze_candidates.py
# 岡畠AI研究所 採用候補分析ツール
#
# 実行方法:
#   python analyze_candidates.py
#
# 読み込み候補:
#   data/research_v06/adopted_candidates.csv
#   data/research/adopted_research_candidates.csv
#   data/backtest/adopted_candidates.csv
#
# 出力:
#   data/analysis/candidate_analysis.csv
#   data/analysis/candidate_analysis_report.txt
# ============================================================

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "analysis"

CANDIDATE_PATHS = [
    BASE_DIR / "data" / "research_v06" / "adopted_candidates.csv",
    BASE_DIR / "data" / "research_v06" / "adopted_research_candidates.csv",
    BASE_DIR / "data" / "research" / "adopted_research_candidates.csv",
    BASE_DIR / "data" / "backtest" / "adopted_candidates.csv",
]


def find_candidate_file() -> Path:
    for path in CANDIDATE_PATHS:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in CANDIDATE_PATHS)
    raise FileNotFoundError(
        "採用候補CSVが見つかりません。先に research_v06.py を実行してください。\n"
        f"探した場所:\n{searched}"
    )


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for name in candidates:
        if name in cols:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    col_pf = pick_col(out, ["test_pf", "検証PF", "pf_test", "PF", "profit_factor"])
    col_profit = pick_col(out, ["test_profit_pips", "検証損益", "検証損益pips", "profit_pips", "pips", "損益pips"])
    col_trades = pick_col(out, ["test_trades", "取引回数", "trades", "n_trades"])
    col_score = pick_col(out, ["score", "スコア"])
    col_train_pf = pick_col(out, ["train_pf", "学習PF", "pf_train"])
    col_strategy = pick_col(out, ["strategy", "戦略", "戦略タイプ", "strategy_type"])
    col_time = pick_col(out, ["time_filter", "時間帯", "time_band", "session"])
    col_weekday = pick_col(out, ["weekday_filter", "曜日", "weekday"])
    col_trend = pick_col(out, ["trend_filter", "トレンド環境", "trend"])
    col_vol = pick_col(out, ["vol_filter", "ボラ環境", "vol", "volatility"])

    out["__pf"] = to_num(out[col_pf]) if col_pf else math.nan
    out["__profit"] = to_num(out[col_profit]) if col_profit else math.nan
    out["__trades"] = to_num(out[col_trades]) if col_trades else math.nan
    out["__score"] = to_num(out[col_score]) if col_score else math.nan
    out["__train_pf"] = to_num(out[col_train_pf]) if col_train_pf else math.nan

    out["__strategy"] = out[col_strategy].astype(str) if col_strategy else "不明"
    out["__time"] = out[col_time].astype(str) if col_time else "不明"
    out["__weekday"] = out[col_weekday].astype(str) if col_weekday else "不明"
    out["__trend"] = out[col_trend].astype(str) if col_trend else "不明"
    out["__vol"] = out[col_vol].astype(str) if col_vol else "不明"

    # 1取引あたり期待pips
    out["期待pips/回"] = out["__profit"] / out["__trades"].replace(0, pd.NA)

    # 過剰最適化疑い: 学習PFが高いのに検証PFが低い
    out["PF劣化"] = out["__train_pf"] - out["__pf"]

    # 研究所スコア。単純すぎるPF偏重を避ける。
    pf_part = out["__pf"].fillna(0).clip(0, 3) * 30
    profit_part = (out["__profit"].fillna(0).clip(-500, 500) / 10)
    trade_part = (out["__trades"].fillna(0).clip(0, 300) / 10)
    expectancy_part = out["期待pips/回"].fillna(0).clip(-5, 5) * 10
    overfit_penalty = out["PF劣化"].fillna(0).clip(0, 5) * 8

    out["研究所評価"] = pf_part + profit_part + trade_part + expectancy_part - overfit_penalty

    def judge(row) -> str:
        pf = row.get("__pf", 0)
        profit = row.get("__profit", 0)
        trades = row.get("__trades", 0)
        exp = row.get("期待pips/回", 0)
        decay = row.get("PF劣化", 0)

        if pd.isna(pf) or pd.isna(profit) or pd.isna(trades):
            return "要確認"
        if trades < 100:
            return "取引数不足"
        if profit <= 0:
            return "見送り"
        if pf >= 1.3 and profit >= 150 and trades >= 120 and exp >= 0.5 and (pd.isna(decay) or decay <= 1.0):
            return "本命候補"
        if pf >= 1.15 and profit >= 80 and trades >= 100:
            return "準候補"
        return "要注意"

    out["判定"] = out.apply(judge, axis=1)
    return out.sort_values("研究所評価", ascending=False)


def summarize_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(col, dropna=False)
        .agg(
            件数=("__pf", "size"),
            平均PF=("__pf", "mean"),
            平均損益=("__profit", "mean"),
            平均取引数=("__trades", "mean"),
            本命数=("判定", lambda s: (s == "本命候補").sum()),
            準候補数=("判定", lambda s: (s == "準候補").sum()),
        )
        .sort_values(["平均PF", "平均損益"], ascending=False)
    )
    return g


def make_report(df: pd.DataFrame, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("岡畠AI研究所 採用候補分析レポート")
    lines.append("=" * 60)
    lines.append(f"読み込み元: {source}")
    lines.append(f"候補数: {len(df):,}件")
    lines.append("")

    lines.append("判定内訳:")
    for name, count in df["判定"].value_counts().items():
        lines.append(f"  {name}: {count:,}件")
    lines.append("")

    lines.append("上位10件:")
    show_cols = [
        "判定", "研究所評価", "__strategy", "__time", "__weekday", "__trend", "__vol",
        "__pf", "__profit", "__trades", "期待pips/回", "PF劣化"
    ]
    existing = [c for c in show_cols if c in df.columns]
    top = df[existing].head(10).copy()
    for i, row in top.iterrows():
        lines.append(
            f"{len(lines)}. [{row.get('判定','')}] "
            f"戦略={row.get('__strategy','')} / 時間={row.get('__time','')} / 曜日={row.get('__weekday','')} / "
            f"PF={row.get('__pf', float('nan')):.3f} / 損益={row.get('__profit', float('nan')):.1f}pips / "
            f"取引={row.get('__trades', float('nan')):.0f}回 / 期待={row.get('期待pips/回', float('nan')):.3f}pips/回"
        )
    lines.append("")

    for title, col in [
        ("戦略別", "__strategy"),
        ("時間帯別", "__time"),
        ("曜日別", "__weekday"),
        ("トレンド環境別", "__trend"),
        ("ボラ環境別", "__vol"),
    ]:
        lines.append(f"{title}の傾向:")
        g = summarize_group(df, col).head(10)
        if g.empty:
            lines.append("  データなし")
        else:
            for idx, row in g.iterrows():
                lines.append(
                    f"  {idx}: 件数{row['件数']:.0f} / 平均PF{row['平均PF']:.3f} / "
                    f"平均損益{row['平均損益']:.1f} / 本命{row['本命数']:.0f} / 準候補{row['準候補数']:.0f}"
                )
        lines.append("")

    lines.append("次にやるべきこと:")
    lines.append("1. 本命候補だけを別期間で再検証する。")
    lines.append("2. 期待pips/回が高い条件を優先して深掘りする。")
    lines.append("3. PF劣化が大きい条件は過剰最適化疑いとして除外する。")
    lines.append("4. v0.6では本命候補周辺だけを追加探索する。")
    lines.append("")
    lines.append("注意: これは研究用であり、利益を保証するものではありません。")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = find_candidate_file()
    df_raw = read_csv_safely(source)
    df = normalize(df_raw)

    out_csv = OUTPUT_DIR / "candidate_analysis.csv"
    out_report = OUTPUT_DIR / "candidate_analysis_report.txt"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    report = make_report(df, source)
    out_report.write_text(report, encoding="utf-8-sig")

    print(report)
    print("=" * 60)
    print("出力しました:")
    print(out_csv)
    print(out_report)


if __name__ == "__main__":
    main()
