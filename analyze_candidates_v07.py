# =====================================================
# analyze_candidates_v07.py
# 岡畠AI研究所 採用候補分析ツール Ver.0.7
#
# 実行方法:
#   python analyze_candidates_v07.py
#
# 読み込み候補:
#   data/research_v06/adopted_candidates.csv
#   data/research_v06/research_ranking.csv
#   data/research/adopted_research_candidates.csv
#   data/backtest/adopted_candidates.csv
#
# 出力先:
#   data/analysis_v07/
# =====================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data" / "analysis_v07"

CANDIDATE_PATHS = [
    BASE_DIR / "data" / "research_v06" / "adopted_candidates.csv",
    BASE_DIR / "data" / "research_v06" / "research_ranking.csv",
    BASE_DIR / "data" / "research_v04" / "adopted_candidates.csv",
    BASE_DIR / "data" / "research" / "adopted_research_candidates.csv",
    BASE_DIR / "data" / "backtest" / "adopted_candidates.csv",
]


def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"CSVを読めませんでした: {path}\n{last_error}")


def find_source() -> Path:
    existing = [p for p in CANDIDATE_PATHS if p.exists()]
    if not existing:
        msg = "候補CSVが見つかりません。先に python research_v06.py を実行してください。\n探した場所:\n"
        msg += "\n".join(str(p) for p in CANDIDATE_PATHS)
        raise FileNotFoundError(msg)

    # v06の採用候補を最優先。空ならrankingにフォールバック。
    for p in existing:
        try:
            df = read_csv_safely(p)
            if len(df) > 0:
                return p
        except Exception:
            continue
    return existing[0]


def pick_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def numeric(df: pd.DataFrame, col: Optional[str], default: float = np.nan) -> pd.Series:
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def text(df: pd.DataFrame, col: Optional[str], default: str = "") -> pd.Series:
    if col is None:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def classify(row: pd.Series) -> str:
    n = row.get("取引回数", 0)
    pf = row.get("PF", 0)
    total = row.get("総損益pips", 0)
    ev = row.get("期待値pips/回", 0)
    win = row.get("勝率%", 0)
    dd = row.get("最大DD_pips", 0)

    if pd.isna(n) or n < 100:
        return "要確認_取引数不足"
    if pd.isna(pf) or pd.isna(total) or pd.isna(ev):
        return "要確認_数値不足"
    if pf >= 1.3 and total > 0 and ev > 0 and win >= 45:
        return "本命候補"
    if pf >= 1.1 and total > 0 and ev > 0:
        return "準候補"
    if total > 0 and ev > 0:
        return "様子見"
    return "除外候補"


def build_score(df: pd.DataFrame) -> pd.Series:
    def rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            return pd.Series(0.0, index=s.index)
        return s.rank(pct=True, ascending=ascending).fillna(0)

    pf = df["PF"].clip(upper=5)
    total = df["総損益pips"]
    ev = df["期待値pips/回"]
    win = df["勝率%"]
    n = df["取引回数"]
    dd = df["最大DD_pips"].abs()
    loss_streak = df["最大連敗"]

    score = (
        0.25 * rank_pct(pf)
        + 0.25 * rank_pct(ev)
        + 0.20 * rank_pct(total)
        + 0.10 * rank_pct(win)
        + 0.10 * rank_pct(n)
        + 0.05 * rank_pct(dd, ascending=False)
        + 0.05 * rank_pct(loss_streak, ascending=False)
    ) * 100
    return score.round(1)


def normalize_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    strategy_col = pick_col(df, ["戦略", "strategy", "name"])
    type_col = pick_col(df, ["戦略タイプ", "strategy_type", "type"])
    tp_col = pick_col(df, ["利確pips", "TP", "tp", "tp_pips"])
    sl_col = pick_col(df, ["損切りpips", "SL", "sl", "sl_pips"])
    time_col = pick_col(df, ["時間帯", "time", "session"])
    weekday_col = pick_col(df, ["曜日", "weekday"])
    trend_col = pick_col(df, ["トレンド環境", "trend"])
    vol_col = pick_col(df, ["ボラ環境", "vol", "volatility"])
    verdict_col = pick_col(df, ["判定", "verdict", "judge"])

    pf_col = pick_col(df, ["検証_PF", "PF", "pf", "test_pf"])
    train_pf_col = pick_col(df, ["学習_PF", "train_pf"])
    n_col = pick_col(df, ["検証_取引回数", "取引回数", "trades", "test_trades"])
    win_col = pick_col(df, ["検証_勝率%", "勝率%", "win_rate", "test_win_rate"])
    total_col = pick_col(df, ["検証_総損益pips", "総損益pips", "total_pips", "test_total_pips"])
    avg_profit_col = pick_col(df, ["検証_平均利益pips", "平均利益pips", "avg_profit_pips"])
    avg_loss_col = pick_col(df, ["検証_平均損失pips", "平均損失pips", "avg_loss_pips"])
    dd_col = pick_col(df, ["検証_最大DD_pips", "最大DD_pips", "max_dd_pips"])
    loss_streak_col = pick_col(df, ["検証_最大連敗", "最大連敗", "max_loss_streak"])
    score_col = pick_col(df, ["総合スコア", "score"])

    out = pd.DataFrame(index=df.index)
    out["判定元"] = text(df, verdict_col, "")
    out["戦略"] = text(df, strategy_col, "")
    out["戦略タイプ"] = text(df, type_col, "")
    out["利確pips"] = numeric(df, tp_col)
    out["損切りpips"] = numeric(df, sl_col)
    out["時間帯"] = text(df, time_col, "全時間")
    out["曜日"] = text(df, weekday_col, "全曜日")
    out["トレンド環境"] = text(df, trend_col, "なし")
    out["ボラ環境"] = text(df, vol_col, "なし")

    out["学習PF"] = numeric(df, train_pf_col)
    out["PF"] = numeric(df, pf_col)
    out["取引回数"] = numeric(df, n_col, 0).fillna(0).astype(int)
    out["勝率%"] = numeric(df, win_col)
    out["総損益pips"] = numeric(df, total_col)
    out["平均利益pips"] = numeric(df, avg_profit_col)
    out["平均損失pips"] = numeric(df, avg_loss_col)
    out["最大DD_pips"] = numeric(df, dd_col)
    out["最大連敗"] = numeric(df, loss_streak_col)

    out["期待値pips/回"] = out["総損益pips"] / out["取引回数"].replace(0, np.nan)
    out["PF乖離"] = out["PF"] / out["学習PF"].replace(0, np.nan)
    out["旧スコア"] = numeric(df, score_col)
    out["v07スコア"] = build_score(out)
    out["v07判定"] = out.apply(classify, axis=1)

    cols = [
        "v07判定", "v07スコア", "判定元", "戦略", "戦略タイプ", "利確pips", "損切りpips",
        "時間帯", "曜日", "トレンド環境", "ボラ環境", "PF", "学習PF", "PF乖離",
        "総損益pips", "期待値pips/回", "勝率%", "取引回数", "平均利益pips", "平均損失pips",
        "最大DD_pips", "最大連敗", "旧スコア",
    ]
    out = out[cols]
    return out.sort_values(["v07スコア", "総損益pips", "PF"], ascending=False).reset_index(drop=True)


def summarize_dimension(df: pd.DataFrame, col: str, min_count: int = 1) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return pd.DataFrame()
    g = df.groupby(col).agg(
        件数=("PF", "size"),
        平均PF=("PF", "mean"),
        平均期待値=("期待値pips/回", "mean"),
        平均損益=("総損益pips", "mean"),
        プラス率=("総損益pips", lambda s: (s > 0).mean() * 100),
        本命数=("v07判定", lambda s: (s == "本命候補").sum()),
        準候補数=("v07判定", lambda s: (s == "準候補").sum()),
    ).round(3)
    return g[g["件数"] >= min_count].sort_values(["平均期待値", "平均PF"], ascending=False)


def build_report(source: Path, df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("岡畠AI研究所 採用候補分析レポート Ver.0.7")
    lines.append("=" * 70)
    lines.append(f"読み込み元: {source}")
    lines.append(f"候補数: {len(df)}件")
    lines.append("")
    lines.append("判定内訳:")
    for k, v in df["v07判定"].value_counts().items():
        lines.append(f"  {k}: {v}件")

    lines.append("")
    lines.append("上位20件:")
    for i, (_, r) in enumerate(df.head(20).iterrows(), 1):
        lines.append(
            f"{i}. [{r['v07判定']}] {r['戦略']} TP{r['利確pips']} SL{r['損切りpips']} "
            f"{r['時間帯']} {r['曜日']} {r['トレンド環境']} {r['ボラ環境']} "
            f"PF{r['PF']:.3f} 損益{r['総損益pips']:.1f}pips "
            f"期待値{r['期待値pips/回']:.3f}pips/回 勝率{r['勝率%']:.1f}% "
            f"取引{int(r['取引回数'])}回 スコア{r['v07スコア']:.1f}"
        )

    lines.append("")
    lines.append("条件別の傾向:")
    for col in ["戦略タイプ", "時間帯", "曜日", "トレンド環境", "ボラ環境", "利確pips", "損切りpips"]:
        tbl = summarize_dimension(df, col)
        if not tbl.empty:
            lines.append(f"\n[{col}]")
            lines.append(tbl.head(10).to_string())

    lines.append("")
    lines.append("次にやるべきこと:")
    lines.append("1. 本命候補と準候補だけを別期間で再検証する。")
    lines.append("2. 期待値がプラスの条件だけを深掘りする。")
    lines.append("3. PFだけ高くて期待値が低い条件は除外する。")
    lines.append("4. v0.8では候補周辺のTP/SLと時間帯を細かく再探索する。")
    lines.append("")
    lines.append("注意: これは研究用であり、利益を保証するものではありません。")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = find_source()
    raw = read_csv_safely(source)
    analyzed = normalize_candidates(raw)

    out_csv = OUT_DIR / "candidate_analysis_v07.csv"
    out_txt = OUT_DIR / "candidate_analysis_report_v07.txt"
    analyzed.to_csv(out_csv, index=False, encoding="utf-8-sig")
    report = build_report(source, analyzed)
    out_txt.write_text(report, encoding="utf-8")

    print(report)
    print()
    print("出力しました:")
    print(out_csv)
    print(out_txt)


if __name__ == "__main__":
    main()
