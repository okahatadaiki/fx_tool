# ============================================================
# research_v11.py
# 岡畠AI研究所 Ver.0.11 トレード履歴フィルター深掘り
# 目的:
#   v09 の trades_v09.csv を使い、負けやすい曜日・時間・月を除外した場合の改善度を比較する。
# 実行:
#   python research_v11.py
# 出力:
#   data/research_v11/
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    base_dir: Path = Path(__file__).parent
    input_trades: Path = Path(__file__).parent / "data" / "research_v09" / "trades_v09.csv"
    out_dir: Path = Path(__file__).parent / "data" / "research_v11"
    top_n: int = 30
    min_trades_after_filter: int = 250


WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def read_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"trades_v09.csv がありません: {path}\n"
            "先に python research_v09.py を実行してください。"
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("trades_v09.csv が空です。")

    required = {"entry_time", "pnl_pips"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {missing}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["entry_time", "pnl_pips"]).copy()
    df["pnl_pips"] = pd.to_numeric(df["pnl_pips"], errors="coerce")
    df = df.dropna(subset=["pnl_pips"]).copy()

    df["year"] = df["entry_time"].dt.year
    df["month"] = df["entry_time"].dt.to_period("M").astype(str)
    df["month_num"] = df["entry_time"].dt.month
    df["weekday_num"] = df["entry_time"].dt.weekday
    df["weekday"] = df["weekday_num"].map(lambda x: WEEKDAY_JP[int(x)] if 0 <= int(x) <= 6 else str(x))
    df["hour"] = df["entry_time"].dt.hour
    return df


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min()) if len(dd) else 0.0


def max_losing_streak(pnl: pd.Series) -> int:
    max_streak = 0
    cur = 0
    for x in pnl:
        if x < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def monthly_positive_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    m = df.groupby("month")["pnl_pips"].sum()
    if len(m) == 0:
        return 0.0
    return float((m > 0).mean() * 100)


def yearly_positive_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    y = df.groupby("year")["pnl_pips"].sum()
    if len(y) == 0:
        return 0.0
    return float((y > 0).mean() * 100)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    pnl = df["pnl_pips"]
    total = float(pnl.sum())
    trades = int(len(df))
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    pf = profit_factor(pnl)
    exp = float(total / trades) if trades else 0.0
    win_rate = float(wins / trades * 100) if trades else 0.0
    dd = max_drawdown(pnl)
    ml = max_losing_streak(pnl)
    month_rate = monthly_positive_rate(df)
    year_rate = yearly_positive_rate(df)
    worst_month = float(df.groupby("month")["pnl_pips"].sum().min()) if trades else 0.0
    return {
        "取引数": trades,
        "損益pips": round(total, 1),
        "PF": round(pf, 3) if np.isfinite(pf) else 999.0,
        "期待値": round(exp, 3),
        "勝率%": round(win_rate, 1),
        "最大DD": round(dd, 1),
        "最大連敗": ml,
        "月プラス率%": round(month_rate, 1),
        "年プラス率%": round(year_rate, 1),
        "最悪月pips": round(worst_month, 1),
        "勝ち数": wins,
        "負け数": losses,
    }


def score_summary(s: Dict[str, float], base_trades: int) -> float:
    # 実運用向けの雑な採点。利益だけでなく安定性も見る。
    trade_penalty = max(0, (base_trades - s["取引数"]) / max(base_trades, 1)) * 10
    score = 0.0
    score += min(max(s["PF"] - 1.0, 0), 1.0) * 35
    score += min(max(s["期待値"], 0), 5.0) * 6
    score += min(max(s["月プラス率%"] - 45, 0), 40) * 0.6
    score += min(max(s["年プラス率%"] - 50, 0), 50) * 0.35
    score += min(max(s["損益pips"], 0), 1500) / 1500 * 20
    score += max(0, 10 + s["最大DD"] / 30)  # DDが小さいほど加点
    score -= trade_penalty
    if s["取引数"] < 250:
        score -= 20
    if s["PF"] < 1.1:
        score -= 15
    if s["期待値"] < 1.0:
        score -= 10
    return round(score, 2)


def apply_rule(df: pd.DataFrame, exclude_weekdays=(), exclude_hours=(), exclude_month_nums=()) -> pd.DataFrame:
    out = df.copy()
    if exclude_weekdays:
        out = out[~out["weekday"].isin(exclude_weekdays)]
    if exclude_hours:
        out = out[~out["hour"].isin(exclude_hours)]
    if exclude_month_nums:
        out = out[~out["month_num"].isin(exclude_month_nums)]
    return out.copy()


def make_rule_name(wds=(), hrs=(), mons=()) -> str:
    parts = []
    if wds:
        parts.append("除外曜日=" + ",".join(map(str, wds)))
    if hrs:
        parts.append("除外時間=" + ",".join(f"{h}時" for h in hrs))
    if mons:
        parts.append("除外月=" + ",".join(f"{m}月" for m in mons))
    return " / ".join(parts) if parts else "フィルターなし"


def generate_rules(df: pd.DataFrame) -> List[Tuple[str, Tuple[str, ...], Tuple[int, ...], Tuple[int, ...]]]:
    rules = [("フィルターなし", tuple(), tuple(), tuple())]

    weekdays = list(df.groupby("weekday")["pnl_pips"].sum().sort_values().index)
    bad_weekdays = [w for w in weekdays if df.loc[df["weekday"] == w, "pnl_pips"].sum() < 0]

    hours = list(df.groupby("hour")["pnl_pips"].sum().sort_values().index)
    bad_hours = [int(h) for h in hours if df.loc[df["hour"] == h, "pnl_pips"].sum() < 0]

    months = list(df.groupby("month_num")["pnl_pips"].sum().sort_values().index)
    bad_months = [int(m) for m in months if df.loc[df["month_num"] == m, "pnl_pips"].sum() < 0]

    # 単体除外
    for w in bad_weekdays:
        rules.append((make_rule_name((w,), (), ()), (w,), tuple(), tuple()))
    for h in bad_hours:
        rules.append((make_rule_name((), (h,), ()), tuple(), (h,), tuple()))
    for m in bad_months:
        rules.append((make_rule_name((), (), (m,)), tuple(), tuple(), (m,)))

    # 2条件組み合わせ
    for w in bad_weekdays:
        for h in bad_hours:
            rules.append((make_rule_name((w,), (h,), ()), (w,), (h,), tuple()))
    for w in bad_weekdays:
        for m in bad_months:
            rules.append((make_rule_name((w,), (), (m,)), (w,), tuple(), (m,)))
    for h in bad_hours:
        for m in bad_months:
            rules.append((make_rule_name((), (h,), (m,)), tuple(), (h,), (m,)))

    # 悪い上位2つをまとめて除外。ただしやりすぎ防止で各カテゴリ最大2個。
    for combo in combinations(bad_weekdays[:3], 2):
        rules.append((make_rule_name(combo, (), ()), tuple(combo), tuple(), tuple()))
    for combo in combinations(bad_hours[:3], 2):
        rules.append((make_rule_name((), combo, ()), tuple(), tuple(combo), tuple()))
    for combo in combinations(bad_months[:3], 2):
        rules.append((make_rule_name((), (), combo), tuple(), tuple(), tuple(combo)))

    # 重複削除
    seen = set()
    unique = []
    for name, w, h, m in rules:
        key = (tuple(sorted(w)), tuple(sorted(h)), tuple(sorted(m)))
        if key not in seen:
            seen.add(key)
            unique.append((name, tuple(w), tuple(h), tuple(m)))
    return unique


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_trades(cfg.input_trades)
    base = summarize(df)
    base_score = score_summary(base, len(df))

    rows = []
    for name, wds, hrs, mons in generate_rules(df):
        filtered = apply_rule(df, wds, hrs, mons)
        s = summarize(filtered)
        s["ルール"] = name
        s["除外曜日"] = ",".join(map(str, wds))
        s["除外時間"] = ",".join(map(str, hrs))
        s["除外月"] = ",".join(map(str, mons))
        s["スコア"] = score_summary(s, len(df))
        s["元との差損益"] = round(s["損益pips"] - base["損益pips"], 1)
        s["元との差PF"] = round(s["PF"] - base["PF"], 3)
        s["元との差DD"] = round(s["最大DD"] - base["最大DD"], 1)
        if s["取引数"] < cfg.min_trades_after_filter:
            s["判定"] = "除外候補"
        elif s["PF"] >= 1.25 and s["期待値"] >= 2.0 and s["月プラス率%"] >= 55 and s["損益pips"] > base["損益pips"]:
            s["判定"] = "改善候補"
        elif s["PF"] > base["PF"] and s["最大DD"] > base["最大DD"] and s["損益pips"] >= base["損益pips"] * 0.9:
            s["判定"] = "監視候補"
        else:
            s["判定"] = "参考"
        rows.append(s)

    res = pd.DataFrame(rows)
    cols = [
        "判定", "スコア", "ルール", "取引数", "損益pips", "PF", "期待値", "勝率%", "最大DD", "最大連敗",
        "月プラス率%", "年プラス率%", "最悪月pips", "元との差損益", "元との差PF", "元との差DD",
        "除外曜日", "除外時間", "除外月", "勝ち数", "負け数",
    ]
    res = res[cols].sort_values(["判定", "スコア", "損益pips"], ascending=[True, False, False])

    # 見やすいランキングはスコア順
    ranking = res.sort_values(["スコア", "損益pips", "PF"], ascending=[False, False, False]).reset_index(drop=True)

    # グループ別分析
    by_weekday = df.groupby("weekday").apply(lambda g: pd.Series(summarize(g))).reset_index()
    by_hour = df.groupby("hour").apply(lambda g: pd.Series(summarize(g))).reset_index()
    by_month_num = df.groupby("month_num").apply(lambda g: pd.Series(summarize(g))).reset_index()
    by_exit = df.groupby("exit_reason").apply(lambda g: pd.Series(summarize(g))).reset_index() if "exit_reason" in df.columns else pd.DataFrame()

    # 出力
    ranking.to_csv(cfg.out_dir / "filter_ranking_v11.csv", index=False, encoding="utf-8-sig")
    by_weekday.to_csv(cfg.out_dir / "weekday_analysis_v11.csv", index=False, encoding="utf-8-sig")
    by_hour.to_csv(cfg.out_dir / "hour_analysis_v11.csv", index=False, encoding="utf-8-sig")
    by_month_num.to_csv(cfg.out_dir / "monthnum_analysis_v11.csv", index=False, encoding="utf-8-sig")
    if not by_exit.empty:
        by_exit.to_csv(cfg.out_dir / "exit_reason_analysis_v11.csv", index=False, encoding="utf-8-sig")

    top = ranking.head(cfg.top_n)
    best = ranking.iloc[0].to_dict()
    improve = ranking[ranking["判定"].isin(["改善候補", "監視候補"])].head(10)

    lines = []
    lines.append("=" * 70)
    lines.append("岡畠AI研究所 Ver.0.11 フィルター深掘りレポート")
    lines.append("=" * 70)
    lines.append(f"入力トレード数: {len(df)}件")
    lines.append(f"元データ: 損益{base['損益pips']}pips / PF{base['PF']} / 期待値{base['期待値']} / DD{base['最大DD']} / 月プラス率{base['月プラス率%']}%")
    lines.append("")
    lines.append("上位候補:")
    for i, r in top.head(10).iterrows():
        lines.append(
            f"{i+1}. [{r['判定']}] {r['ルール']} / score={r['スコア']} / 損益{r['損益pips']}pips "
            f"/ PF{r['PF']} / 期待値{r['期待値']} / DD{r['最大DD']} / 月率{r['月プラス率%']}% / 取引{r['取引数']}回"
        )
    lines.append("")
    lines.append("次にやること:")
    lines.append("1. filter_ranking_v11.csv の上位候補を確認する。")
    lines.append("2. 改善候補があれば v12 でその条件だけを使い、詳細トレード履歴を再出力する。")
    lines.append("3. 改善候補がなければ、MACD以外の戦略候補へ戻す。")
    lines.append("")
    lines.append("注意: これは研究用であり、利益を保証するものではありません。")
    lines.append("")
    lines.append("出力:")
    for fn in ["filter_ranking_v11.csv", "weekday_analysis_v11.csv", "hour_analysis_v11.csv", "monthnum_analysis_v11.csv", "research_v11_report.txt"]:
        lines.append(str(cfg.out_dir / fn))

    report = "\n".join(lines)
    (cfg.out_dir / "research_v11_report.txt").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
