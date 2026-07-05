# ============================================================
# research_v10.py
# 岡畠AI研究所 Ver.0.10 フィルター改善比較ツール
#
# 実行方法:
#   python research_v10.py
#
# 入力:
#   data/research_v09/trades_v09.csv
#
# 出力:
#   data/research_v10/filter_comparison_v10.csv
#   data/research_v10/best_trades_v10.csv
#   data/research_v10/monthly_v10.csv
#   data/research_v10/yearly_v10.csv
#   data/research_v10/research_v10_report.txt
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple
import math
import sys

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = BASE_DIR / "data" / "research_v08" / "watch_candidates_v08.csv"
INPUT_TRADES = BASE_DIR / "data" / "research_v09" / "trades_v09.csv"
OUT_DIR = BASE_DIR / "data" / "research_v10"


def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_trades() -> pd.DataFrame:
    if not INPUT_TRADES.exists():
        raise FileNotFoundError(
            f"trades_v09.csv がありません: {INPUT_TRADES}\n"
            "先に python research_v09.py を実行してください。"
        )

    df = pd.read_csv(INPUT_TRADES)
    if df.empty:
        raise ValueError("trades_v09.csv が空です。v09を再実行してください。")

    required = ["entry_time", "pnl_pips"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trades_v09.csv に必要な列がありません: {missing}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df.get("exit_time"), errors="coerce") if "exit_time" in df.columns else pd.NaT
    df["pnl_pips"] = pd.to_numeric(df["pnl_pips"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["entry_time"]).copy()

    df["year"] = df["entry_time"].dt.year
    df["month"] = df["entry_time"].dt.to_period("M").astype(str)
    df["weekday_num"] = df["entry_time"].dt.weekday
    df["weekday"] = df["weekday_num"].map({0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"})
    df["hour"] = df["entry_time"].dt.hour

    # 念のため時系列順に並べる
    df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def calc_pf(pnls: pd.Series) -> float:
    profit = pnls[pnls > 0].sum()
    loss = -pnls[pnls < 0].sum()
    if loss == 0:
        return float("inf") if profit > 0 else 0.0
    return profit / loss


def calc_max_dd(pnls: pd.Series) -> float:
    equity = pnls.cumsum()
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min()) if len(dd) else 0.0


def calc_max_losing_streak(pnls: pd.Series) -> int:
    max_streak = 0
    streak = 0
    for v in pnls:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def calc_metrics(df: pd.DataFrame) -> Dict[str, float]:
    pnls = df["pnl_pips"] if not df.empty else pd.Series(dtype=float)
    trades = int(len(df))
    total = float(pnls.sum()) if trades else 0.0
    pf = calc_pf(pnls) if trades else 0.0
    win_rate = float((pnls > 0).mean() * 100) if trades else 0.0
    expectancy = float(pnls.mean()) if trades else 0.0
    max_dd = calc_max_dd(pnls) if trades else 0.0
    max_ls = calc_max_losing_streak(pnls) if trades else 0
    avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0
    avg_loss = float(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0.0

    monthly = df.groupby("month")["pnl_pips"].sum() if trades else pd.Series(dtype=float)
    month_plus_rate = float((monthly > 0).mean() * 100) if len(monthly) else 0.0
    worst_month = float(monthly.min()) if len(monthly) else 0.0
    bad_months = int((monthly < 0).sum()) if len(monthly) else 0

    yearly = df.groupby("year")["pnl_pips"].sum() if trades else pd.Series(dtype=float)
    profitable_years = int((yearly > 0).sum()) if len(yearly) else 0
    losing_years = int((yearly < 0).sum()) if len(yearly) else 0

    return {
        "取引回数": trades,
        "総損益pips": round(total, 3),
        "PF": round(pf, 3) if math.isfinite(pf) else 999.0,
        "勝率%": round(win_rate, 2),
        "期待値pips/回": round(expectancy, 3),
        "平均利益pips": round(avg_win, 3),
        "平均損失pips": round(avg_loss, 3),
        "最大DDpips": round(max_dd, 3),
        "最大連敗": max_ls,
        "月プラス率%": round(month_plus_rate, 2),
        "最悪月pips": round(worst_month, 3),
        "負け月数": bad_months,
        "利益年数": profitable_years,
        "負け年数": losing_years,
    }


def score_metrics(m: Dict[str, float]) -> float:
    # pips・PF・安定性・DDを総合した研究用スコア。利益保証ではない。
    total = safe_float(m["総損益pips"])
    pf = safe_float(m["PF"])
    exp = safe_float(m["期待値pips/回"])
    month_rate = safe_float(m["月プラス率%"])
    max_dd = abs(safe_float(m["最大DDpips"]))
    losing_years = safe_float(m["負け年数"])
    trades = safe_float(m["取引回数"])

    score = 0.0
    score += min(total / 10.0, 50.0)          # 利益
    score += min(pf * 18.0, 35.0)            # PF
    score += min(max(exp, 0) * 6.0, 25.0)    # 期待値
    score += min(month_rate * 0.35, 25.0)    # 月安定性
    score -= min(max_dd / 20.0, 20.0)        # DDペナルティ
    score -= losing_years * 8.0              # 年単位で負ける場合のペナルティ
    if trades < 100:
        score -= 20.0
    elif trades < 200:
        score -= 8.0
    return round(score, 2)


def apply_variants(df: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
    variants: List[Tuple[str, str, pd.DataFrame]] = []

    variants.append(("baseline", "フィルターなし", df.copy()))
    variants.append(("exclude_thu", "木曜日を除外", df[df["weekday"] != "木"].copy()))
    variants.append(("exclude_14h", "14時エントリーを除外", df[df["hour"] != 14].copy()))
    variants.append(("exclude_thu_14h", "木曜日＋14時エントリーを除外", df[(df["weekday"] != "木") & (df["hour"] != 14)].copy()))

    # 参考: 曜日を1つずつ除外
    for wd in ["月", "火", "水", "木", "金"]:
        variants.append((f"exclude_weekday_{wd}", f"{wd}曜日を除外", df[df["weekday"] != wd].copy()))

    # 参考: 東京時間の各時間を1つずつ除外
    for h in sorted(df["hour"].dropna().unique()):
        if 0 <= int(h) <= 23:
            variants.append((f"exclude_hour_{int(h)}", f"{int(h)}時エントリーを除外", df[df["hour"] != int(h)].copy()))

    return variants


def make_monthly_yearly(best_name: str, best_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    for month, g in best_df.groupby("month"):
        m = calc_metrics(g)
        monthly_rows.append({"variant": best_name, "month": month, **m})
    monthly = pd.DataFrame(monthly_rows)

    yearly_rows = []
    for year, g in best_df.groupby("year"):
        m = calc_metrics(g)
        yearly_rows.append({"variant": best_name, "year": year, **m})
    yearly = pd.DataFrame(yearly_rows)
    return monthly, yearly


def verdict_from_metrics(m: Dict[str, float]) -> str:
    if (
        m["PF"] >= 1.25
        and m["期待値pips/回"] >= 1.0
        and m["月プラス率%"] >= 55
        and m["負け年数"] == 0
        and m["最大DDpips"] >= -250
        and m["取引回数"] >= 200
    ):
        return "実戦候補"
    if (
        m["PF"] >= 1.15
        and m["期待値pips/回"] > 0
        and m["月プラス率%"] >= 45
        and m["取引回数"] >= 150
    ):
        return "監視候補"
    return "除外候補"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_trades()

    rows = []
    variant_map: Dict[str, pd.DataFrame] = {}
    for key, desc, vdf in apply_variants(trades):
        m = calc_metrics(vdf)
        row = {
            "variant": key,
            "説明": desc,
            **m,
            "改善スコア": score_metrics(m),
            "判定": verdict_from_metrics(m),
        }
        rows.append(row)
        variant_map[key] = vdf

    comp = pd.DataFrame(rows).sort_values(["判定", "改善スコア", "総損益pips"], ascending=[True, False, False])
    # 判定の並びを見やすくする
    order = {"実戦候補": 0, "監視候補": 1, "除外候補": 2}
    comp["_order"] = comp["判定"].map(order).fillna(9)
    comp = comp.sort_values(["_order", "改善スコア", "総損益pips"], ascending=[True, False, False]).drop(columns=["_order"])

    best_key = str(comp.iloc[0]["variant"])
    best_desc = str(comp.iloc[0]["説明"])
    best_df = variant_map[best_key].copy()
    best_df["v10_variant"] = best_key
    best_df["v10_filter"] = best_desc

    monthly, yearly = make_monthly_yearly(best_key, best_df)

    comp_path = OUT_DIR / "filter_comparison_v10.csv"
    best_path = OUT_DIR / "best_trades_v10.csv"
    monthly_path = OUT_DIR / "monthly_v10.csv"
    yearly_path = OUT_DIR / "yearly_v10.csv"
    report_path = OUT_DIR / "research_v10_report.txt"

    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("岡畠AI研究所 Ver.0.10 フィルター改善比較レポート")
    lines.append("=" * 70)
    lines.append(f"入力トレード数: {len(trades)}件")
    lines.append(f"比較パターン数: {len(comp)}件")
    lines.append("")
    lines.append("上位候補:")
    for i, (_, r) in enumerate(comp.head(10).iterrows(), 1):
        lines.append(
            f"{i}. [{r['判定']}] {r['説明']} / score={r['改善スコア']} / "
            f"損益={r['総損益pips']}pips / PF={r['PF']} / "
            f"期待値={r['期待値pips/回']} / DD={r['最大DDpips']} / "
            f"月プラス率={r['月プラス率%']}% / 取引={r['取引回数']}回"
        )

    lines.append("")
    lines.append("最良パターン:")
    best_row = comp.iloc[0].to_dict()
    for k in ["説明", "判定", "改善スコア", "取引回数", "総損益pips", "PF", "勝率%", "期待値pips/回", "最大DDpips", "最大連敗", "月プラス率%", "最悪月pips", "負け年数"]:
        lines.append(f"  {k}: {best_row.get(k)}")

    lines.append("")
    lines.append("次にやること:")
    lines.append("1. filter_comparison_v10.csv を確認して、除外条件で本当に改善しているか見る。")
    lines.append("2. best_trades_v10.csv を確認して、残ったエントリーの時刻・月・連敗箇所を見る。")
    lines.append("3. v0.11では、最良パターンを対象に資産曲線と負け局面をさらに分解する。")
    lines.append("")
    lines.append("注意: これは研究用であり、利益を保証するものではありません。")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 70)
    print("岡畠AI研究所 Ver.0.10 フィルター改善比較レポート")
    print("=" * 70)
    print(f"入力トレード数: {len(trades)}件")
    print(f"比較パターン数: {len(comp)}件")
    print("")
    print("上位候補:")
    for i, (_, r) in enumerate(comp.head(8).iterrows(), 1):
        print(
            f"{i}. [{r['判定']}] {r['説明']} / score={r['改善スコア']} / "
            f"損益={r['総損益pips']}pips / PF={r['PF']} / 期待値={r['期待値pips/回']} / "
            f"DD={r['最大DDpips']} / 月+率={r['月プラス率%']}% / 取引={r['取引回数']}回"
        )
    print("")
    print("出力しました:")
    print(comp_path)
    print(best_path)
    print(monthly_path)
    print(yearly_path)
    print(report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("エラーが発生しました。")
        print(type(e).__name__ + ":", e)
        sys.exit(1)
