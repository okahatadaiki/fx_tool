# =====================================================
# research_v09.py
# 岡畠AI研究所 Ver.0.9
# v08監視候補の「実トレード履歴・月別・資産曲線」出力ツール
#
# 実行方法:
#   python research_v09.py
#
# 入力:
#   data/research_v08/watch_candidates_v08.csv
#   data/research_v08/final_candidates_v08.csv
#   data/USDJPY_5min_2021-2026.csv
#
# 出力:
#   data/research_v09/trades_v09.csv
#   data/research_v09/monthly_v09.csv
#   data/research_v09/equity_curve_v09.csv
#   data/research_v09/research_v09_report.txt
# =====================================================

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research_v06 import (
    Config as BaseConfig,
    PIP,
    REASON_NAMES,
    load_data,
    IndicatorCache,
    build_spread_array,
    build_signal,
    build_candidates,
    compute_outcomes,
    filter_mask,
    select_exclusive,
    calc_stats,
)


BASE_DIR = Path(__file__).resolve().parent
IN_DIR = BASE_DIR / "data" / "research_v08"
OUT_DIR = BASE_DIR / "data" / "research_v09"

FINAL_CSV = IN_DIR / "final_candidates_v08.csv"
WATCH_CSV = IN_DIR / "watch_candidates_v08.csv"


def parse_strategy(text: str) -> Tuple:
    text = str(text)

    m = re.search(r"EMAクロス\((\d+),(\d+)\)", text)
    if m:
        return ("EMAクロス", int(m.group(1)), int(m.group(2)))

    m = re.search(r"RSI逆張り\(期間(\d+),\s*(\d+)/(\d+)\)", text)
    if m:
        return ("RSI逆張り", int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(r"MACDクロス\((\d+),(\d+),(\d+)\)", text)
    if m:
        return ("MACDクロス", int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(r"BB逆張り\(期間(\d+),\s*([0-9.]+)σ\)", text)
    if m:
        return ("BB逆張り", int(m.group(1)), float(m.group(2)))

    m = re.search(r"BBブレイク\(期間(\d+),\s*([0-9.]+)σ\)", text)
    if m:
        return ("BBブレイク", int(m.group(1)), float(m.group(2)))

    m = re.search(r"押し目\(EMA200\+RSI(\d+)/(\d+)\)", text)
    if m:
        return ("押し目", 14, int(m.group(1)), int(m.group(2)))

    raise ValueError(f"戦略を解析できません: {text}")


def load_candidates() -> pd.DataFrame:
    frames = []

    if FINAL_CSV.exists():
        f = pd.read_csv(FINAL_CSV)
        if not f.empty:
            f["v09元ファイル"] = "final_candidates_v08.csv"
            frames.append(f)

    if WATCH_CSV.exists():
        w = pd.read_csv(WATCH_CSV)
        if not w.empty:
            w["v09元ファイル"] = "watch_candidates_v08.csv"
            frames.append(w)

    if not frames:
        raise FileNotFoundError(
            "v08候補CSVがありません、または空です。\n"
            f"確認場所: {IN_DIR}\n"
            "先に python research_v08.py を実行してください。"
        )

    df = pd.concat(frames, ignore_index=True)

    # v08スコア順。基本は監視候補1件だが、複数あっても対応。
    if "v08スコア" in df.columns:
        df["v08スコア"] = pd.to_numeric(df["v08スコア"], errors="coerce").fillna(0)
        df = df.sort_values("v08スコア", ascending=False)

    return df.reset_index(drop=True)


def monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    t = trades.copy()
    t["month"] = pd.to_datetime(t["entry_time"]).dt.to_period("M").astype(str)

    rows = []
    for month, g in t.groupby("month"):
        pnl = g["pnl_pips"].to_numpy(dtype=float)
        st = calc_stats(pnl, pd.to_datetime(g["entry_time"]).astype("int64").to_numpy())
        rows.append({
            "month": month,
            "trades": int(st["取引回数"]),
            "pnl_pips": st["総損益pips"],
            "pf": st["PF"],
            "expectancy": st["期待値pips/回"],
            "win_rate_pct": st["勝率%"],
            "max_dd_pips": st["最大DD_pips"],
            "max_losing_streak": st["最大連敗"],
        })

    return pd.DataFrame(rows)


def yearly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    t = trades.copy()
    t["year"] = pd.to_datetime(t["entry_time"]).dt.year

    rows = []
    for year, g in t.groupby("year"):
        pnl = g["pnl_pips"].to_numpy(dtype=float)
        st = calc_stats(pnl, pd.to_datetime(g["entry_time"]).astype("int64").to_numpy())
        rows.append({
            "year": int(year),
            "trades": int(st["取引回数"]),
            "pnl_pips": st["総損益pips"],
            "pf": st["PF"],
            "expectancy": st["期待値pips/回"],
            "win_rate_pct": st["勝率%"],
            "max_dd_pips": st["最大DD_pips"],
            "max_losing_streak": st["最大連敗"],
        })

    return pd.DataFrame(rows)


def make_trades_for_candidate(df: pd.DataFrame, ind: IndicatorCache, spread: np.ndarray, row: pd.Series, candidate_no: int) -> pd.DataFrame:
    sig_key = parse_strategy(row["戦略"])
    tp = int(row["利確pips"])
    sl = int(row["損切りpips"])
    time_f = str(row["時間帯"])
    weekday_f = str(row["曜日"])
    trend_f = str(row["トレンド環境"])
    vol_f = str(row["ボラ環境"])

    sig = build_signal(sig_key, ind)
    ctx = build_candidates(df, sig)

    exit_idx, pnl_all, reason_all = compute_outcomes(df, ctx, tp, sl, spread)
    mask = filter_mask(ctx, time_f, weekday_f, trend_f, vol_f)
    selected = select_exclusive(ctx["entry_idx"], exit_idx, mask)

    if len(selected) == 0:
        return pd.DataFrame()

    entry_idx = ctx["entry_idx"][selected]
    ex_idx = exit_idx[selected]
    pnl = pnl_all[selected]
    reasons = reason_all[selected]
    side = ctx["side"][selected]

    idx = df.index
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()

    trades = pd.DataFrame({
        "candidate_no": candidate_no,
        "v08判定": row.get("v08判定", ""),
        "v08スコア": row.get("v08スコア", ""),
        "strategy": row["戦略"],
        "strategy_type": row.get("戦略タイプ", ""),
        "tp_pips": tp,
        "sl_pips": sl,
        "time_filter": time_f,
        "weekday_filter": weekday_f,
        "trend_filter": trend_f,
        "vol_filter": vol_f,
        "entry_time": idx[entry_idx].astype(str),
        "exit_time": idx[ex_idx].astype(str),
        "side": np.where(side > 0, "BUY", "SELL"),
        "entry_open": o[entry_idx],
        "exit_close": c[ex_idx],
        "pnl_pips": np.round(pnl.astype(float), 3),
        "exit_reason": [REASON_NAMES.get(int(x), str(x)) for x in reasons],
    })

    trades["equity_pips"] = trades["pnl_pips"].cumsum().round(3)
    trades["running_max_pips"] = trades["equity_pips"].cummax().round(3)
    trades["drawdown_pips"] = (trades["equity_pips"] - trades["running_max_pips"]).round(3)

    return trades


def build_report(cands: pd.DataFrame, trades: pd.DataFrame, monthly: pd.DataFrame, yearly: pd.DataFrame, elapsed: float) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("岡畠AI研究所 Ver.0.9 実トレード履歴・資産曲線レポート")
    lines.append("=" * 70)
    lines.append(f"実行時間: {elapsed:.0f}秒")
    lines.append(f"検証候補数: {len(cands)}件")
    lines.append(f"出力トレード数: {len(trades)}件")
    lines.append("")

    if trades.empty:
        lines.append("トレードがありません。条件が厳しすぎる可能性があります。")
        return "\n".join(lines)

    pnl = trades["pnl_pips"].to_numpy(dtype=float)
    ts = pd.to_datetime(trades["entry_time"]).astype("int64").to_numpy()
    st = calc_stats(pnl, ts)

    lines.append("総合成績:")
    lines.append(f"  総損益: {st['総損益pips']} pips")
    lines.append(f"  PF: {st['PF']}")
    lines.append(f"  期待値: {st['期待値pips/回']} pips/回")
    lines.append(f"  勝率: {st['勝率%']}%")
    lines.append(f"  取引回数: {st['取引回数']}回")
    lines.append(f"  最大DD: {st['最大DD_pips']} pips")
    lines.append(f"  最大連敗: {st['最大連敗']}回")
    lines.append("")

    lines.append("候補:")
    for i, (_, r) in enumerate(cands.iterrows(), 1):
        lines.append(
            f"  {i}. {r['戦略']} TP{int(r['利確pips'])} SL{int(r['損切りpips'])} "
            f"{r['時間帯']} {r['曜日']} {r['トレンド環境']} {r['ボラ環境']} "
            f"v08={r.get('v08判定','')} score={r.get('v08スコア','')}"
        )
    lines.append("")

    if not yearly.empty:
        lines.append("年別:")
        for _, r in yearly.iterrows():
            lines.append(
                f"  {int(r['year'])}: {r['pnl_pips']}pips / PF{r['pf']} / "
                f"期待値{r['expectancy']} / {int(r['trades'])}回 / DD{r['max_dd_pips']}"
            )
        lines.append("")

    if not monthly.empty:
        bad_months = monthly.sort_values("pnl_pips").head(5)
        lines.append("悪い月ワースト5:")
        for _, r in bad_months.iterrows():
            lines.append(
                f"  {r['month']}: {r['pnl_pips']}pips / PF{r['pf']} / "
                f"{int(r['trades'])}回 / DD{r['max_dd_pips']}"
            )
        lines.append("")

    lines.append("次にやること:")
    lines.append("  1. trades_v09.csv を見て、実際のエントリー時刻が偏っていないか確認する。")
    lines.append("  2. 月別で負けが大きい月の相場環境を確認する。")
    lines.append("  3. v1.0ではこの1候補をリアルタイム監視ツールに変換する。")
    lines.append("")
    lines.append("注意: これは研究用であり、利益を保証するものではありません。")

    return "\n".join(lines)


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cands = load_candidates()

    base_cfg = BaseConfig()
    df = load_data(base_cfg)
    ind = IndicatorCache(df, base_cfg)
    df = ind.prepare_base_env()
    spread = build_spread_array(df, base_cfg)

    all_trades = []
    for i, (_, row) in enumerate(cands.iterrows(), 1):
        print(f"{i}/{len(cands)} トレード履歴作成: {row['戦略']}")
        tr = make_trades_for_candidate(df, ind, spread, row, i)
        all_trades.append(tr)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    monthly = monthly_breakdown(trades)
    yearly = yearly_breakdown(trades)

    if not trades.empty:
        equity = trades[["entry_time", "exit_time", "candidate_no", "strategy", "pnl_pips", "equity_pips", "drawdown_pips"]].copy()
    else:
        equity = pd.DataFrame(columns=["entry_time", "exit_time", "candidate_no", "strategy", "pnl_pips", "equity_pips", "drawdown_pips"])

    cands.to_csv(OUT_DIR / "input_candidates_v09.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUT_DIR / "trades_v09.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_v09.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT_DIR / "yearly_v09.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(OUT_DIR / "equity_curve_v09.csv", index=False, encoding="utf-8-sig")

    report = build_report(cands, trades, monthly, yearly, time.time() - start)
    (OUT_DIR / "research_v09_report.txt").write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print("出力しました:")
    print(OUT_DIR / "input_candidates_v09.csv")
    print(OUT_DIR / "trades_v09.csv")
    print(OUT_DIR / "monthly_v09.csv")
    print(OUT_DIR / "yearly_v09.csv")
    print(OUT_DIR / "equity_curve_v09.csv")
    print(OUT_DIR / "research_v09_report.txt")


if __name__ == "__main__":
    main()
