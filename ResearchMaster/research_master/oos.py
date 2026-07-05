import pandas as pd
from config import SPREADS, TRAIN_END_YEAR, OOS_START_YEAR
from research_master.strategy import build_strategy_trades
from research_master.metrics import summarize_trades


def run_oos_validation(df: pd.DataFrame):
    """2021-2024を開発期間、2025以降を完全OOSとして固定検証する。"""
    train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
    oos_df = df[df["year"] >= OOS_START_YEAR].copy()

    rows = []
    trades_by_name = {}
    for spread in SPREADS:
        train_trades = build_strategy_trades(train_df, spread=spread)
        oos_trades = build_strategy_trades(oos_df, spread=spread)

        train_summary = summarize_trades(train_trades, spread=spread)
        oos_summary = summarize_trades(oos_trades, spread=spread)

        row = {
            "spread": spread,
            "train_trades": train_summary["trades"],
            "train_pf": train_summary["pf"],
            "train_expectancy": train_summary["expectancy_pips"],
            "train_dd": train_summary["max_dd_pips"],
            "oos_trades": oos_summary["trades"],
            "oos_pf": oos_summary["pf"],
            "oos_expectancy": oos_summary["expectancy_pips"],
            "oos_dd": oos_summary["max_dd_pips"],
            "oos_winrate_pct": oos_summary["winrate_pct"],
            "判定": judge_oos(oos_summary),
        }
        rows.append(row)
        trades_by_name[f"train_spread_{spread}"] = train_trades
        trades_by_name[f"oos_spread_{spread}"] = oos_trades

    return pd.DataFrame(rows), trades_by_name


def judge_oos(s):
    if s["trades"] < 300:
        return "取引不足"
    if s["pf"] >= 1.15 and s["expectancy_pips"] > 0:
        return "合格候補"
    if s["pf"] >= 1.05 and s["expectancy_pips"] > 0:
        return "弱いが生存"
    return "不合格"


def monthly_breakdown(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()
    d = trades.copy()
    d["ym"] = d["datetime"].dt.to_period("M").astype(str)
    rows = []
    for ym, g in d.groupby("ym"):
        p = g["pips"]
        rows.append({
            "ym": ym,
            "trades": len(g),
            "total_pips": round(float(p.sum()), 3),
            "expectancy_pips": round(float(p.mean()), 4),
            "winrate_pct": round(float((p > 0).mean() * 100), 2),
        })
    return pd.DataFrame(rows)
