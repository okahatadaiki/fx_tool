from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research_master.metrics import summarize_trades, max_drawdown


def daily_pivot(trades_map: Dict[str, pd.DataFrame], names: List[str]) -> pd.DataFrame:
    parts = []
    for name in names:
        t = trades_map[name]
        if len(t) == 0:
            continue
        x = t.copy()
        x["day"] = pd.to_datetime(x["datetime"]).dt.date
        s = x.groupby("day")["pips"].sum().rename(name)
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).fillna(0.0).sort_index()


def calc_weights(daily: pd.DataFrame, selected_rows: pd.DataFrame, mode: str) -> Dict[str, float]:
    names = list(daily.columns)
    if not names:
        return {}
    if mode == "equal":
        raw = pd.Series(1.0, index=names)
    elif mode == "inv_dd":
        vals = {}
        for n in names:
            eq = daily[n].cumsum()
            dd = abs(max_drawdown(eq))
            vals[n] = 1.0 / max(dd, 1.0)
        raw = pd.Series(vals)
    elif mode == "score":
        sr = selected_rows.set_index("strategy")
        raw = sr.reindex(names)["score"].fillna(1.0).clip(lower=1.0)
    elif mode == "risk_adjusted":
        vals = {}
        sr = selected_rows.set_index("strategy")
        for n in names:
            eq = daily[n].cumsum()
            dd = abs(max_drawdown(eq))
            mean = max(float(daily[n].mean()), 0.0)
            score = float(sr.loc[n, "score"]) if n in sr.index else 1.0
            vals[n] = (mean + 0.01) * max(score, 1.0) / max(dd, 1.0)
        raw = pd.Series(vals)
    else:
        raise ValueError(f"unknown weight mode: {mode}")

    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=names)
    w = raw / raw.sum()
    # 1戦略に寄りすぎないように上限45%。残りを再配分。
    cap = 0.45
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] = w[under] + excess * (w[under] / w[under].sum())
    w = w / w.sum()
    return {k: float(v) for k, v in w.items()}


def apply_weights(trades_map: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    parts = []
    for name, weight in weights.items():
        t = trades_map[name].copy()
        if len(t) == 0:
            continue
        t["raw_pips"] = t["pips"]
        t["weight"] = weight
        t["pips"] = t["raw_pips"] * weight
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["datetime", "pips", "strategy", "equity_pips", "weight"])
    out = pd.concat(parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    out["equity_pips"] = out["pips"].cumsum()
    return out


def correlation_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    corr = daily.corr().fillna(0.0)
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({"strategy_a": cols[i], "strategy_b": cols[j], "corr": round(float(corr.iloc[i, j]), 4)})
    return pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def run_portfolio_optimizer(selected_df: pd.DataFrame, trades_map: Dict[str, pd.DataFrame], spread: float = 0.5):
    names = selected_df["strategy"].tolist() if len(selected_df) else []
    daily = daily_pivot(trades_map, names)
    modes = ["equal", "inv_dd", "score", "risk_adjusted"]
    rows = []
    weighted_trades = {}
    weight_rows = []

    for mode in modes:
        weights = calc_weights(daily, selected_df, mode)
        trades = apply_weights(trades_map, weights)
        s = summarize_trades(trades, spread=spread)
        s["weight_mode"] = mode
        # 運用目線の追加スコア。DDを強めに減点。
        s["optimizer_score"] = round((s["pf"] - 1.0) * 1000 + s["expectancy_pips"] * 120 - abs(s["max_dd_pips"]) / 40, 3)
        rows.append(s)
        weighted_trades[mode] = trades
        for k, v in weights.items():
            weight_rows.append({"weight_mode": mode, "strategy": k, "weight": round(v, 5)})

    opt_df = pd.DataFrame(rows).sort_values("optimizer_score", ascending=False).reset_index(drop=True)
    weights_df = pd.DataFrame(weight_rows)
    corr_df = correlation_table(daily)
    best_mode = opt_df.iloc[0]["weight_mode"] if len(opt_df) else "equal"
    best_trades = weighted_trades.get(best_mode, pd.DataFrame())
    best_weights = weights_df[weights_df["weight_mode"] == best_mode].copy() if len(weights_df) else pd.DataFrame()
    return opt_df, weights_df, corr_df, best_mode, best_trades, best_weights
