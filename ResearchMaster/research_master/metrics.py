import numpy as np
import pandas as pd
from config import INITIAL_BALANCE, YEN_PER_PIP


def max_drawdown(equity):
    if len(equity) == 0:
        return 0.0
    arr = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = arr - peak
    return float(dd.min())


def streaks(values):
    max_win = max_loss = cur_win = cur_loss = 0
    for v in values:
        if v > 0:
            cur_win += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def summarize_trades(trades: pd.DataFrame, spread: float):
    p = trades["pips"] if len(trades) else pd.Series(dtype=float)
    wins = p[p > 0]
    losses = p[p <= 0]
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    total = float(p.sum())
    dd = max_drawdown(trades["equity_pips"]) if len(trades) else 0.0
    max_win, max_loss = streaks(p)
    return {
        "spread": spread,
        "trades": int(len(trades)),
        "total_pips": round(total, 3),
        "pf": round(pf, 3),
        "expectancy_pips": round(float(p.mean()) if len(p) else 0.0, 4),
        "winrate_pct": round(float((p > 0).mean() * 100) if len(p) else 0.0, 2),
        "max_dd_pips": round(dd, 1),
        "max_dd_money": round(dd * YEN_PER_PIP, 0),
        "ending_balance": round(INITIAL_BALANCE + total * YEN_PER_PIP, 0),
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
        "avg_win_pips": round(float(wins.mean()) if len(wins) else 0.0, 4),
        "avg_loss_pips": round(float(losses.mean()) if len(losses) else 0.0, 4),
    }
