import pandas as pd
from config import SESSION_START_HOUR, SESSION_END_HOUR, BREAKOUT_LOOKBACK, EXIT_BARS, PIP_SIZE


def build_strategy_trades(df: pd.DataFrame, spread: float) -> pd.DataFrame:
    d = df.copy()
    d["prev_low"] = d["low"].shift(1).rolling(BREAKOUT_LOOKBACK).min()
    d["entry"] = d["close"]
    d["exit"] = d["close"].shift(-EXIT_BARS)

    cond = (
        (d["hour"] >= SESSION_START_HOUR) &
        (d["hour"] <= SESSION_END_HOUR) &
        (d["low"] <= d["prev_low"])
    )

    trades = d.loc[cond, ["datetime", "entry", "exit", "hour", "weekday", "month", "year"]].dropna().copy()
    trades["pips"] = ((trades["exit"] - trades["entry"]) / PIP_SIZE) - spread
    trades["equity_pips"] = trades["pips"].cumsum()
    return trades.reset_index(drop=True)
