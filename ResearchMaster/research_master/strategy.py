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


# --- 10.0 Portfolio Lab support functions ---
# lab.py imports these names from strategy.py. They are duplicated here intentionally
# so older project folders can be overwritten safely without import errors.
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class StrategySpec:
    name: str
    direction: str
    session_start: int
    session_end: int
    rule_type: str
    param: int
    exit_bars: int


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period).mean()
    loss = (-diff.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_candidate_specs() -> List[StrategySpec]:
    specs: List[StrategySpec] = []
    sessions = [(15, 21), (16, 21), (15, 22), (15, 20), (21, 23), (9, 15)]
    exits = [12, 24, 48]
    lows = [12, 24, 48, 96]
    rsi_lows = [25, 30, 35]
    high_breaks = [48, 96]

    for s, e in sessions:
        for ex in exits:
            for lb in lows:
                specs.append(StrategySpec(f"LONG 欧州系 {s}-{e} 安値ブレイク{lb} exit{ex}", "LONG", s, e, "LOW_BREAK", lb, ex))
            for r in rsi_lows:
                specs.append(StrategySpec(f"LONG 欧州系 {s}-{e} RSI<{r} exit{ex}", "LONG", s, e, "RSI_LOW", r, ex))
            for hb in high_breaks:
                specs.append(StrategySpec(f"SHORT {s}-{e} 高値ブレイク{hb} exit{ex}", "SHORT", s, e, "HIGH_BREAK", hb, ex))
    return specs


def build_trades_for_spec(df: pd.DataFrame, spec: StrategySpec, spread: float) -> pd.DataFrame:
    d = df.copy()
    d["entry"] = d["close"]
    d["exit"] = d["close"].shift(-spec.exit_bars)
    session = (d["hour"] >= spec.session_start) & (d["hour"] <= spec.session_end)

    if spec.rule_type == "LOW_BREAK":
        prev_low = d["low"].shift(1).rolling(spec.param).min()
        cond = session & (d["low"] <= prev_low)
    elif spec.rule_type == "HIGH_BREAK":
        prev_high = d["high"].shift(1).rolling(spec.param).max()
        cond = session & (d["high"] >= prev_high)
    elif spec.rule_type == "RSI_LOW":
        r = _rsi(d["close"], 14)
        cond = session & (r < spec.param)
    else:
        raise ValueError(f"unknown rule_type: {spec.rule_type}")

    trades = d.loc[cond, ["datetime", "entry", "exit", "hour", "weekday", "month", "year"]].dropna().copy()
    if spec.direction == "LONG":
        trades["pips"] = ((trades["exit"] - trades["entry"]) / PIP_SIZE) - spread
    else:
        trades["pips"] = ((trades["entry"] - trades["exit"]) / PIP_SIZE) - spread
    trades["strategy"] = spec.name
    trades["direction"] = spec.direction
    trades["exit_bars"] = spec.exit_bars
    trades["rule_type"] = spec.rule_type
    trades["param"] = spec.param
    trades["session"] = f"{spec.session_start}-{spec.session_end}"
    trades = trades.sort_values("datetime").reset_index(drop=True)
    trades["equity_pips"] = trades["pips"].cumsum()
    return trades
