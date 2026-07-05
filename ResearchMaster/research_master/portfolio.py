import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import PIP_SIZE, SPREADS
from research_master.metrics import summarize_trades, max_drawdown


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


def _score(summary: Dict) -> float:
    if summary["trades"] < 1000:
        return -9999
    pf = summary["pf"]
    exp = summary["expectancy_pips"]
    dd = abs(summary["max_dd_pips"])
    return float((pf - 1.0) * 1000 + exp * 80 - dd / 80)


def evaluate_candidates(df: pd.DataFrame, spread: float = 0.5, top_n: int = 40) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    trades_map: Dict[str, pd.DataFrame] = {}
    specs = make_candidate_specs()
    print(f"候補戦略評価: {len(specs)}本 / spread {spread}")
    for i, spec in enumerate(specs, 1):
        if i == 1 or i % 50 == 0 or i == len(specs):
            print(f"進捗: {i}/{len(specs)}")
        trades = build_trades_for_spec(df, spec, spread=spread)
        s = summarize_trades(trades, spread=spread)
        s.update({
            "strategy": spec.name,
            "direction": spec.direction,
            "session": f"{spec.session_start}-{spec.session_end}",
            "rule_type": spec.rule_type,
            "param": spec.param,
            "exit_bars": spec.exit_bars,
            "score": round(_score(s), 3),
        })
        rows.append(s)
        trades_map[spec.name] = trades
    ranking = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    keep_names = ranking.head(top_n)["strategy"].tolist()
    trades_map = {k: v for k, v in trades_map.items() if k in keep_names}
    return ranking, trades_map


def _daily_returns(trades: pd.DataFrame) -> pd.Series:
    if len(trades) == 0:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["day"] = pd.to_datetime(t["datetime"]).dt.date
    return t.groupby("day")["pips"].sum().astype(float)


def _corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).fillna(0.0)
    if len(x) < 5:
        return 1.0
    c = x.iloc[:, 0].corr(x.iloc[:, 1])
    return float(c) if pd.notna(c) else 1.0


def select_low_corr_portfolio(ranking: pd.DataFrame, trades_map: Dict[str, pd.DataFrame], max_strategies: int = 6, max_corr: float = 0.65) -> List[str]:
    selected: List[str] = []
    daily_cache = {name: _daily_returns(trades) for name, trades in trades_map.items()}
    for _, row in ranking.iterrows():
        name = row["strategy"]
        if name not in trades_map:
            continue
        if len(selected) >= max_strategies:
            break
        ok = True
        for s in selected:
            if abs(_corr(daily_cache[name], daily_cache[s])) > max_corr:
                ok = False
                break
        if ok:
            selected.append(name)
    return selected


def combine_trades(selected: List[str], trades_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name in selected:
        t = trades_map[name].copy()
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["datetime", "pips", "strategy", "equity_pips"])
    out = pd.concat(parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    out["equity_pips"] = out["pips"].cumsum()
    return out


def portfolio_summary(selected: List[str], combined: pd.DataFrame, spread: float) -> Dict:
    s = summarize_trades(combined, spread=spread)
    s["strategies"] = len(selected)
    s["selected"] = " | ".join(selected)
    return s


def run_portfolio_engine(df: pd.DataFrame, spread: float = 0.5):
    ranking, trades_map = evaluate_candidates(df, spread=spread, top_n=50)
    selected = select_low_corr_portfolio(ranking, trades_map, max_strategies=6, max_corr=0.65)
    combined = combine_trades(selected, trades_map)
    summary = portfolio_summary(selected, combined, spread=spread)

    selected_rows = ranking[ranking["strategy"].isin(selected)].copy()
    selected_rows["採用"] = "YES"
    return ranking, selected_rows, combined, summary, trades_map
