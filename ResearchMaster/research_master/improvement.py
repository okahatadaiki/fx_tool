from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from config import PIP_SIZE, TRAIN_END_YEAR, OOS_START_YEAR
from research_master.metrics import summarize_trades, max_drawdown


@dataclass(frozen=True)
class ImprovedSpec:
    name: str
    direction: str
    session_start: int
    session_end: int
    rule_type: str
    param: int
    exit_bars: int
    filter_type: str = "NONE"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period).mean()
    loss = (-diff.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi14"] = _rsi(d["close"], 14)
    d["ma24"] = d["close"].rolling(24).mean()
    d["ma96"] = d["close"].rolling(96).mean()
    d["ma288"] = d["close"].rolling(288).mean()
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr14_pips"] = tr.rolling(14).mean() / PIP_SIZE
    d["mom12_pips"] = (d["close"] - d["close"].shift(12)) / PIP_SIZE
    d["mom24_pips"] = (d["close"] - d["close"].shift(24)) / PIP_SIZE
    return d


def make_improvement_specs() -> List[ImprovedSpec]:
    specs: List[ImprovedSpec] = []
    sessions = [(15, 21), (16, 21), (16, 20), (17, 22), (21, 23)]
    exits = [12, 18, 24, 36, 48, 72]
    filters = ["NONE", "MA_UP", "TREND_UP", "ATR_MID", "NO_FRIDAY", "TUE_THU"]

    # 利幅を狙うため、決済本数を増やしつつ、条件を絞る候補を多めに作る。
    for s, e in sessions:
        for ex in exits:
            for f in filters:
                for lb in [12, 24, 48, 96]:
                    specs.append(ImprovedSpec(f"LONG {s}-{e} 安値B{lb} exit{ex} {f}", "LONG", s, e, "LOW_BREAK", lb, ex, f))
                for r in [20, 25, 30, 35]:
                    specs.append(ImprovedSpec(f"LONG {s}-{e} RSI<{r} exit{ex} {f}", "LONG", s, e, "RSI_LOW", r, ex, f))
                for th in [8, 12, 16]:
                    specs.append(ImprovedSpec(f"LONG {s}-{e} MOM_DOWN{th} exit{ex} {f}", "LONG", s, e, "MOM_DOWN", th, ex, f))
                for hb in [48, 96]:
                    specs.append(ImprovedSpec(f"SHORT {s}-{e} 高値B{hb} exit{ex} {f}", "SHORT", s, e, "HIGH_BREAK", hb, ex, f))
    return specs


def _filter_condition(d: pd.DataFrame, filter_type: str) -> pd.Series:
    cond = pd.Series(True, index=d.index)
    if filter_type == "NONE":
        return cond
    if filter_type == "MA_UP":
        return d["ma24"] > d["ma96"]
    if filter_type == "TREND_UP":
        return d["close"] > d["ma288"]
    if filter_type == "ATR_MID":
        lo = d["atr14_pips"].quantile(0.20)
        hi = d["atr14_pips"].quantile(0.85)
        return (d["atr14_pips"] >= lo) & (d["atr14_pips"] <= hi)
    if filter_type == "NO_FRIDAY":
        return d["weekday"] != "Friday"
    if filter_type == "TUE_THU":
        return d["weekday"].isin(["Tuesday", "Wednesday", "Thursday"])
    return cond


def build_improved_trades(enriched_df: pd.DataFrame, spec: ImprovedSpec, spread: float) -> pd.DataFrame:
    d = enriched_df.copy()
    d["entry"] = d["close"]
    d["exit"] = d["close"].shift(-spec.exit_bars)
    session = (d["hour"] >= spec.session_start) & (d["hour"] <= spec.session_end)

    if spec.rule_type == "LOW_BREAK":
        prev_low = d["low"].shift(1).rolling(spec.param).min()
        rule = d["low"] <= prev_low
    elif spec.rule_type == "HIGH_BREAK":
        prev_high = d["high"].shift(1).rolling(spec.param).max()
        rule = d["high"] >= prev_high
    elif spec.rule_type == "RSI_LOW":
        rule = d["rsi14"] < spec.param
    elif spec.rule_type == "MOM_DOWN":
        rule = d["mom12_pips"] <= -abs(spec.param)
    else:
        raise ValueError(f"unknown rule_type: {spec.rule_type}")

    cond = session & rule & _filter_condition(d, spec.filter_type)
    cols = ["datetime", "entry", "exit", "hour", "weekday", "month", "year", "atr14_pips", "rsi14"]
    trades = d.loc[cond, cols].dropna().copy()
    if spec.direction == "LONG":
        trades["pips"] = ((trades["exit"] - trades["entry"]) / PIP_SIZE) - spread
    else:
        trades["pips"] = ((trades["entry"] - trades["exit"]) / PIP_SIZE) - spread
    trades["strategy"] = spec.name
    trades["direction"] = spec.direction
    trades["session"] = f"{spec.session_start}-{spec.session_end}"
    trades["rule_type"] = spec.rule_type
    trades["param"] = spec.param
    trades["exit_bars"] = spec.exit_bars
    trades["filter_type"] = spec.filter_type
    trades = trades.sort_values("datetime").reset_index(drop=True)
    trades["equity_pips"] = trades["pips"].cumsum()
    return trades


def _summary(trades: pd.DataFrame, spread: float) -> Dict:
    if trades is None or len(trades) == 0:
        return summarize_trades(pd.DataFrame(columns=["pips", "equity_pips"]), spread=spread)
    t = trades.copy().sort_values("datetime").reset_index(drop=True)
    t["equity_pips"] = t["pips"].cumsum()
    return summarize_trades(t, spread=spread)


def _rolling_fixed_score(enriched: pd.DataFrame, spec: ImprovedSpec, spread: float) -> Dict:
    years = sorted(int(y) for y in enriched["year"].dropna().unique())
    test_years = [y for y in years if y >= min(years) + 2]
    rows = []
    for y in test_years:
        test_df = enriched[enriched["year"] == y]
        tr = build_improved_trades(test_df, spec, spread)
        s = _summary(tr, spread)
        passed = bool(s["trades"] >= 80 and s["pf"] >= 1.05 and s["expectancy_pips"] > 0)
        rows.append({
            "test_year": y,
            "trades": s["trades"],
            "pf": s["pf"],
            "expectancy_pips": s["expectancy_pips"],
            "max_dd_pips": s["max_dd_pips"],
            "pass": passed,
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {"rwf_windows": 0, "rwf_pass_rate_pct": 0.0, "rwf_worst_pf": 0.0, "rwf_worst_exp": 0.0, "rwf_detail": df}
    return {
        "rwf_windows": int(len(df)),
        "rwf_pass_rate_pct": round(float(df["pass"].mean() * 100), 2),
        "rwf_worst_pf": round(float(df["pf"].min()), 3),
        "rwf_worst_exp": round(float(df["expectancy_pips"].min()), 4),
        "rwf_detail": df,
    }


def _score_row(full_s: Dict, oos_s: Dict, stress_s: Dict, rwf: Dict) -> float:
    # 利幅は大事。ただし、DDと期間安定性をより強く見る。
    score = 0.0
    score += max(full_s["pf"] - 1.0, -1.0) * 700
    score += max(oos_s["pf"] - 1.0, -1.0) * 900
    score += max(stress_s["pf"] - 1.0, -1.0) * 450
    score += full_s["expectancy_pips"] * 70
    score += oos_s["expectancy_pips"] * 110
    score += rwf["rwf_pass_rate_pct"] * 5
    score -= abs(full_s["max_dd_pips"]) / 70
    if full_s["trades"] < 1200:
        score -= 120
    if oos_s["trades"] < 250:
        score -= 100
    if stress_s["expectancy_pips"] <= 0:
        score -= 150
    return round(float(score), 3)


def _decision(top: pd.Series) -> str:
    if top.empty:
        return "判定不能: 改善候補なし"
    if top["rwf_pass_rate_pct"] >= 80 and top["oos_pf"] >= 1.12 and top["stress_pf_1_0"] >= 1.03 and top["oos_expectancy"] > 0:
        return "改善候補: 小ロット検証に進める"
    if top["rwf_pass_rate_pct"] >= 60 and top["oos_pf"] >= 1.08 and top["stress_pf_1_0"] >= 1.00:
        return "準改善候補: まだ小ロット限定"
    return "保留: 利幅か期間安定性がまだ不足"


def run_improvement_engine(df: pd.DataFrame, spread: float = 0.5, stress_spread: float = 1.0, top_limit: int = 20):
    enriched = _enrich(df)
    specs = make_improvement_specs()
    rows = []
    trades_cache: Dict[str, pd.DataFrame] = {}
    detail_cache: Dict[str, pd.DataFrame] = {}

    print(f"11.0 改善候補探索: {len(specs)}本 / spread {spread} / stress {stress_spread}")
    for i, spec in enumerate(specs, 1):
        if i == 1 or i % 200 == 0 or i == len(specs):
            print(f"進捗: {i}/{len(specs)}")
        trades = build_improved_trades(enriched, spec, spread)
        full_s = _summary(trades, spread)
        if full_s["trades"] < 700 or full_s["pf"] < 1.04 or full_s["expectancy_pips"] <= 0:
            continue
        stress_trades = build_improved_trades(enriched, spec, stress_spread)
        stress_s = _summary(stress_trades, stress_spread)
        train_tr = trades[trades["year"] <= TRAIN_END_YEAR]
        oos_tr = trades[trades["year"] >= OOS_START_YEAR]
        train_s = _summary(train_tr, spread)
        oos_s = _summary(oos_tr, spread)
        if oos_s["trades"] < 150 or oos_s["expectancy_pips"] <= 0:
            continue
        rwf = _rolling_fixed_score(enriched, spec, spread)
        score = _score_row(full_s, oos_s, stress_s, rwf)
        row = {
            "strategy": spec.name,
            "direction": spec.direction,
            "session": f"{spec.session_start}-{spec.session_end}",
            "rule_type": spec.rule_type,
            "param": spec.param,
            "exit_bars": spec.exit_bars,
            "filter_type": spec.filter_type,
            "trades": full_s["trades"],
            "pf": full_s["pf"],
            "expectancy_pips": full_s["expectancy_pips"],
            "max_dd_pips": full_s["max_dd_pips"],
            "train_pf": train_s["pf"],
            "train_expectancy": train_s["expectancy_pips"],
            "oos_trades": oos_s["trades"],
            "oos_pf": oos_s["pf"],
            "oos_expectancy": oos_s["expectancy_pips"],
            "stress_pf_1_0": stress_s["pf"],
            "stress_expectancy_1_0": stress_s["expectancy_pips"],
            "rwf_pass_rate_pct": rwf["rwf_pass_rate_pct"],
            "rwf_worst_pf": rwf["rwf_worst_pf"],
            "rwf_worst_exp": rwf["rwf_worst_exp"],
            "improvement_score": score,
        }
        rows.append(row)
        trades_cache[spec.name] = trades
        detail_cache[spec.name] = rwf["rwf_detail"]

    ranking = pd.DataFrame(rows)
    if len(ranking) == 0:
        return {
            "improvement_ranking": ranking,
            "improvement_top_trades": pd.DataFrame(),
            "improvement_rolling_detail": pd.DataFrame(),
            "improvement_decision": pd.DataFrame([{"decision": "保留: 改善候補なし"}]),
        }

    ranking = ranking.sort_values("improvement_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    top = ranking.iloc[0]
    top_name = top["strategy"]
    top_trades = trades_cache[top_name].copy()
    top_detail = detail_cache.get(top_name, pd.DataFrame()).copy()
    decision = _decision(top)
    decision_df = pd.DataFrame([{
        "decision": decision,
        "top_strategy": top_name,
        "top_pf": top["pf"],
        "top_oos_pf": top["oos_pf"],
        "top_stress_pf_1_0": top["stress_pf_1_0"],
        "top_rwf_pass_rate_pct": top["rwf_pass_rate_pct"],
        "recommended_action": "EA化前にデモ/極小ロット検証" if "候補" in decision else "改善探索を継続",
    }])
    return {
        "improvement_ranking": ranking.head(top_limit),
        "improvement_all_candidates": ranking,
        "improvement_top_trades": top_trades,
        "improvement_rolling_detail": top_detail,
        "improvement_decision": decision_df,
    }
