from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from config import PIP_SIZE, TRAIN_END_YEAR, OOS_START_YEAR
from research_master.metrics import summarize_trades
from research_master.improvement import ImprovedSpec, _enrich, _filter_condition, build_improved_trades


@dataclass(frozen=True)
class ValidationSpec:
    name: str
    direction: str
    session_start: int
    session_end: int
    rule_type: str
    param: int
    exit_bars: int
    filter_type: str


def _row_to_spec(row: pd.Series) -> ImprovedSpec:
    session = str(row.get("session", "21-23"))
    try:
        s, e = session.split("-")
        start, end = int(s), int(e)
    except Exception:
        start, end = 21, 23
    return ImprovedSpec(
        name=str(row.get("strategy", "LONG 21-23 安値B48 exit18 MA_UP")),
        direction=str(row.get("direction", "LONG")),
        session_start=start,
        session_end=end,
        rule_type=str(row.get("rule_type", "LOW_BREAK")),
        param=int(row.get("param", 48)),
        exit_bars=int(row.get("exit_bars", 18)),
        filter_type=str(row.get("filter_type", "MA_UP")),
    )


def _summ(trades: pd.DataFrame, spread: float = 0.5) -> Dict:
    if trades is None or len(trades) == 0:
        return summarize_trades(pd.DataFrame(columns=["pips", "equity_pips"]), spread=spread)
    t = trades.copy().sort_values("datetime").reset_index(drop=True)
    t["equity_pips"] = t["pips"].cumsum()
    return summarize_trades(t, spread=spread)


def _build_with_delay(enriched: pd.DataFrame, spec: ImprovedSpec, spread: float, slippage: float = 0.0, entry_delay: int = 0, exit_extra_delay: int = 0) -> pd.DataFrame:
    """候補戦略を、遅延・追加スリッページ込みで作る。
    entry_delay=1ならシグナルの1本後closeで入る。
    exit_extra_delay=1なら決済をさらに1本遅らせる。
    """
    d = enriched.copy()
    exit_bars = int(spec.exit_bars + exit_extra_delay)
    d["entry"] = d["close"].shift(-entry_delay)
    d["exit"] = d["close"].shift(-exit_bars)
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
        rule = pd.Series(False, index=d.index)

    cond = session & rule & _filter_condition(d, spec.filter_type)
    cols = ["datetime", "entry", "exit", "hour", "weekday", "month", "year", "atr14_pips", "rsi14", "close", "ma96", "ma288"]
    trades = d.loc[cond, cols].dropna().copy()
    total_cost = spread + slippage
    if spec.direction == "LONG":
        trades["pips"] = ((trades["exit"] - trades["entry"]) / PIP_SIZE) - total_cost
    else:
        trades["pips"] = ((trades["entry"] - trades["exit"]) / PIP_SIZE) - total_cost
    trades["strategy"] = spec.name
    trades["spread"] = spread
    trades["slippage"] = slippage
    trades["entry_delay"] = entry_delay
    trades["exit_extra_delay"] = exit_extra_delay
    trades = trades.sort_values("datetime").reset_index(drop=True)
    trades["equity_pips"] = trades["pips"].cumsum()
    return trades


def _cost_stress(enriched: pd.DataFrame, spec: ImprovedSpec) -> pd.DataFrame:
    rows = []
    spreads = [0.2, 0.4, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    slips = [0.0, 0.2, 0.5, 1.0, 2.0]
    for sp in spreads:
        for sl in slips:
            tr = _build_with_delay(enriched, spec, sp, slippage=sl)
            s = _summ(tr, spread=sp)
            rows.append({
                "spread": sp,
                "slippage": sl,
                "total_cost": round(sp + sl, 2),
                "trades": s["trades"],
                "pf": s["pf"],
                "expectancy_pips": s["expectancy_pips"],
                "max_dd_pips": s["max_dd_pips"],
                "pass": bool(s["trades"] >= 300 and s["pf"] >= 1.05 and s["expectancy_pips"] > 0),
            })
    return pd.DataFrame(rows)


def _delay_stress(enriched: pd.DataFrame, spec: ImprovedSpec, spread: float = 0.5, slippage: float = 0.2) -> pd.DataFrame:
    rows = []
    for entry_delay in [0, 1, 2, 3]:
        for exit_delay in [0, 1, 2, 3]:
            tr = _build_with_delay(enriched, spec, spread, slippage=slippage, entry_delay=entry_delay, exit_extra_delay=exit_delay)
            s = _summ(tr, spread=spread)
            rows.append({
                "entry_delay_bars": entry_delay,
                "exit_extra_delay_bars": exit_delay,
                "spread": spread,
                "slippage": slippage,
                "trades": s["trades"],
                "pf": s["pf"],
                "expectancy_pips": s["expectancy_pips"],
                "max_dd_pips": s["max_dd_pips"],
                "pass": bool(s["trades"] >= 300 and s["pf"] >= 1.05 and s["expectancy_pips"] > 0),
            })
    return pd.DataFrame(rows)


def _rolling_year_detail(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades is None or len(trades) == 0:
        return pd.DataFrame(rows)
    for y in sorted(trades["year"].dropna().unique()):
        t = trades[trades["year"] == y]
        s = _summ(t)
        rows.append({
            "year": int(y),
            "trades": s["trades"],
            "pf": s["pf"],
            "expectancy_pips": s["expectancy_pips"],
            "max_dd_pips": s["max_dd_pips"],
            "pass": bool(s["trades"] >= 80 and s["pf"] >= 1.05 and s["expectancy_pips"] > 0),
        })
    return pd.DataFrame(rows)


def _calendar_breakdown(trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if trades is None or len(trades) == 0:
        return out
    for key in ["year", "month", "weekday", "hour"]:
        rows = []
        for val, g in trades.groupby(key):
            s = _summ(g)
            rows.append({key: val, "trades": s["trades"], "pf": s["pf"], "expectancy_pips": s["expectancy_pips"], "total_pips": s["total_pips"], "max_dd_pips": s["max_dd_pips"], "pass": bool(s["trades"] >= 20 and s["pf"] >= 1.0 and s["expectancy_pips"] > 0)})
        out[f"by_{key}"] = pd.DataFrame(rows).sort_values(key).reset_index(drop=True)
    return out


def _regime_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    # ATRはトレード時点の分位で分類
    q1 = t["atr14_pips"].quantile(0.33)
    q2 = t["atr14_pips"].quantile(0.66)
    t["atr_regime"] = np.select([t["atr14_pips"] <= q1, t["atr14_pips"] <= q2], ["LOW_ATR", "MID_ATR"], default="HIGH_ATR")
    t["trend_regime"] = np.select([t["close"] > t["ma288"], t["close"] < t["ma288"]], ["UP_TREND", "DOWN_TREND"], default="RANGE")
    rows = []
    for key in ["atr_regime", "trend_regime"]:
        for val, g in t.groupby(key):
            s = _summ(g)
            rows.append({
                "regime_type": key,
                "regime": val,
                "trades": s["trades"],
                "pf": s["pf"],
                "expectancy_pips": s["expectancy_pips"],
                "total_pips": s["total_pips"],
                "max_dd_pips": s["max_dd_pips"],
                "pass": bool(s["trades"] >= 50 and s["pf"] >= 1.0 and s["expectancy_pips"] > 0),
            })
    return pd.DataFrame(rows)


def _monte_carlo(trades: pd.DataFrame, runs: int = 5000, seed: int = 1021) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    pips = trades["pips"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    totals = []
    dds = []
    for _ in range(runs):
        sample = rng.choice(pips, size=len(pips), replace=True)
        eq = np.cumsum(sample)
        dd = eq - np.maximum.accumulate(eq)
        totals.append(float(eq[-1]))
        dds.append(float(dd.min()))
    totals = np.array(totals)
    dds = np.array(dds)
    return pd.DataFrame([{
        "runs": runs,
        "total_pips_p05": round(float(np.percentile(totals, 5)), 3),
        "total_pips_median": round(float(np.percentile(totals, 50)), 3),
        "total_pips_p95": round(float(np.percentile(totals, 95)), 3),
        "dd_p50": round(abs(float(np.percentile(dds, 50))), 3),
        "dd_p90": round(abs(float(np.percentile(dds, 10))), 3),
        "dd_p95": round(abs(float(np.percentile(dds, 5))), 3),
        "dd_p99": round(abs(float(np.percentile(dds, 1))), 3),
        "risk_of_loss_pct": round(float((totals <= 0).mean() * 100), 2),
        "risk_dd_over_3000pips_pct": round(float((dds <= -3000).mean() * 100), 2),
    }])


def _final_decision(cost: pd.DataFrame, delay: pd.DataFrame, year: pd.DataFrame, regime: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    base = cost[(cost["spread"] == 0.5) & (cost["slippage"] == 0.0)]
    stress_1 = cost[(cost["spread"] == 1.0) & (cost["slippage"] == 0.5)]
    stress_15 = cost[(cost["spread"] == 1.5) & (cost["slippage"] == 0.5)]
    base_pf = float(base["pf"].iloc[0]) if len(base) else 0.0
    stress_pf = float(stress_1["pf"].iloc[0]) if len(stress_1) else 0.0
    hard_pf = float(stress_15["pf"].iloc[0]) if len(stress_15) else 0.0
    delay_pass = float(delay["pass"].mean() * 100) if len(delay) else 0.0
    year_pass = float(year["pass"].mean() * 100) if len(year) else 0.0
    regime_pass = float(regime["pass"].mean() * 100) if len(regime) else 0.0
    loss_risk = float(mc["risk_of_loss_pct"].iloc[0]) if len(mc) else 100.0
    dd_risk = float(mc["risk_dd_over_3000pips_pct"].iloc[0]) if len(mc) else 100.0

    score = 0
    score += min(max((base_pf - 1) * 80, 0), 25)
    score += min(max((stress_pf - 1) * 100, 0), 25)
    score += min(delay_pass / 4, 20)
    score += min(year_pass / 5, 20)
    score += min(regime_pass / 10, 10)
    score -= min(loss_risk * 2, 20)
    score -= min(dd_risk / 2, 20)
    score = round(float(score), 2)

    if score >= 75 and stress_pf >= 1.10 and delay_pass >= 60 and year_pass >= 70:
        decision = "合格: デモEA化に進める。ただし極小ロットから"
    elif score >= 55 and stress_pf >= 1.03:
        decision = "準合格: デモ検証は可。本番はまだ不可"
    else:
        decision = "保留: まだ壊れやすい。条件追加か別候補へ"

    return pd.DataFrame([{
        "decision": decision,
        "validation_score": score,
        "base_pf_spread0_5": round(base_pf, 3),
        "stress_pf_spread1_0_slip0_5": round(stress_pf, 3),
        "hard_pf_spread1_5_slip0_5": round(hard_pf, 3),
        "delay_pass_rate_pct": round(delay_pass, 2),
        "year_pass_rate_pct": round(year_pass, 2),
        "regime_pass_rate_pct": round(regime_pass, 2),
        "mc_risk_of_loss_pct": round(loss_risk, 2),
        "mc_risk_dd_over_3000pips_pct": round(dd_risk, 2),
    }])


def run_validation_engine(df: pd.DataFrame, improvement_results: Dict, spread: float = 0.5) -> Dict[str, pd.DataFrame]:
    ranking = improvement_results.get("improvement_ranking", pd.DataFrame()) if improvement_results else pd.DataFrame()
    if ranking is None or len(ranking) == 0:
        return {"validation_decision": pd.DataFrame([{"decision": "判定不能: 11.0改善候補がありません"}])}

    top = ranking.iloc[0]
    spec = _row_to_spec(top)
    print(f"12.0 本命候補を検証: {spec.name}")
    enriched = _enrich(df)

    base_trades = _build_with_delay(enriched, spec, spread=spread, slippage=0.0)
    cost = _cost_stress(enriched, spec)
    delay = _delay_stress(enriched, spec, spread=spread, slippage=0.2)
    year = _rolling_year_detail(base_trades)
    cal = _calendar_breakdown(base_trades)
    regime = _regime_breakdown(base_trades)
    mc = _monte_carlo(base_trades, runs=5000)
    decision = _final_decision(cost, delay, year, regime, mc)

    selected = pd.DataFrame([{
        "strategy": spec.name,
        "direction": spec.direction,
        "session": f"{spec.session_start}-{spec.session_end}",
        "rule_type": spec.rule_type,
        "param": spec.param,
        "exit_bars": spec.exit_bars,
        "filter_type": spec.filter_type,
    }])

    out = {
        "validation_selected_strategy": selected,
        "validation_decision": decision,
        "validation_cost_stress": cost,
        "validation_delay_stress": delay,
        "validation_year_detail": year,
        "validation_regime_detail": regime,
        "validation_monte_carlo": mc,
        "validation_top_trades": base_trades,
    }
    out.update({f"validation_{k}": v for k, v in cal.items()})
    return out
