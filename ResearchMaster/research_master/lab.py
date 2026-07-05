
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from config import INITIAL_BALANCE, YEN_PER_PIP, MONTE_CARLO_RUNS, MONTE_CARLO_SEED, MAX_RISK_PCT
from research_master.metrics import summarize_trades, max_drawdown
from research_master.optimizer import daily_pivot, correlation_table, apply_weights, calc_weights
from research_master.strategy import make_candidate_specs, build_trades_for_spec


def _safe_summary(trades: pd.DataFrame, spread: float = 0.5) -> Dict:
    if trades is None or len(trades) == 0:
        return summarize_trades(pd.DataFrame(columns=["pips", "equity_pips"]), spread=spread)
    t = trades.copy().sort_values("datetime").reset_index(drop=True)
    t["equity_pips"] = t["pips"].cumsum()
    return summarize_trades(t, spread=spread)


def portfolio_correlation_lab(selected_df: pd.DataFrame, trades_map: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    names = selected_df["strategy"].tolist() if len(selected_df) else []
    daily = daily_pivot(trades_map, names)
    corr_pairs = correlation_table(daily)
    if daily.empty:
        matrix = pd.DataFrame()
    else:
        matrix = daily.corr().fillna(0.0).round(4)
    if len(corr_pairs):
        corr_pairs["判定"] = np.where(corr_pairs["corr"].abs() >= 0.75, "高相関 注意", np.where(corr_pairs["corr"].abs() >= 0.55, "中相関", "低相関"))
    return corr_pairs, matrix


def monte_carlo_lab(trades: pd.DataFrame, runs: int = MONTE_CARLO_RUNS, seed: int = MONTE_CARLO_SEED) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    pips = trades["pips"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(runs):
        sample = rng.permutation(pips)
        eq = np.cumsum(sample)
        rows.append({
            "run": i + 1,
            "total_pips": round(float(eq[-1]), 3),
            "max_dd_pips": round(float(max_drawdown(eq)), 3),
            "min_equity_pips": round(float(eq.min()), 3),
            "risk_of_loss": bool(eq[-1] < 0),
            "risk_dd_over_20pct": bool(abs(max_drawdown(eq)) * YEN_PER_PIP > INITIAL_BALANCE * (MAX_RISK_PCT / 100.0)),
        })
    return pd.DataFrame(rows)


def monte_carlo_summary(mc: pd.DataFrame) -> pd.DataFrame:
    if mc is None or len(mc) == 0:
        return pd.DataFrame()
    dd = mc["max_dd_pips"].abs()
    out = {
        "runs": int(len(mc)),
        "total_pips_p05": round(float(mc["total_pips"].quantile(0.05)), 3),
        "total_pips_median": round(float(mc["total_pips"].median()), 3),
        "total_pips_p95": round(float(mc["total_pips"].quantile(0.95)), 3),
        "dd_p50": round(float(dd.quantile(0.50)), 3),
        "dd_p90": round(float(dd.quantile(0.90)), 3),
        "dd_p95": round(float(dd.quantile(0.95)), 3),
        "dd_p99": round(float(dd.quantile(0.99)), 3),
        "risk_of_loss_pct": round(float(mc["risk_of_loss"].mean() * 100), 2),
        "risk_dd_over_20pct": round(float(mc["risk_dd_over_20pct"].mean() * 100), 2),
    }
    return pd.DataFrame([out])


def position_sizing_lab(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    p = trades["pips"].astype(float)
    wins = p[p > 0]
    losses = p[p <= 0]
    winrate = float((p > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
    rr = avg_win / avg_loss if avg_loss > 0 else 0.0
    kelly = winrate - (1 - winrate) / rr if rr > 0 else 0.0
    kelly = max(0.0, min(kelly, 0.20))
    eq = p.cumsum()
    dd = abs(max_drawdown(eq))
    rows = []
    methods = [
        ("fixed", 1.0),
        ("half_kelly", max(kelly / 2, 0.01)),
        ("quarter_kelly", max(kelly / 4, 0.005)),
    ]
    # DDが大きい場合は自動で縮小する目安
    target_dd_pips = INITIAL_BALANCE * 0.20 / YEN_PER_PIP
    vol_scale = min(1.0, target_dd_pips / max(dd, 1.0))
    methods.append(("dd_20pct_cap", max(vol_scale, 0.01)))
    for name, mult in methods:
        t = trades.copy()
        t["pips"] = t["pips"] * mult
        t["equity_pips"] = t["pips"].cumsum()
        s = summarize_trades(t, spread=0.5)
        s["position_method"] = name
        s["lot_multiplier"] = round(float(mult), 5)
        s["kelly_estimate"] = round(float(kelly), 5)
        rows.append(s)
    return pd.DataFrame(rows).sort_values(["pf", "max_dd_pips"], ascending=[False, False]).reset_index(drop=True)


def rolling_walk_forward_lab(df: pd.DataFrame, spread: float = 0.5, top_n: int = 8) -> pd.DataFrame:
    years = sorted([int(y) for y in df["year"].dropna().unique()])
    if len(years) < 3:
        return pd.DataFrame()
    specs = make_candidate_specs()
    rows = []
    # 2年学習→翌1年検証。軽量化のため全候補を使うが上位だけ詳細化。
    for i in range(0, len(years) - 2):
        train_years = years[i:i+2]
        test_year = years[i+2]
        train_df = df[df["year"].isin(train_years)].copy()
        test_df = df[df["year"] == test_year].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        scored = []
        trade_cache = {}
        for spec in specs:
            tr = build_trades_for_spec(train_df, spec, spread=spread)
            s = _safe_summary(tr, spread=spread)
            score = (s["pf"] - 1.0) * 1000 + s["expectancy_pips"] * 100 - abs(s["max_dd_pips"]) / 60
            scored.append((score, spec, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, spec, train_s) in enumerate(scored[:top_n], 1):
            test_tr = build_trades_for_spec(test_df, spec, spread=spread)
            test_s = _safe_summary(test_tr, spread=spread)
            rows.append({
                "train_years": "-".join(map(str, train_years)),
                "test_year": test_year,
                "rank": rank,
                "strategy": spec.name,
                "train_trades": train_s["trades"],
                "train_pf": train_s["pf"],
                "train_exp": train_s["expectancy_pips"],
                "train_dd": train_s["max_dd_pips"],
                "test_trades": test_s["trades"],
                "test_pf": test_s["pf"],
                "test_exp": test_s["expectancy_pips"],
                "test_dd": test_s["max_dd_pips"],
                "pass": bool(test_s["pf"] >= 1.05 and test_s["expectancy_pips"] > 0 and test_s["trades"] >= 100),
            })
    out = pd.DataFrame(rows)
    return out


def rolling_summary(rwf: pd.DataFrame) -> pd.DataFrame:
    if rwf is None or len(rwf) == 0:
        return pd.DataFrame()
    best_each = rwf.sort_values(["test_year", "rank"]).groupby("test_year").head(1)
    return pd.DataFrame([{
        "windows": int(best_each["test_year"].nunique()),
        "pass_windows": int(best_each["pass"].sum()),
        "pass_rate_pct": round(float(best_each["pass"].mean() * 100), 2),
        "avg_test_pf": round(float(best_each["test_pf"].mean()), 3),
        "avg_test_exp": round(float(best_each["test_exp"].mean()), 4),
        "worst_test_pf": round(float(best_each["test_pf"].min()), 3),
        "worst_test_exp": round(float(best_each["test_exp"].min()), 4),
    }])


def lab_decision(optimizer_summary: pd.DataFrame, mc_sum: pd.DataFrame, rwf_sum: pd.DataFrame) -> str:
    reasons = []
    ok = True
    if optimizer_summary is None or len(optimizer_summary) == 0:
        return "判定不能: optimizer結果がありません"
    best = optimizer_summary.iloc[0]
    if best.get("pf", 0) < 1.15:
        ok = False; reasons.append("PF不足")
    if best.get("expectancy_pips", 0) <= 0:
        ok = False; reasons.append("期待値不足")
    if mc_sum is not None and len(mc_sum):
        m = mc_sum.iloc[0]
        if m.get("risk_of_loss_pct", 100) > 5:
            ok = False; reasons.append("MonteCarlo損失確率が高い")
        if m.get("risk_dd_over_20pct", 100) > 20:
            ok = False; reasons.append("DD20%超リスクが高い")
    if rwf_sum is not None and len(rwf_sum):
        r = rwf_sum.iloc[0]
        if r.get("pass_rate_pct", 0) < 60:
            ok = False; reasons.append("Rolling WF通過率不足")
    if ok:
        return "合格候補: 小ロット検証へ進める"
    return "保留: " + " / ".join(reasons)


def run_portfolio_lab(df: pd.DataFrame, selected_df: pd.DataFrame, trades_map: Dict[str, pd.DataFrame], optimizer_best_trades: pd.DataFrame, optimizer_summary: pd.DataFrame):
    corr_pairs, corr_matrix = portfolio_correlation_lab(selected_df, trades_map)
    mc = monte_carlo_lab(optimizer_best_trades)
    mc_sum = monte_carlo_summary(mc)
    pos = position_sizing_lab(optimizer_best_trades)
    rwf = rolling_walk_forward_lab(df, spread=0.5, top_n=5)
    rwf_sum = rolling_summary(rwf)
    decision = lab_decision(optimizer_summary, mc_sum, rwf_sum)
    decision_df = pd.DataFrame([{"final_decision": decision}])
    return {
        "corr_pairs": corr_pairs,
        "corr_matrix": corr_matrix,
        "monte_carlo": mc,
        "monte_carlo_summary": mc_sum,
        "position_sizing": pos,
        "rolling_walk_forward": rwf,
        "rolling_summary": rwf_sum,
        "decision": decision_df,
    }
