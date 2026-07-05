# =====================================================
# research_v07.py
# 岡畠AI研究所 Ver.0.7 ChatGPT版
# 本命候補の再検証・月別/年別耐久チェック
#
# 実行方法:
#   python research_v07.py
#
# 入力:
#   data/analysis_v07/candidate_analysis_v07.csv
#   data/USDJPY_5min_2021-2026.csv
#
# 出力先:
#   data/research_v07/
# =====================================================

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# v06の計算エンジンを使う
from research_v06 import (
    Config as BaseConfig,
    PIP,
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


@dataclass
class Config:
    base_dir: Path = Path(__file__).parent
    candidate_csv: Path = Path(__file__).parent / "data" / "analysis_v07" / "candidate_analysis_v07.csv"
    out_dir: Path = Path(__file__).parent / "data" / "research_v07"

    # 本命候補だけ深掘り。全部見たいなら False。
    only_honmei: bool = True

    # 2026年が途中までしかないため、この期間でチェック
    forward_start: str = "2025-01-01"
    forward_end: str = "2026-06-30"

    # 何件まで詳細トレードを出すか
    detail_top_n: int = 20


def parse_strategy(text: str) -> Tuple:
    """戦略文字列を v06 の sig_key に戻す。"""
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

    # v06の押し目表記は RSI期間が消えているため、期間14として再現する
    m = re.search(r"押し目\(EMA200\+RSI(\d+)/(\d+)\)", text)
    if m:
        return ("押し目", 14, int(m.group(1)), int(m.group(2)))

    raise ValueError(f"戦略を解析できません: {text}")


def stats_for_period(pnl: np.ndarray, ts: np.ndarray, start: str, end: str) -> Dict[str, float]:
    s = pd.Timestamp(start).value
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).value
    idx = (ts >= s) & (ts < e)
    st = calc_stats(pnl[idx], ts[idx])
    st["期待値pips/回"] = round(st["総損益pips"] / st["取引回数"], 3) if st["取引回数"] else 0.0
    return st


def monthly_stats(pnl: np.ndarray, ts: np.ndarray, start: str, end: str) -> pd.DataFrame:
    if len(pnl) == 0:
        return pd.DataFrame()
    dt = pd.to_datetime(ts)
    frame = pd.DataFrame({"datetime": dt, "pnl": pnl})
    frame = frame[(frame["datetime"] >= pd.Timestamp(start)) & (frame["datetime"] <= pd.Timestamp(end) + pd.Timedelta(days=1))]
    if frame.empty:
        return pd.DataFrame()
    frame["month"] = frame["datetime"].dt.to_period("M").astype(str)
    rows = []
    for month, g in frame.groupby("month"):
        p = g["pnl"].to_numpy(dtype=float)
        gross_w = p[p > 0].sum()
        gross_l = -p[p <= 0].sum()
        pf = gross_w / gross_l if gross_l > 0 else (999.0 if gross_w > 0 else 0.0)
        rows.append({
            "月": month,
            "取引回数": int(len(p)),
            "損益pips": round(float(p.sum()), 1),
            "期待値pips/回": round(float(p.mean()), 3) if len(p) else 0.0,
            "PF": round(float(min(pf, 999.0)), 3),
            "勝率%": round(float((p > 0).mean() * 100), 1) if len(p) else 0.0,
        })
    return pd.DataFrame(rows)


def score_candidate(base_row: pd.Series, st25: Dict[str, float], st26: Dict[str, float], mon: pd.DataFrame) -> Tuple[str, float, str]:
    pf = float(base_row.get("PF", 0) or 0)
    ev = float(base_row.get("期待値pips/回", 0) or 0)
    dd = abs(float(base_row.get("最大DD_pips", 0) or 0))
    trades = int(base_row.get("取引回数", 0) or 0)

    pos_month_rate = 0.0
    worst_month = 0.0
    if not mon.empty:
        pos_month_rate = float((mon["損益pips"] > 0).mean() * 100)
        worst_month = float(mon["損益pips"].min())

    s = 0.0
    s += min(pf, 2.0) / 2.0 * 25
    s += max(min(ev, 2.0), -2.0) / 2.0 * 20
    s += min(max(trades / 150, 0), 1) * 10
    s += min(max(pos_month_rate / 70, 0), 1) * 20
    s += 10 if st25["総損益pips"] > 0 else -10
    s += 10 if st26["総損益pips"] > 0 else -10
    s += 5 if worst_month > -80 else -5
    s -= min(dd / 300, 1) * 10
    s = round(float(s), 1)

    reasons = []
    reasons.append(f"月勝率{pos_month_rate:.1f}%")
    reasons.append(f"2025損益{st25['総損益pips']}pips")
    reasons.append(f"2026損益{st26['総損益pips']}pips")
    reasons.append(f"最悪月{worst_month:.1f}pips")

    if s >= 70 and st25["総損益pips"] > 0 and st26["総損益pips"] > 0 and pos_month_rate >= 55:
        verdict = "実戦候補"
    elif s >= 55 and pos_month_rate >= 45:
        verdict = "監視候補"
    else:
        verdict = "保留"

    return verdict, s, " / ".join(reasons)


def run(cfg: Config) -> None:
    t0 = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.candidate_csv.exists():
        raise FileNotFoundError(f"候補CSVがありません: {cfg.candidate_csv}\n先に python analyze_candidates_v07.py を実行してください。")

    cands = pd.read_csv(cfg.candidate_csv)
    if cfg.only_honmei and "v07判定" in cands.columns:
        cands = cands[cands["v07判定"] == "本命候補"].copy()

    print("=" * 70)
    print("岡畠AI研究所 Ver.0.7 本命候補 再検証")
    print("=" * 70)
    print(f"読み込み候補: {len(cands)}件")
    print(f"入力CSV: {cfg.candidate_csv}")

    base_cfg = BaseConfig()
    df = load_data(base_cfg)
    ind = IndicatorCache(df, base_cfg)
    df = ind.prepare_base_env()
    spread = build_spread_array(df, base_cfg)

    signal_cache = {}
    outcome_cache = {}
    rows: List[Dict] = []
    trade_rows: List[Dict] = []
    month_all: List[pd.DataFrame] = []

    for no, (_, r) in enumerate(cands.iterrows(), 1):
        sig_key = parse_strategy(r["戦略"])
        tp = int(r["利確pips"])
        sl = int(r["損切りpips"])
        time_f = str(r["時間帯"])
        weekday_f = str(r["曜日"])
        trend_f = str(r["トレンド環境"])
        vol_f = str(r["ボラ環境"])

        if sig_key not in signal_cache:
            sig = build_signal(sig_key, ind)
            ctx = build_candidates(df, sig)
            signal_cache[sig_key] = ctx
        ctx = signal_cache[sig_key]

        out_key = (sig_key, tp, sl)
        if out_key not in outcome_cache:
            outcome_cache[out_key] = compute_outcomes(df, ctx, tp, sl, spread)
        exit_idx, pnl_all, reason = outcome_cache[out_key]

        m = filter_mask(ctx, time_f, weekday_f, trend_f, vol_f)
        selected = select_exclusive(ctx["entry_idx"], exit_idx, m)
        pnl = pnl_all[selected]
        ts = ctx["ts"][selected]

        st25 = stats_for_period(pnl, ts, "2025-01-01", "2025-12-31")
        st26 = stats_for_period(pnl, ts, "2026-01-01", "2026-06-30")
        stfw = stats_for_period(pnl, ts, cfg.forward_start, cfg.forward_end)
        mon = monthly_stats(pnl, ts, cfg.forward_start, cfg.forward_end)
        verdict, score, reason_txt = score_candidate(r, st25, st26, mon)

        pos_month_rate = round(float((mon["損益pips"] > 0).mean() * 100), 1) if not mon.empty else 0.0
        worst_month = round(float(mon["損益pips"].min()), 1) if not mon.empty else 0.0

        out = {
            "v07最終判定": verdict,
            "v07最終スコア": score,
            "理由": reason_txt,
            "元v07判定": r.get("v07判定", ""),
            "元v07スコア": r.get("v07スコア", ""),
            "戦略": r["戦略"],
            "戦略タイプ": r["戦略タイプ"],
            "利確pips": tp,
            "損切りpips": sl,
            "時間帯": time_f,
            "曜日": weekday_f,
            "トレンド環境": trend_f,
            "ボラ環境": vol_f,
            "元PF": r.get("PF", np.nan),
            "元期待値pips/回": r.get("期待値pips/回", np.nan),
            "元勝率%": r.get("勝率%", np.nan),
            "元取引回数": r.get("取引回数", np.nan),
            "月プラス率%": pos_month_rate,
            "最悪月pips": worst_month,
            "2025_損益pips": st25["総損益pips"],
            "2025_PF": st25["PF"],
            "2025_期待値": st25["期待値pips/回"],
            "2025_取引回数": st25["取引回数"],
            "2026_損益pips": st26["総損益pips"],
            "2026_PF": st26["PF"],
            "2026_期待値": st26["期待値pips/回"],
            "2026_取引回数": st26["取引回数"],
            "フォワード損益pips": stfw["総損益pips"],
            "フォワードPF": stfw["PF"],
            "フォワード期待値": stfw["期待値pips/回"],
            "フォワード取引回数": stfw["取引回数"],
        }
        rows.append(out)

        if not mon.empty:
            mon2 = mon.copy()
            mon2.insert(0, "候補No", no)
            mon2.insert(1, "戦略", r["戦略"])
            mon2.insert(2, "条件", f"TP{tp} SL{sl} {time_f} {weekday_f} {trend_f} {vol_f}")
            month_all.append(mon2)

        print(f"{no:02d}/{len(cands)} {verdict} score={score} {r['戦略']} TP{tp} SL{sl}")

    result = pd.DataFrame(rows).sort_values("v07最終スコア", ascending=False).reset_index(drop=True)
    result.to_csv(cfg.out_dir / "final_candidates.csv", index=False, encoding="utf-8-sig")

    if month_all:
        pd.concat(month_all, ignore_index=True).to_csv(cfg.out_dir / "monthly_breakdown.csv", index=False, encoding="utf-8-sig")

    report = build_report(result, time.time() - t0, cfg)
    (cfg.out_dir / "research_v07_report.txt").write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print(f"最終候補CSV : {cfg.out_dir / 'final_candidates.csv'}")
    print(f"月別分析CSV : {cfg.out_dir / 'monthly_breakdown.csv'}")
    print(f"レポート     : {cfg.out_dir / 'research_v07_report.txt'}")


def build_report(result: pd.DataFrame, elapsed: float, cfg: Config) -> str:
    L: List[str] = []
    L.append("=" * 70)
    L.append("岡畠AI研究所 Ver.0.7 本命候補 再検証レポート")
    L.append("=" * 70)
    L.append(f"実行時間: {elapsed:.0f}秒")
    L.append(f"検証対象: {len(result)}件")
    L.append("")
    L.append("判定内訳:")
    for k, v in result["v07最終判定"].value_counts().items():
        L.append(f"  {k}: {v}件")

    L.append("")
    L.append("上位候補:")
    for i, (_, r) in enumerate(result.head(10).iterrows(), 1):
        L.append(
            f"{i}. [{r['v07最終判定']}] score{r['v07最終スコア']} "
            f"{r['戦略']} TP{r['利確pips']} SL{r['損切りpips']} "
            f"{r['時間帯']} {r['曜日']} {r['トレンド環境']} {r['ボラ環境']} / "
            f"2025 {r['2025_損益pips']}pips PF{r['2025_PF']} / "
            f"2026 {r['2026_損益pips']}pips PF{r['2026_PF']} / "
            f"月プラス率{r['月プラス率%']}% / 最悪月{r['最悪月pips']}pips"
        )

    L.append("")
    L.append("次にやること:")
    L.append("  1. 実戦候補だけ残す。")
    L.append("  2. 月別でマイナスが大きい候補は除外する。")
    L.append("  3. 残った候補だけをリアルタイム監視ツールに回す。")
    L.append("  4. いきなり自動売買にはせず、まず通知だけにする。")
    L.append("")
    L.append("注意: これは研究用であり、利益を保証するものではありません。")
    return "\n".join(L)


if __name__ == "__main__":
    run(Config())
