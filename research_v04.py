# =====================================================
# research_v04.py
# 岡畠AI研究所 Ver.0.4 ChatGPT版
# USD/JPY 5分足 条件研究ツール
#
# 実行方法:
#   python research_v04.py
#
# 出力先:
#   data/research_v04/
# =====================================================

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PIP = 0.01  # USD/JPY の 1pips = 0.01円
R_TP, R_SL, R_EOD = 0, 1, 2
REASON_NAMES = {R_TP: "利確", R_SL: "損切り", R_EOD: "日終わり決済"}
WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


@dataclass
class Config:
    csv_path: Path = Path(__file__).parent / "data" / "USDJPY_5min_2021-2026.csv"
    out_dir: Path = Path(__file__).parent / "data" / "research_v04"

    # まずは50,000。重ければ 10000 に下げてOK。
    n_patterns: int = 50_000
    random_seed: int = 42

    train_start: str = "2021-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-06-30"

    # 時間帯別スプレッド（銭 = pips）
    spread_tokyo: float = 0.2      # 9-15
    spread_london: float = 0.3     # 16-21
    spread_ny: float = 0.4         # 21-24
    spread_other: float = 0.5

    # 探索プール
    ema_fast_pool: Tuple[int, ...] = (3, 5, 8, 10, 12, 15, 20, 25)
    ema_slow_pool: Tuple[int, ...] = (30, 50, 75, 100, 150, 200)
    rsi_period_pool: Tuple[int, ...] = (7, 10, 14, 21)
    rsi_low_pool: Tuple[int, ...] = (20, 25, 30, 35, 40)
    rsi_high_pool: Tuple[int, ...] = (60, 65, 70, 75, 80)
    macd_fast_pool: Tuple[int, ...] = (8, 12, 16)
    macd_slow_pool: Tuple[int, ...] = (21, 26, 35)
    macd_signal_pool: Tuple[int, ...] = (6, 9, 12)
    bb_period_pool: Tuple[int, ...] = (14, 20, 25, 30)
    bb_dev_pool: Tuple[float, ...] = (1.5, 2.0, 2.5)
    tp_pool: Tuple[int, ...] = (4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
    sl_pool: Tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25, 30)

    time_pool: Tuple[str, ...] = (
        "全時間", "東京(9-15時)", "ロンドン(16-21時)", "NY(21-23時)",
        "9時台", "10時台", "15時台", "16時台", "21時台", "22時台"
    )
    weekday_pool: Tuple[str, ...] = ("全曜日", "月", "火", "水", "木", "金")
    trend_pool: Tuple[str, ...] = ("なし", "上昇のみ", "下降のみ", "レンジのみ", "順方向のみ")
    vol_pool: Tuple[str, ...] = ("なし", "ATR高", "ATR低", "直近30分変動大")

    # 新規エントリー締切。これ以降は建てない。日終わりで強制決済。
    entry_cutoff_hour: int = 23

    min_trades_train: int = 300
    min_trades_test: int = 100
    pf_cap: float = 10.0
    top_n_report: int = 10


# -----------------------------------------------------
# データと指標
# -----------------------------------------------------
def load_data(cfg: Config) -> pd.DataFrame:
    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSVがありません: {cfg.csv_path}\n先に python main.py を実行してください。")

    df = pd.read_csv(cfg.csv_path, parse_dates=["datetime_utc", "datetime_jst"])
    df = df.set_index("datetime_jst").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df["date_jst"] = df.index.date
    date_arr = df["date_jst"].to_numpy()
    is_eod = np.empty(len(df), dtype=bool)
    is_eod[:-1] = date_arr[:-1] != date_arr[1:]
    is_eod[-1] = True
    df["is_eod"] = is_eod

    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["can_enter"] = df["hour"] < cfg.entry_cutoff_hour

    print(f"データ読み込み: {len(df):,}本 ({df.index.min():%Y-%m-%d} ～ {df.index.max():%Y-%m-%d} JST)")
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


class IndicatorCache:
    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df = df
        self.cfg = cfg
        self.cache: Dict[Tuple, object] = {}
        self.close = df["close"]

    def get_ema(self, n: int) -> pd.Series:
        key = ("ema", n)
        if key not in self.cache:
            self.cache[key] = ema(self.close, n)
        return self.cache[key]

    def get_rsi(self, n: int) -> pd.Series:
        key = ("rsi", n)
        if key not in self.cache:
            self.cache[key] = rsi(self.close, n)
        return self.cache[key]

    def get_macd(self, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series]:
        key = ("macd", fast, slow, signal)
        if key not in self.cache:
            self.cache[key] = macd(self.close, fast, slow, signal)
        return self.cache[key]

    def get_bb(self, period: int, dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        key = ("bb", period, dev)
        if key not in self.cache:
            mid = self.close.rolling(period).mean()
            sd = self.close.rolling(period).std(ddof=0)
            self.cache[key] = (mid, mid + dev * sd, mid - dev * sd)
        return self.cache[key]

    def prepare_base_env(self) -> pd.DataFrame:
        df = self.df
        ema20 = self.get_ema(20)
        ema50 = self.get_ema(50)
        ema200 = self.get_ema(200)
        df["trend_up"] = (ema20 > ema50) & (ema50 > ema200)
        df["trend_dn"] = (ema20 < ema50) & (ema50 < ema200)
        df["trend_rng"] = ~(df["trend_up"] | df["trend_dn"])

        df["atr14"] = atr(df, 14)
        atr_med = df["atr14"].rolling(5760, min_periods=1000).median()
        df["vol_high"] = df["atr14"] > atr_med * 1.2
        df["vol_low"] = df["atr14"] < atr_med * 0.8
        df["vol_move30"] = (df["close"] - df["close"].shift(6)).abs() >= 8 * PIP
        return df


# -----------------------------------------------------
# 条件生成
# -----------------------------------------------------
def generate_patterns(cfg: Config) -> List[Tuple]:
    rng = random.Random(cfg.random_seed)
    patterns = set()
    limit = cfg.n_patterns * 200
    tries = 0

    kinds = ["EMAクロス", "RSI逆張り", "MACDクロス", "BB逆張り", "BBブレイク", "押し目"]

    while len(patterns) < cfg.n_patterns and tries < limit:
        tries += 1
        kind = rng.choice(kinds)

        if kind == "EMAクロス":
            f = rng.choice(cfg.ema_fast_pool)
            s = rng.choice(cfg.ema_slow_pool)
            if f >= s:
                continue
            sig_key = (kind, f, s)

        elif kind == "RSI逆張り":
            p = rng.choice(cfg.rsi_period_pool)
            lo = rng.choice(cfg.rsi_low_pool)
            hi = rng.choice(cfg.rsi_high_pool)
            if lo >= hi:
                continue
            sig_key = (kind, p, lo, hi)

        elif kind == "MACDクロス":
            f = rng.choice(cfg.macd_fast_pool)
            s = rng.choice(cfg.macd_slow_pool)
            g = rng.choice(cfg.macd_signal_pool)
            if f >= s:
                continue
            sig_key = (kind, f, s, g)

        elif kind in ("BB逆張り", "BBブレイク"):
            sig_key = (kind, rng.choice(cfg.bb_period_pool), rng.choice(cfg.bb_dev_pool))

        else:  # 押し目
            sig_key = (kind, rng.choice(cfg.rsi_period_pool), rng.choice((35, 40, 45)), rng.choice((55, 60, 65)))

        pat = (
            sig_key,
            rng.choice(cfg.tp_pool), rng.choice(cfg.sl_pool),
            rng.choice(cfg.time_pool), rng.choice(cfg.weekday_pool),
            rng.choice(cfg.trend_pool), rng.choice(cfg.vol_pool),
        )
        patterns.add(pat)

    return sorted(patterns, key=repr)


def sig_key_to_text(sig_key: Tuple) -> str:
    kind = sig_key[0]
    if kind == "EMAクロス":
        return f"EMAクロス({sig_key[1]},{sig_key[2]})"
    if kind == "RSI逆張り":
        return f"RSI逆張り(期間{sig_key[1]}, {sig_key[2]}/{sig_key[3]})"
    if kind == "MACDクロス":
        return f"MACDクロス({sig_key[1]},{sig_key[2]},{sig_key[3]})"
    if kind == "BB逆張り":
        return f"BB逆張り(期間{sig_key[1]}, {sig_key[2]}σ)"
    if kind == "BBブレイク":
        return f"BBブレイク(期間{sig_key[1]}, {sig_key[2]}σ)"
    if kind == "押し目":
        return f"押し目(EMA200+RSI{sig_key[2]}/{sig_key[3]})"
    return repr(sig_key)


# -----------------------------------------------------
# シグナル生成
# -----------------------------------------------------
def build_signal(sig_key: Tuple, ind: IndicatorCache) -> pd.Series:
    kind = sig_key[0]
    df = ind.df
    c = df["close"]

    if kind == "EMAクロス":
        _, f, s = sig_key
        fast = ind.get_ema(f)
        slow = ind.get_ema(s)
        up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)

    if kind == "RSI逆張り":
        _, p, lo, hi = sig_key
        rr = ind.get_rsi(p)
        up = (rr.shift(1) < lo) & (rr >= lo)
        dn = (rr.shift(1) > hi) & (rr <= hi)
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)

    if kind == "MACDクロス":
        _, f, s, g = sig_key
        line, sig = ind.get_macd(f, s, g)
        up = (line > sig) & (line.shift(1) <= sig.shift(1))
        dn = (line < sig) & (line.shift(1) >= sig.shift(1))
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)

    if kind == "BB逆張り":
        _, p, dev = sig_key
        mid, upper, lower = ind.get_bb(p, dev)
        up = (c.shift(1) < lower.shift(1)) & (c >= lower)
        dn = (c.shift(1) > upper.shift(1)) & (c <= upper)
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)

    if kind == "BBブレイク":
        _, p, dev = sig_key
        mid, upper, lower = ind.get_bb(p, dev)
        up = (c.shift(1) <= upper.shift(1)) & (c > upper)
        dn = (c.shift(1) >= lower.shift(1)) & (c < lower)
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)

    if kind == "押し目":
        _, rsi_p, buy_level, sell_level = sig_key
        rr = ind.get_rsi(rsi_p)
        ema200 = ind.get_ema(200)
        above = c > ema200
        below = c < ema200
        up = above & (rr.shift(1) < buy_level) & (rr >= buy_level)
        dn = below & (rr.shift(1) > sell_level) & (rr <= sell_level)
        return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)

    raise ValueError(f"unknown signal: {sig_key}")


# -----------------------------------------------------
# エントリー候補・フィルター・スプレッド
# -----------------------------------------------------
def build_spread_array(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    h = df.index.hour.to_numpy()
    sp = np.full(len(df), cfg.spread_other * PIP, dtype=float)
    sp[(h >= 9) & (h < 15)] = cfg.spread_tokyo * PIP
    sp[(h >= 16) & (h < 21)] = cfg.spread_london * PIP
    sp[(h >= 21) & (h < 24)] = cfg.spread_ny * PIP
    return sp


def build_candidates(df: pd.DataFrame, sig: pd.Series) -> Dict[str, np.ndarray]:
    sig_arr = np.nan_to_num(sig.to_numpy()).astype(np.int8)
    can_enter = df["can_enter"].to_numpy()
    is_eod = df["is_eod"].to_numpy()
    e_all = np.arange(1, len(df))
    ok = (sig_arr[e_all - 1] != 0) & can_enter[e_all] & ~is_eod[e_all]
    entry_idx = e_all[ok]
    sig_idx = entry_idx - 1
    ts = df.index.as_unit("ns").asi8

    return {
        "entry_idx": entry_idx,
        "signal_idx": sig_idx,
        "side": sig_arr[sig_idx],
        "ts": ts[entry_idx],
        "hour": df.index.hour.to_numpy()[entry_idx],
        "weekday": df.index.weekday.to_numpy()[entry_idx],
        "trend_up": df["trend_up"].to_numpy()[sig_idx],
        "trend_dn": df["trend_dn"].to_numpy()[sig_idx],
        "trend_rng": df["trend_rng"].to_numpy()[sig_idx],
        "vol_high": df["vol_high"].to_numpy()[sig_idx],
        "vol_low": df["vol_low"].to_numpy()[sig_idx],
        "vol_move30": df["vol_move30"].to_numpy()[sig_idx],
    }


def filter_mask(ctx: Dict[str, np.ndarray], time_f: str, weekday_f: str, trend_f: str, vol_f: str) -> np.ndarray:
    n = len(ctx["entry_idx"])
    m = np.ones(n, dtype=bool)
    h = ctx["hour"]
    wd = ctx["weekday"]
    side = ctx["side"]

    if time_f == "東京(9-15時)":
        m &= (h >= 9) & (h < 15)
    elif time_f == "ロンドン(16-21時)":
        m &= (h >= 16) & (h < 21)
    elif time_f == "NY(21-23時)":
        m &= (h >= 21) & (h < 23)
    elif time_f.endswith("時台"):
        try:
            hh = int(time_f.replace("時台", ""))
            m &= h == hh
        except ValueError:
            pass

    if weekday_f != "全曜日":
        idx = WEEKDAYS.index(weekday_f)
        m &= wd == idx

    if trend_f == "上昇のみ":
        m &= ctx["trend_up"]
    elif trend_f == "下降のみ":
        m &= ctx["trend_dn"]
    elif trend_f == "レンジのみ":
        m &= ctx["trend_rng"]
    elif trend_f == "順方向のみ":
        m &= ((side > 0) & ctx["trend_up"]) | ((side < 0) & ctx["trend_dn"])

    if vol_f == "ATR高":
        m &= ctx["vol_high"]
    elif vol_f == "ATR低":
        m &= ctx["vol_low"]
    elif vol_f == "直近30分変動大":
        m &= ctx["vol_move30"]

    return m


# -----------------------------------------------------
# 約定計算・成績
# -----------------------------------------------------
def compute_outcomes(df: pd.DataFrame, ctx: Dict[str, np.ndarray], tp: int, sl: int, spread: np.ndarray):
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    is_eod = df["is_eod"].to_numpy()

    entry_idx = ctx["entry_idx"]
    side = ctx["side"]
    n = len(entry_idx)
    exit_idx = np.empty(n, dtype=np.int64)
    pnl = np.empty(n, dtype=np.float64)
    reason = np.empty(n, dtype=np.int8)

    tp_v = tp * PIP
    sl_v = sl * PIP

    for k in range(n):
        e = entry_idx[k]
        if side[k] > 0:
            entry = o[e] + spread[e]
            sl_level = entry - sl_v
            tp_level = entry + tp_v
            i = e
            while True:
                if l[i] <= sl_level:
                    exit_idx[k] = i; pnl[k] = -sl; reason[k] = R_SL; break
                if h[i] >= tp_level:
                    exit_idx[k] = i; pnl[k] = tp; reason[k] = R_TP; break
                if is_eod[i]:
                    exit_idx[k] = i; pnl[k] = (c[i] - entry) / PIP; reason[k] = R_EOD; break
                i += 1
        else:
            entry = o[e]
            i = e
            while True:
                sp = spread[i]
                sl_level_bid = entry + sl_v - sp
                tp_level_bid = entry - tp_v - sp
                if h[i] >= sl_level_bid:
                    exit_idx[k] = i; pnl[k] = -sl; reason[k] = R_SL; break
                if l[i] <= tp_level_bid:
                    exit_idx[k] = i; pnl[k] = tp; reason[k] = R_TP; break
                if is_eod[i]:
                    exit_idx[k] = i; pnl[k] = (entry - (c[i] + sp)) / PIP; reason[k] = R_EOD; break
                i += 1

    return exit_idx, pnl, reason


def select_exclusive(entry_idx: np.ndarray, exit_idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    take = []
    last_exit = -1
    for k in np.flatnonzero(mask):
        if entry_idx[k] > last_exit:
            take.append(k)
            last_exit = exit_idx[k]
    return np.asarray(take, dtype=np.int64)


def max_loss_streak(pnl: np.ndarray) -> int:
    if len(pnl) == 0:
        return 0
    loss = pnl <= 0
    if not loss.any():
        return 0
    padded = np.concatenate(([False], loss, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def calc_stats(pnl: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    if len(pnl) == 0:
        return {"取引回数": 0, "1日平均回数": 0.0, "勝率%": 0.0, "総損益pips": 0.0,
                "平均利益pips": 0.0, "平均損失pips": 0.0, "PF": 0.0,
                "最大DD_pips": 0.0, "最大連敗": 0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_w = float(wins.sum())
    gross_l = float(-losses.sum())
    cum = np.cumsum(pnl)
    dd = float(np.min(cum - np.maximum.accumulate(cum)))
    n_days = max(1, len(np.unique(ts // (86400 * 10**9))))
    return {
        "取引回数": int(len(pnl)),
        "1日平均回数": round(float(len(pnl) / n_days), 1),
        "勝率%": round(float(len(wins) / len(pnl) * 100), 1),
        "総損益pips": round(float(pnl.sum()), 1),
        "平均利益pips": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "平均損失pips": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "PF": round(gross_w / gross_l, 3) if gross_l > 0 else 999.0,
        "最大DD_pips": round(dd, 1),
        "最大連敗": max_loss_streak(pnl),
    }


def add_score_and_verdict(rank: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    r = rank.copy()
    pf_test = r["検証_PF"].clip(upper=cfg.pf_cap)
    pf_train = r["学習_PF"].clip(upper=cfg.pf_cap)
    r["乖離率(検証PF/学習PF)"] = (pf_test / pf_train.replace(0, np.nan)).clip(upper=1.5).fillna(0).round(3)

    def pct(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, method="average")

    score = (
        0.25 * pct(r["検証_総損益pips"])
        + 0.25 * pct(pf_test)
        + 0.15 * pct(r["検証_最大DD_pips"])
        + 0.10 * pct(-r["検証_最大連敗"])
        + 0.10 * pct(r["検証_取引回数"])
        + 0.15 * pct(r["乖離率(検証PF/学習PF)"])
    ) * 100

    enough = (r["学習_取引回数"] >= cfg.min_trades_train) & (r["検証_取引回数"] >= cfg.min_trades_test)
    r["総合スコア"] = np.where(enough, score, score * 0.3).round(1)

    def judge(row):
        tr_pf = row["学習_PF"]
        te_pf = row["検証_PF"]
        te_total = row["検証_総損益pips"]
        n_ok = row["学習_取引回数"] >= cfg.min_trades_train and row["検証_取引回数"] >= cfg.min_trades_test
        if not n_ok:
            return "要注意(取引数不足)" if te_pf >= 1.2 and te_total > 0 else "不採用"
        if tr_pf >= 1.15 and (te_pf < 1.0 or te_total <= 0):
            return "過剰最適化疑い"
        if te_pf >= 1.15 and te_total > 0 and tr_pf >= 1.05 and te_pf >= tr_pf * 0.70:
            return "採用候補"
        if te_pf >= 1.0 and te_total > 0:
            return "要注意"
        return "不採用"

    r["判定"] = r.apply(judge, axis=1)
    return r


# -----------------------------------------------------
# 実行
# -----------------------------------------------------
def run(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_data(cfg)
    ind = IndicatorCache(df, cfg)
    df = ind.prepare_base_env()
    spread = build_spread_array(df, cfg)

    train_s = pd.Timestamp(cfg.train_start).value
    train_e = (pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1)).value
    test_s = pd.Timestamp(cfg.test_start).value
    test_e = (pd.Timestamp(cfg.test_end) + pd.Timedelta(days=1)).value

    patterns = generate_patterns(cfg)
    print(f"自動生成パターン数: {len(patterns):,}通り")
    print(f"学習: {cfg.train_start}～{cfg.train_end} / 検証: {cfg.test_start}～{cfg.test_end}")
    print(f"スプレッド: 東京{cfg.spread_tokyo}銭 / ロンドン{cfg.spread_london}銭 / NY{cfg.spread_ny}銭 / その他{cfg.spread_other}銭")
    print()

    rows = []
    signal_cache: Dict[Tuple, Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
    outcome_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for i, pat in enumerate(patterns, 1):
        sig_key, tp, sl, time_f, weekday_f, trend_f, vol_f = pat

        if sig_key not in signal_cache:
            sig = build_signal(sig_key, ind)
            ctx = build_candidates(df, sig)
            signal_cache[sig_key] = (ctx, ctx["entry_idx"], ctx["ts"])
        ctx, entry_idx, ts_arr = signal_cache[sig_key]

        out_key = (sig_key, tp, sl)
        if out_key not in outcome_cache:
            outcome_cache[out_key] = compute_outcomes(df, ctx, tp, sl, spread)
        exit_idx, pnl, reason = outcome_cache[out_key]

        m = filter_mask(ctx, time_f, weekday_f, trend_f, vol_f)
        selected = select_exclusive(entry_idx, exit_idx, m)
        train_sel = selected[(ts_arr[selected] >= train_s) & (ts_arr[selected] < train_e)]
        test_sel = selected[(ts_arr[selected] >= test_s) & (ts_arr[selected] < test_e)]

        row = {
            "戦略": sig_key_to_text(sig_key),
            "戦略タイプ": sig_key[0],
            "利確pips": tp,
            "損切りpips": sl,
            "時間帯": time_f,
            "曜日": weekday_f,
            "トレンド環境": trend_f,
            "ボラ環境": vol_f,
        }
        row.update({f"学習_{k}": v for k, v in calc_stats(pnl[train_sel], ts_arr[train_sel]).items()})
        row.update({f"検証_{k}": v for k, v in calc_stats(pnl[test_sel], ts_arr[test_sel]).items()})
        rows.append(row)

        if i % 1000 == 0 or i == len(patterns):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(patterns) - i) if i else 0
            print(f"  進捗 {i:,}/{len(patterns):,} 経過{elapsed:.0f}秒 残り約{eta:.0f}秒")

    rank = pd.DataFrame(rows)
    rank = add_score_and_verdict(rank, cfg).sort_values("総合スコア", ascending=False).reset_index(drop=True)

    adopted = rank[rank["判定"] == "採用候補"]
    overfit = rank[rank["判定"] == "過剰最適化疑い"].sort_values("学習_PF", ascending=False).head(20)

    rank.to_csv(cfg.out_dir / "research_ranking.csv", index=False, encoding="utf-8-sig")
    adopted.to_csv(cfg.out_dir / "adopted_candidates.csv", index=False, encoding="utf-8-sig")
    overfit.to_csv(cfg.out_dir / "overfit_examples.csv", index=False, encoding="utf-8-sig")

    report = build_report(rank, adopted, overfit, cfg, time.time() - t0)
    (cfg.out_dir / "research_report.txt").write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print(f"全ランキング : {cfg.out_dir / 'research_ranking.csv'}")
    print(f"採用候補     : {cfg.out_dir / 'adopted_candidates.csv'}")
    print(f"過剰最適化例 : {cfg.out_dir / 'overfit_examples.csv'}")
    print(f"研究レポート : {cfg.out_dir / 'research_report.txt'}")


def summarize_dimension(rank: pd.DataFrame, col: str, min_count: int = 20) -> pd.DataFrame:
    r = rank[rank["検証_取引回数"] >= 100].copy()
    if r.empty:
        return pd.DataFrame()
    r["pf_c"] = r["検証_PF"].clip(upper=10)
    g = r.groupby(col).agg(
        件数=("pf_c", "size"),
        平均検証PF=("pf_c", "mean"),
        検証プラス率=("検証_総損益pips", lambda s: (s > 0).mean() * 100),
        採用率=("判定", lambda s: (s == "採用候補").mean() * 100),
    ).round(3)
    return g[g["件数"] >= min_count].sort_values("平均検証PF", ascending=False)


def build_report(rank: pd.DataFrame, adopted: pd.DataFrame, overfit: pd.DataFrame, cfg: Config, elapsed: float) -> str:
    L: List[str] = []
    L.append("=" * 62)
    L.append("岡畠AI研究所 条件研究レポート Ver.0.4 ChatGPT版")
    L.append("=" * 62)
    L.append(f"検証パターン数 : {len(rank):,}通り")
    L.append(f"実行時間       : {elapsed:.0f}秒")
    L.append(f"学習期間       : {cfg.train_start} ～ {cfg.train_end}")
    L.append(f"検証期間       : {cfg.test_start} ～ {cfg.test_end}")
    L.append(f"スプレッド     : 東京{cfg.spread_tokyo}銭 / ロンドン{cfg.spread_london}銭 / NY{cfg.spread_ny}銭 / その他{cfg.spread_other}銭")
    L.append("")
    L.append("判定の内訳:")
    for k, v in rank["判定"].value_counts().items():
        L.append(f"  {k}: {v:,}件")

    L.append("")
    L.append("=" * 62)
    if adopted.empty:
        L.append("採用候補: なし")
        L.append("今回の条件空間では、検証期間まで安定して利益が残る条件は見つかりませんでした。")
        L.append("これは失敗ではなく、現時点の戦略群では優位性が弱いという研究結果です。")
    else:
        L.append(f"採用候補: あり（{len(adopted):,}件）")
        for i, (_, r) in enumerate(adopted.head(cfg.top_n_report).iterrows(), 1):
            L.append(f"{i}. {r['戦略']} TP{r['利確pips']} SL{r['損切りpips']} "
                     f"{r['時間帯']} {r['曜日']} {r['トレンド環境']} {r['ボラ環境']} "
                     f"検証PF{r['検証_PF']} 検証損益{r['検証_総損益pips']}pips "
                     f"取引{r['検証_取引回数']}回 スコア{r['総合スコア']}")

    L.append("")
    L.append("上位10件（採用候補に限らない）:")
    for i, (_, r) in enumerate(rank.head(10).iterrows(), 1):
        L.append(f"{i}. [{r['判定']}] {r['戦略']} TP{r['利確pips']} SL{r['損切りpips']} "
                 f"{r['時間帯']} {r['曜日']} {r['トレンド環境']} {r['ボラ環境']} "
                 f"検証PF{r['検証_PF']} 検証損益{r['検証_総損益pips']}pips "
                 f"取引{r['検証_取引回数']}回 スコア{r['総合スコア']}")

    L.append("")
    L.append("条件別の強弱（平均検証PF順・取引数不足除外）:")
    for col in ["戦略タイプ", "時間帯", "曜日", "トレンド環境", "ボラ環境", "利確pips", "損切りpips"]:
        tbl = summarize_dimension(rank, col)
        if tbl.empty:
            continue
        L.append(f"\n[{col}]")
        L.append(tbl.head(8).to_string())

    L.append("")
    L.append("過剰最適化の例:")
    if overfit.empty:
        L.append("  該当なし")
    else:
        for _, r in overfit.head(10).iterrows():
            L.append(f"  {r['戦略']} TP{r['利確pips']} SL{r['損切りpips']} "
                     f"学習PF{r['学習_PF']} → 検証PF{r['検証_PF']} "
                     f"検証損益{r['検証_総損益pips']}pips")

    L.append("")
    L.append("次の改善案:")
    L.append("  1. 採用候補が出ない場合は、EMA/RSI/MACDだけでなくブレイクアウトや高安値更新系を追加する。")
    L.append("  2. 固定TP/SLだけでなくATR倍率による利確・損切りを追加する。")
    L.append("  3. 指標発表時間の除外、東京仲値、ロンドン初動など時間構造を追加する。")
    L.append("  4. 上位条件が一点だけ良い場合は採用せず、似た条件がまとまって強いか確認する。")
    L.append("")
    L.append("注意: これは研究用であり、自動売買や利益を保証するものではありません。")
    return "\n".join(L)


if __name__ == "__main__":
    run(Config())
