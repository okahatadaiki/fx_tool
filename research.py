# =====================================================
# research.py (Ver.0.3)
# 岡畠AI研究所: USD/JPY 5分足 研究モード
#
# 目的:
# - 自動売買ではなく、売買条件を大量生成して検証する
# - 既存の main.py / fetcher.py / quality.py / backtest.py は変更しない
# - data/USDJPY_5min_2021-2026.csv を読み込み、研究結果を data/research/ に出力する
#
# 検証の約束:
# - シグナルは足の確定時点で判定
# - エントリーは次の足の始値
# - 未来データは見ない
# - 同一足で利確・損切りの両方に届いたら損切り優先
# - 日をまたがず、その日の最終足で強制決済
# - Bid価格データなので、買いはAsk約定、売り決済もAsk基準
# - スプレッドは時間帯別に設定
# =====================================================

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import math
import numpy as np
import pandas as pd

PIP = 0.01  # USD/JPY: 1pip = 0.01円 = 1銭
REASON_TP = "利確"
REASON_SL = "損切り"
REASON_EOD = "日終わり強制決済"
WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


@dataclass
class ResearchConfig:
    csv_path: Path = Path(__file__).parent / "data" / "USDJPY_5min_2021-2026.csv"
    out_dir: Path = Path(__file__).parent / "data" / "research"

    train_start: str = "2021-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-06-30"

    # 多すぎると初心者環境で時間がかかるため、まずは5000件に制限
    # 増やす場合は 10000 などに変更
    max_patterns: int = 5000

    # 採用候補の最低条件
    min_test_trades: int = 100
    min_train_trades: int = 300
    min_test_pf: float = 1.15
    min_test_total_pips: float = 0.0
    max_pf_divergence: float = 0.70  # 検証PFが学習PFの70%以上なら再現性あり

    # 日跨ぎ防止。23時以降は新規エントリーしない
    entry_cutoff_jst: str = "23:00"

    # 時間帯別スプレッド（単位: 銭 = pips）
    spread_tokyo: float = 0.2     # 9〜15時
    spread_london: float = 0.3    # 16〜21時
    spread_ny: float = 0.4        # 21〜24時
    spread_other: float = 0.5

    # 研究するパラメータ候補
    ema_fast_list: tuple = (5, 8, 10, 12, 15, 20)
    ema_slow_list: tuple = (20, 30, 50, 75, 100, 200)
    rsi_low_list: tuple = (20, 25, 30, 35, 40)
    rsi_high_list: tuple = (60, 65, 70, 75, 80)
    tp_list: tuple = (5, 8, 10, 12, 15, 20, 25)
    sl_list: tuple = (5, 8, 10, 12, 15, 20)
    time_filters: tuple = ("全時間", "東京(9-15)", "ロンドン(16-21)", "NY(21-24)", "9時台", "10時台", "16時台", "21時台", "22時台")
    weekday_filters: tuple = ("全曜日", "月", "火", "水", "木", "金")
    trend_filters: tuple = ("なし", "上昇のみ", "下降のみ", "レンジのみ", "順方向のみ")
    vol_filters: tuple = ("なし", "ATR高", "ATR低", "直近30分変動大")


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
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


def load_data(cfg: ResearchConfig) -> pd.DataFrame:
    if not cfg.csv_path.exists():
        raise FileNotFoundError(
            f"{cfg.csv_path} が見つかりません。先に python main.py を実行してCSVを作ってください。"
        )
    df = pd.read_csv(cfg.csv_path, parse_dates=["datetime_utc", "datetime_jst"])
    df = df.set_index("datetime_jst").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # 日終わり判定
    df["date_jst"] = df.index.date
    df["is_eod"] = df["date_jst"] != pd.Series(df["date_jst"], index=df.index).shift(-1)
    df.iloc[-1, df.columns.get_loc("is_eod")] = True

    cutoff_h, cutoff_m = map(int, cfg.entry_cutoff_jst.split(":"))
    minutes = df.index.hour * 60 + df.index.minute
    df["can_enter"] = minutes < (cutoff_h * 60 + cutoff_m)
    print(f"データ読み込み: {len(df):,}本 ({df.index.min()} 〜 {df.index.max()} JST)")
    return df


def add_base_indicators(df: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    c = df["close"]
    # EMA候補を全部作る
    all_emas = sorted(set(cfg.ema_fast_list + cfg.ema_slow_list + (20, 50, 200)))
    for n in all_emas:
        df[f"ema{n}"] = ema(c, n)
    df["rsi14"] = rsi(c, 14)
    df["macd"], df["macd_sig"] = macd(c)
    df["atr14"] = atr(df, 14)

    df["trend_up"] = (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    df["trend_dn"] = (df["ema20"] < df["ema50"]) & (df["ema50"] < df["ema200"])
    df["trend_rng"] = ~df["trend_up"] & ~df["trend_dn"]

    atr_med = df["atr14"].rolling(5760, min_periods=1000).median()  # 約20営業日
    df["vol_high"] = df["atr14"] > atr_med * 1.2
    df["vol_low"] = df["atr14"] < atr_med * 0.8
    df["vol_move30"] = (df["close"] - df["close"].shift(6)).abs() >= 8 * PIP
    return df


def spread_pips_by_hour(hour: int, cfg: ResearchConfig) -> float:
    if 9 <= hour < 15:
        return cfg.spread_tokyo
    if 16 <= hour < 21:
        return cfg.spread_london
    if 21 <= hour < 24:
        return cfg.spread_ny
    return cfg.spread_other


def make_time_mask(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    h = index.hour
    if name == "全時間": return np.ones(len(index), dtype=bool)
    if name == "東京(9-15)": return (h >= 9) & (h < 15)
    if name == "ロンドン(16-21)": return (h >= 16) & (h < 21)
    if name == "NY(21-24)": return (h >= 21) & (h < 24)
    if name == "9時台": return h == 9
    if name == "10時台": return h == 10
    if name == "16時台": return h == 16
    if name == "21時台": return h == 21
    if name == "22時台": return h == 22
    raise ValueError(f"未知の時間帯フィルター: {name}")


def make_weekday_mask(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    if name == "全曜日": return np.ones(len(index), dtype=bool)
    return index.weekday == WEEKDAYS.index(name)


def make_vol_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "なし": return np.ones(len(df), dtype=bool)
    if name == "ATR高": return df["vol_high"].to_numpy()
    if name == "ATR低": return df["vol_low"].to_numpy()
    if name == "直近30分変動大": return df["vol_move30"].to_numpy()
    raise ValueError(f"未知のボラフィルター: {name}")


def generate_signal(df: pd.DataFrame, rule: dict) -> pd.Series:
    """足確定時点のシグナル。+1買い / -1売り / 0見送り"""
    kind = rule["kind"]
    if kind == "ema_cross":
        f, s = df[f"ema{rule['fast']}"], df[f"ema{rule['slow']}"]
        up = (f > s) & (f.shift(1) <= s.shift(1))
        dn = (f < s) & (f.shift(1) >= s.shift(1))
    elif kind == "ema_trend":
        f, s = df[f"ema{rule['fast']}"], df[f"ema{rule['slow']}"]
        up = (f > s) & (df["close"] > f) & (df["close"].shift(1) <= f.shift(1))
        dn = (f < s) & (df["close"] < f) & (df["close"].shift(1) >= f.shift(1))
    elif kind == "rsi_reversal":
        r = df["rsi14"]
        up = (r.shift(1) < rule["low"]) & (r >= rule["low"])
        dn = (r.shift(1) > rule["high"]) & (r <= rule["high"])
    elif kind == "macd_cross":
        m, s = df["macd"], df["macd_sig"]
        up = (m > s) & (m.shift(1) <= s.shift(1))
        dn = (m < s) & (m.shift(1) >= s.shift(1))
    elif kind == "rsi_trend_pullback":
        r = df["rsi14"]
        up = df["trend_up"] & (r.shift(1) < rule["low"]) & (r >= rule["low"])
        dn = df["trend_dn"] & (r.shift(1) > rule["high"]) & (r <= rule["high"])
    else:
        raise ValueError(f"未知の戦略: {kind}")
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def trend_mask_for_entries(df: pd.DataFrame, entry_idx: np.ndarray, sig_idx: np.ndarray, side: np.ndarray, name: str) -> np.ndarray:
    if name == "なし": return np.ones(len(entry_idx), dtype=bool)
    if name == "上昇のみ": return df["trend_up"].to_numpy()[sig_idx]
    if name == "下降のみ": return df["trend_dn"].to_numpy()[sig_idx]
    if name == "レンジのみ": return df["trend_rng"].to_numpy()[sig_idx]
    if name == "順方向のみ":
        up = df["trend_up"].to_numpy()[sig_idx]
        dn = df["trend_dn"].to_numpy()[sig_idx]
        return ((side > 0) & up) | ((side < 0) & dn)
    raise ValueError(f"未知のトレンドフィルター: {name}")


def simulate_rule(df: pd.DataFrame, rule: dict, cfg: ResearchConfig) -> pd.DataFrame:
    sig = generate_signal(df, rule).to_numpy()
    can_enter = df["can_enter"].to_numpy()
    is_eod = df["is_eod"].to_numpy()

    # エントリーはシグナルの次足
    entry_idx = np.arange(1, len(df))
    sig_idx = entry_idx - 1
    side = sig[sig_idx]
    base_ok = (side != 0) & can_enter[entry_idx] & ~is_eod[entry_idx]

    # エントリー時刻で見るフィルター
    time_ok = make_time_mask(df.index, rule["time_filter"])[entry_idx]
    wd_ok = make_weekday_mask(df.index, rule["weekday_filter"])[entry_idx]
    vol_ok = make_vol_mask(df, rule["vol_filter"])[sig_idx]
    trend_ok = trend_mask_for_entries(df, entry_idx, sig_idx, side, rule["trend_filter"])

    ok = base_ok & time_ok & wd_ok & vol_ok & trend_ok
    candidates = entry_idx[ok]
    sides = side[ok].astype(int)

    if len(candidates) == 0:
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "pnl_pips", "exit_reason"])

    open_a = df["open"].to_numpy()
    high_a = df["high"].to_numpy()
    low_a = df["low"].to_numpy()
    close_a = df["close"].to_numpy()
    times = df.index

    trades = []
    last_exit = -1
    tp_v = rule["tp"] * PIP
    sl_v = rule["sl"] * PIP

    for e, sd in zip(candidates, sides):
        # ポジション重複を避ける
        if e <= last_exit:
            continue

        sp = spread_pips_by_hour(times[e].hour, cfg) * PIP
        if sd > 0:
            entry = open_a[e] + sp  # 買いはAskで建てる
            tp_level = entry + tp_v
            sl_level = entry - sl_v
            i = e
            while True:
                # 同一足なら損切り優先
                if low_a[i] <= sl_level:
                    pnl, reason = -rule["sl"], REASON_SL
                    break
                if high_a[i] >= tp_level:
                    pnl, reason = rule["tp"], REASON_TP
                    break
                if is_eod[i]:
                    pnl, reason = (close_a[i] - entry) / PIP, REASON_EOD
                    break
                i += 1
            side_name = "買い"
        else:
            entry = open_a[e]  # 売りはBidで建てる
            # 買い戻しはAskなので、Bid換算の判定水準にする
            tp_level_bid = entry - tp_v - sp
            sl_level_bid = entry + sl_v - sp
            i = e
            while True:
                if high_a[i] >= sl_level_bid:
                    pnl, reason = -rule["sl"], REASON_SL
                    break
                if low_a[i] <= tp_level_bid:
                    pnl, reason = rule["tp"], REASON_TP
                    break
                if is_eod[i]:
                    pnl, reason = (entry - (close_a[i] + sp)) / PIP, REASON_EOD
                    break
                i += 1
            side_name = "売り"

        trades.append({
            "entry_time": times[e], "exit_time": times[i], "side": side_name,
            "pnl_pips": round(float(pnl), 2), "exit_reason": reason,
        })
        last_exit = i

    return pd.DataFrame(trades)


def calc_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"取引回数": 0, "勝率%": 0.0, "総損益pips": 0.0, "平均利益pips": 0.0,
                "平均損失pips": 0.0, "PF": 0.0, "最大DD_pips": 0.0, "最大連敗": 0}
    pnl = trades["pnl_pips"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_w = wins.sum()
    gross_l = -losses.sum()
    cum = np.cumsum(pnl)
    dd = float(np.min(cum - np.maximum.accumulate(cum)))

    # 最大連敗
    loss = pnl <= 0
    max_streak = 0
    cur = 0
    for x in loss:
        if x:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    return {
        "取引回数": int(len(pnl)),
        "勝率%": round(float((pnl > 0).mean() * 100), 1),
        "総損益pips": round(float(pnl.sum()), 1),
        "平均利益pips": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "平均損失pips": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "PF": round(float(gross_w / gross_l), 3) if gross_l > 0 else 999.0,
        "最大DD_pips": round(dd, 1),
        "最大連敗": int(max_streak),
    }


def split_trades(trades: pd.DataFrame, cfg: ResearchConfig):
    if trades.empty:
        return trades, trades
    train_end = pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1)
    test_end = pd.Timestamp(cfg.test_end) + pd.Timedelta(days=1)
    train = trades[(trades["entry_time"] >= cfg.train_start) & (trades["entry_time"] < train_end)]
    test = trades[(trades["entry_time"] >= cfg.test_start) & (trades["entry_time"] < test_end)]
    return train, test


def generate_rules(cfg: ResearchConfig):
    rules = []
    # EMA系
    for fast, slow in product(cfg.ema_fast_list, cfg.ema_slow_list):
        if fast >= slow:
            continue
        for kind in ("ema_cross", "ema_trend"):
            for tp, sl, tf, wf, trf, vf in product(cfg.tp_list, cfg.sl_list, cfg.time_filters, cfg.weekday_filters, cfg.trend_filters, cfg.vol_filters):
                rules.append({"kind": kind, "fast": fast, "slow": slow, "tp": tp, "sl": sl,
                              "time_filter": tf, "weekday_filter": wf, "trend_filter": trf, "vol_filter": vf})
                if len(rules) >= cfg.max_patterns: return rules
    # RSI系
    for low, high in product(cfg.rsi_low_list, cfg.rsi_high_list):
        if low >= high:
            continue
        for kind in ("rsi_reversal", "rsi_trend_pullback"):
            for tp, sl, tf, wf, trf, vf in product(cfg.tp_list, cfg.sl_list, cfg.time_filters, cfg.weekday_filters, cfg.trend_filters, cfg.vol_filters):
                rules.append({"kind": kind, "low": low, "high": high, "tp": tp, "sl": sl,
                              "time_filter": tf, "weekday_filter": wf, "trend_filter": trf, "vol_filter": vf})
                if len(rules) >= cfg.max_patterns: return rules
    # MACD系
    for tp, sl, tf, wf, trf, vf in product(cfg.tp_list, cfg.sl_list, cfg.time_filters, cfg.weekday_filters, cfg.trend_filters, cfg.vol_filters):
        rules.append({"kind": "macd_cross", "tp": tp, "sl": sl,
                      "time_filter": tf, "weekday_filter": wf, "trend_filter": trf, "vol_filter": vf})
        if len(rules) >= cfg.max_patterns: return rules
    return rules


def rule_name(rule: dict) -> str:
    if rule["kind"] in ("ema_cross", "ema_trend"):
        base = f"{rule['kind']}(EMA{rule['fast']}/{rule['slow']})"
    elif rule["kind"] in ("rsi_reversal", "rsi_trend_pullback"):
        base = f"{rule['kind']}(RSI{rule['low']}/{rule['high']})"
    else:
        base = "macd_cross(12/26/9)"
    return base


def judge(row, cfg: ResearchConfig) -> str:
    train_pf = row["学習_PF"]
    test_pf = row["検証_PF"]
    test_total = row["検証_総損益pips"]
    enough = row["学習_取引回数"] >= cfg.min_train_trades and row["検証_取引回数"] >= cfg.min_test_trades
    if not enough:
        return "要注意(取引数不足)" if test_pf >= 1.15 and test_total > 0 else "不採用"
    if train_pf >= 1.2 and (test_pf < 1.0 or test_total <= 0):
        return "過剰最適化疑い"
    if (test_total > cfg.min_test_total_pips and test_pf >= cfg.min_test_pf
            and train_pf >= 1.05 and test_pf >= train_pf * cfg.max_pf_divergence):
        return "採用候補"
    if test_total > 0 and test_pf >= 1.0:
        return "要注意"
    return "不採用"


def score_row(row, cfg: ResearchConfig) -> float:
    # 簡易スコア。ランキング後に相対評価ではなく単独で見ても意味があるようにする。
    test_total = row["検証_総損益pips"]
    test_pf = min(row["検証_PF"], 5.0)
    train_pf = max(row["学習_PF"], 0.01)
    dd_penalty = abs(row["検証_最大DD_pips"])
    streak_penalty = row["検証_最大連敗"]
    trades = row["検証_取引回数"]
    divergence = min(row["検証_PF"] / train_pf, 1.2)
    score = 0
    score += max(test_total, -1000) * 0.03
    score += test_pf * 25
    score += divergence * 20
    score += min(trades / 100, 5) * 5
    score -= dd_penalty * 0.02
    score -= streak_penalty * 1.5
    return round(float(score), 1)


def analyze_conditions(rank: pd.DataFrame) -> tuple[str, str]:
    if rank.empty:
        return "分析不可", "分析不可"
    candidates = rank[rank["判定"].isin(["採用候補", "要注意"])]
    target = candidates if not candidates.empty else rank.head(100)

    strong = []
    weak = []
    for col in ["戦略種別", "時間帯", "曜日", "トレンド条件", "ボラ条件", "利確pips", "損切りpips"]:
        tbl = target.groupby(col)["検証_総損益pips"].mean().sort_values(ascending=False)
        if not tbl.empty:
            strong.append(f"{col}: {tbl.index[0]} が比較的強い（平均 {tbl.iloc[0]:.1f}pips）")
            weak.append(f"{col}: {tbl.index[-1]} が弱い（平均 {tbl.iloc[-1]:.1f}pips）")
    return "\n".join(strong), "\n".join(weak)


def build_report(rank: pd.DataFrame, adopted: pd.DataFrame, overfit: pd.DataFrame, cfg: ResearchConfig) -> str:
    strong, weak = analyze_conditions(rank)
    lines = []
    lines.append("=" * 70)
    lines.append("岡畠AI研究所 Ver.0.3 研究レポート")
    lines.append("=" * 70)
    lines.append(f"検証パターン数: {len(rank):,} 通り")
    lines.append(f"学習期間: {cfg.train_start}〜{cfg.train_end}")
    lines.append(f"検証期間: {cfg.test_start}〜{cfg.test_end}")
    lines.append("時間帯別スプレッド: 東京0.2銭 / ロンドン0.3銭 / NY0.4銭 / その他0.5銭")
    lines.append("")
    lines.append("判定内訳:")
    for k, v in rank["判定"].value_counts().items():
        lines.append(f"  {k}: {v:,}件")

    lines.append("\n採用候補 上位10件:")
    if adopted.empty:
        lines.append("  採用候補なし。これは失敗ではなく、現在の条件群に安定優位性が弱いという研究結果です。")
    else:
        for i, (_, r) in enumerate(adopted.head(10).iterrows(), 1):
            lines.append(f"  {i}. {r['戦略名']} TP{r['利確pips']} SL{r['損切りpips']} "
                         f"[{r['時間帯']}|{r['曜日']}|{r['トレンド条件']}|{r['ボラ条件']}] "
                         f"検証PF{r['検証_PF']} 総損益{r['検証_総損益pips']}pips 取引{r['検証_取引回数']}回")

    lines.append("\n過剰最適化の例:")
    if overfit.empty:
        lines.append("  該当なし。")
    else:
        for _, r in overfit.head(10).iterrows():
            lines.append(f"  ・{r['戦略名']} 学習PF{r['学習_PF']}→検証PF{r['検証_PF']} "
                         f"学習{r['学習_総損益pips']}pips→検証{r['検証_総損益pips']}pips")

    lines.append("\n強かった条件の傾向:")
    lines.append(strong)
    lines.append("\n弱かった条件の傾向:")
    lines.append(weak)

    lines.append("\n次に試すべき改善案:")
    lines.append("  1. 採用候補が少ない場合は、時間帯を絞るより『全時間＋トレンド順方向』を優先して確認する。")
    lines.append("  2. 取引数不足の良成績は信用しすぎず、最低100回以上の検証を維持する。")
    lines.append("  3. 採用候補が出たら、その条件だけで年別・月別のブレをさらに確認する。")
    lines.append("  4. 次版では経済指標前後の取引禁止フィルターを追加する。")
    lines.append("  5. 実運用ではなく、まだ研究段階。実資金投入はデモ検証後にする。")

    lines.append("\n注意点:")
    lines.append("  ・バックテストは未来の利益を保証しません。")
    lines.append("  ・スプレッドは時間帯別に入れていますが、指標時や急変時の滑りは未反映です。")
    lines.append("  ・採用候補は『すぐ使うルール』ではなく『次に深掘りする研究対象』です。")
    return "\n".join(lines)


def run_research(cfg: ResearchConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(cfg)
    df = add_base_indicators(df, cfg)
    rules = generate_rules(cfg)
    print(f"研究パターン生成: {len(rules):,}通り")

    rows = []
    for i, rule in enumerate(rules, 1):
        trades = simulate_rule(df, rule, cfg)
        train, test = split_trades(trades, cfg)
        train_stats = calc_stats(train)
        test_stats = calc_stats(test)
        row = {
            "戦略名": rule_name(rule),
            "戦略種別": rule["kind"],
            "利確pips": rule["tp"],
            "損切りpips": rule["sl"],
            "時間帯": rule["time_filter"],
            "曜日": rule["weekday_filter"],
            "トレンド条件": rule["trend_filter"],
            "ボラ条件": rule["vol_filter"],
        }
        if "fast" in rule:
            row["EMA短期"] = rule["fast"]
            row["EMA長期"] = rule["slow"]
        if "low" in rule:
            row["RSI下限"] = rule["low"]
            row["RSI上限"] = rule["high"]
        row.update({f"学習_{k}": v for k, v in train_stats.items()})
        row.update({f"検証_{k}": v for k, v in test_stats.items()})
        rows.append(row)
        if i % 250 == 0 or i == len(rules):
            print(f"  進捗 {i:,}/{len(rules):,}")

    rank = pd.DataFrame(rows)
    rank["判定"] = rank.apply(lambda r: judge(r, cfg), axis=1)
    rank["総合スコア"] = rank.apply(lambda r: score_row(r, cfg), axis=1)
    rank = rank.sort_values(["判定", "総合スコア"], ascending=[True, False]).reset_index(drop=True)
    # 採用候補を上に出したいので、判定順を明示
    order = {"採用候補": 0, "要注意": 1, "要注意(取引数不足)": 2, "過剰最適化疑い": 3, "不採用": 4}
    rank["_order"] = rank["判定"].map(order).fillna(9)
    rank = rank.sort_values(["_order", "総合スコア"], ascending=[True, False]).drop(columns=["_order"]).reset_index(drop=True)

    adopted = rank[rank["判定"] == "採用候補"]
    overfit = rank[rank["判定"] == "過剰最適化疑い"].sort_values("学習_PF", ascending=False)

    rank.to_csv(cfg.out_dir / "research_ranking.csv", index=False, encoding="utf-8-sig")
    adopted.to_csv(cfg.out_dir / "adopted_research_candidates.csv", index=False, encoding="utf-8-sig")
    overfit.to_csv(cfg.out_dir / "overfit_research_examples.csv", index=False, encoding="utf-8-sig")
    report = build_report(rank, adopted, overfit, cfg)
    (cfg.out_dir / "research_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print("\n出力先:")
    print(f"  {cfg.out_dir / 'research_ranking.csv'}")
    print(f"  {cfg.out_dir / 'adopted_research_candidates.csv'}")
    print(f"  {cfg.out_dir / 'overfit_research_examples.csv'}")
    print(f"  {cfg.out_dir / 'research_report.txt'}")


if __name__ == "__main__":
    run_research(ResearchConfig())
