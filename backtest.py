# =====================================================
# backtest.py  (Ver.0.2.1)
# USD/JPY 5分足 デイトレ条件研究ツール
#
# Ver.0.2からの変更点:
# - 検証パターンを大幅拡張（初期設定で5,400通り、設定変更で13万通り超まで）
# - 時間帯・曜日・ボラティリティ・トレンド環境のフィルターを追加
# - 総合スコアによるランキング（PF単独評価をやめた）
# - 「採用候補/要注意/過剰最適化疑い/不採用」の判定列を追加
# - 学習で良く検証で崩れた失敗例も出力
#
# 維持している検証精度のルール:
# - シグナルは足の確定時点で計算し、エントリーは次の足の始値
# - 買いはAsk(Bid+スプレッド)で建て、売りの決済もAsk基準で判定
#   → 全トレードに往復スプレッドが必ず乗る
# - 同一足で利確と損切り両方に届いたら損切り優先（保守的）
# - 日本時間の日の終わりで強制決済、持ち越しなし
# - 学習期間でランキング、検証期間で再評価
#
# 高速化の仕組み（結果は逐次処理と同一）:
# 従来は「1パターンごとに全40万本を走査」していたが、
# 本版は「戦略×利確×損切り」ごとにトレード候補の結末を1回だけ計算し、
# 時間帯・曜日等のフィルターは候補の取捨選択として適用する。
# フィルターはエントリー可否を決めるだけで約定結果を変えないため、
# この分解をしても結果は変わらない。
# =====================================================
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PIP = 0.01  # USD/JPYの1pips = 0.01円（1銭）

# 決済理由コード
R_TP, R_SL, R_EOD = 0, 1, 2
REASON_NAMES = {R_TP: "利確", R_SL: "損切り", R_EOD: "日終わり強制決済"}
WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


# -----------------------------------------------------
# 設定
# -----------------------------------------------------
@dataclass
class BTConfig:
    csv_path: Path = Path(__file__).parent / "data" / "USDJPY_5min_2021-2026.csv"
    out_dir: Path = Path(__file__).parent / "data" / "backtest"

    # スプレッド（銭 = pips）
    spread_pips: float = 0.2

    # 学習期間と検証期間（日本時間の日付）
    train_start: str = "2021-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-06-30"

    # 利確・損切りの探索候補（pips）
    tp_list: tuple = (5, 8, 10, 15, 20)
    sl_list: tuple = (5, 10, 15, 20)

    # --- フィルターの探索対象 ---
    # ここに入れた名前の全組み合わせを検証する。
    # 初期設定: 6戦略 × 5利確 × 4損切り × 9時間帯 × 5トレンド = 5,400通り
    # 曜日やボラ条件も総当たりしたい場合は下のリストに追加する。例:
    #   weekday_filters を ("全曜日","月","火","水","木","金") に → 32,400通り
    #   さらに vol_filters に "ATR高","ATR低","直近30分変動大" を追加 → 129,600通り
    # 処理時間は組み合わせ数にほぼ比例する（目安: 5,400通りで2〜5分）
    time_filters: tuple = ("全時間", "東京(9-15時)", "ロンドン(16-21時)", "NY(21-23時)",
                           "9時台", "10時台", "16時台", "21時台", "22時台")
    weekday_filters: tuple = ("全曜日",)
    vol_filters: tuple = ("なし",)
    trend_filters: tuple = ("なし", "上昇トレンドのみ", "下降トレンドのみ",
                            "レンジのみ", "トレンド順方向のみ")

    # ボラ条件のパラメータ
    vol_move30_pips: float = 8.0   # 「直近30分変動大」の基準pips
    atr_roll_bars: int = 5760      # ATR高低の基準となる移動中央値の窓（約20営業日）

    # 1日の終わり関連（日本時間）
    entry_cutoff_jst: str = "23:00"  # これ以降は新規エントリーしない

    # 信頼できる最低取引数
    min_trades_train: int = 300
    min_trades_test: int = 100

    # 総合スコアの重み（合計1.0）
    w_total: float = 0.25    # 検証期間の総損益
    w_pf: float = 0.25       # 検証期間のPF
    w_dd: float = 0.15       # 最大DDの小ささ
    w_streak: float = 0.10   # 最大連敗の少なさ
    w_trades: float = 0.10   # 取引回数（多いほど統計的に信頼できる）
    w_diverge: float = 0.15  # 学習と検証の成績乖離の小ささ

    pf_cap: float = 10.0     # スコア計算時のPF上限（極端値の影響を抑える）

    # 出力件数
    detail_top_n: int = 3         # 採用候補上位の詳細分析
    overfit_examples_n: int = 10  # 過剰最適化の失敗例


# -----------------------------------------------------
# 1. データ読み込み
# -----------------------------------------------------
def load_data(cfg: BTConfig) -> pd.DataFrame:
    if not cfg.csv_path.exists():
        raise FileNotFoundError(
            f"{cfg.csv_path} が見つかりません。先に main.py（Ver.0.1）を実行してください。")
    df = pd.read_csv(cfg.csv_path, parse_dates=["datetime_utc", "datetime_jst"])
    df = df.set_index("datetime_jst").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df["date_jst"] = df.index.date
    df["is_eod"] = df["date_jst"] != np.roll(df["date_jst"], -1)
    df.iloc[-1, df.columns.get_loc("is_eod")] = True

    cutoff_h, cutoff_m = map(int, cfg.entry_cutoff_jst.split(":"))
    minutes = df.index.hour * 60 + df.index.minute
    df["can_enter"] = minutes < (cutoff_h * 60 + cutoff_m)

    print(f"データ読み込み: {len(df):,}本 "
          f"({df.index.min():%Y-%m-%d} 〜 {df.index.max():%Y-%m-%d} JST)")
    return df


# -----------------------------------------------------
# 2. テクニカル指標（すべて過去データのみで計算される）
# -----------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df, cfg: BTConfig):
    c = df["close"]
    for n in (5, 10, 20, 50, 200):
        df[f"ema{n}"] = ema(c, n)
    df["rsi14"] = rsi(c, 14)
    df["macd"], df["macd_sig"] = macd(c)
    df["atr14"] = atr(df, 14)

    # --- トレンド環境判定（EMA20/50/200の並び）---
    df["trend_up"] = (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    df["trend_dn"] = (df["ema20"] < df["ema50"]) & (df["ema50"] < df["ema200"])
    df["trend_rng"] = ~df["trend_up"] & ~df["trend_dn"]

    # --- ボラティリティ環境判定 ---
    # 「高い/低い」の基準は過去約20営業日のATR中央値。
    # rolling（過去方向のみの窓）なので未来のデータは一切使わない。
    atr_med = df["atr14"].rolling(cfg.atr_roll_bars, min_periods=1000).median()
    df["vol_high"] = df["atr14"] > atr_med * 1.2
    df["vol_low"] = df["atr14"] < atr_med * 0.8
    df["vol_move30"] = (c - c.shift(6)).abs() >= cfg.vol_move30_pips * PIP
    return df


# -----------------------------------------------------
# 3. 売買判定ロジック
#    +1(買い) / -1(売り) / 0(見送り)。判定は足の確定時点。
#    新しい戦略を追加したいときは、関数を書いて
#    build_strategies() の辞書に1行足すだけでよい。
# -----------------------------------------------------
def sig_ema_cross(df, fast, slow):
    f, s = df[f"ema{fast}"], df[f"ema{slow}"]
    up = (f > s) & (f.shift(1) <= s.shift(1))
    dn = (f < s) & (f.shift(1) >= s.shift(1))
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def sig_rsi_reversal(df, low_th, high_th):
    r = df["rsi14"]
    up = (r.shift(1) < low_th) & (r >= low_th)
    dn = (r.shift(1) > high_th) & (r <= high_th)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def sig_macd_cross(df):
    m, s = df["macd"], df["macd_sig"]
    up = (m > s) & (m.shift(1) <= s.shift(1))
    dn = (m < s) & (m.shift(1) >= s.shift(1))
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def sig_trend_pullback(df, rsi_buy, rsi_sell):
    above = df["close"] > df["ema200"]
    below = df["close"] < df["ema200"]
    r = df["rsi14"]
    up = above & (r.shift(1) < rsi_buy) & (r >= rsi_buy)
    dn = below & (r.shift(1) > rsi_sell) & (r <= rsi_sell)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def build_strategies(df) -> dict:
    return {
        "EMAクロス(5,20)":        sig_ema_cross(df, 5, 20),
        "EMAクロス(10,50)":       sig_ema_cross(df, 10, 50),
        "RSI逆張り(30,70)":       sig_rsi_reversal(df, 30, 70),
        "RSI逆張り(25,75)":       sig_rsi_reversal(df, 25, 75),
        "MACDクロス(12,26,9)":    sig_macd_cross(df),
        "トレンド押し目(EMA200+RSI40/60)": sig_trend_pullback(df, 40, 60),
    }


# -----------------------------------------------------
# 4. トレード候補の事前計算
# -----------------------------------------------------
def build_entry_candidates(df, sig):
    """
    戦略ごとに「エントリー候補」の位置と、その時点の環境情報を作る。
    候補 = シグナル確定の次の足（＝エントリー執行足）。
    環境情報（トレンド・ボラ）はシグナル足の確定値を使う（未来参照なし）。
    時間・曜日はエントリー足の時刻を使う（始値約定の瞬間に確定している情報）。
    """
    sig_arr = np.nan_to_num(sig.to_numpy())
    can_enter = df["can_enter"].to_numpy()
    is_eod = df["is_eod"].to_numpy()

    # エントリー執行足 e: 前の足でシグナルがあり、カットオフ前で、日の最終足でない
    e_all = np.arange(1, len(df))
    ok = (sig_arr[e_all - 1] != 0) & can_enter[e_all] & ~is_eod[e_all]
    entry_idx = e_all[ok]
    sig_idx = entry_idx - 1

    # pandasのバージョンによってはindexの内部単位がns以外になることがあるため、
    # ns単位に明示変換してから整数化する（学習/検証期間の判定はns前提）
    ts = df.index.as_unit("ns").asi8
    ctx = {
        "entry_idx": entry_idx,
        "side": sig_arr[sig_idx].astype(np.int8),
        "hour": df.index.hour.to_numpy()[entry_idx],
        "weekday": df.index.weekday.to_numpy()[entry_idx],
        "ts": ts[entry_idx],
        "trend_up": df["trend_up"].to_numpy()[sig_idx],
        "trend_dn": df["trend_dn"].to_numpy()[sig_idx],
        "trend_rng": df["trend_rng"].to_numpy()[sig_idx],
        "vol_high": df["vol_high"].to_numpy()[sig_idx],
        "vol_low": df["vol_low"].to_numpy()[sig_idx],
        "vol_move30": df["vol_move30"].to_numpy()[sig_idx],
    }
    return ctx


def compute_outcomes(df_arrays, ctx, tp, sl, spread_pips):
    """
    各エントリー候補について「もしここで建てたら」の結末を計算する。
    ロジックはVer.0.2のsimulate()と同一:
      買い: entry = Bid始値 + spread（Ask約定）
            損切り水準 entry-sl / 利確水準 entry+tp をBidのhigh/lowで判定
      売り: entry = Bid始値。決済はAsk(=Bid+spread)なのでBid換算水準で判定
      同一足で両方到達なら損切り優先。日の最終足の終値で強制決済。
    戻り値: exit_idx, pnl_pips, reason の配列
    """
    o, h, l, c, is_eod = df_arrays
    spread = spread_pips * PIP
    tp_v, sl_v = tp * PIP, sl * PIP

    n = len(ctx["entry_idx"])
    entry_arr = ctx["entry_idx"]
    side_arr = ctx["side"]
    exit_idx = np.empty(n, dtype=np.int64)
    pnl = np.empty(n, dtype=np.float64)
    reason = np.empty(n, dtype=np.int8)

    for k in range(n):
        e = entry_arr[k]
        if side_arr[k] > 0:
            entry = o[e] + spread
            sl_level = entry - sl_v
            tp_level = entry + tp_v
            i = e
            while True:
                if l[i] <= sl_level:
                    pnl[k], reason[k] = -sl, R_SL
                    break
                if h[i] >= tp_level:
                    pnl[k], reason[k] = tp, R_TP
                    break
                if is_eod[i]:
                    pnl[k], reason[k] = (c[i] - entry) / PIP, R_EOD
                    break
                i += 1
            exit_idx[k] = i
        else:
            entry = o[e]
            sl_level_bid = entry + sl_v - spread
            tp_level_bid = entry - tp_v - spread
            i = e
            while True:
                if h[i] >= sl_level_bid:
                    pnl[k], reason[k] = -sl, R_SL
                    break
                if l[i] <= tp_level_bid:
                    pnl[k], reason[k] = tp, R_TP
                    break
                if is_eod[i]:
                    pnl[k], reason[k] = (entry - (c[i] + spread)) / PIP, R_EOD
                    break
                i += 1
            exit_idx[k] = i

    return exit_idx, pnl, reason


def select_exclusive(entry_idx, exit_idx, pass_mask):
    """
    フィルターを通過した候補から、ポジション重複しないように順番に採用する。
    決済した足の次の足以降でないと新規エントリーできない
    （Ver.0.2の逐次シミュレーションと同じ挙動）。
    """
    take = []
    last_exit = -1
    for k in np.flatnonzero(pass_mask):
        if entry_idx[k] > last_exit:
            take.append(k)
            last_exit = exit_idx[k]
    return np.asarray(take, dtype=np.int64)


# -----------------------------------------------------
# 5. フィルター定義
#    追加したい条件はこの辞書に足すだけでよい
# -----------------------------------------------------
def make_filter_masks(ctx, cfg: BTConfig) -> dict:
    """候補ごとの通過可否(bool配列)をフィルター名→配列で返す"""
    hour = ctx["hour"]
    wd = ctx["weekday"]
    n = len(hour)
    all_true = np.ones(n, dtype=bool)

    time_masks = {
        "全時間": all_true,
        "東京(9-15時)": (hour >= 9) & (hour < 15),
        "ロンドン(16-21時)": (hour >= 16) & (hour < 21),
        "NY(21-23時)": (hour >= 21) & (hour < 23),
        "9時台": hour == 9, "10時台": hour == 10, "16時台": hour == 16,
        "21時台": hour == 21, "22時台": hour == 22,
    }
    weekday_masks = {"全曜日": all_true}
    for i, name in enumerate(WEEKDAYS[:5]):
        weekday_masks[name] = wd == i

    vol_masks = {
        "なし": all_true,
        "ATR高": ctx["vol_high"],
        "ATR低": ctx["vol_low"],
        "直近30分変動大": ctx["vol_move30"],
    }
    side = ctx["side"]
    trend_masks = {
        "なし": all_true,
        "上昇トレンドのみ": ctx["trend_up"],
        "下降トレンドのみ": ctx["trend_dn"],
        "レンジのみ": ctx["trend_rng"],
        # トレンド順方向のみ: 上昇中は買いシグナルだけ、下降中は売りシグナルだけ採用
        "トレンド順方向のみ": ((side > 0) & ctx["trend_up"]) | ((side < 0) & ctx["trend_dn"]),
    }
    return {"時間帯": time_masks, "曜日": weekday_masks,
            "ボラ": vol_masks, "トレンド": trend_masks}


# -----------------------------------------------------
# 6. 成績集計（numpyベースの高速版）
# -----------------------------------------------------
def _max_loss_streak(pnl):
    loss = pnl <= 0
    if not loss.any():
        return 0
    padded = np.concatenate(([False], loss, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def calc_stats(pnl: np.ndarray, ts: np.ndarray) -> dict:
    """pnl: pips配列 / ts: エントリー時刻(ns int64)"""
    if len(pnl) == 0:
        return {"取引回数": 0, "1日平均回数": 0.0, "勝率%": 0.0, "総損益pips": 0.0,
                "平均利益pips": 0.0, "平均損失pips": 0.0, "PF": 0.0,
                "最大DD_pips": 0.0, "最大連敗": 0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_w = wins.sum()
    gross_l = -losses.sum()
    cum = np.cumsum(pnl)
    dd = float(np.min(cum - np.maximum.accumulate(cum)))
    n_days = len(np.unique(ts // (86400 * 10**9)))
    return {
        "取引回数": len(pnl),
        "1日平均回数": round(len(pnl) / max(n_days, 1), 1),
        "勝率%": round(len(wins) / len(pnl) * 100, 1),
        "総損益pips": round(float(pnl.sum()), 1),
        "平均利益pips": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "平均損失pips": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "PF": round(gross_w / gross_l, 3) if gross_l > 0 else 999.0,
        "最大DD_pips": round(dd, 1),
        "最大連敗": _max_loss_streak(pnl),
    }


# -----------------------------------------------------
# 7. 総合スコアと判定
# -----------------------------------------------------
def add_score_and_verdict(rank: pd.DataFrame, cfg: BTConfig) -> pd.DataFrame:
    r = rank.copy()
    pf_test = r["検証_PF"].clip(upper=cfg.pf_cap)
    pf_train = r["学習_PF"].clip(upper=cfg.pf_cap)

    # 乖離率: 検証PF ÷ 学習PF。1に近い（または上回る）ほど再現性が高い
    r["乖離率(検証PF/学習PF)"] = (
        (pf_test / pf_train.replace(0, np.nan)).clip(upper=1.5).fillna(0).round(3))

    # 各指標を順位パーセンタイル(0〜1)に変換してから重み付けする。
    # pips・PF・回数という単位の違う指標を混ぜるための正規化。
    def pct(s):
        return s.rank(pct=True, method="average")

    score = (
        cfg.w_total * pct(r["検証_総損益pips"])
        + cfg.w_pf * pct(pf_test)
        + cfg.w_dd * pct(r["検証_最大DD_pips"])      # DDは負値。0に近いほど高順位
        + cfg.w_streak * pct(-r["検証_最大連敗"])
        + cfg.w_trades * pct(r["検証_取引回数"])
        + cfg.w_diverge * pct(r["乖離率(検証PF/学習PF)"])
    ) * 100

    # 取引数が足りないものは統計的に信用できないのでスコアを大きく減点
    enough = ((r["学習_取引回数"] >= cfg.min_trades_train)
              & (r["検証_取引回数"] >= cfg.min_trades_test))
    score = np.where(enough, score, score * 0.3)
    r["総合スコア"] = np.round(score, 1)

    # --- 判定 ---
    def judge(row):
        tr_pf, te_pf = row["学習_PF"], row["検証_PF"]
        te_total = row["検証_総損益pips"]
        n_ok = (row["学習_取引回数"] >= cfg.min_trades_train
                and row["検証_取引回数"] >= cfg.min_trades_test)
        if not n_ok:
            # 数が少ないのに成績が良く見えるものは判断保留扱い
            return "要注意(取引数不足)" if (te_pf >= 1.2 and te_total > 0) else "不採用"
        if tr_pf >= 1.15 and (te_pf < 1.0 or te_total <= 0):
            return "過剰最適化疑い"  # 学習では明確に良いのに検証で崩壊
        if (te_pf >= 1.1 and te_total > 0 and tr_pf >= 1.05
                and te_pf >= tr_pf * 0.7):
            return "採用候補"        # 両期間で利益が残り、乖離も許容内
        if te_pf >= 1.0 and te_total > 0:
            return "要注意"          # プラスだが優位性が弱い/乖離が大きい
        return "不採用"

    r["判定"] = r.apply(judge, axis=1)
    return r


# -----------------------------------------------------
# 8. 年別・月別・曜日別・時間帯別の内訳
# -----------------------------------------------------
def breakdown(trades: pd.DataFrame) -> dict:
    t = trades.copy()
    t["年"] = t["entry_time"].dt.year
    t["月"] = t["entry_time"].dt.month
    t["曜日"] = t["entry_time"].dt.weekday.map(lambda w: WEEKDAYS[w])
    t["時間帯"] = t["entry_time"].dt.hour

    def agg(g):
        pnl = g["pnl_pips"]
        gw = pnl[pnl > 0].sum()
        gl = -pnl[pnl <= 0].sum()
        return pd.Series({
            "取引回数": len(g),
            "総損益pips": round(pnl.sum(), 1),
            "勝率%": round((pnl > 0).mean() * 100, 1),
            "PF": round(gw / gl, 3) if gl > 0 else 999.0,
        })

    out = {}
    for key in ["年", "月", "曜日", "時間帯"]:
        tbl = t.groupby(key).apply(agg, include_groups=False)
        if key == "曜日":
            tbl = tbl.reindex([w for w in WEEKDAYS if w in tbl.index])
        out[key] = tbl
    return out


# -----------------------------------------------------
# 9. グリッド探索本体
# -----------------------------------------------------
def run_grid(cfg: BTConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(cfg)
    df = add_indicators(df, cfg)
    strategies = build_strategies(df)

    df_arrays = (df["open"].to_numpy(), df["high"].to_numpy(),
                 df["low"].to_numpy(), df["close"].to_numpy(),
                 df["is_eod"].to_numpy())
    times = df.index

    train_s = pd.Timestamp(cfg.train_start).value
    train_e = (pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1)).value
    test_s = pd.Timestamp(cfg.test_start).value
    test_e = (pd.Timestamp(cfg.test_end) + pd.Timedelta(days=1)).value

    filter_combos = list(product(cfg.time_filters, cfg.weekday_filters,
                                 cfg.vol_filters, cfg.trend_filters))
    n_base = len(strategies) * len(cfg.tp_list) * len(cfg.sl_list)
    n_total = n_base * len(filter_combos)
    print(f"検証パターン数: 戦略{len(strategies)} × 利確{len(cfg.tp_list)} × 損切り{len(cfg.sl_list)}"
          f" × フィルター{len(filter_combos)} = {n_total:,}通り")
    print(f"学習: {cfg.train_start}〜{cfg.train_end} / 検証: {cfg.test_start}〜{cfg.test_end}"
          f" / スプレッド: {cfg.spread_pips}銭")
    print()

    rows = []
    outcome_cache = {}   # 上位の詳細分析で再利用する
    ctx_cache = {}
    done = 0

    for name, sig in strategies.items():
        ctx = build_entry_candidates(df, sig)
        masks = make_filter_masks(ctx, cfg)
        ctx_cache[name] = (ctx, masks)
        in_train = (ctx["ts"] >= train_s) & (ctx["ts"] < train_e)
        in_test = (ctx["ts"] >= test_s) & (ctx["ts"] < test_e)

        for tp, sl in product(cfg.tp_list, cfg.sl_list):
            exit_idx, pnl, reason = compute_outcomes(df_arrays, ctx, tp, sl, cfg.spread_pips)
            outcome_cache[(name, tp, sl)] = (exit_idx, pnl, reason)

            for tf, wf, vf, trf in filter_combos:
                pass_mask = (masks["時間帯"][tf] & masks["曜日"][wf]
                             & masks["ボラ"][vf] & masks["トレンド"][trf])
                sel = select_exclusive(ctx["entry_idx"], exit_idx, pass_mask)

                sel_train = sel[in_train[sel]]
                sel_test = sel[in_test[sel]]

                row = {"戦略": name, "利確pips": tp, "損切りpips": sl,
                       "時間帯": tf, "曜日": wf, "ボラ条件": vf, "トレンド条件": trf}
                row.update({f"学習_{k}": v for k, v in
                            calc_stats(pnl[sel_train], ctx["ts"][sel_train]).items()})
                row.update({f"検証_{k}": v for k, v in
                            calc_stats(pnl[sel_test], ctx["ts"][sel_test]).items()})
                rows.append(row)
                done += 1

            if done % 2000 < len(filter_combos):
                print(f"  進捗 {done:,}/{n_total:,}")

    print(f"  進捗 {done:,}/{n_total:,} 完了")
    rank = pd.DataFrame(rows)
    rank = add_score_and_verdict(rank, cfg)
    rank = rank.sort_values("総合スコア", ascending=False).reset_index(drop=True)

    # --- 出力1: 全ランキング ---
    rank_path = cfg.out_dir / "ranking.csv"
    rank.to_csv(rank_path, index=False, encoding="utf-8-sig")

    # --- 出力2: 採用候補のみ ---
    adopted = rank[rank["判定"] == "採用候補"]
    adopted.to_csv(cfg.out_dir / "adopted_candidates.csv", index=False, encoding="utf-8-sig")

    # --- 出力3: 過剰最適化の失敗例（学習成績が良い順）---
    overfit = (rank[rank["判定"] == "過剰最適化疑い"]
               .sort_values("学習_PF", ascending=False)
               .head(cfg.overfit_examples_n))
    overfit.to_csv(cfg.out_dir / "overfit_examples.csv", index=False, encoding="utf-8-sig")

    # --- レポート作成 ---
    report = build_report(rank, adopted, overfit, cfg, ctx_cache, outcome_cache, times)
    (cfg.out_dir / "detail_report.txt").write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print(f"全ランキング : {rank_path}")
    print(f"採用候補     : {cfg.out_dir / 'adopted_candidates.csv'}")
    print(f"過剰最適化例 : {cfg.out_dir / 'overfit_examples.csv'}")
    print(f"詳細レポート : {cfg.out_dir / 'detail_report.txt'}")


def rebuild_trades(row, ctx_cache, outcome_cache, times, cfg, period=None):
    """ランキングの1行から実際のトレード一覧を復元する"""
    name = row["戦略"]
    ctx, masks = ctx_cache[name]
    exit_idx, pnl, reason = outcome_cache[(name, row["利確pips"], row["損切りpips"])]
    pass_mask = (masks["時間帯"][row["時間帯"]] & masks["曜日"][row["曜日"]]
                 & masks["ボラ"][row["ボラ条件"]] & masks["トレンド"][row["トレンド条件"]])
    sel = select_exclusive(ctx["entry_idx"], exit_idx, pass_mask)
    tr = pd.DataFrame({
        "entry_time": times[ctx["entry_idx"][sel]],
        "exit_time": times[exit_idx[sel]],
        "side": np.where(ctx["side"][sel] > 0, "買い", "売り"),
        "pnl_pips": np.round(pnl[sel], 2),
        "exit_reason": [REASON_NAMES[x] for x in reason[sel]],
    })
    if period == "test":
        end = pd.Timestamp(cfg.test_end) + pd.Timedelta(days=1)
        tr = tr[(tr["entry_time"] >= cfg.test_start) & (tr["entry_time"] < end)]
    return tr


def build_report(rank, adopted, overfit, cfg, ctx_cache, outcome_cache, times) -> str:
    L = []
    L.append("=" * 62)
    L.append("USD/JPY 5分足 条件研究レポート (Ver.0.2.1)")
    L.append("=" * 62)
    L.append(f"検証パターン総数: {len(rank):,}通り")
    L.append("")
    L.append("判定の内訳:")
    for v, n in rank["判定"].value_counts().items():
        L.append(f"  {v}: {n:,}件")
    L.append("")
    L.append("判定基準:")
    L.append("  採用候補       … 学習・検証の両方で利益が残り、検証PF>=1.1、")
    L.append("                   かつ検証PFが学習PFの7割以上（再現性あり）")
    L.append("  要注意         … 検証はプラスだが優位性が弱い、または取引数不足")
    L.append("  過剰最適化疑い … 学習PF>=1.15なのに検証で崩壊（この差が研究材料）")
    L.append("  不採用         … 検証期間で利益が残らない")

    # --- 採用候補の上位詳細 ---
    L.append("")
    L.append("=" * 62)
    L.append(f"採用候補 上位{cfg.detail_top_n}（総合スコア順 / 内訳は検証期間）")
    L.append("=" * 62)
    if adopted.empty:
        L.append("採用候補なし。判定基準を満たす組み合わせがなかった。")
        L.append("これは「この戦略群には安定した優位性がない」という結果であり、")
        L.append("検証としては意味のある結論。要注意カテゴリと失敗例を確認すること。")
    for i, (_, r) in enumerate(adopted.head(cfg.detail_top_n).iterrows(), 1):
        L.append("")
        L.append(f"◆ 第{i}位 [スコア{r['総合スコア']}] {r['戦略']} "
                 f"利確{r['利確pips']}/損切り{r['損切りpips']}")
        L.append(f"  条件: 時間帯={r['時間帯']} 曜日={r['曜日']} "
                 f"ボラ={r['ボラ条件']} トレンド={r['トレンド条件']}")
        L.append(f"  学習: 取引{r['学習_取引回数']:,} PF{r['学習_PF']} "
                 f"総損益{r['学習_総損益pips']}pips DD{r['学習_最大DD_pips']} 連敗{r['学習_最大連敗']}")
        L.append(f"  検証: 取引{r['検証_取引回数']:,} PF{r['検証_PF']} "
                 f"総損益{r['検証_総損益pips']}pips DD{r['検証_最大DD_pips']} 連敗{r['検証_最大連敗']}"
                 f" 乖離率{r['乖離率(検証PF/学習PF)']}")
        trades = rebuild_trades(r, ctx_cache, outcome_cache, times, cfg, period="test")
        if not trades.empty:
            for label, tbl in breakdown(trades).items():
                L.append(f"\n  [{label}別・検証期間]")
                L.append("  " + tbl.to_string().replace("\n", "\n  "))
            path = cfg.out_dir / f"trades_top{i}.csv"
            full = rebuild_trades(r, ctx_cache, outcome_cache, times, cfg)
            full.to_csv(path, index=False, encoding="utf-8-sig")
            L.append(f"\n  全期間トレード履歴: {path.name}")

    # --- 失敗例（過剰最適化）---
    L.append("")
    L.append("=" * 62)
    L.append(f"過剰最適化の失敗例 上位{cfg.overfit_examples_n}（学習PF順）")
    L.append("学習で最も良く見えた条件が検証でどう崩れたかの記録。")
    L.append("「なぜ崩れたか」を考えることが次の条件設計の材料になる。")
    L.append("=" * 62)
    if overfit.empty:
        L.append("該当なし。")
    for _, r in overfit.iterrows():
        L.append(f"・{r['戦略']} 利確{r['利確pips']}/損切り{r['損切りpips']} "
                 f"[{r['時間帯']}|{r['曜日']}|{r['ボラ条件']}|{r['トレンド条件']}]")
        L.append(f"    学習: PF{r['学習_PF']} {r['学習_総損益pips']}pips ({r['学習_取引回数']:,}回)"
                 f" → 検証: PF{r['検証_PF']} {r['検証_総損益pips']}pips ({r['検証_取引回数']:,}回)")

    L.append("")
    L.append(f"備考: スプレッド{cfg.spread_pips}銭を全トレードに往復コストとして適用済み。")
    L.append("      同一足で利確・損切り両方に届いた場合は損切り扱い（保守的）。")
    return "\n".join(L)


if __name__ == "__main__":
    run_grid(BTConfig())
