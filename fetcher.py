# =====================================================
# fetcher.py
# DukascopyからUSD/JPYの5分足を取得する。
#
# 設計方針:
# - 5年半を一気に取りにいくと途中で失敗したとき全部やり直しになるので、
#   「1ヶ月ずつ取得 → data/chunks/ に保存」する方式にした。
# - 一度取得済みの月はスキップするので、途中で止めても
#   もう一度 main.py を実行すれば続きから再開できる。
# =====================================================
import time
from datetime import datetime

import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_USD_JPY

import config


def _month_ranges(start: datetime, end: datetime):
    """開始〜終了を1ヶ月ごとの(開始, 終了)ペアに分割する"""
    ranges = []
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1)
        chunk_start = max(cur, start)
        chunk_end = min(nxt, end)
        ranges.append((chunk_start, chunk_end))
        cur = nxt
    return ranges


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    ライブラリが返すDataFrameを標準形に整える。
    - インデックス: UTCのタイムスタンプ（tz-aware）
    - 列: open, high, low, close, volume
    ライブラリのバージョン差で列名の大文字小文字などが変わる可能性があるため、
    ここで吸収しておく。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # timestampが列に入っているバージョンへの対応
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"

    # volume列の名前ゆれ対応
    for cand in ("volume", "vol", "tick_volume"):
        if cand in df.columns:
            df = df.rename(columns={cand: "volume"})
            break
    if "volume" not in df.columns:
        df["volume"] = pd.NA

    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"取得データに想定した列がありません: {missing} / 実際の列: {list(df.columns)}"
        )
    return df[need]


def _fetch_one_month(start: datetime, end: datetime) -> pd.DataFrame:
    """1ヶ月分をリトライ付きで取得する"""
    last_err = None
    for attempt in range(1, config.MAX_RETRY + 1):
        try:
            df = dukascopy_python.fetch(
                INSTRUMENT_FX_MAJORS_USD_JPY,
                dukascopy_python.INTERVAL_MIN_5,
                dukascopy_python.OFFER_SIDE_BID,
                start,
                end,
            )
            return _normalize(df)
        except Exception as e:
            last_err = e
            print(f"    取得失敗 ({attempt}/{config.MAX_RETRY}回目): {e}")
            if attempt < config.MAX_RETRY:
                time.sleep(config.RETRY_WAIT_SEC)
    raise RuntimeError(f"{start:%Y-%m} の取得に{config.MAX_RETRY}回失敗しました: {last_err}")


def fetch_all() -> pd.DataFrame:
    """
    設定期間の全データを取得して1つのDataFrameにまとめて返す。
    月ごとの中間ファイルがあればダウンロードせず再利用する。
    """
    config.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    ranges = _month_ranges(config.START, config.END)
    frames = []

    print(f"取得期間: {config.START:%Y-%m-%d} 〜 {config.END:%Y-%m-%d}（{len(ranges)}ヶ月分）")

    for i, (s, e) in enumerate(ranges, 1):
        chunk_path = config.CHUNK_DIR / f"{config.PAIR_NAME}_{s:%Y-%m}.csv"

        if chunk_path.exists():
            print(f"[{i}/{len(ranges)}] {s:%Y-%m} は取得済み（スキップ）")
            df = pd.read_csv(chunk_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "datetime_utc"
        else:
            print(f"[{i}/{len(ranges)}] {s:%Y-%m} をダウンロード中...")
            df = _fetch_one_month(s, e)
            if df.empty:
                print(f"    ※ {s:%Y-%m} はデータが空でした（週末のみ等の可能性）")
            df.to_csv(chunk_path)

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("データが1件も取得できませんでした。ネット接続とライブラリのバージョンを確認してください。")

    all_df = pd.concat(frames)
    all_df = all_df.sort_index()
    print(f"取得完了: 合計 {len(all_df):,} 本")
    return all_df
