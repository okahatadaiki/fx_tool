# =====================================================
# quality.py
# データ品質チェック（欠損・重複・異常値）
#
# 方針:
# - 問題を「検出して報告」するのが役割。勝手にデータを書き換えない。
#   （重複の除去だけは機械的に安全なので main.py 側で行う）
# - FX市場は週末（おおよそ 金曜21:00 UTC 〜 日曜21:00 UTC）休場なので、
#   週末のギャップは欠損ではなく正常。ここを区別しないと
#   欠損レポートが週末だらけになって使い物にならない。
# =====================================================
import pandas as pd

import config


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """同じ時刻の足が複数ある行を返す（最初の1本以外）"""
    dup_mask = df.index.duplicated(keep="first")
    return df[dup_mask]


def find_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    連続する足の間隔が5分を超えている箇所を洗い出す。
    週末休場によるギャップかどうかも判定して列に持たせる。
    戻り値の列: gap_start(最後にデータがある足), gap_end(再開した足),
               gap_minutes, missing_bars, type
    """
    idx = df.index
    diffs = idx.to_series().diff()

    gaps = []
    for i in range(1, len(idx)):
        d = diffs.iloc[i]
        if d is pd.NaT or d <= pd.Timedelta(minutes=5):
            continue

        gap_start = idx[i - 1]
        gap_end = idx[i]
        gap_min = d.total_seconds() / 60
        missing_bars = int(gap_min // 5) - 1

        # 週末判定: 金曜の夜に止まり、日曜の夜〜月曜に再開し、
        # 期間が3日以内ならFX市場の通常休場とみなす
        is_weekend = (
            gap_start.weekday() == 4          # 金曜
            and gap_end.weekday() in (6, 0)   # 日曜 or 月曜
            and d <= pd.Timedelta(days=3)
        )
        gap_type = "週末休場(正常)" if is_weekend else "要確認"

        gaps.append({
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_minutes": round(gap_min, 1),
            "missing_bars": missing_bars,
            "type": gap_type,
        })

    return pd.DataFrame(gaps)


def find_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    異常値の候補を検出する。戻り値は「時刻 + 理由」の一覧。
    チェック項目:
    1. 価格が0以下
    2. OHLCの整合性が壊れている（high < low など）
    3. 価格またはvolumeが欠損(NaN)
    4. 前の終値から SPIKE_THRESHOLD_PCT % 以上の急変動
       （エラーとは限らない。指標発表・介入などの本物の急変も含まれる）
    """
    records = []

    price_cols = ["open", "high", "low", "close"]

    # 1. 0以下の価格
    bad = df[(df[price_cols] <= 0).any(axis=1)]
    for t in bad.index:
        records.append({"datetime_utc": t, "reason": "価格が0以下"})

    # 2. OHLC整合性
    bad = df[
        (df["high"] < df["low"])
        | (df["high"] < df["open"]) | (df["high"] < df["close"])
        | (df["low"] > df["open"]) | (df["low"] > df["close"])
    ]
    for t in bad.index:
        records.append({"datetime_utc": t, "reason": "OHLCの大小関係が不正"})

    # 3. NaN
    bad = df[df[price_cols].isna().any(axis=1)]
    for t in bad.index:
        records.append({"datetime_utc": t, "reason": "価格にNaNが含まれる"})

    # 4. 急変動（前の足の終値と比較）
    pct_change = df["close"].pct_change().abs() * 100
    spike = df[pct_change > config.SPIKE_THRESHOLD_PCT]
    for t in spike.index:
        records.append({
            "datetime_utc": t,
            "reason": f"前足比 {pct_change.loc[t]:.2f}% の急変動（閾値{config.SPIKE_THRESHOLD_PCT}%）",
        })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("datetime_utc").reset_index(drop=True)
    return out


def build_report(df: pd.DataFrame, dups: pd.DataFrame,
                 gaps: pd.DataFrame, anomalies: pd.DataFrame) -> str:
    """人間が読むためのサマリーレポート文字列を作る"""
    lines = []
    lines.append("=" * 50)
    lines.append(f"データ品質レポート: {config.PAIR_NAME} 5分足")
    lines.append("=" * 50)
    lines.append(f"期間        : {df.index.min()} 〜 {df.index.max()}")
    lines.append(f"総本数      : {len(df):,} 本")
    lines.append("")
    lines.append(f"重複        : {len(dups)} 本（重複分は削除済み）")

    if gaps.empty:
        lines.append("欠損        : なし")
    else:
        weekend = gaps[gaps["type"] == "週末休場(正常)"]
        suspect = gaps[gaps["type"] == "要確認"]
        lines.append(f"ギャップ    : 全{len(gaps)}箇所")
        lines.append(f"  うち週末休場(正常): {len(weekend)}箇所")
        lines.append(f"  うち要確認        : {len(suspect)}箇所 → 詳細は gaps.csv")
        if not suspect.empty:
            lines.append("  要確認ギャップの上位（欠損本数順）:")
            top = suspect.sort_values("missing_bars", ascending=False).head(10)
            for _, r in top.iterrows():
                lines.append(
                    f"    {r['gap_start']} → {r['gap_end']}  約{r['missing_bars']}本欠損"
                )

    lines.append("")
    if anomalies.empty:
        lines.append("異常値候補  : なし")
    else:
        lines.append(f"異常値候補  : {len(anomalies)}件 → 詳細は anomalies.csv")
        lines.append("  ※ 急変動フラグは指標発表・介入等の本物の値動きも含みます。")
        lines.append("    削除はせずフラグのみです。検証時に個別判断してください。")

    lines.append("")
    lines.append("備考:")
    lines.append("- 価格はDukascopyのBid価格です。")
    lines.append("- volumeは出来高そのものではなくティックベースの取引量指標です。")
    lines.append("  相対的な活況度の比較には使えますが、絶対量として扱わないでください。")
    return "\n".join(lines)
