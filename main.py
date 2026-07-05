# =====================================================
# main.py
# 実行の入り口。これだけ実行すればよい。
#
#   python main.py
#
# 処理の流れ:
#   1. Dukascopyから5分足を取得（月単位・再開可能）
#   2. 重複を除去してソート
#   3. 品質チェック（欠損・重複・異常値）→ レポート出力
#   4. datetime_utc / datetime_jst 付きのCSVとして保存
#
# Ver.0.1 の役割はここまで。
# バックテスト機能を追加するときは、この main.py には手を入れず、
# 出力されたCSVを読み込む backtest.py を新規に作る想定。
# =====================================================
import pandas as pd

import config
import fetcher
import quality


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 取得
    df = fetcher.fetch_all()

    # 2. 重複チェック → 除去（同一時刻は最初の1本を残す）
    dups = quality.find_duplicates(df)
    if len(dups) > 0:
        print(f"重複を {len(dups)} 本検出 → 除去します")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # 3. 品質チェック
    print("品質チェック中...")
    gaps = quality.find_gaps(df)
    anomalies = quality.find_anomalies(df)

    report = quality.build_report(df, dups, gaps, anomalies)
    print()
    print(report)

    config.REPORT_TXT.write_text(report, encoding="utf-8")
    if not gaps.empty:
        gaps.to_csv(config.GAP_CSV, index=False, encoding="utf-8-sig")
    if not anomalies.empty:
        anomalies.to_csv(config.ANOMALY_CSV, index=False, encoding="utf-8-sig")

    # 4. CSV保存
    out = df.copy()
    out["datetime_jst"] = out.index.tz_convert("Asia/Tokyo")

    # タイムゾーン表記(+00:00等)を外し、素直な文字列にして保存する。
    # pandasで読み込むときに扱いやすいのはこの形。
    out_csv = pd.DataFrame({
        "datetime_utc": out.index.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
        "datetime_jst": out["datetime_jst"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": out["open"].values,
        "high": out["high"].values,
        "low": out["low"].values,
        "close": out["close"].values,
        "volume": out["volume"].values,
    })
    out_csv.to_csv(config.OUTPUT_CSV, index=False, encoding="utf-8")

    print()
    print(f"保存完了: {config.OUTPUT_CSV}")
    print(f"レポート: {config.REPORT_TXT}")
    if not gaps.empty:
        print(f"欠損一覧: {config.GAP_CSV}")
    if not anomalies.empty:
        print(f"異常値一覧: {config.ANOMALY_CSV}")


if __name__ == "__main__":
    main()
