# =====================================================
# config.py
# 設定はすべてここに集約する。
# 期間や通貨ペアを変えたいときはこのファイルだけ触ればよい。
# =====================================================
from datetime import datetime
from pathlib import Path

# --- 取得対象 ---
PAIR_NAME = "USDJPY"

# 取得期間（UTC基準）
# Dukascopyのデータは内部的にUTCで管理されているため、期間指定もUTCで行う
START = datetime(2021, 1, 1)
END = datetime(2026, 6, 30, 23, 59, 59)

# --- 保存先 ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHUNK_DIR = DATA_DIR / "chunks"          # 月ごとの中間ファイル置き場（再開用）
OUTPUT_CSV = DATA_DIR / f"{PAIR_NAME}_5min_2021-2026.csv"
REPORT_TXT = DATA_DIR / "quality_report.txt"
GAP_CSV = DATA_DIR / "gaps.csv"          # 欠損区間の一覧
ANOMALY_CSV = DATA_DIR / "anomalies.csv" # 異常値候補の一覧

# --- 通信まわり ---
MAX_RETRY = 3        # 1チャンクあたりの再試行回数
RETRY_WAIT_SEC = 15  # 再試行までの待機秒数

# --- 品質チェックの基準 ---
# 5分間で終値がこの割合(%)以上動いたら「異常値候補」としてフラグを立てる。
# USD/JPYの5分足で1%(約1.5円)動くのは指標発表時などごく稀なので、
# フラグ＝即エラーではなく「人間が目視確認すべき足」という位置づけ。
SPIKE_THRESHOLD_PCT = 1.0
