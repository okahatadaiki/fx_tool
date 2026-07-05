import pandas as pd

DATE_CANDIDATES = ["datetime_utc", "datetime", "time", "date", "Date", "Time"]
CLOSE_CANDIDATES = ["close", "Close", "終値"]
LOW_CANDIDATES = ["low", "Low", "安値"]
HIGH_CANDIDATES = ["high", "High", "高値"]


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"必要な列が見つかりません: {candidates}")


def load_price_csv(path):
    df = pd.read_csv(path)
    date_col = _find_col(df, DATE_CANDIDATES)
    close_col = _find_col(df, CLOSE_CANDIDATES)
    low_col = _find_col(df, LOW_CANDIDATES)
    high_col = _find_col(df, HIGH_CANDIDATES)

    out = pd.DataFrame({
        "datetime": pd.to_datetime(df[date_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
    }).dropna().sort_values("datetime").reset_index(drop=True)

    out["hour"] = out["datetime"].dt.hour
    out["weekday"] = out["datetime"].dt.day_name()
    out["month"] = out["datetime"].dt.month
    out["year"] = out["datetime"].dt.year
    return out
