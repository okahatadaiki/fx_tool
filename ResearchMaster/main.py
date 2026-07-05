import pandas as pd

from config import DEFAULT_CSV, OUTPUT_DIR, SPREADS
from research_master.data_loader import load_price_csv
from research_master.strategy import build_strategy_trades
from research_master.metrics import summarize_trades
from research_master.oos import run_oos_validation, monthly_breakdown
from research_master.report import save_reports


def main():
    print("Research Master Project版 + 7.5 OOS固定検証 開始")
    csv_path = DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    df = load_price_csv(csv_path)
    results = []
    all_trades = {}

    print("全期間検証")
    for spread in SPREADS:
        trades = build_strategy_trades(df, spread=spread)
        summary = summarize_trades(trades, spread=spread)
        results.append(summary)
        all_trades[str(spread)] = trades
        print(f"spread {spread}: trades={summary['trades']} PF={summary['pf']:.3f} EXP={summary['expectancy_pips']:.4f} DD={summary['max_dd_pips']:.1f}")

    result_df = pd.DataFrame(results)

    print("\nOOS固定検証 2021-2024 / 2025以降")
    oos_df, oos_trades = run_oos_validation(df)
    print(oos_df.to_string(index=False))

    # OOSの本命スプレッド0.2だけ月別を出す
    monthly_oos = monthly_breakdown(oos_trades.get("oos_spread_0.2", pd.DataFrame()))

    save_reports(result_df, all_trades, OUTPUT_DIR, oos_df=oos_df, oos_trades=oos_trades, monthly_oos=monthly_oos)
    print("\n完了")
    print(f"出力先: {OUTPUT_DIR}")
    print("見るファイル: summary.csv / oos_validation.csv / oos_monthly.csv / report.html")


if __name__ == "__main__":
    main()
