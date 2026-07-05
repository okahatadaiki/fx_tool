import pandas as pd

from config import DEFAULT_CSV, OUTPUT_DIR, SPREADS
from research_master.data_loader import load_price_csv
from research_master.strategy import build_strategy_trades
from research_master.metrics import summarize_trades
from research_master.oos import run_oos_validation, monthly_breakdown
from research_master.portfolio import run_portfolio_engine
from research_master.optimizer import run_portfolio_optimizer
from research_master.report import save_reports


def main():
    print("Research Master 9.0 Portfolio Optimizer 開始")
    csv_path = DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    df = load_price_csv(csv_path)
    results = []
    all_trades = {}

    print("\n固定本命ルールの全期間検証")
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
    monthly_oos = monthly_breakdown(oos_trades.get("oos_spread_0.2", pd.DataFrame()))

    print("\n8.0 ポートフォリオ探索")
    portfolio_ranking, selected_df, portfolio_trades, portfolio_summary, portfolio_trades_map = run_portfolio_engine(df, spread=0.5)
    print("\n採用戦略")
    if len(selected_df):
        print(selected_df[["strategy", "trades", "pf", "expectancy_pips", "max_dd_pips", "score"]].to_string(index=False))
    print("\nポートフォリオ成績")
    print(pd.DataFrame([portfolio_summary]).to_string(index=False))

    print("\n9.0 ポートフォリオ最適化")
    optimizer_summary, optimizer_weights, optimizer_corr, best_mode, optimizer_best_trades, best_weights = run_portfolio_optimizer(selected_df, portfolio_trades_map, spread=0.5)
    print("\n最適化結果")
    print(optimizer_summary.to_string(index=False))
    print(f"\n最終採用ウェイト方式: {best_mode}")
    if len(best_weights):
        print(best_weights[["strategy", "weight"]].to_string(index=False))

    save_reports(
        result_df,
        all_trades,
        OUTPUT_DIR,
        oos_df=oos_df,
        oos_trades=oos_trades,
        monthly_oos=monthly_oos,
        portfolio_ranking=portfolio_ranking,
        selected_strategies=selected_df,
        portfolio_trades=portfolio_trades,
        portfolio_summary=pd.DataFrame([portfolio_summary]),
        optimizer_summary=optimizer_summary,
        optimizer_weights=optimizer_weights,
        optimizer_corr=optimizer_corr,
        optimizer_best_trades=optimizer_best_trades,
    )
    print("\n完了")
    print(f"出力先: {OUTPUT_DIR}")
    print("見るファイル: optimizer_summary.csv / optimizer_weights.csv / strategy_correlations.csv / report.html")


if __name__ == "__main__":
    main()
