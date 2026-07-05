
import pandas as pd

from config import DEFAULT_CSV, OUTPUT_DIR, SPREADS
from research_master.data_loader import load_price_csv
from research_master.strategy import build_strategy_trades
from research_master.metrics import summarize_trades
from research_master.oos import run_oos_validation, monthly_breakdown
from research_master.portfolio import run_portfolio_engine
from research_master.optimizer import run_portfolio_optimizer
from research_master.lab import run_portfolio_lab
from research_master.improvement import run_improvement_engine
from research_master.validation import run_validation_engine
from research_master.ea_factory import run_ea_factory
from research_master.report import save_reports


def main():
    print("Research Master 13.0 EA Factory 開始")
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

    print("\n10.0 Portfolio Lab")
    lab_results = run_portfolio_lab(df, selected_df, portfolio_trades_map, optimizer_best_trades, optimizer_summary)
    if len(lab_results.get("decision", pd.DataFrame())):
        print("最終判定:", lab_results["decision"].iloc[0, 0])
    if len(lab_results.get("monte_carlo_summary", pd.DataFrame())):
        print("\nMonte Carlo Summary")
        print(lab_results["monte_carlo_summary"].to_string(index=False))
    if len(lab_results.get("rolling_summary", pd.DataFrame())):
        print("\nRolling Walk Forward Summary")
        print(lab_results["rolling_summary"].to_string(index=False))
    if len(lab_results.get("position_sizing", pd.DataFrame())):
        print("\nPosition Sizing")
        print(lab_results["position_sizing"][["position_method", "lot_multiplier", "pf", "expectancy_pips", "max_dd_pips", "ending_balance"]].to_string(index=False))

    print("\n11.0 AI Improvement Engine")
    improvement_results = run_improvement_engine(df, spread=0.5, stress_spread=1.0, top_limit=20)
    imp_decision = improvement_results.get("improvement_decision", pd.DataFrame())
    imp_rank = improvement_results.get("improvement_ranking", pd.DataFrame())
    if len(imp_decision):
        print("改善判定:", imp_decision.iloc[0]["decision"])
        print("最上位:", imp_decision.iloc[0].get("top_strategy", ""))
    if len(imp_rank):
        cols = ["rank", "strategy", "trades", "pf", "expectancy_pips", "oos_pf", "oos_expectancy", "stress_pf_1_0", "rwf_pass_rate_pct", "improvement_score"]
        print("\n改善候補ランキング")
        print(imp_rank[cols].head(10).to_string(index=False))

    print("\n12.0 Validation Engine")
    validation_results = run_validation_engine(df, improvement_results, spread=0.5)
    val_decision = validation_results.get("validation_decision", pd.DataFrame())
    if len(val_decision):
        print("最終検証判定:", val_decision.iloc[0].get("decision", ""))
        print(val_decision.to_string(index=False))
    cost_stress = validation_results.get("validation_cost_stress", pd.DataFrame())
    if len(cost_stress):
        print("\nコスト耐性 上位抜粋")
        print(cost_stress[(cost_stress["slippage"].isin([0.0,0.5])) & (cost_stress["spread"].isin([0.5,1.0,1.5,2.0]))].to_string(index=False))
    delay_stress = validation_results.get("validation_delay_stress", pd.DataFrame())
    if len(delay_stress):
        print("\n遅延耐性")
        print(delay_stress.to_string(index=False))
    year_detail = validation_results.get("validation_year_detail", pd.DataFrame())
    if len(year_detail):
        print("\n年別検証")
        print(year_detail.to_string(index=False))


    print("\n13.0 EA Factory")
    ea_results = run_ea_factory(OUTPUT_DIR, validation_results)
    print("MT5 EAを作成しました:", ea_results["mq5_path"])
    print("出力フォルダ:", ea_results["ea_dir"])
    print("次はMT5のデモ口座で0.01ロット検証です。")

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
        lab_results=lab_results,
        improvement_results=improvement_results,
        validation_results=validation_results,
    )
    print("\n完了")
    print(f"出力先: {OUTPUT_DIR}")
    print("見るファイル: report.html / portfolio_lab / improvement / validation / ea_factory フォルダ")


if __name__ == "__main__":
    main()
