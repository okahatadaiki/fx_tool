
from pathlib import Path


def _save_df(df, path):
    if df is not None and len(df):
        df.to_csv(path, index=False, encoding="utf-8-sig")


def _html_table(title, df, limit=None):
    if df is None or len(df) == 0:
        return ""
    show = df.head(limit) if limit else df
    return f"<h2>{title}</h2>" + show.to_html(index=False)


def save_reports(summary_df, all_trades, out_dir, oos_df=None, oos_trades=None, monthly_oos=None, portfolio_ranking=None, selected_strategies=None, portfolio_trades=None, portfolio_summary=None, optimizer_summary=None, optimizer_weights=None, optimizer_corr=None, optimizer_best_trades=None, lab_results=None, improvement_results=None, validation_results=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    for spread, trades in all_trades.items():
        safe = str(spread).replace('.', '_')
        trades.to_csv(out_dir / f"trades_spread_{safe}.csv", index=False, encoding="utf-8-sig")

    html_parts = ["<html><meta charset='utf-8'><style>body{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;} table{border-collapse:collapse;font-size:12px;} th,td{border:1px solid #ddd;padding:5px;} th{background:#f2f2f2;} h1{border-bottom:3px solid #333;} .ok{font-size:20px;font-weight:bold;padding:12px;background:#eef8ee;border:1px solid #9c9;}</style><body><h1>Research Master 12.0 Validation Engine</h1>"]
    text_parts = ["Research Master 12.0 Validation Engine", "", "全期間検証", summary_df.to_string(index=False)]

    html_parts.append(_html_table("全期間検証", summary_df))

    if oos_df is not None:
        oos_df.to_csv(out_dir / "oos_validation.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("OOS固定検証 2021-2024 / 2025以降", oos_df))
        text_parts.extend(["", "OOS固定検証", oos_df.to_string(index=False)])

    if oos_trades:
        oos_dir = out_dir / "oos_trades"
        oos_dir.mkdir(exist_ok=True)
        for name, trades in oos_trades.items():
            safe = name.replace('.', '_')
            trades.to_csv(oos_dir / f"{safe}.csv", index=False, encoding="utf-8-sig")

    if monthly_oos is not None and len(monthly_oos):
        monthly_oos.to_csv(out_dir / "oos_monthly.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("OOS月別", monthly_oos))
        text_parts.extend(["", "OOS月別", monthly_oos.to_string(index=False)])

    if portfolio_ranking is not None:
        portfolio_ranking.to_csv(out_dir / "portfolio_ranking.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("8.0 ポートフォリオ候補ランキング", portfolio_ranking, 30))
        text_parts.extend(["", "8.0 ポートフォリオ候補ランキング", portfolio_ranking.head(30).to_string(index=False)])

    if selected_strategies is not None:
        selected_strategies.to_csv(out_dir / "selected_strategies.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("採用戦略", selected_strategies))
        text_parts.extend(["", "採用戦略", selected_strategies.to_string(index=False)])

    if portfolio_trades is not None:
        portfolio_trades.to_csv(out_dir / "portfolio_trades.csv", index=False, encoding="utf-8-sig")

    if portfolio_summary is not None:
        portfolio_summary.to_csv(out_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("ポートフォリオ成績", portfolio_summary))
        text_parts.extend(["", "ポートフォリオ成績", portfolio_summary.to_string(index=False)])

    if optimizer_summary is not None:
        optimizer_summary.to_csv(out_dir / "optimizer_summary.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("9.0 ポートフォリオ最適化", optimizer_summary))
        text_parts.extend(["", "9.0 ポートフォリオ最適化", optimizer_summary.to_string(index=False)])

    if optimizer_weights is not None:
        optimizer_weights.to_csv(out_dir / "optimizer_weights.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("最適化ウェイト", optimizer_weights))
        text_parts.extend(["", "最適化ウェイト", optimizer_weights.to_string(index=False)])

    if optimizer_corr is not None:
        optimizer_corr.to_csv(out_dir / "strategy_correlations.csv", index=False, encoding="utf-8-sig")
        html_parts.append(_html_table("戦略相関 9.0", optimizer_corr, 30))

    if optimizer_best_trades is not None:
        optimizer_best_trades.to_csv(out_dir / "optimizer_best_trades.csv", index=False, encoding="utf-8-sig")

    if lab_results:
        lab_dir = out_dir / "portfolio_lab"
        lab_dir.mkdir(exist_ok=True)
        for key, df in lab_results.items():
            if key == "corr_matrix" and df is not None and len(df):
                df.to_csv(lab_dir / "correlation_matrix.csv", encoding="utf-8-sig")
                html_parts.append("<h2>10.0 相関マトリクス</h2>" + df.to_html())
                continue
            if df is not None and hasattr(df, "to_csv") and len(df):
                df.to_csv(lab_dir / f"{key}.csv", index=False, encoding="utf-8-sig")
        decision = lab_results.get("decision")
        if decision is not None and len(decision):
            msg = str(decision.iloc[0, 0])
            html_parts.append(f"<div class='ok'>最終判定: {msg}</div>")
            text_parts.extend(["", "10.0 最終判定", msg])
        html_parts.append(_html_table("10.0 Monte Carlo Summary", lab_results.get("monte_carlo_summary")))
        html_parts.append(_html_table("10.0 Position Sizing", lab_results.get("position_sizing")))
        html_parts.append(_html_table("10.0 Rolling Walk Forward Summary", lab_results.get("rolling_summary")))
        html_parts.append(_html_table("10.0 Rolling Walk Forward Detail", lab_results.get("rolling_walk_forward"), 50))
        html_parts.append(_html_table("10.0 戦略相関ペア", lab_results.get("corr_pairs"), 50))
        if lab_results.get("monte_carlo_summary") is not None and len(lab_results.get("monte_carlo_summary")):
            text_parts.extend(["", "10.0 Monte Carlo Summary", lab_results.get("monte_carlo_summary").to_string(index=False)])
        if lab_results.get("rolling_summary") is not None and len(lab_results.get("rolling_summary")):
            text_parts.extend(["", "10.0 Rolling Walk Forward Summary", lab_results.get("rolling_summary").to_string(index=False)])
        if lab_results.get("position_sizing") is not None and len(lab_results.get("position_sizing")):
            text_parts.extend(["", "10.0 Position Sizing", lab_results.get("position_sizing").to_string(index=False)])


    if improvement_results:
        imp_dir = out_dir / "improvement"
        imp_dir.mkdir(exist_ok=True)
        for key, df in improvement_results.items():
            if df is not None and hasattr(df, "to_csv") and len(df):
                df.to_csv(imp_dir / f"{key}.csv", index=False, encoding="utf-8-sig")
        decision = improvement_results.get("improvement_decision")
        if decision is not None and len(decision):
            msg = str(decision.iloc[0].get("decision", ""))
            html_parts.append(f"<div class='ok'>11.0 改善判定: {msg}</div>")
            text_parts.extend(["", "11.0 改善判定", decision.to_string(index=False)])
        ranking = improvement_results.get("improvement_ranking")
        if ranking is not None and len(ranking):
            html_parts.append(_html_table("11.0 改善候補ランキング", ranking, 20))
            text_parts.extend(["", "11.0 改善候補ランキング", ranking.head(20).to_string(index=False)])
        rolling_detail = improvement_results.get("improvement_rolling_detail")
        if rolling_detail is not None and len(rolling_detail):
            html_parts.append(_html_table("11.0 最上位候補 Rolling Detail", rolling_detail, 30))
        top_trades = improvement_results.get("improvement_top_trades")
        if top_trades is not None and len(top_trades):
            # Equity curveを別CSVで保存。HTMLは重くなるので先頭だけ。
            top_trades.to_csv(imp_dir / "top_strategy_trades.csv", index=False, encoding="utf-8-sig")
            html_parts.append(_html_table("11.0 最上位候補 トレード先頭", top_trades.head(50)))



    if validation_results:
        val_dir = out_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        for key, df in validation_results.items():
            if df is not None and hasattr(df, "to_csv") and len(df):
                name = key.replace("validation_", "")
                if key == "validation_top_trades":
                    df.to_csv(val_dir / "top_strategy_trades.csv", index=False, encoding="utf-8-sig")
                else:
                    df.to_csv(val_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        decision = validation_results.get("validation_decision")
        if decision is not None and len(decision):
            msg = str(decision.iloc[0].get("decision", ""))
            html_parts.append(f"<div class='ok'>12.0 最終検証判定: {msg}</div>")
            text_parts.extend(["", "12.0 最終検証判定", decision.to_string(index=False)])
        html_parts.append(_html_table("12.0 採用候補", validation_results.get("validation_selected_strategy")))
        html_parts.append(_html_table("12.0 コスト耐性", validation_results.get("validation_cost_stress"), 80))
        html_parts.append(_html_table("12.0 遅延耐性", validation_results.get("validation_delay_stress"), 80))
        html_parts.append(_html_table("12.0 年別検証", validation_results.get("validation_year_detail"), 30))
        html_parts.append(_html_table("12.0 レジーム検証", validation_results.get("validation_regime_detail"), 30))
        html_parts.append(_html_table("12.0 Monte Carlo", validation_results.get("validation_monte_carlo")))
        if validation_results.get("validation_cost_stress") is not None and len(validation_results.get("validation_cost_stress")):
            text_parts.extend(["", "12.0 コスト耐性", validation_results.get("validation_cost_stress").head(80).to_string(index=False)])
        if validation_results.get("validation_delay_stress") is not None and len(validation_results.get("validation_delay_stress")):
            text_parts.extend(["", "12.0 遅延耐性", validation_results.get("validation_delay_stress").to_string(index=False)])
        if validation_results.get("validation_year_detail") is not None and len(validation_results.get("validation_year_detail")):
            text_parts.extend(["", "12.0 年別検証", validation_results.get("validation_year_detail").to_string(index=False)])

    html_parts.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")
    (out_dir / "report.txt").write_text("\n".join(text_parts), encoding="utf-8")
