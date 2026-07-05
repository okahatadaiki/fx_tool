from pathlib import Path


def save_reports(summary_df, all_trades, out_dir, oos_df=None, oos_trades=None, monthly_oos=None, portfolio_ranking=None, selected_strategies=None, portfolio_trades=None, portfolio_summary=None, optimizer_summary=None, optimizer_weights=None, optimizer_corr=None, optimizer_best_trades=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    for spread, trades in all_trades.items():
        safe = str(spread).replace('.', '_')
        trades.to_csv(out_dir / f"trades_spread_{safe}.csv", index=False, encoding="utf-8-sig")

    html_parts = ["<html><meta charset='utf-8'><body><h1>Research Master Project Report</h1>"]
    html_parts.append("<h2>全期間検証</h2>")
    html_parts.append(summary_df.to_html(index=False))

    text_parts = ["Research Master Project Report", "", "全期間検証", summary_df.to_string(index=False)]

    if oos_df is not None:
        oos_df.to_csv(out_dir / "oos_validation.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>OOS固定検証 2021-2024 / 2025以降</h2>")
        html_parts.append(oos_df.to_html(index=False))
        text_parts.extend(["", "OOS固定検証", oos_df.to_string(index=False)])

    if oos_trades:
        oos_dir = out_dir / "oos_trades"
        oos_dir.mkdir(exist_ok=True)
        for name, trades in oos_trades.items():
            safe = name.replace('.', '_')
            trades.to_csv(oos_dir / f"{safe}.csv", index=False, encoding="utf-8-sig")

    if monthly_oos is not None and len(monthly_oos):
        monthly_oos.to_csv(out_dir / "oos_monthly.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>OOS月別</h2>")
        html_parts.append(monthly_oos.to_html(index=False))
        text_parts.extend(["", "OOS月別", monthly_oos.to_string(index=False)])

    if portfolio_ranking is not None:
        portfolio_ranking.to_csv(out_dir / "portfolio_ranking.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>8.0 ポートフォリオ候補ランキング</h2>")
        html_parts.append(portfolio_ranking.head(30).to_html(index=False))
        text_parts.extend(["", "8.0 ポートフォリオ候補ランキング", portfolio_ranking.head(30).to_string(index=False)])

    if selected_strategies is not None:
        selected_strategies.to_csv(out_dir / "selected_strategies.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>採用戦略</h2>")
        html_parts.append(selected_strategies.to_html(index=False))
        text_parts.extend(["", "採用戦略", selected_strategies.to_string(index=False)])

    if portfolio_trades is not None:
        portfolio_trades.to_csv(out_dir / "portfolio_trades.csv", index=False, encoding="utf-8-sig")

    if portfolio_summary is not None:
        portfolio_summary.to_csv(out_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>ポートフォリオ成績</h2>")
        html_parts.append(portfolio_summary.to_html(index=False))
        text_parts.extend(["", "ポートフォリオ成績", portfolio_summary.to_string(index=False)])


    if optimizer_summary is not None:
        optimizer_summary.to_csv(out_dir / "optimizer_summary.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>9.0 ポートフォリオ最適化</h2>")
        html_parts.append(optimizer_summary.to_html(index=False))
        text_parts.extend(["", "9.0 ポートフォリオ最適化", optimizer_summary.to_string(index=False)])

    if optimizer_weights is not None:
        optimizer_weights.to_csv(out_dir / "optimizer_weights.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>最適化ウェイト</h2>")
        html_parts.append(optimizer_weights.to_html(index=False))
        text_parts.extend(["", "最適化ウェイト", optimizer_weights.to_string(index=False)])

    if optimizer_corr is not None:
        optimizer_corr.to_csv(out_dir / "strategy_correlations.csv", index=False, encoding="utf-8-sig")
        html_parts.append("<h2>戦略相関</h2>")
        html_parts.append(optimizer_corr.head(30).to_html(index=False))

    if optimizer_best_trades is not None:
        optimizer_best_trades.to_csv(out_dir / "optimizer_best_trades.csv", index=False, encoding="utf-8-sig")

    html_parts.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")
    (out_dir / "report.txt").write_text("\n".join(text_parts), encoding="utf-8")
