from pathlib import Path


def save_reports(summary_df, all_trades, out_dir, oos_df=None, oos_trades=None, monthly_oos=None):
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

    html_parts.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")
    (out_dir / "report.txt").write_text("\n".join(text_parts), encoding="utf-8")
