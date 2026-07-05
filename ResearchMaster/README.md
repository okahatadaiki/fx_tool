# Research Master 9.0 Portfolio Optimizer

USDJPY 5分足CSVを使って、固定ルール検証、OOS固定検証、ポートフォリオ探索、ポートフォリオ最適化まで実行します。

## 実行

```powershell
cd C:\FX\fx_tool\ResearchMaster
python main.py
```

## 入力CSV

既定では次を探します。

```text
C:\FX\fx_tool\data\USDJPY_5min_2021-2026.csv
```

## 主な出力

```text
C:\FX\fx_tool\data\research_master_project
```

- summary.csv
- oos_validation.csv
- portfolio_ranking.csv
- selected_strategies.csv
- portfolio_summary.csv
- optimizer_summary.csv
- optimizer_weights.csv
- strategy_correlations.csv
- optimizer_best_trades.csv
- report.html

## 保存

```powershell
git add .
git commit -m "add Research Master 9.0 portfolio optimizer"
```
