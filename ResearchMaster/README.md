# Research Master Project

USDJPY 5分足CSVを使い、固定戦略の全期間検証とOOS固定検証を行うプロジェクト版です。

## 実行

```powershell
cd C:\FX\fx_tool\ResearchMaster
python main.py
```

## 出力先

```text
C:\FX\fx_tool\data\research_master_project
```

## 主な出力

- summary.csv 全期間検証
- oos_validation.csv 2021-2024 / 2025以降の固定検証
- oos_monthly.csv OOSの月別成績
- report.html レポート
- report.txt テキストレポート

## Git保存

```powershell
git add .
git commit -m "add OOS validation"
```
