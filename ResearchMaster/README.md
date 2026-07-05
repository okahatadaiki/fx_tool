# Research Master 13.0 EA Factory

12.0で合格した本命候補を、MT5デモ口座で検証するためのEAファイルとして出力します。

## 実行

```powershell
cd C:\FX\fx_tool\ResearchMaster
python main.py
```

## 出力

`C:\FX\fx_tool\data\research_master_project\ea_factory` に以下を出力します。

- `ResearchMaster_USDJPY_LowBreak48_MAUP.mq5`
- `ea_settings.json`
- `demo_checklist.txt`
- `strategy_signal_reference.py`

## 注意

最初は必ずデモ口座、0.01ロットのみ。利益保証ではありません。
