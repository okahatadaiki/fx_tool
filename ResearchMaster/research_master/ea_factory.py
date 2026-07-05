from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

EA_NAME = "ResearchMaster_USDJPY_LowBreak48_MAUP"


def _mq5_code() -> str:
    return r'''//+------------------------------------------------------------------+
//| Research Master 13.0 EA Factory                                  |
//| Strategy: LONG 21-23 LowBreak48 exit18 MA_UP                     |
//| NOTE: demo testing only. No profit guarantee.                    |
//+------------------------------------------------------------------+
#property strict
#property version   "13.00"
#property description "Research Master generated EA - DEMO only"

#include <Trade/Trade.mqh>
CTrade trade;

input double InpLots = 0.01;
input int    InpMagic = 102113;
input int    SessionStartHour = 21;
input int    SessionEndHour   = 23;
input int    LookbackBars     = 48;
input int    ExitBars         = 18;
input int    FastMAPeriod     = 96;
input int    SlowMAPeriod     = 288;
input int    MaxSpreadPoints  = 25;
input int    MaxPositions     = 1;
input bool   OneTradePerBar   = true;

string TRADE_COMMENT = "RM13_LOW48_MAUP";
datetime last_bar_time = 0;

int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if(_Symbol != "USDJPY") return;
   if(_Period != PERIOD_M5) return;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int need = MathMax(SlowMAPeriod + 5, LookbackBars + ExitBars + 10);
   if(CopyRates(_Symbol, PERIOD_M5, 0, need, rates) < need) return;

   if(OneTradePerBar)
   {
      if(rates[0].time == last_bar_time) return;
      last_bar_time = rates[0].time;
   }

   ManageExit(rates);

   if(CurrentPositions() >= MaxPositions) return;
   if(!IsSession(rates[1].time)) return;
   if(CurrentSpreadPoints() > MaxSpreadPoints) return;
   if(!IsMAUp()) return;
   if(!IsLowBreak(rates)) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   trade.Buy(InpLots, _Symbol, ask, 0, 0, TRADE_COMMENT);
}

bool IsSession(datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   int h = dt.hour;
   return (h >= SessionStartHour && h <= SessionEndHour);
}

int CurrentSpreadPoints()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return (int)MathRound((ask - bid) / _Point);
}

bool IsMAUp()
{
   double fast[], slow[];
   ArraySetAsSeries(fast, true);
   ArraySetAsSeries(slow, true);
   int hFast = iMA(_Symbol, PERIOD_M5, FastMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   int hSlow = iMA(_Symbol, PERIOD_M5, SlowMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if(hFast == INVALID_HANDLE || hSlow == INVALID_HANDLE) return false;
   if(CopyBuffer(hFast, 0, 1, 1, fast) <= 0) return false;
   if(CopyBuffer(hSlow, 0, 1, 1, slow) <= 0) return false;
   return fast[0] > slow[0];
}

bool IsLowBreak(MqlRates &rates[])
{
   double prevLow = rates[2].low;
   for(int i = 2; i < 2 + LookbackBars; i++)
   {
      if(rates[i].low < prevLow) prevLow = rates[i].low;
   }
   return rates[1].low <= prevLow;
}

int CurrentPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         count++;
   }
   return count;
}

void ManageExit(MqlRates &rates[])
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      int barsSinceOpen = iBarShift(_Symbol, PERIOD_M5, openTime, false);
      if(barsSinceOpen >= ExitBars)
      {
         trade.PositionClose(ticket);
      }
   }
}
//+------------------------------------------------------------------+
'''


def _python_bot_code() -> str:
    return '''"""Research Master 13.0 Python signal reference.
This is not an execution bot. It only shows the exact signal logic.
"""
import pandas as pd

PIP_SIZE = 0.01

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ma96"] = d["close"].rolling(96).mean()
    d["ma288"] = d["close"].rolling(288).mean()
    d["prev_low48"] = d["low"].shift(1).rolling(48).min()
    d["signal_long"] = (
        (d["hour"] >= 21) &
        (d["hour"] <= 23) &
        (d["low"] <= d["prev_low48"]) &
        (d["ma96"] > d["ma288"])
    )
    return d
'''


def run_ea_factory(output_dir: Path, validation_results: dict | None = None) -> dict:
    ea_dir = output_dir / "ea_factory"
    ea_dir.mkdir(parents=True, exist_ok=True)

    mq5_path = ea_dir / f"{EA_NAME}.mq5"
    py_path = ea_dir / "strategy_signal_reference.py"
    settings_path = ea_dir / "ea_settings.json"
    checklist_path = ea_dir / "demo_checklist.txt"

    mq5_path.write_text(_mq5_code(), encoding="utf-8")
    py_path.write_text(_python_bot_code(), encoding="utf-8")

    settings = {
        "version": "Research Master 13.0 EA Factory",
        "symbol": "USDJPY",
        "timeframe": "M5",
        "direction": "LONG",
        "session": "21-23",
        "rule": "LowBreak48 + MA_UP",
        "exit_bars": 18,
        "recommended_start_lot": 0.01,
        "demo_only": True,
        "notes": [
            "Start with demo account only.",
            "Use USDJPY M5 only.",
            "Check broker server time. If server time differs from CSV time, adjust SessionStartHour/SessionEndHour.",
            "Stop if live/demo behavior diverges from backtest materially."
        ]
    }
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    checklist_path.write_text(
        "Research Master 13.0 デモ運用チェックリスト\n"
        "1. MT5を開く\n"
        "2. ファイル > データフォルダを開く\n"
        "3. MQL5 > Experts に mq5 をコピー\n"
        "4. MetaEditorでコンパイル\n"
        "5. USDJPY M5チャートに適用\n"
        "6. 最初は0.01ロット\n"
        "7. 最低1〜2週間はデモのみ\n"
        "8. 約定時間・スプレッド・実際の損益をCSVに保存\n",
        encoding="utf-8"
    )

    summary = pd.DataFrame([{
        "artifact": "MT5 EA",
        "path": str(mq5_path),
        "strategy": "LONG 21-23 LowBreak48 exit18 MA_UP",
        "status": "created",
        "next_action": "MT5 demo compile and run with 0.01 lot"
    },{
        "artifact": "Settings",
        "path": str(settings_path),
        "strategy": "configuration",
        "status": "created",
        "next_action": "confirm broker server time"
    }])
    summary.to_csv(ea_dir / "ea_factory_summary.csv", index=False, encoding="utf-8-sig")
    return {"ea_dir": ea_dir, "mq5_path": mq5_path, "summary": summary}
