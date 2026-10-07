//+------------------------------------------------------------------+
//|                                                    ExportBars.mq5 |
//|   Э13.2: dump broker history into a CSV the smc-zero loader reads. |
//|                                                                  |
//| The MT5 "Save As" dialog of the terminal stops around 100 000    |
//| bars, which is 1.4 years of M5 - too short for the walk-forward  |
//| of SPEC_SMC.md §7.10 п.61.  This script walks the requested range |
//| in calendar chunks and writes every bar it gets, so a four year   |
//| M5 tape comes out as one file.                                    |
//|                                                                  |
//| Output (in MQL5\Files of the terminal, one file per timeframe):   |
//|   symbol,timeframe,time,open,high,low,close,tick_volume,spread,   |
//|   real_volume                                                     |
//| - the raw MetaTrader 5 layout, which                              |
//| smc_zero.data_loader.load_ohlcv(format="auto") detects and reads. |
//| "time" is the OPEN time of the bar, as the project convention     |
//| requires (SPEC_SMC.md §7.10 п.62).                                |
//|                                                                  |
//| Before running: Symbols -> EURUSD -> Bars -> Request the history, |
//| and set Tools -> Options -> Charts -> "Max bars in chart" to      |
//| Unlimited, otherwise the terminal answers with fewer bars.        |
//+------------------------------------------------------------------+
#property copyright "smc-zero project"
#property version   "1.00"
#property script_show_inputs
#property description "Dump MT5 history into a CSV the smc-zero loader reads (Э13.2)."

input string          InpSymbol     = "EURUSD";               // symbol to dump
input ENUM_TIMEFRAMES InpTimeframe  = PERIOD_M5;              // timeframe to dump
input datetime        InpFrom       = D'2022.08.15 00:00';    // first bar (open time)
input datetime        InpTo         = D'2026.10.06 23:59';    // last bar (open time)
input int             InpChunkDays  = 30;                     // calendar days per request
input int             InpMaxRetries = 20;                      // waits for a slow download

//+------------------------------------------------------------------+
//| Return the project's timeframe label ("M5", "M15", "H1", "H4").  |
//+------------------------------------------------------------------+
string TimeframeLabel(const ENUM_TIMEFRAMES timeframe)
{
   switch(timeframe)
   {
      case PERIOD_M5:  return("M5");
      case PERIOD_M15: return("M15");
      case PERIOD_H1:  return("H1");
      case PERIOD_H4:  return("H4");
      case PERIOD_D1:  return("D1");
      default:         return(EnumToString(timeframe));
   }
}

//+------------------------------------------------------------------+
//| Return a bar stamp as "YYYY-MM-DD HH:MM:SS", the loader's format.|
//+------------------------------------------------------------------+
string Stamp(const datetime time)
{
   MqlDateTime parts;
   TimeToStruct(time, parts);
   return(StringFormat("%04d-%02d-%02d %02d:%02d:%02d",
                       parts.year, parts.mon, parts.day, parts.hour, parts.min, parts.sec));
}

//+------------------------------------------------------------------+
//| Return the bars of one window, waiting while the terminal downloads.
//+------------------------------------------------------------------+
int FetchChunk(const datetime from, const datetime to, MqlRates &rates[])
{
   for(int attempt = 0; attempt < InpMaxRetries; attempt++)
   {
      const int copied = CopyRates(InpSymbol, InpTimeframe, from, to, rates);
      if(copied > 0)
         return(copied);
      if(_LastError != ERR_HISTORY_WILL_UPDATED && _LastError != ERR_NO_HISTORY_DATA)
         break;
      PrintFormat("waiting for %s %s history (%d/%d)",
                  InpSymbol, TimeframeLabel(InpTimeframe), attempt + 1, InpMaxRetries);
      Sleep(500);
   }
   return(0);
}

//+------------------------------------------------------------------+
//| Walk the range in chunks and write one row per bar.              |
//+------------------------------------------------------------------+
void OnStart()
{
   const string label = TimeframeLabel(InpTimeframe);
   const string name  = InpSymbol + "_" + label + ".csv";
   const int handle = FileOpen(name, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("cannot open %s: error %d", name, GetLastError());
      return;
   }

   FileWrite(handle, "symbol", "timeframe", "time", "open", "high", "low", "close",
             "tick_volume", "spread", "real_volume");

   int written = 0;
   datetime cursor = InpFrom;
   datetime last_stamp = 0;
   double close_out = 0.0;

   while(cursor <= InpTo)
   {
      const datetime chunk_end = (datetime)MathMin((long)cursor + (long)InpChunkDays * 86400,
                                                   (long)InpTo);
      MqlRates rates[];
      const int copied = FetchChunk(cursor, chunk_end, rates);
      if(copied <= 0)
         PrintFormat("no bars in %s .. %s (error %d)",
                     Stamp(cursor), Stamp(chunk_end), GetLastError());
      for(int i = 0; i < copied; i++)
      {
         if(rates[i].time <= last_stamp)   // the chunk edges overlap: one bar, one row
            continue;
         FileWrite(handle, InpSymbol, label, Stamp(rates[i].time),
                   DoubleToString(rates[i].open, 5), DoubleToString(rates[i].high, 5),
                   DoubleToString(rates[i].low, 5), DoubleToString(rates[i].close, 5),
                   (string)rates[i].tick_volume, (string)rates[i].spread,
                   (string)rates[i].real_volume);
         last_stamp = rates[i].time;
         close_out = rates[i].close;
         written++;
      }
      cursor = chunk_end + (datetime)PeriodSeconds(InpTimeframe);
   }

   FileClose(handle);
   if(written == 0)
   {
      PrintFormat("%s: no bar in %s .. %s - request the history in Symbols first",
                  name, Stamp(InpFrom), Stamp(InpTo));
      return;
   }
   PrintFormat("%s: %d bars written, %s .. %s, last close %.5f",
               name, written, Stamp(InpFrom), Stamp(last_stamp), close_out);
   PrintFormat("copy MQL5\\Files\\%s into the tape base, e.g. ~/_data/mt5/%s", name, name);
}
//+------------------------------------------------------------------+

