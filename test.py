import MetaTrader5 as mt5
from datetime import datetime

if not mt5.initialize(path=r"C:\Program Files\ExclusiveMarkets MetaTrader5\terminal64.exe"):
    print("MT5 init gagal")
else:
    rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 5)
    mt5.shutdown()
    print("Rates:", rates)
