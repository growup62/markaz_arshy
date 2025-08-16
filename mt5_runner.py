
"""
mt5_runner.py — Live/MT5 runner with pair-specific SPREAD GATE
- Blocks entries when spread is too wide for the symbol (relative to ATR and/or absolute cap)
- Safe defaults: if spread or ATR cannot be computed, block the trade
- Designed to be imported and used by your live trading script / signal executor
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import time
import pandas as pd
import numpy as np

# Optional: TA-Lib for ATR (falls back to numpy if not available)
try:
    import talib as ta
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False

# MetaTrader5 library (not available in this sandbox, but used in live environment)
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None  # to avoid import error here

# --- Pair-specific spread rules (same semantics as analyze_market_v2) ---
SPREAD_RULES = {
    "XAUUSD": {"max_abs": 0.5, "max_rel_atr": 0.25},
    "BTCUSD": {"max_abs": 30.0, "max_rel_atr": 0.12},
    "DEFAULT": {"max_rel_atr": 0.20},
}

def _match_spread_key(symbol: str) -> str:
    if not symbol:
        return "DEFAULT"
    s = str(symbol).upper()
    if s in SPREAD_RULES:
        return s
    for k in SPREAD_RULES.keys():
        if k != "DEFAULT" and s.startswith(k):
            return k
    return "DEFAULT"

def is_spread_ok_for_symbol(symbol: str, spread: float, atr: float) -> bool:
    """
    Gate entry by spread with pair-specific rules.
    Returns True if spread acceptable for the given symbol.
    """
    try:
        if spread is None or np.isnan(spread):
            return False
        key = _match_spread_key(symbol)
        rules = SPREAD_RULES.get(key, SPREAD_RULES["DEFAULT"])

        # Relative cap (spread / ATR)
        if atr is None or not np.isfinite(atr) or atr <= 0:
            return False  # cannot judge; safer to block
        rel = float(spread) / float(atr)
        max_rel = rules.get("max_rel_atr")
        if (max_rel is not None) and (rel > max_rel):
            return False

        # Absolute cap (if defined for the pair)
        max_abs = rules.get("max_abs")
        if (max_abs is not None) and (float(spread) > float(max_abs)):
            return False

        return True
    except Exception:
        return False

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    if _HAS_TALIB:
        atr = ta.ATR(high, low, close, timeperiod=period)
        return pd.Series(atr, index=df.index)
    # numpy fallback
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.max(np.vstack([tr1, tr2, tr3]), axis=0)
    atr = pd.Series(tr).rolling(period, min_periods=period).mean()
    atr.index = df.index
    return atr

def _get_symbol_spread(symbol: str) -> Optional[float]:
    """
    Get live spread in PRICE units (not points).
    Prefers tick.ask - tick.bid; falls back to symbol_info.spread * point.
    Returns None if not available.
    """
    if mt5 is None:
        return None
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None and tick.ask and tick.bid:
            spr = float(tick.ask - tick.bid)
            if spr > 0:
                return spr
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if hasattr(info, "spread") and hasattr(info, "point"):
            # 'spread' is quoted in points; convert to price
            if info.spread is not None and info.point is not None:
                return float(info.spread) * float(info.point)
        return None
    except Exception:
        return None

def _rates_to_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(list(rates), columns=["time","open","high","low","close","tick_volume","spread","real_volume"][:len(rates[0])])
    # normalize names
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if "tick_volume" in df.columns:
        df.rename(columns={"tick_volume":"volume"}, inplace=True)
    # NOTE: MT5 'spread' here is in points; we'll prefer live tick spread
    return df[["time","open","high","low","close","volume"]]

@dataclass
class SpreadGateResult:
    allow: bool
    spread: Optional[float]
    atr: Optional[float]
    rel: Optional[float]
    reason: str

class MT5SpreadGate:
    """
    Drop-in gate you can call right before placing orders.
    Usage:
        gate = MT5SpreadGate()
        ok, detail = gate.allow_entry("XAUUSD", mt5.TIMEFRAME_M5, bars=300)
        if ok:
            # proceed to send order
        else:
            # skip and log detail.reason
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def _latest_atr(self, df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        atr = _compute_atr(df, period=self.atr_period)
        if atr is None or atr.empty:
            return None
        val = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else None
        return val

    def allow_entry(self, symbol: str, timeframe, bars: int = 500) -> Tuple[bool, SpreadGateResult]:
        if mt5 is None:
            return False, SpreadGateResult(False, None, None, None, "MT5 module not available in this environment")

        # Ensure connection
        if not mt5.initialize():
            return False, SpreadGateResult(False, None, None, None, "Failed to initialize MT5")

        try:
            # Make sure symbol is available
            info = mt5.symbol_info(symbol)
            if info is None:
                return False, SpreadGateResult(False, None, None, None, f"Symbol {symbol} not found")
            if not info.visible:
                mt5.symbol_select(symbol, True)

            # Pull candles
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) < max(60, self.atr_period + 10):
                return False, SpreadGateResult(False, None, None, None, "Not enough bars for ATR")

            df = _rates_to_df(rates)

            # Compute ATR
            atr_val = self._latest_atr(df)

            # Pull live spread
            spr_val = _get_symbol_spread(symbol)

            if spr_val is None or atr_val is None or not np.isfinite(atr_val) or atr_val <= 0:
                return False, SpreadGateResult(False, spr_val, atr_val, None, "Spread/ATR unavailable")

            # Evaluate pair-specific rules
            allow = is_spread_ok_for_symbol(symbol, spr_val, atr_val)
            rel = spr_val / atr_val if atr_val else None
            if allow:
                return True, SpreadGateResult(True, spr_val, atr_val, rel, "OK")
            else:
                key = _match_spread_key(symbol)
                rules = SPREAD_RULES.get(key, SPREAD_RULES["DEFAULT"])
                reason = f"Blocked by spread gate for {symbol}: spread={spr_val:.6f}, ATR={atr_val:.6f}, rel={rel:.4f}, rules={rules}"
                return False, SpreadGateResult(False, spr_val, atr_val, rel, reason)
        finally:
            try:
                mt5.shutdown()
            except Exception:
                pass

# --- Example integration points ------------------------------------------------

def send_market_order_with_spread_gate(symbol: str, lot: float, order_type: str, sl: Optional[float]=None, tp: Optional[float]=None, deviation: int = 20, timeframe=None):
    """
    Example wrapper to demonstrate where to call the gate.
    - order_type: 'buy' or 'sell'
    - timeframe: e.g., mt5.TIMEFRAME_M5
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module not available here; run this on your live machine.")

    gate = MT5SpreadGate(atr_period=14)
    ok, detail = gate.allow_entry(symbol, timeframe=timeframe, bars=500)
    if not ok:
        print(f"[SPREAD-GATE] {detail.reason}")
        return {"ok": False, "reason": detail.reason}

    # If allowed, proceed to send order
    if not mt5.initialize():
        return {"ok": False, "reason": "Failed to initialize MT5 for order"}
    try:
        info = mt5.symbol_info(symbol)
        if not info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type.lower() == "buy" else tick.bid
        order_type_map = {
            "buy": mt5.ORDER_TYPE_BUY,
            "sell": mt5.ORDER_TYPE_SELL
        }
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type_map[order_type.lower()],
            "price": price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "deviation": deviation,
            "magic": 123456,
            "comment": "live_entry_with_spread_gate",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "reason": "order_send returned None"}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"ok": False, "reason": f"Order rejected: {result.retcode}", "result": result._asdict()}
        return {"ok": True, "result": result._asdict(), "gate": detail.__dict__}
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

# --- How to wire into your existing live runner --------------------------------
# 1) Import:
#   from mt5_runner import send_market_order_with_spread_gate, SPREAD_RULES
# 2) Before any order send, call the wrapper above instead of directly calling mt5.order_send.
# 3) Adjust SPREAD_RULES per your pairs and broker conditions.
# 4) For multi-threaded runners, reuse a single MT5 connection per thread and move gate.initialize/shutdown accordingly.
# 5) Log detail.rel (spread/ATR) to tune thresholds after a week of live data.
