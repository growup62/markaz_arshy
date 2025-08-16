# signal_generator.py

import logging
import pandas as pd
import hashlib
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

# --- DEPENDENSI UTAMA ---
from data_fetching import get_candlestick_data 
from technical_indicators import analyze_tf_opportunity as analyze_from_indicators
from technical_indicators_smc import detect_equal_highs_lows, calculate_atr
from server_comm import send_signal_to_server

# --- FUNGSI PEMBANTU (HELPER) ---

def build_signal_format(symbol: str, direction: str, entry_price: float, sl: float, tp: float, order_type: str) -> Dict[str, Any]:
    """Membangun format JSON standar untuk sinyal yang sesuai dengan EA."""
    prefix = direction.capitalize()
    signal_id = make_signal_id(symbol, direction, entry_price)
    # Gunakan field BuyLimit/SellLimit untuk kompatibilitas dengan EA
    specific_order_type = direction.upper()  # BUY atau SELL untuk OrderType
    logging.debug(f"Building signal: symbol={symbol}, direction={direction}, order_type={specific_order_type}")
    signal = {
        "OrderType": specific_order_type,
        "Symbol": symbol,
        "Direction": direction,
        "signal_id": signal_id
    }
    # Gunakan BuyLimit/SellLimit alih-alih BuyEntry/SellEntry
    if direction.lower() == "buy":
        signal["BuyLimit"] = round(entry_price, 2)
        signal["BuyLimitSL"] = round(sl, 2)
        signal["BuyLimitTP"] = round(tp, 2)
    else:  # sell
        signal["SellLimit"] = round(entry_price, 2)
        signal["SellLimitSL"] = round(sl, 2)
        signal["SellLimitTP"] = round(tp, 2)
    return signal

def make_signal_id(symbol: str, direction: str, entry_price: float) -> str:
    """Membuat ID unik untuk sebuah sinyal dari kontennya."""
    s = f"{symbol}:{direction}:{entry_price}".encode('utf-8')
    return hashlib.md5(s).hexdigest()

def get_active_orders(symbol: str, mt5_path: str) -> List[Dict[str, Any]]:
    """Placeholder: Fungsi ini seharusnya terhubung ke MT5 untuk mendapatkan order aktif."""
    return []

def is_far_enough(new_entry: float, existing_orders: List[Dict[str, Any]], pip_size: float, min_pips: int) -> bool:
    """Memeriksa apakah harga entry baru cukup jauh dari order yang sudah ada."""
    min_distance = min_pips * pip_size
    for order in existing_orders:
        if abs(new_entry - order.get('price_open', 0)) < min_distance:
            return False
    return True

# --- FUNGSI PEMROSESAN & ANALISIS ---

def handle_opportunity(
    opp: Dict[str, Any], symbol: str, tf: str, symbol_config: Dict[str, Any],
    global_config: Dict[str, Any], xgb_model: any, profile_name: str,
    signal_cooldown: Dict[Tuple[str, str], datetime]
) -> bool:
    """
    Memvalidasi hasil analisis dan mengirimkan sinyal ke server.
    Fungsi ini dipanggil oleh main.py setelah analyze_tf_opportunity berhasil.
    """
    if opp.get('signal') not in ["BUY", "SELL"]:
        logging.debug(f"Sinyal {opp.get('signal', 'UNKNOWN')} untuk {symbol}|{tf} dilewati, tidak dikirim ke server.")
        return False

    profile_config = symbol_config['strategy_profiles'][profile_name]
    
    try:
        entry_price = float(opp['entry_price_chosen'])
        sl_price = float(opp['sl'])
        tp_price = float(opp['tp'])
    except (ValueError, TypeError, KeyError) as e:
        logging.error(f"❌ Data harga tidak valid dalam hasil analisis: {e}")
        return False

    pip_size = global_config.get('pip_size_by_symbol', {}).get(symbol.upper(), 0.0001)
    min_pips = profile_config.get('min_distance_pips_per_tf', {}).get(tf, 10)
    
    try:
        existing_orders = get_active_orders(symbol, global_config['mt5_terminal_path'])
        if not is_far_enough(entry_price, existing_orders, pip_size, min_pips):
            logging.info(f"🟡 Sinyal dilewati karena terlalu dekat dengan order aktif.")
            return False
    except Exception as e:
        logging.error(f"❌ Gagal memeriksa order aktif: {e}")

    signal_json = build_signal_format(
        symbol=symbol, direction=opp['signal'].lower(),
        entry_price=entry_price, sl=sl_price, tp=tp_price,
        order_type=opp['signal']  # Gunakan BUY atau SELL
    )

    payload = {
        "symbol": symbol,
        "signal_json": signal_json,
        "api_key": global_config['api_key'],
        "server_url": global_config['server_url'],
        "secret_key": global_config['secret_key'],
        "score": opp.get('score'),
        "info": opp.get('info'),
        "profile_name": profile_name,
        "order_type": signal_json['OrderType']  # BUY atau SELL
    }

    logging.info(f"📤 Menyiapkan pengiriman sinyal ke server: OrderType={signal_json['OrderType']}")
    send_status = send_signal_to_server(**payload)
    if send_status == 'SUCCESS':
        logging.info(f"✅ [{profile_name}|{symbol}|{tf}] Sinyal BERHASIL dikirim!")
        signal_cooldown[(symbol, profile_name)] = datetime.now()
        return True
    elif send_status == 'SKIPPED':
        logging.info(f"🟡 [{profile_name}|{symbol}|{tf}] Sinyal dilewati karena tidak valid.")
        return False
    else:
        logging.error(f"❌ [{profile_name}|{symbol}|{tf}] Pengiriman sinyal GAGAL. Status dari server: {send_status}")
        return False

def analyze_tf_opportunity(
    symbol: str, tf: str, mt5_path: str, weights: dict, confidence_threshold: float, **kwargs
) -> dict:
    """
    Fungsi ini bertindak sebagai jembatan, memanggil 'otak' analisis dari technical_indicators.py.
    """
    try:
        df = get_candlestick_data(symbol=symbol, tf=tf, count=300, mt5_path=mt5_path)
        if df is None or df.empty or len(df) < 100:
            logging.debug(f"Data candle tidak cukup untuk analisis di {symbol}|{tf}.")
            return {"signal": "WAIT"}
    except Exception as e:
        logging.error(f"Gagal mengambil data untuk {symbol}|{tf}: {e}")
        return {"signal": "WAIT"}

    opportunity = analyze_from_indicators(
        df=df,
        symbol=symbol,
        tf=tf,
        mt5_path=mt5_path,
        start_date="",
        end_date="",
        profile_type=kwargs.get("profile_name", "intraday"),
        weights=weights
    )
    
    if not opportunity:
        return {"signal": "WAIT"}
    
    score = opportunity.get('score', 0.0)
    
    signal_type = "WAIT"
    if score >= confidence_threshold:
        signal_type = "BUY"
    elif score <= -confidence_threshold:
        signal_type = "SELL"
    
    if signal_type == "WAIT":
        return {"signal": "WAIT"}

    entry_price = df['close'].iloc[-1]
    atr = calculate_atr(df, period=14)
    stop_loss = entry_price - (2 * atr) if signal_type == "BUY" else entry_price + (2 * atr)
    take_profit = entry_price + (4 * atr) if signal_type == "BUY" else entry_price - (4 * atr)

    logging.info(f"📈 Peluang {signal_type} Ditemukan oleh Analisis Utama. Skor: {score:.2f}")

    return {
        "signal": signal_type,
        "score": score,
        "info": ", ".join(opportunity.get('info_list', ["Advanced Analysis"])),
        "entry_price_chosen": round(entry_price, 2),
        "sl": round(stop_loss, 2),
        "tp": round(take_profit, 2),
        "order_type": signal_type  # BUY atau SELL
    }