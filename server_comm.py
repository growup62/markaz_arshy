# server_comm.py
#
# Deskripsi:
# Versi ini disesuaikan untuk mendukung format JSON yang diharapkan EA (BuyLimit/SellLimit).

from __future__ import annotations
import logging
import requests
import json
from typing import Dict, Any

def send_signal_to_server(**payload: Any) -> str:
    """Mengirim sinyal trading ke server dan mengembalikan status keberhasilan."""
    
    server_url = payload.pop("server_url", None)
    secret_key = payload.pop("secret_key", None)
    
    if not server_url:
        logging.error("server_url tidak ditemukan di payload. Sinyal tidak dikirim.")
        return 'FAILED'
        
    headers = {}
    if secret_key:
        headers['X-Secret-Key'] = secret_key
    else:
        logging.warning("secret_key tidak ditemukan di payload. Mengirim tanpa otorisasi.")
        
    signal_json = payload.get("signal_json", {})
    if not isinstance(signal_json, dict):
        logging.error("Tipe data signal_json tidak valid (harus dictionary). Sinyal tidak dikirim.")
        return 'FAILED'
    
    symbol = payload.get("symbol", "UNKNOWN")
    order_type = signal_json.get("OrderType")
    
    # Validasi bahwa order_type ada dan valid
    valid_order_types = ["BUY", "SELL", "CANCEL"]
    if not order_type or order_type.upper() not in valid_order_types:
        logging.warning(f"OrderType tidak valid ({order_type}) untuk {symbol}, sinyal dilewati.")
        return 'SKIPPED'
    
    # Validasi field wajib dalam signal_json
    required_fields = ["Symbol", "OrderType", "Direction"]
    for field in required_fields:
        if field not in signal_json:
            logging.error(f"Field {field} tidak ada di signal_json untuk {symbol}. Sinyal tidak dikirim.")
            return 'FAILED'

    payload['signal'] = order_type.upper()

    try:
        logging.debug(f"Mengirim payload ke server: {json.dumps(payload, indent=2)}")
        response = requests.post(server_url, json=payload, headers=headers, timeout=10)
        log_message = f"Sinyal {payload.get('signal', 'UNKNOWN')} ({order_type}) untuk {symbol} dikirim."

        if response.status_code == 200:
            logging.info(f"✅ {log_message} Status: BERHASIL.")
            return 'SUCCESS'
        elif 400 <= response.status_code < 500:
            logging.error(f"❌ {log_message} Status: DITOLAK. Respons: {response.text}")
            return 'REJECTED'
        else:
            logging.error(f"❌ {log_message} Status: GAGAL. Kode: {response.status_code}, Respons: {response.text}")
            return 'FAILED'
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error koneksi saat mengirim sinyal: {e}")
        return 'FAILED'

def cancel_signal(signal_id: str, active_signals: Dict[str, Dict[str, any]], api_key: str, server_url: str, secret_key: str) -> None:
    """Membangun dan mengirim sinyal pembatalan untuk semua tipe order."""
    if signal_id not in active_signals:
        logging.warning(f"Sinyal ID {signal_id} tidak ditemukan di active_signals.")
        return

    original = active_signals[signal_id]['signal_json']
    symbol = original.get("Symbol")

    entry_val = original.get("BuyLimit") or original.get("SellLimit")

    if not symbol or not entry_val:
        logging.error(f"Data tidak lengkap untuk membatalkan sinyal ID {signal_id}.")
        return

    cancel_json = {
        "Symbol": symbol,
        "DeleteLimit/Stop": entry_val,
        "BuyLimit": "", "BuyLimitSL": "", "BuyLimitTP": "",
        "SellLimit": "", "SellLimitSL": "", "SellLimitTP": "",
        "OrderType": "CANCEL",
        "Direction": original.get("Direction", "")
    }

    payload = {
        "symbol": symbol,
        "signal_json": cancel_json,
        "api_key": api_key,
        "server_url": server_url,
        "secret_key": secret_key,
        "order_type": "CANCEL"
    }
    send_signal_to_server(**payload)
    del active_signals[signal_id]