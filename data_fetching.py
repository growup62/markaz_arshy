# data_fetching.py

from __future__ import annotations
import pandas as pd
import MetaTrader5 as mt5
import time
import threading
import logging
from functools import lru_cache
from typing import Optional, List, Dict, Tuple

def validate_timeframe(tf: str):
    """Memvalidasi dan mengubah string timeframe ke konstanta MT5."""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_map.get(tf.upper())
    if not mt5_tf:
        raise ValueError(f"Timeframe tidak valid: {tf}")
    return mt5_tf

def validate_symbol(symbol: str):
    """Memvalidasi format simbol."""
    if not isinstance(symbol, str) or len(symbol) < 3:
        raise ValueError(f"Simbol tidak valid: {symbol}")

def validate_count(count: int):
    """Memvalidasi jumlah candle."""
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Jumlah candle (count) harus angka positif, diterima: {count}")

def robust_mt5_init(mt5_path: str, retry: int = 5, sleep_sec: float = 3.0) -> bool:
    """Mencoba menginisialisasi koneksi ke MT5 dengan retry dan timeout."""
    for attempt in range(1, retry + 1):
        logging.debug(f"[Koneksi MT5] Upaya {attempt}/{retry}: Menghubungkan ke terminal...")
        try:
            if mt5.initialize(path=mt5_path, timeout=15000):  # Tingkatkan timeout
                logging.debug("[Koneksi MT5] Berhasil terhubung.")
                return True
        except Exception as e:
            logging.warning(f"[Koneksi MT5] Upaya {attempt} GAGAL: {e}. Mencoba lagi...")
        time.sleep(sleep_sec)
    logging.error("[Koneksi MT5] Semua upaya untuk menginisialisasi koneksi MT5 gagal.")
    return False

def get_candlestick_data(symbol: str, tf: str, count: int, mt5_path: str, retry: int = 3) -> Optional[pd.DataFrame]:
    """Mengambil data candlestick dari MetaTrader 5 dengan log Bahasa Indonesia."""
    try:
        validate_symbol(symbol)
        tf_mt5 = validate_timeframe(tf)
        validate_count(count)
    except ValueError as ve:
        logging.error(f"❌ Parameter tidak valid: {ve}")
        return None

    for attempt in range(1, retry + 1):
        if not robust_mt5_init(mt5_path, retry=1):
            time.sleep(1)
            continue
        try:
            logging.info(f"📥 Mengambil {count} candle untuk {symbol} di timeframe {tf}...")
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, count)
            if rates is None or len(rates) == 0:
                logging.warning(f"Tidak ada data candle yang diterima dari MT5 untuk {symbol} di {tf}.")
                mt5.shutdown()
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
            df = df.dropna()  # Tambah drop NaN
            if df.empty:
                logging.warning("Data setelah drop NaN kosong.")
                return None
            logging.info(f"✅ Berhasil mengambil {len(df)} candle.")
            mt5.shutdown()
            return df
        except Exception as e:
            logging.error(f"❌ Terjadi error saat mengambil data: {e}", exc_info=True)
            mt5.shutdown()
            time.sleep(1)
            
    logging.error(f"Gagal total mengambil data untuk {symbol} di {tf} setelah {retry} kali percobaan.")
    return None

class DataCache:
    """Kelas sederhana untuk caching data (opsional, untuk optimasi)."""
    def __init__(self, expiry_seconds: int = 60):
        self._cache = {}
        self._lock = threading.Lock()
        self.expiry_seconds = expiry_seconds

    @lru_cache(maxsize=128)
    def get_data(self, symbol: str, timeframe: str, count: int, mt5_path: str) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{timeframe}_{count}"
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached['timestamp']) < self.expiry_seconds:
                logging.debug("Mengambil data dari cache untuk %s", cache_key)
                return cached['data']
            
            logging.debug("Data tidak ada di cache atau sudah kadaluarsa, mengambil data baru...")
            data = get_candlestick_data(symbol, timeframe, count, mt5_path)
            if data is not None:  # Hanya cache jika sukses
                self._cache[cache_key] = {'data': data, 'timestamp': now}
            return data