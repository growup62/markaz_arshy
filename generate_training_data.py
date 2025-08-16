# generate_training_data.py
#
# Deskripsi:
# Skrip ini membaca file riwayat trading yang SUDAH DIBERSIHKAN ('history_cleaned.csv'),
# lalu merekonstruksi fitur-fitur pasar pada saat setiap trade dibuka.
# Hasilnya adalah file JSON yang siap digunakan untuk melatih model AI (XGBoost).
#
# Cara Penggunaan:
# 1. Pastikan Anda sudah menjalankan 'clean_history_csv.py' terlebih dahulu.
# 2. Jalankan skrip ini dari terminal: python generate_training_data.py

import pandas as pd
import numpy as np
import json
import logging
import re
from datetime import datetime

# Impor fungsi-fungsi yang kita butuhkan dari skrip lain
from data_fetching import get_candlestick_data
from technical_indicators import (
    detect_structure,
    detect_order_blocks_multi,
    detect_fvg_multi,
    detect_engulfing,
    detect_pinbar,
    get_daily_high_low,
    get_pivot_points,
    extract_features_full,
    detect_liquidity_sweep
)

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Membaca file yang sudah bersih ---
HISTORY_FILE_PATH = 'history_cleaned.csv'
MT5_TERMINAL_PATH = r"C:\\Program Files\\ExclusiveMarkets MetaTrader5\\terminal64.exe"
OUTPUT_JSON_PATH = 'trade_feedback_generated1.json'
CANDLES_TO_FETCH = 50000

def reconstruct_features_for_trade(trade_row):
    """
    Mengambil data historis dan merekonstruksi fitur untuk satu baris trade.
    """
    try:
        # --- PERUBAHAN: Menggunakan nama kolom yang sudah distandarkan ---
        open_time = pd.to_datetime(trade_row['open_time'])
        symbol = trade_row['symbol']
        # Membersihkan string profit dari pemisah ribuan sebelum konversi
        profit_str = str(trade_row['profit']).replace('.', '')
        profit = float(profit_str)
        ticket = int(trade_row['ticket']) # <-- Menggunakan 'ticket' bukan 'position'
        price = float(trade_row['open_price'])

        logging.info(f"Memproses trade #{ticket} ({symbol}) pada {open_time}...")
        clean_symbol = re.sub(r'[cm]$', '', symbol).upper()

        timeframes_to_try = ["M1", "M5", "M15", "H1"]
        df_history = None
        timeframe_found = None

        for tf in timeframes_to_try:
            logging.info(f"Mencoba mengambil data untuk timeframe {tf}...")
            temp_df = get_candlestick_data(clean_symbol, tf, CANDLES_TO_FETCH, MT5_TERMINAL_PATH)
            # Check if we got data AND if there are candles before the trade's open time
            if temp_df is not None and not temp_df[temp_df['time'] < open_time].empty:
                df_history = temp_df
                timeframe_found = tf
                logging.info(f"Data ditemukan dan valid untuk timeframe {tf}.")
                break # Found valid data, exit loop
            else:
                logging.warning(f"Tidak ada data valid untuk timeframe {tf}.")

        if df_history is None:
            logging.error(f"Gagal total: Tidak dapat menemukan data historis yang cukup untuk trade #{ticket} pada timeframe manapun. Dilewati.")
            return None

        # Filter dataframe hingga tepat sebelum waktu trade dibuka
        df_snapshot = df_history[df_history['time'] < open_time].tail(CANDLES_TO_FETCH)
        if len(df_snapshot) < 50:
            logging.warning(f"Data snapshot tidak cukup untuk analisis pada trade #{ticket} (timeframe {timeframe_found}). Dilewati.")
            return None

        structure, _ = detect_structure(df_snapshot)
        order_blocks = detect_order_blocks_multi(df_snapshot, structure_filter=structure)
        fvg_zones = detect_fvg_multi(df_snapshot)
        patterns = detect_engulfing(df_snapshot) + detect_pinbar(df_snapshot)
        boundary = get_daily_high_low(df_snapshot)
        pivot = get_pivot_points(df_snapshot)
        liquidity_sweeps = detect_liquidity_sweep(df_snapshot)

        # ======== INI BAGIAN YANG DIPERBAIKI ========
        features_vector = extract_features_full(
            df_snapshot, structure, order_blocks, fvg_zones, patterns, boundary, pivot
        )
        # ===========================================
        result = "WIN" if profit > 0 else "LOSS"

        return {
            "symbol": symbol,
            "tf": timeframe_found, # Use the timeframe where data was found
            "entry": price,
            "result": result,
            "pnl": profit,
            "gng_input_features_on_signal": features_vector.tolist(),
            "ticket": ticket
        }

    except Exception as e:
        ticket_for_log = trade_row.get('ticket', 'UNKNOWN')
        logging.error(f"Error saat memproses trade #{ticket_for_log}: {e}", exc_info=False)
        return None

def main():
    """
    Fungsi utama untuk membaca file riwayat yang sudah bersih dan menghasilkan data latih.
    """
    try:
        history_df = pd.read_csv(HISTORY_FILE_PATH)
        logging.info(f"Berhasil membaca {len(history_df)} baris data yang sudah bersih dari '{HISTORY_FILE_PATH}'.")
    except FileNotFoundError:
        logging.error(f"File riwayat '{HISTORY_FILE_PATH}' tidak ditemukan. Silakan jalankan 'clean_history_csv.py' terlebih dahulu.")
        return
    except Exception as e:
        logging.error(f"Gagal membaca file CSV yang sudah bersih: {e}")
        return

    # --- PERUBAHAN: Memeriksa kolom 'ticket' bukan 'position' ---
    required_cols = ['open_time', 'ticket', 'symbol', 'type', 'open_price', 'profit']
    missing_cols = [col for col in required_cols if col not in history_df.columns]
    if missing_cols:
        logging.error(f"FATAL: Kolom yang dibutuhkan tidak ditemukan: {missing_cols}")
        return

    # Data sudah bersih, jadi kita hanya filter untuk trade yang sudah ditutup
    closed_trades_df = history_df[history_df['type'].isin(['buy', 'sell'])].copy()
    logging.info(f"Ditemukan {len(closed_trades_df)} trade yang sudah ditutup untuk diproses.")

    training_data = []
    for index, row in closed_trades_df.iterrows():
        reconstructed_data = reconstruct_features_for_trade(row)
        if reconstructed_data:
            training_data.append(reconstructed_data)

    if not training_data:
        logging.warning("Tidak ada data latih yang berhasil dibuat. Periksa kembali file riwayat atau log error.")
        return

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(training_data, f, indent=2)

    logging.info(f"Berhasil! {len(training_data)} data latih telah dibuat dan disimpan di '{OUTPUT_JSON_PATH}'.")
    logging.info("Anda sekarang bisa menggunakan file ini untuk melatih model XGBoost.")

if __name__ == '__main__':
    main()