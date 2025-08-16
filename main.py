# main.py

import logging
import time
import xgboost as xgb
from datetime import datetime
from typing import Dict, Any, Tuple
import json
import os

# Dependensi yang sudah disinkronkan
from signal_generator import analyze_tf_opportunity, handle_opportunity
from log_handler import WebServerHandler
from learning import analyze_and_adapt_profiles

def load_config(filepath: str = "config.json") -> Dict[str, Any]:
    """Memuat konfigurasi dari file JSON dan mengatur logging."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        log_config = config.get('logging', {})
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        log_level = getattr(logging, log_config.get('level', 'INFO').upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=log_format, force=True)

        if config.get('global_settings', {}).get('server_url'):
            root_logger = logging.getLogger()
            log_server_url = config['global_settings']['server_url'].replace('/submit_signal', '')
            web_handler = WebServerHandler(url=log_server_url)
            web_handler.setFormatter(logging.Formatter(log_format))
            if not any(isinstance(h, WebServerHandler) for h in root_logger.handlers):
                root_logger.addHandler(web_handler)

        logging.info("✅ Konfigurasi berhasil dimuat dari %s.", filepath)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.critical("❌ KRITIS: File konfigurasi '%s' tidak ditemukan atau rusak. Bot berhenti.", filepath, e)
        raise SystemExit(1)

def initialize_models(config: Dict[str, Any]) -> Dict[str, xgb.XGBClassifier]:
    """Inisialisasi semua model XGBoost untuk simbol yang relevan."""
    xgb_models = {}
    symbols = [s for s in config if s.isupper()]

    logging.info("--- 🧠 Menginisialisasi Model AI (XGBoost) ---")
    for symbol in symbols:
        try:
            model_path = f"xgboost_model_{symbol}.json"
            if not os.path.exists(model_path):
                logging.warning("⚠️ Model file %s tidak ditemukan. Skip model untuk %s.", model_path, symbol)
                xgb_models[symbol] = None
                continue
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            xgb_models[symbol] = model
            logging.info("👍 Model AI untuk %s berhasil dimuat.", symbol)
        except Exception as e:
            logging.warning("⚠️ Gagal memuat model AI untuk %s: %s. Analisis akan berjalan tanpa validasi AI.", symbol, e)
            xgb_models[symbol] = None
    return xgb_models

def process_symbol(
    symbol: str, symbol_config: Dict[str, Any], global_config: Dict[str, Any],
    models: Dict[str, Any], adapted_weights_per_profile: Dict[str, Any],
    signal_cooldown: Dict[Tuple[str, str], datetime]
):
    """Menjalankan seluruh siklus analisis untuk satu simbol."""
    logging.info("➡️ Menganalisis Simbol: '%s'", symbol)
    for profile_name, profile_config in symbol_config.get("strategy_profiles", {}).items():
        if not profile_config.get("enabled", False):
            continue

        cd_key = (symbol, profile_name)
        if cd_key in signal_cooldown:
            cooldown_minutes = profile_config.get('signal_cooldown_minutes', 1)
            elapsed = (datetime.now() - signal_cooldown[cd_key]).total_seconds() / 60
            if elapsed < cooldown_minutes:
                logging.debug("⏳ Profil '%s' untuk %s masih dalam cooldown.", profile_name, symbol)
                continue

        logging.info("♟️ Profil Aktif: '%s' | Simbol: '%s'", profile_name, symbol)
        adapted_weights = adapted_weights_per_profile.get(symbol, {}).get(profile_name, symbol_config.get("base_weights", {}))
        
        for tf in profile_config.get('timeframes', []):
            logging.debug("--- Menganalisis Timeframe: %s ---", tf)
            try:
                opp = analyze_tf_opportunity(
                    symbol=symbol,
                    tf=tf,
                    mt5_path=global_config['mt5_terminal_path'],
                    weights=adapted_weights,
                    confidence_threshold=profile_config.get('confidence_threshold', 5.0)
                )
                if opp and opp.get('signal') != "WAIT":
                    logging.info("💡 Peluang ditemukan di %s | %s! Memproses lebih lanjut...", symbol, tf)
                    if handle_opportunity(opp, symbol, tf, symbol_config, global_config, models.get(symbol), profile_name, signal_cooldown):
                        logging.info("Sinyal berhasil dikirim, berhenti menganalisis timeframe lain untuk profil '%s'.", profile_name)
                        break 
            except Exception as e:
                logging.error("❌ Terjadi Error saat menganalisis %s|%s|%s: %s", profile_name, symbol, tf, e, exc_info=True)
                continue

def main():
    """Fungsi utama untuk menjalankan loop bot."""
    config = load_config()
    global_config = config.get('global_settings', {})
    xgb_models = initialize_models(config)
    signal_cooldown: Dict[Tuple[str, str], datetime] = {}
    
    logging.info("==================================================")
    logging.info(" Bot Trading AI v5.0 (Esteh AI) Siap Beraksi! 🚀 ")
    logging.info("==================================================")

    try:
        while True:
            logging.info("--- 👨‍🏫 Memulai Siklus Pembelajaran Adaptif ---")
            adapted_weights = analyze_and_adapt_profiles(config)
            logging.info("--- ✅ Siklus Pembelajaran Adaptif Selesai ---")

            logging.info("--- 🔎 Memulai Siklus Analisis Pasar Baru ---")
            symbols_to_process = [s for s in config if s.isupper()]
            
            for symbol in symbols_to_process:
                process_symbol(symbol, config[symbol], global_config, xgb_models, adapted_weights, signal_cooldown)

            sleep_duration = int(global_config.get('main_loop_sleep_seconds', 20))
            logging.info("--- Siklus Selesai. Istirahat selama %d detik... ---\n", sleep_duration)
            time.sleep(sleep_duration)
    except KeyboardInterrupt:
        logging.info("🔴 Perintah berhenti diterima (Ctrl+C). Bot akan dimatikan.")
    finally:
        logging.info("🏁 Aplikasi Selesai.")

if __name__ == '__main__':
    main()