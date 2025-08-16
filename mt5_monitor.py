import MetaTrader5 as mt5
import requests
import time
from datetime import datetime
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_mt5():
    """Inisialisasi koneksi ke MetaTrader5."""
    if not mt5.initialize():
        logger.error("Gagal menginisialisasi MT5. Pastikan terminal MT5 berjalan dan login valid.")
        return False
    logger.info("MT5 berhasil diinisialisasi.")
    return True

def on_trade_close(position):
    """Dipanggil saat trade ditutup (manual, SL, atau TP)."""
    result = "win" if position.profit >= 0 else "loss"
    signal_id = position.comment if position.comment else "unknown"
    context = {}
    try:
        response = requests.get(f"http://localhost:5000/api/get_signal_context/{signal_id}")
        if response.status_code == 200:
            context = response.json().get("context", {})
        else:
            logger.warning(f"Gagal mengambil konteks untuk signal_id {signal_id}: {response.text}")
    except Exception as e:
        logger.warning(f"Error mengambil konteks: {e}")
    
    feedback_data = {
        "signal_id": signal_id,
        "result": result,
        "score": position.magic / 1000.0 if position.magic else 0.0,
        "score_components": context.get("score_components", []),
        "profile_name": context.get("profile_name", "scalping"),
        "symbol": position.symbol,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        response = requests.post("http://localhost:5000/api/feedback_trade", json=feedback_data)
        if response.status_code == 200:
            logger.info(f"Feedback trade {position.ticket} dikirim: {result}, Data: {feedback_data}")
        else:
            logger.error(f"Gagal mengirim feedback: {response.text}")
    except Exception as e:
        logger.error(f"Error saat mengirim feedback: {e}")

def monitor_trades():
    """Monitor posisi terbuka dan deteksi penutupan."""
    if not initialize_mt5():
        logger.error("Monitor trades dihentikan karena gagal inisialisasi MT5.")
        return
    
    last_positions = set()
    while True:
        try:
            positions = mt5.positions_get()
            if positions is None:
                logger.error("Gagal mengambil posisi. Mungkin MT5 tidak terkoneksi.")
                time.sleep(1)
                continue
            
            current_positions = {pos.ticket for pos in positions}
            closed_positions = last_positions - current_positions
            
            for ticket in closed_positions:
                history = mt5.history_deals_get(position=ticket)
                if history:
                    on_trade_close(history[-1])
                else:
                    logger.warning(f"Tidak ada history deal untuk ticket {ticket}")
            
            last_positions = current_positions
            time.sleep(1)  # Jeda 1 detik
        except Exception as e:
            logger.error(f"Error dalam loop monitor_trades: {e}")
            time.sleep(1)

if __name__ == "__main__":
    logger.info("Memulai monitor_trades pada %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    monitor_trades()