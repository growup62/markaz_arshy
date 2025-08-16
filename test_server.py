# test_server.py

import os
import requests
from dotenv import load_dotenv

# 1. Muat file .env dari direktori saat ini
load_dotenv()

# 2. Baca kunci rahasia dari environment, sama seperti yang dilakukan server
SECRET_KEY = os.environ.get('INTERNAL_SECRET_KEY')

print("--- Memulai Tes Diagnostik ---")

if not SECRET_KEY:
    print("❌ GAGAL: Tidak dapat menemukan INTERNAL_SECRET_KEY di file .env")
else:
    print(f"✅ Ditemukan kunci di .env: '{SECRET_KEY}'")
    
    # 3. Siapkan data untuk dikirim (payload)
    url = "http://127.0.0.1:5000/api/internal/submit_signal"
    payload = {
        "symbol": "XAUUSD_TEST",
        "signal_json": {"BuyEntry": 2300, "SL": 2290, "TP": 2320},
        "api_key": "test_api_key",
        "secret_key": SECRET_KEY, # Menggunakan kunci yang baru saja kita baca
        "order_type": "BUY"
    }
    
    print("🚀 Mengirim sinyal ke server...")
    
    # 4. Kirim permintaan ke server Anda yang sedang berjalan
    try:
        response = requests.post(url, json=payload)
        print(f"📬 Respons Server (Status Code: {response.status_code}):")
        print(response.json())
    except requests.exceptions.ConnectionError as e:
        print(f"❌ GAGAL KONEKSI: Tidak bisa terhubung ke {url}.")
        print("   Pastikan server app_rev4.py Anda sedang berjalan.")
    except Exception as e:
        print(f"❌ Terjadi error tak terduga: {e}")

print("--- Tes Selesai ---")