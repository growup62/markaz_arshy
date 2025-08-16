# train_xgboost.py
#
# Deskripsi:
# Skrip ini melatih model XGBoost untuk setiap simbol berdasarkan data latih
# yang dihasilkan oleh generate_training_data.py.

import json
import logging
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRAINING_DATA_PATH = 'trade_feedback.json'
MODEL_OUTPUT_PATH = 'xgboost_model_{symbol}.json'

def load_training_data(filepath):
    """Memuat data latih dari file JSON."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error saat memuat data latih: {e}")
        return []

def train_model_for_symbol(symbol, trades):
    """Melatih model XGBoost untuk simbol tertentu."""
    if not trades:
        logging.warning(f"Tidak ada data untuk simbol {symbol}. Melewati pelatihan.")
        return None

    # Siapkan data dengan validasi
    try:
        X = np.array([trade['gng_input_features_on_signal'] for trade in trades])
        y = np.array([1 if trade['result'].upper() == "WIN" else 0 for trade in trades])
    except (KeyError, AttributeError) as e:
        logging.error(f"Format data tidak valid untuk {symbol}: {e}")
        return None

    # Validasi jumlah data dan rasio kelas
    if len(X) < 100:  # Minimum sampel yang dibutuhkan
        logging.warning(f"Data untuk {symbol} terlalu sedikit ({len(X)}). Minimal 100 sampel diperlukan.")
        return None

    class_ratio = np.mean(y)
    if class_ratio < 0.2 or class_ratio > 0.8:
        logging.warning(f"Ketidakseimbangan kelas terdeteksi untuk {symbol}. Win rate: {class_ratio:.2%}")

    # Split data dengan stratifikasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load hyperparameters dari hasil tuning jika ada
    try:
        with open(f'tuning_results_{symbol}.json', 'r') as f:
            best_params = json.load(f)
            logging.info(f"Menggunakan parameter hasil tuning untuk {symbol}")
    except FileNotFoundError:
        logging.warning(f"File tuning tidak ditemukan untuk {symbol}, menggunakan parameter default")
        best_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    # Inisialisasi dan latih model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        **best_params
    )

    # Early stopping untuk mencegah overfitting
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Evaluasi model secara menyeluruh
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Log hasil evaluasi
    logging.info(f"\n--- Evaluasi Model {symbol} ---")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")
    logging.info("\nClassification Report:")
    logging.info(f"\n{classification_report(y_test, y_pred)}")

    # Simpan model dan metrik
    model_path = MODEL_OUTPUT_PATH.format(symbol=symbol)
    model.save_model(model_path)
    
    # Simpan metrik evaluasi
    metrics_path = f'model_metrics_{symbol}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"Model dan metrik untuk {symbol} disimpan di {model_path} dan {metrics_path}")
    
    return model, metrics

def main():
    """Fungsi utama untuk melatih model XGBoost untuk semua simbol."""
    training_data = load_training_data(TRAINING_DATA_PATH)
    if not training_data:
        logging.error("Tidak ada data latih yang valid. Keluar.")
        return

    # Pisahkan data per simbol
    trades_by_symbol = {}
    for trade in training_data:
        symbol = trade['symbol']
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)
    
    # Latih model untuk setiap simbol
    for symbol, trades in trades_by_symbol.items():
        logging.info(f"Melatih model untuk simbol {symbol} ({len(trades)} trade)...")
        train_model_for_symbol(symbol, trades)

    logging.info("Pelatihan model selesai untuk semua simbol.")

if __name__ == '__main__':
    main()