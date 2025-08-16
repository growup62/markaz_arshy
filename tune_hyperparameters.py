# tune_hyperparameters.py
#
# Deskripsi:
# Versi ini disesuaikan untuk strategi SCALPING.
# Skrip ini melakukan 'Hyperparameter Tuning' menggunakan RandomizedSearchCV
# untuk menemukan kombinasi pengaturan terbaik bagi model XGBoost Anda secara
# lebih cepat dan efisien.

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import re
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tune_model_for_symbol(symbol_data, symbol_name):
    """
    Fungsi untuk melakukan tuning hyperparameter untuk satu simbol spesifik
    menggunakan RandomizedSearchCV yang lebih cepat.
    """
    logging.info(f"--- Memulai Tuning Hyperparameter (Scalping) untuk: {symbol_name} ---")

    # --- Persiapan Data ---
    features = [record['gng_input_features_on_signal'] for record in symbol_data]
    labels = [1 if record['result'].upper() == "WIN" else 0 for record in symbol_data]

    X = np.array(features)
    y = np.array(labels)

    if len(X) < 100:
        logging.warning(f"Data untuk {symbol_name} ({len(X)} baris) tidak cukup untuk tuning yang efektif.")
        return

    # Split data dengan stratifikasi untuk menjaga keseimbangan kelas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Parameter grid yang lebih adaptif untuk scalping
    param_dist = {
        'max_depth': randint(2, 6),
        'learning_rate': uniform(0.005, 0.195), # Range: 0.005 - 0.2
        'n_estimators': randint(50, 300),
        'subsample': uniform(0.5, 0.5), # Range: 0.5 - 1.0
        'colsample_bytree': uniform(0.5, 0.5), # Range: 0.5 - 1.0
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5)
    }

    # Inisialisasi model dengan parameter dasar untuk scalping
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['logloss', 'auc', 'error'],
        use_label_encoder=False,
        tree_method='hist'  # Lebih cepat untuk dataset besar
    )

    # Implementasi RandomizedSearchCV dengan cross-validation yang lebih ketat
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=param_dist,
        n_iter=75,  # Lebih banyak iterasi untuk hasil lebih baik
        scoring=['accuracy', 'f1', 'roc_auc'],
        refit='f1',  # Optimisasi berdasarkan F1-score
        cv=5,  # 5-fold cross validation
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )

    # Tambahkan early stopping dalam fitting
    eval_set = [(X_test, y_test)]
    
    logging.info(f"Memulai RandomizedSearchCV untuk {symbol_name}...")
    random_search.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=20,
        verbose=True
    )

    # Simpan hasil tuning ke file JSON
    tuning_results = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': {
            metric: random_search.cv_results_[f'mean_test_{metric}'][random_search.best_index_]
            for metric in ['accuracy', 'f1', 'roc_auc']
        }
    }

    with open(f'tuning_results_{symbol_name}.json', 'w') as f:
        json.dump(tuning_results, f, indent=4)

    logging.info(f"Tuning selesai untuk {symbol_name}. Hasil disimpan ke tuning_results_{symbol_name}.json")
    return random_search.best_params_


def main():
    """
    Fungsi utama untuk memuat data dan mengorkestrasi tuning per simbol.
    """
    feedback_file_path = "trade_feedback_generated.json"

    try:
        with open(feedback_file_path, "r") as f:
            trade_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Gagal memuat '{feedback_file_path}': {e}.")
        return

    logging.info(f"Berhasil memuat {len(trade_data)} total record data latih.")
    df = pd.DataFrame(trade_data)
    df['clean_symbol'] = df['symbol'].apply(lambda s: re.sub(r'[cm]$', '', s).upper())
    
    grouped = df.groupby('clean_symbol')
    logging.info(f"Data akan di-tuning untuk simbol: {list(grouped.groups.keys())}")

    for symbol_name, group_df in grouped:
        symbol_records = group_df.to_dict('records')
        tune_model_for_symbol(symbol_records, symbol_name)

if __name__ == '__main__':
    main()
