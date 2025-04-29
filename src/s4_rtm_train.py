# -*- coding: utf-8 -*-
# File: rtm_train_s4.py (con loop per tuning clauses/T)
import numpy as np
import pickle
import os
import time
import sys
import errno
from sklearn.metrics import mean_squared_error
from ecg_utils import calculate_snr, plot_signal_comparison
try:
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
except ImportError: print("Errore: pyTsetlinMachine non trovata."); exit()
except NameError: print("Errore: RegressionTsetlinMachine non definita."); exit()

print("--- RTM Training Script for S4 (Noise Estimation) ---")
print("--- TUNING LOOP FOR CLAUSES / T ---")

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = True
# -------------------------

# --- Definizione Percorsi ---
# ... (Blocco if/else invariato) ...
if RUNNING_ON_COLAB:
    # ... (definizioni Colab) ...
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data"); MODEL_DIR = os.path.join(GDRIVE_BASE, "models")
else:
    # ... (definizioni Locali) ...
    PROJECT_ROOT = "."; DATA_DIR = os.path.join(PROJECT_ROOT, "data"); MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RTM_S4_DATA_DIR = os.path.join(DATA_DIR, "rtm_processed_s4")
RTM_S4_MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "rtm_noise_estimator")
X_SCALER_PATH = os.path.join(RTM_S4_DATA_DIR, "rtm_s4_x_scaler.pkl")

print(f"INFO: Usando RTM S4 DATA DIR: {RTM_S4_DATA_DIR}")
print(f"INFO: Usando RTM S4 MODEL DIR: {RTM_S4_MODEL_OUTPUT_DIR}")
try: os.makedirs(RTM_S4_MODEL_OUTPUT_DIR, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {RTM_S4_MODEL_OUTPUT_DIR}.")
except NameError: pass

# --- Configurazioni RTM ---

# --- PARAMETRI FISSI DURANTE QUESTO TUNING ---
BASE_RTM_PARAMS = {
    "s": 5.0,                  # Fisso per ora
    "number_of_state_bits": 8,
    "boost_true_positive_feedback": 0,
}
RTM_EPOCHS = 20 # Fisso per ora
# FORCE_RETRAIN_RTM = True # Sempre True durante il tuning

# --- COMBINAZIONI CLAUSES/T DA PROVARE ---
PARAMS_TO_TRY = [
    {"number_of_clauses": 2000, "T": 1000}, # Riferimento precedente
    {"number_of_clauses": 4000, "T": 2000}, # Test 1.1
    {"number_of_clauses": 4000, "T": 3000}, # Test 1.2
    {"number_of_clauses": 6000, "T": 3000}, # Test 1.3
    {"number_of_clauses": 6000, "T": 4500}, # Test 1.4
]
# ---------------------------------------


# --- Main Script Logic ---
if __name__ == "__main__":

    # --- 1. Caricamento Dati (una sola volta) ---
    print("--- Fase 1: Caricamento Dati RTM S4 Processati ---")
    try:
        # ... (codice caricamento dati come prima) ...
        print(f"INFO: Caricamento da {RTM_S4_DATA_DIR}...")
        X_train_int = np.load(os.path.join(RTM_S4_DATA_DIR, "X_train_rtm_s4_int.npy"))
        y_train_noise_num = np.load(os.path.join(RTM_S4_DATA_DIR, "y_train_rtm_s4_noise_num.npy"))
        X_test_int = np.load(os.path.join(RTM_S4_DATA_DIR, "X_test_rtm_s4_int.npy"))
        y_test_clean_num = np.load(os.path.join(RTM_S4_DATA_DIR, "y_test_rtm_s4_clean_num.npy"))
        y_test_noisy_num = np.load(os.path.join(RTM_S4_DATA_DIR, "y_test_noisy_center_num.npy"))
        with open(X_SCALER_PATH, 'rb') as f: x_scaler = pickle.load(f)
        print("✅ Dati RTM S4 caricati.")
        print(f"  Train: X_int={X_train_int.shape}, y_noise={y_train_noise_num.shape}")
        print(f"  Test:  X_int={X_test_int.shape}, y_clean={y_test_clean_num.shape}, y_noisy={y_test_noisy_num.shape}")
    except FileNotFoundError as e: print(f"❌ Errore: File dati RTM S4 non trovato: {e.filename}."); exit()
    except Exception as e: print(f"❌ Errore caricamento dati RTM S4: {e}"); exit()

    results = {} # Dizionario per salvare i risultati

    # --- Loop di Tuning su Combinazioni Clausole/T ---
    for params_combo in PARAMS_TO_TRY:
        clauses = params_combo["number_of_clauses"]
        t_value = params_combo["T"]
        config_key = f"C{clauses}_T{t_value}" # Chiave per dizionario risultati
        print(f"\n{'='*10} INIZIO TUNING {config_key} {'='*10}")

        # Aggiorna i parametri RTM
        current_rtm_params = BASE_RTM_PARAMS.copy()
        current_rtm_params["number_of_clauses"] = clauses
        current_rtm_params["T"] = t_value

        # --- 2. Training Modello RTM ---
        print(f"--- Fase 2: Training Modello RTM ({config_key}) ---")
        rtm = None
        print("🔹 Inizializzazione e Training nuovo modello RTM...")
        try:
            rtm = RegressionTsetlinMachine(**current_rtm_params)
            print(f"  Training per {RTM_EPOCHS} epoche...")
            start_train_time = time.time()
            rtm.fit(X_train_int, y_train_noise_num, epochs=RTM_EPOCHS)
            end_train_time = time.time()
            print(f"✅ Training RTM completato in {end_train_time - start_train_time:.2f}s.")
        except Exception as e:
            print(f"❌ Errore durante training per {config_key}: {e}")
            results[config_key] = {'RMSE': np.nan, 'SNR': np.nan, 'Pred': None, 'Params': current_rtm_params}
            continue

        # --- 3. Valutazione Denoising sul Set di Test ---
        print(f"\n--- Fase 3: Valutazione Denoising RTM ({config_key}) ---")
        y_pred_denoised_num = None
        rmse_denoised = np.nan
        snr_output = np.nan
        noise_pred_num = None # Inizializza
        try:
            print("  Predicting noise values on test set...")
            start_pred_time = time.time()
            noise_pred_num = rtm.predict(X_test_int) # Predice il rumore
            end_pred_time = time.time()
            print(f"  Predizione rumore completata in {end_pred_time - start_pred_time:.2f}s.")
            print(f"  Range output predict() (rumore predetto): min={noise_pred_num.min():.4f}, max={noise_pred_num.max():.4f}")

            print("  Calculating denoised signal...")
            if len(y_test_noisy_num) == len(noise_pred_num):
                y_pred_denoised_num = y_test_noisy_num - noise_pred_num
            else: print("❌ Errore: Lunghezza segnale rumoroso != predizione rumore."); continue

            print("\n  Calculating denoising metrics...")
            if len(y_pred_denoised_num) == len(y_test_clean_num):
                mse_denoised = mean_squared_error(y_test_clean_num, y_pred_denoised_num)
                rmse_denoised = np.sqrt(mse_denoised)
                snr_output = calculate_snr(y_test_clean_num, y_pred_denoised_num)
                print(f"    RMSE (S4 Denoised vs Clean): {rmse_denoised:.6f}")
                print(f"    SNR Output (S4 Denoised vs Clean): {snr_output:.2f} dB")
            else: print("❌ Errore: Lunghezza segnale denoisato != target pulito.")

        except Exception as e:
            print(f"❌ Errore durante valutazione per {config_key}: {e}")

        # Salva i risultati per questa configurazione
        results[config_key] = {
            'RMSE': rmse_denoised,
            'SNR': snr_output,
            'Pred_Example': y_pred_denoised_num[:1000] if y_pred_denoised_num is not None else None,
            'Params': current_rtm_params
        }
        print(f"{'='*10} FINE TUNING {config_key} {'='*10}")


    # --- Stampa Riepilogo Risultati Tuning ---
    print("\n--- Riepilogo Risultati Tuning Clausole/T ---")
    print(" Config         | RMSE      | SNR (dB)")
    print("----------------|-----------|---------")
    best_config_key = None
    min_rmse = float('inf')
    # Ordina i risultati per RMSE (dal migliore al peggiore)
    sorted_configs = sorted(results.keys(), key=lambda k: results[k]['RMSE'] if not np.isnan(results[k]['RMSE']) else float('inf'))

    for config_key in sorted_configs:
        res = results[config_key]
        print(f" {config_key:<14} | {res['RMSE']:.6f} | {res['SNR']:7.2f}")
        if not np.isnan(res['RMSE']) and res['RMSE'] < min_rmse:
             min_rmse = res['RMSE']
             best_config_key = config_key

    if best_config_key is not None:
        print(f"\n🏆 Miglior Configurazione Trovata: {best_config_key} (RMSE: {results[best_config_key]['RMSE']:.6f}, SNR: {results[best_config_key]['SNR']:.2f} dB)")
        print(f"   Parametri: {results[best_config_key]['Params']}")

        # --- Visualizzazione per la configurazione migliore ---
        print(f"\n--- Visualizzazione per {best_config_key} ---")
        y_pred_best = results[best_config_key]['Pred_Example']
        if y_pred_best is not None:
             N_plot_samples = len(y_pred_best)
             if len(y_test_clean_num) >= N_plot_samples and len(y_test_noisy_num) >= N_plot_samples:
                 plot_signal_comparison(
                     clean=y_test_clean_num[:N_plot_samples],
                     noisy=y_test_noisy_num[:N_plot_samples],
                     denoised=y_pred_best,
                     title_prefix=f"ECG Denoising RTM S4 ({best_config_key}, Test Samples 0-{N_plot_samples-1})"
                 )
             else: print("⚠️ Dati test insufficienti per plot.")
        else: print("⚠️ Nessuna predizione valida salvata.")

        # --- SALVA IL MODELLO MIGLIORE (Opzionale) ---
        # Se vuoi salvare il modello corrispondente alla config migliore,
        # dovresti ri-addestrarlo un'ultima volta con quei parametri
        # e poi salvarne lo stato.
        # print("\nRi-addestramento e salvataggio modello migliore...")
        # best_params = results[best_config_key]['Params']
        # rtm_best = RegressionTsetlinMachine(**best_params)
        # rtm_best.fit(X_train_int, y_train_noise_num, epochs=RTM_EPOCHS)
        # try:
        #     with open(RTM_S4_MODEL_SAVE_PATH, "wb") as f: pickle.dump(rtm_best.get_state(), f)
        #     print(f"💾 Modello RTM S4 migliore salvato in: {RTM_S4_MODEL_SAVE_PATH}")
        # except Exception as e: print(f"❌ Errore salvataggio stato RTM S4 migliore: {e}")
        # ------------------------------------------

    else:
        print("\n❌ Nessun risultato valido ottenuto durante il tuning.")

    print("\n🏁 Script execution finished.")