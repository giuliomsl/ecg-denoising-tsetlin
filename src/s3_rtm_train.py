# -*- coding: utf-8 -*-
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

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = False
# -------------------------

# --- Definizione Percorsi ---
# ... (Blocco if/else invariato) ...
if RUNNING_ON_COLAB:
    print("INFO: Flag RUNNING_ON_COLAB impostato a True.")
    try: from google.colab import drive; drive.mount('/content/drive', force_remount=True); import time; time.sleep(5)
    except ImportError: print("ERRORE: Impossibile importare google.colab."); exit()
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data"); MODEL_DIR = os.path.join(GDRIVE_BASE, "models")
else:
    print("INFO: Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    PROJECT_ROOT = "."; DATA_DIR = os.path.join(PROJECT_ROOT, "data"); MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RTM_DATA_DIR = os.path.join(DATA_DIR, "rtm_processed")
RTM_MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "rtm_direct_denoiser")
# Non salviamo modelli individuali durante il tuning
X_SCALER_PATH = os.path.join(RTM_DATA_DIR, "rtm_x_scaler.pkl")
Y_SCALER_PATH = os.path.join(RTM_DATA_DIR, "rtm_y_clean_scaler.pkl")

print(f"INFO: Usando RTM_DATA_DIR: {RTM_DATA_DIR}")
print(f"INFO: Usando RTM_MODEL_OUTPUT_DIR: {RTM_MODEL_OUTPUT_DIR}")
try: os.makedirs(RTM_MODEL_OUTPUT_DIR, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {RTM_MODEL_OUTPUT_DIR}.")
except NameError: pass

# --- Configurazioni RTM ---

# --- PARAMETRI DA FISSARE DURANTE IL TUNING DI s ---
BASE_RTM_PARAMS = {
    "number_of_clauses": 2000, # Usa il valore precedente o quello che preferisci
    "T": 1000,                 # Fissa il T che sembrava migliore
    "number_of_state_bits": 8,
    "boost_true_positive_feedback": 0,
}
RTM_EPOCHS = 20
FORCE_RETRAIN_RTM = True # Sempre True durante il tuning

# --- VALORI DI s DA PROVARE ---
S_VALUES_TO_TRY = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0] # Range ottimale
# -----------------------------


# --- Main Script Logic ---
if __name__ == "__main__":

    # --- 1. Caricamento Dati (una sola volta) ---
    print("--- Fase 1: Caricamento Dati RTM Processati ---")
    try:
        # ... (codice caricamento dati come prima) ...
        print(f"INFO: Caricamento da {RTM_DATA_DIR}...")
        X_train_int = np.load(os.path.join(RTM_DATA_DIR, "X_train_rtm_int.npy"))
        y_train_int = np.load(os.path.join(RTM_DATA_DIR, "y_train_rtm_clean_int.npy"))
        X_test_int = np.load(os.path.join(RTM_DATA_DIR, "X_test_rtm_int.npy"))
        y_test_num = np.load(os.path.join(RTM_DATA_DIR, "y_test_rtm_clean_num.npy"))
        y_test_noisy_num = np.load(os.path.join(RTM_DATA_DIR, "y_test_noisy_center_num.npy"))
        with open(X_SCALER_PATH, 'rb') as f: x_scaler = pickle.load(f)
        with open(Y_SCALER_PATH, 'rb') as f: y_scaler = pickle.load(f)
        print("✅ Dati RTM caricati.")
        print(f"  Train: X_int={X_train_int.shape}, y_int={y_train_int.shape}")
        print(f"  Test:  X_int={X_test_int.shape}, y_num={y_test_num.shape}, y_noisy={y_test_noisy_num.shape}")
    except FileNotFoundError as e: print(f"❌ Errore: File dati RTM non trovato: {e.filename}. Esegui prima 'preprocess_rtm.py'."); exit()
    except Exception as e: print(f"❌ Errore caricamento dati RTM: {e}"); exit()

    results = {} # Dizionario per salvare i risultati per ogni s

    # --- Loop di Tuning su s ---
    for s_value in S_VALUES_TO_TRY:
        print(f"\n{'='*10} INIZIO TUNING s = {s_value} {'='*10}")

        # Aggiorna i parametri RTM con s corrente
        current_rtm_params = BASE_RTM_PARAMS.copy()
        current_rtm_params["s"] = s_value

        # --- 2. Training Modello RTM ---
        print(f"--- Fase 2: Training Modello RTM (s={s_value}) ---")
        rtm = None
        print("🔹 Inizializzazione e Training nuovo modello RTM...")
        try:
            rtm = RegressionTsetlinMachine(**current_rtm_params)
            print(f"  Training per {RTM_EPOCHS} epoche...")
            start_train_time = time.time()
            rtm.fit(X_train_int, y_train_int, epochs=RTM_EPOCHS)
            end_train_time = time.time()
            print(f"✅ Training RTM completato in {end_train_time - start_train_time:.2f}s.")
        except Exception as e:
            print(f"❌ Errore durante training per s={s_value}: {e}")
            results[s_value] = {'RMSE': np.nan, 'SNR': np.nan, 'Pred': None}
            continue

        # --- 3. Valutazione Denoising sul Set di Test ---
        print(f"\n--- Fase 3: Valutazione Denoising RTM (s={s_value}) ---")
        y_pred_rtm_num = None
        rmse_denoised = np.nan
        snr_output = np.nan
        try:
            print("  Predicting scaled values on test set...")
            start_pred_time = time.time()
            y_pred_rtm_scaled = rtm.predict(X_test_int)
            end_pred_time = time.time()
            print(f"  Predizione (scalata) completata in {end_pred_time - start_pred_time:.2f}s.")
            print(f"  Range output predict() (scalato): min={y_pred_rtm_scaled.min()}, max={y_pred_rtm_scaled.max()}")

            print("  Inverting scaling...")
            y_pred_rtm_num = y_scaler.inverse_transform(y_pred_rtm_scaled.reshape(-1, 1)).flatten()
            print(f"  Range output numerico finale: min={y_pred_rtm_num.min():.4f}, max={y_pred_rtm_num.max():.4f}")

            print("\n  Calculating denoising metrics...")
            if len(y_pred_rtm_num) == len(y_test_num):
                mse_denoised = mean_squared_error(y_test_num, y_pred_rtm_num)
                rmse_denoised = np.sqrt(mse_denoised)
                snr_output = calculate_snr(y_test_num, y_pred_rtm_num)
                print(f"    RMSE (RTM Denoised vs Clean): {rmse_denoised:.6f}")
                print(f"    SNR Output (RTM Denoised vs Clean): {snr_output:.2f} dB")
            else: print("❌ Errore: Lunghezza predizioni RTM != target numerici.")

        except Exception as e:
            print(f"❌ Errore durante valutazione per s={s_value}: {e}")

        # Salva i risultati per questo s
        results[s_value] = {
            'RMSE': rmse_denoised,
            'SNR': snr_output,
            'Pred_Example': y_pred_rtm_num[:1000] if y_pred_rtm_num is not None else None
        }
        print(f"{'='*10} FINE TUNING s = {s_value} {'='*10}")


    # --- Stampa Riepilogo Risultati Tuning ---
    print("\n--- Riepilogo Risultati Tuning Parametro s ---")
    print(f"(Parametri Fissi: Clauses={BASE_RTM_PARAMS['number_of_clauses']}, T={BASE_RTM_PARAMS['T']})")
    print(" s      | RMSE      | SNR (dB)")
    print("--------|-----------|---------")
    best_s = None
    min_rmse = float('inf')
    for s_value in sorted(results.keys()):
        res = results[s_value]
        print(f" {s_value:<6.1f} | {res['RMSE']:.6f} | {res['SNR']:7.2f}") # Formatta s con una cifra decimale
        if not np.isnan(res['RMSE']) and res['RMSE'] < min_rmse:
             min_rmse = res['RMSE']
             best_s = s_value

    if best_s is not None:
        print(f"\n🏆 Miglior s trovato: {best_s:.1f} (RMSE: {results[best_s]['RMSE']:.6f}, SNR: {results[best_s]['SNR']:.2f} dB)")

        # --- Visualizzazione per s migliore ---
        print(f"\n--- Visualizzazione per s = {best_s:.1f} ---")
        y_pred_best_s = results[best_s]['Pred_Example']
        if y_pred_best_s is not None:
             N_plot_samples = len(y_pred_best_s)
             if len(y_test_num) >= N_plot_samples and len(y_test_noisy_num) >= N_plot_samples:
                 plot_signal_comparison(
                     clean=y_test_num[:N_plot_samples],
                     noisy=y_test_noisy_num[:N_plot_samples],
                     denoised=y_pred_best_s,
                     title_prefix=f"ECG Denoising RTM (s={best_s:.1f}, T={BASE_RTM_PARAMS['T']}, Test Samples 0-{N_plot_samples-1})" # Titolo più informativo
                 )
             else: print("⚠️ Dati di test insufficienti per plottare il segmento completo.")
        else: print("⚠️ Nessuna predizione valida salvata per il s migliore.")
    else:
        print("\n❌ Nessun risultato valido ottenuto durante il tuning di s.")

    print("\n🏁 Script execution finished.")