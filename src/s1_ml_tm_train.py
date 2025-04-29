# -*- coding: utf-8 -*-
import numpy as np
import pickle
from s1_ml_tm_class import MultiLabelTsetlinMachine # Usa il tuo wrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, hamming_loss
from ecg_dataset import ECGSamplewiseDataset
from ecg_utils import calculate_snr, plot_signal_comparison
from s1_ml_binarization import load_binarization_info # Funzione per caricare info
import os
import time
import sys
import errno

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = False # <--- MODIFICA QUESTA RIGA MANUALMENTE
# -------------------------

# --- Definizione Percorsi basata sul Flag ---
# (Blocco if/else per definire DATA_DIR, MODEL_DIR, SAMPLE_DATA_DIR,
#  MODEL_OUTPUT_DIR, BIN_INFO_PATH come prima)
# ... (codice identico a prima per la definizione dei percorsi) ...
if RUNNING_ON_COLAB:
    print("INFO: Flag RUNNING_ON_COLAB impostato a True.")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
             print("INFO: Montaggio Google Drive...")
             drive.mount('/content/drive', force_remount=True)
             import time
             time.sleep(5)
        else:
             print("INFO: Google Drive già montato.")
    except ImportError: print("ERRORE: Impossibile importare google.colab."); exit()
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    REPO_NAME = "denoising_ecg"
    PROJECT_ROOT = f"/content/{REPO_NAME}/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data")
    MODEL_DIR = os.path.join(GDRIVE_BASE, "models")
else:
    print("INFO: Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    PROJECT_ROOT = "."
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samplewise/")
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "multi_tm_denoiser/")
BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")
# Definisci qui anche NOISY_ECG_DIR se serve per caricare y_noisy_numeric_samples
NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/") # Aggiunto per coerenza

print(f"INFO: Usando SAMPLE_DATA_DIR: {SAMPLE_DATA_DIR}")
print(f"INFO: Usando MODEL_OUTPUT_DIR: {MODEL_OUTPUT_DIR}")

try:
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {e.filename}.")
except NameError: pass

# --- Configurazione ---
# Parametri Binarizzazione (devono corrispondere a preprocessing)
BINARIZATION_METHOD = "combined"
N_BINS_QUANT = 13 # <--- IMPOSTA LO STESSO VALORE DI preprocessing_multilabel.py
BINARIZATION_PARAMS = {"quant_n_bins": N_BINS_QUANT} # Solo per verifica caricamento info

# --- Calcola NUM_OUTPUT_BITS dinamicamente ---
# 2 bit da mean_filter + 1 bit da gradient + N_BINS_QUANT bit da quantizzazione
NUM_OUTPUT_BITS = 2 + 1 + N_BINS_QUANT
print(f"INFO: Calcolato NUM_OUTPUT_BITS = {NUM_OUTPUT_BITS} (per n_bins={N_BINS_QUANT})")
# ---------------------------------------------

CONTEXT_K = 10

# Parametri Tsetlin Machine
TM_PARAMS = {
    "number_of_clauses": 2000, "T": 1600, "s": 3.0,
    "number_of_state_bits": 8, "boost_true_positive_feedback": 1,
    "indexed": True,
}
EPOCHS_PER_LABEL = 10
USE_VALIDATION_FOR_BEST_STATE = True
FORCE_RETRAIN = True

# --- Percorsi (definiti dal blocco if/else RUNNING_ON_COLAB) ---
# ... (come prima) ...
MODEL_FILENAME = "noise_tm_manager.pkl"
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME) # Usa MODEL_OUTPUT_DIR definito prima

# --- Funzione di Decodifica Inversa (MODIFICATA) ---
def inverse_binarize_combined(y_pred_bin, binarization_info):
    """
    Riconverte l'output binario predetto in un valore numerico usando
    una mappa precalcolata (pattern -> media valore pulito).
    """
    if binarization_info is None \
       or 'pattern_to_mean_map' not in binarization_info \
       or 'global_mean_clean' not in binarization_info \
       or 'n_bins_quant' not in binarization_info: # Aggiunto controllo n_bins
        print("⚠️ Errore: Info binarizzazione mancanti ('pattern_to_mean_map', 'global_mean_clean', 'n_bins_quant').")
        return np.zeros(y_pred_bin.shape[0])

    pattern_map = binarization_info['pattern_to_mean_map']
    fallback_value = binarization_info['global_mean_clean']
    n_bins_quant_info = binarization_info['n_bins_quant'] # n_bins usato nel preprocessing
    num_samples = y_pred_bin.shape[0]
    y_pred_numeric = np.zeros(num_samples)
    not_found_count = 0

    # Calcola il numero totale di bit atteso in base a n_bins_quant_info
    expected_total_bits = 2 + 1 + n_bins_quant_info

    if y_pred_bin.shape[1] != expected_total_bits:
         print(f"⚠️ Attenzione (inverse_binarize): y_pred_bin ha {y_pred_bin.shape[1]} colonne, ma attesi {expected_total_bits} (basato su n_bins={n_bins_quant_info}).")
         # Potrebbe essere necessario aggiustare gli indici o fermarsi

    # --- Indice di inizio per i bit di quantizzazione ---
    # Rimane 3 perché i primi 3 bit sono mean(2) + gradient(1)
    quant_bits_start_idx = 3
    # ---------------------------------------------------

    for i in range(num_samples):
        pred_pattern_tuple = tuple(y_pred_bin[i])
        y_pred_numeric[i] = pattern_map.get(pred_pattern_tuple, fallback_value)
        if pred_pattern_tuple not in pattern_map:
            not_found_count += 1

    if not_found_count > 0:
        print(f"INFO (inverse_binarize): Pattern non trovato per {not_found_count}/{num_samples} campioni. Usato fallback.")

    return y_pred_numeric

# --- Main Script Logic ---
if __name__ == "__main__":

    # --- 1. Caricamento Dati ---
    print("--- Fase 1: Caricamento Dati ---")
    print("🔹 Loading sample-centric data (binarized)...")
    try:
        dataset = ECGSamplewiseDataset(SAMPLE_DATA_DIR)
        X_samples_bin, y_samples_bin = dataset.get_data()
        dataset.summary()
    except FileNotFoundError: print(f"❌ Errore: File non trovati in '{SAMPLE_DATA_DIR}'."); exit()
    except Exception as e: print(f"❌ Errore caricamento: {e}"); exit()

    if y_samples_bin.shape[1] != NUM_OUTPUT_BITS:
        print(f"❌ Errore Critico: NUM_OUTPUT_BITS ({NUM_OUTPUT_BITS}) != colonne y_samples_bin ({y_samples_bin.shape[1]})!")
        exit()

    print("🔹 Loading original numeric data for evaluation...")
    try:
        y_clean_numeric_samples = np.load(os.path.join(SAMPLE_DATA_DIR, "y_clean_numeric_samples.npy"))
        y_noisy_numeric_samples = np.load(os.path.join(SAMPLE_DATA_DIR, "y_noisy_numeric_samples.npy"))
    except FileNotFoundError as e: print(f"❌ Errore: File numerici '{e.filename}' non trovato."); exit()
    if X_samples_bin.shape[0] != y_clean_numeric_samples.shape[0] or \
       X_samples_bin.shape[0] != y_noisy_numeric_samples.shape[0]:
         print("❌ Errore: Incoerenza campioni binari/numerici."); exit()

    print("🔹 Loading binarization info...")
    binarization_info = load_binarization_info(BIN_INFO_PATH)
    if binarization_info is None or 'pattern_to_mean_map' not in binarization_info:
        print("❌ Errore: Mappa inversione non trovata in binarization_info.pkl."); exit()
    loaded_n_bins = binarization_info.get("n_bins_quant")
    if loaded_n_bins is None: print("⚠️ 'n_bins_quant' non trovato in info.")
    elif loaded_n_bins != N_BINS_QUANT: # Confronta con la costante definita qui
         print(f"⚠️ n_bins info ({loaded_n_bins}) != n_bins config ({N_BINS_QUANT})")

    # --- 2. Split Train/Validation ---
    print("\n--- Fase 2: Split Dati Train/Validation ---")
    indices = np.arange(X_samples_bin.shape[0])
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    X_train_bin, X_val_bin = X_samples_bin[train_indices], X_samples_bin[val_indices]
    y_train_bin, y_val_bin = y_samples_bin[train_indices], y_samples_bin[val_indices]
    y_train_clean_num = y_clean_numeric_samples[train_indices]
    y_val_clean_num = y_clean_numeric_samples[val_indices]
    y_val_noisy_num = y_noisy_numeric_samples[val_indices]
    print(f"Train shapes: X_bin={X_train_bin.shape}, y_bin={y_train_bin.shape}")
    print(f"Val shapes:   X_bin={X_val_bin.shape}, y_bin={y_val_bin.shape}")


    # --- 3. Training o Caricamento Modello ---
    print("\n--- Fase 3: Training o Caricamento Modello ---")
    manager = None
    if os.path.exists(MODEL_SAVE_PATH) and not FORCE_RETRAIN:
        print(f"🔹 Trovato modello esistente in '{MODEL_SAVE_PATH}'. Caricamento...")
        try:
            manager = MultiLabelTsetlinMachine.load(MODEL_SAVE_PATH)
            # Verifica parametri (usa NUM_OUTPUT_BITS calcolato)
            params_match = True
            if manager.n_labels != NUM_OUTPUT_BITS: params_match = False
            if params_match and len(manager.tm_params) == len(TM_PARAMS):
                 for key, value in TM_PARAMS.items():
                      if manager.tm_params.get(key) != value: params_match = False; break
            else: params_match = False
            if not params_match:
                 print("⚠️ Attenzione: Parametri modello caricato != parametri correnti.")
                 if input("   Forzare retrain? (s/N): ").lower() == 's': manager = None
                 else: print("   Utilizzo parametri del modello caricato.")
        except Exception as e: print(f"❌ Errore caricamento: {e}. Procedo con training."); manager = None
    else: print(f"INFO: Modello non trovato o FORCE_RETRAIN=True.")

    if manager is None:
        print("🔹 Inizializzazione e Training nuovo modello...")
        manager = MultiLabelTsetlinMachine(
            n_labels=NUM_OUTPUT_BITS, # Usa NUM_OUTPUT_BITS calcolato
            **TM_PARAMS
        )
        manager.fit(
            X_train_bin, y_train_bin, epochs=EPOCHS_PER_LABEL,
            X_val=X_val_bin, Y_val=y_val_bin,
            use_best_state=USE_VALIDATION_FOR_BEST_STATE,
            n_jobs=-1, verbose=True, verbose_parallel=False
        )
        manager.save(MODEL_SAVE_PATH)
    else: print("✅ Modello caricato.")

    # --- 4. Valutazione Denoising sul Set di Validazione ---
    print("\n--- Fase 4: Valutazione Denoising (Validation Set) ---")

    # 1. Predizione Multi-Label Binarizzata
    print("  Predicting bits on validation set...")
    y_pred_bin_val = manager.predict(X_val_bin, verbose=True) # Passa verbose

    # Valutazione Accuratezza Binaria
    # ... (come prima) ...
    print("\n--- Fase 4: Valutazione Denoising (Validation Set) ---")
    print("  Predicting bits on validation set...")
    y_pred_bin_val = manager.predict(X_val_bin, verbose=True)

    print("\n  Valutazione Accuratezza Binaria (per label):")
    avg_accuracy = 0
    for i in range(NUM_OUTPUT_BITS): # Itera fino al nuovo numero di bit
         acc = accuracy_score(y_val_bin[:, i], y_pred_bin_val[:, i])
         print(f"    Accuratezza Bit {i}: {acc:.4f}")
         avg_accuracy += acc
    print(f"    Accuratezza Media per Bit: {avg_accuracy / NUM_OUTPUT_BITS:.4f}")
    h_loss = hamming_loss(y_val_bin, y_pred_bin_val)
    print(f"    Hamming Loss: {h_loss:.4f}")

    print("\n  Decoding binary predictions to numeric (using stats map)...")
    start_decode_time = time.time()
    # Passa NUM_OUTPUT_BITS corretto (anche se la funzione non lo usa più direttamente)
    y_pred_numeric_val = inverse_binarize_combined(y_pred_bin_val, binarization_info)
    end_decode_time = time.time()
    print(f"  Decoding time: {end_decode_time - start_decode_time:.2f}s")

    # 3. Calcolo Metriche Numeriche Denoising
    # ... (come prima, ma ora usa y_pred_numeric_val aggiornato) ...
    print("\n  Calculating denoising metrics...")
    if len(y_pred_numeric_val) == len(y_val_clean_num):
        try:
            mse_denoised = mean_squared_error(y_val_clean_num, y_pred_numeric_val)
            rmse_denoised = np.sqrt(mse_denoised)
        except Exception as e: print(f"  Errore calcolo MSE/RMSE: {e}"); rmse_denoised = np.nan
        try:
            snr_output = calculate_snr(y_val_clean_num, y_pred_numeric_val)
        except Exception as e: print(f"  Errore calcolo SNR: {e}"); snr_output = np.nan

        print(f"\n  --- Denoising Results (Validation Set) ---")
        print(f"    RMSE (Denoised vs Clean): {rmse_denoised:.6f}")
        print(f"    SNR Output (Denoised vs Clean): {snr_output:.2f} dB")

        # Calcolo Baseline
        try:
            # y_val_noisy_num è già stato definito dallo split
            mse_noisy = mean_squared_error(y_val_clean_num, y_val_noisy_num)
            rmse_noisy = np.sqrt(mse_noisy)
            snr_input = calculate_snr(y_val_clean_num, y_val_noisy_num)
            print(f"\n    --- Baseline (Noisy vs Clean) ---")
            print(f"    RMSE (Noisy vs Clean):    {rmse_noisy:.6f}")
            print(f"    SNR Input (Noisy vs Clean): {snr_input:.2f} dB")
            print(f"\n    --- Improvement ---")
            if not np.isnan(rmse_denoised) and not np.isnan(rmse_noisy): print(f"    RMSE Decrease: {rmse_noisy - rmse_denoised:.6f} {'✅' if rmse_noisy > rmse_denoised else '❌'}")
            else: print(f"    RMSE Decrease: N/A")
            if not np.isnan(snr_output) and not np.isnan(snr_input): print(f"    SNR Improvement: {snr_output - snr_input:.2f} dB {'✅' if snr_output > snr_input else '❌'}")
            else: print(f"    SNR Improvement: N/A")
        except NameError: print("\n    (Skipping baseline: y_val_noisy_num non definito)")
        except Exception as e: print(f"\n    (Skipping baseline due to error: {e})")
    else: print("❌ Errore: Lunghezza predizioni numeriche != target numerici.")

    # 4. Visualizzazione Esempio
    # ... (come prima) ...
    print("\n  Visualizing example reconstruction...")
    try:
        plot_idx = 50
        clean_sample_to_plot = y_val_clean_num[plot_idx]
        denoised_sample_to_plot = y_pred_numeric_val[plot_idx]
        print(f"    Visualizing sample index {plot_idx} from validation set:")
        print(f"      Original Clean Value: {clean_sample_to_plot:.4f}")
        try: noisy_sample_to_plot = y_val_noisy_num[plot_idx]; print(f"      Original Noisy Value: {noisy_sample_to_plot:.4f}")
        except Exception: print("      Original Noisy Value: (non disponibile)")
        print(f"      Denoised Value (TM):  {denoised_sample_to_plot:.4f}")

        N_plot_samples = 1000
        if len(y_val_clean_num) >= N_plot_samples:
             plot_signal_comparison(
                 clean=y_val_clean_num[:N_plot_samples],
                 noisy=y_val_noisy_num[:N_plot_samples] if 'y_val_noisy_num' in locals() else None,
                 denoised=y_pred_numeric_val[:N_plot_samples],
                 title_prefix=f"ECG Denoising TM (Val Set Samples 0-{N_plot_samples-1})"
             )
    except IndexError: print("    (Skipping visualization: plot_idx out of bounds)")
    except Exception as e: print(f"    (Skipping visualization due to error: {e})")

    print("\n🏁 Script execution finished.")