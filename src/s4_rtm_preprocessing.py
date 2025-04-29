# -*- coding: utf-8 -*-
# File: rtm_preprocessing_s4.py
import os
import numpy as np
# Rimuovi wfdb se non più necessario qui
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import sys
import errno
from tqdm import tqdm
import traceback

print("--- RTM Preprocessing Script for S4 (Noise Estimation) ---")

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = True
# -------------------------

# --- Definizione Percorsi ---
if RUNNING_ON_COLAB:
    # ... (definizioni Colab) ...
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data"); MODEL_DIR = os.path.join(GDRIVE_BASE, "models")
else:
    # ... (definizioni Locali) ...
    PROJECT_ROOT = "."; DATA_DIR = os.path.join(PROJECT_ROOT, "data"); MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Directory di INPUT (dove generate_noisy_ecg salva i file completi)
INPUT_SIGNAL_DIR = os.path.join(DATA_DIR, "generated_signals/") # <--- MODIFICATO
# Directory di OUTPUT per dati S4
RTM_S4_DATA_DIR = os.path.join(DATA_DIR, "rtm_processed_s4")

print(f"INFO: Leggendo segnali completi da: {INPUT_SIGNAL_DIR}")
print(f"INFO: Output RTM S4 DATA DIR: {RTM_S4_DATA_DIR}")
try: os.makedirs(RTM_S4_DATA_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {RTM_S4_DATA_DIR}.")
except NameError: pass

# --- Configurazioni ---
WINDOW_SIZE_RTM = 31
X_SCALING_MIN = 0
X_SCALING_MAX = 1023
X_INTEGER_DTYPE = np.uint16
TARGET_NOISE_DTYPE = np.float32

# --- Funzioni Helper ---

def load_all_signals_s4(record_name, input_dir):
    """Carica tutti i segnali numerici necessari per S4."""
    signals = {}
    expected_suffixes = ["_clean.npy", "_noisy.npy", "_noise_bw.npy", "_noise_ma.npy", "_noise_pli.npy"]
    base_path = os.path.join(input_dir, record_name)
    sig_len = -1
    try:
        for suffix in expected_suffixes:
            key = suffix.replace(".npy", "").replace("_", "")
            path = f"{base_path}{suffix}"
            if not os.path.exists(path):
                 # Per S4, tutti i file sono necessari
                 raise FileNotFoundError(f"File necessario non trovato: {path}")
            signals[key] = np.load(path)
            if sig_len < 0: sig_len = len(signals[key])
            elif len(signals[key]) != sig_len:
                 raise ValueError(f"Lunghezze segnali non consistenti per {record_name}")
        if sig_len <= 0: raise ValueError("Lunghezza segnale non valida.")
        return signals
    except Exception as e:
        print(f"❌ Errore caricamento segnali S4 per {record_name}: {e}")
        return None

def create_rtm_windows_noise_target(signals, window_size):
    """Crea finestre da input rumoroso e target rumore (noisy - clean)."""
    noisy_signal = signals.get('noisy')
    clean_signal = signals.get('clean')
    if noisy_signal is None or clean_signal is None: return None, None, None, None # Errore

    if window_size % 2 == 0: window_size += 1
    half_window = window_size // 2
    n_total_samples = len(noisy_signal)
    if n_total_samples < window_size: return None, None, None, None

    X_windows = []; y_noise_targets = []; y_clean_centers = []; noisy_centers = []
    noise_signal = noisy_signal - clean_signal # Calcola rumore target

    for i in range(half_window, n_total_samples - half_window):
        X_windows.append(noisy_signal[i - half_window : i + half_window + 1])
        y_noise_targets.append(noise_signal[i])
        y_clean_centers.append(clean_signal[i])
        noisy_centers.append(noisy_signal[i])

    return (np.array(X_windows),
            np.array(y_noise_targets, dtype=TARGET_NOISE_DTYPE),
            np.array(y_clean_centers),
            np.array(noisy_centers))

# --- Funzione Principale di Preprocessing per RTM S4 ---
def preprocess_rtm_s4_data():
    print(f"INFO: Cerco file _clean.npy in: {INPUT_SIGNAL_DIR}")
    try:
        clean_files = sorted([f for f in os.listdir(INPUT_SIGNAL_DIR) if f.endswith("_clean.npy")])
        record_ids = sorted(list(set(f.replace("_clean.npy", "") for f in clean_files)))
        print(f"INFO: Trovati {len(clean_files)} file _clean.npy per {len(record_ids)} record.")
    except Exception as e: print(f"❌ ERRORE leggendo '{INPUT_SIGNAL_DIR}': {e}"); return
    if not record_ids: print("❌ ERRORE: Nessun record trovato."); return

    all_X_windows_num = []
    all_y_noise_num = []
    all_y_clean_center_num = []
    all_y_noisy_center_num = []

    print(f"▶️ Estrazione finestre e target rumore (window_size={WINDOW_SIZE_RTM})...")
    for record in tqdm(record_ids, desc="Processing Records"):
        signals = load_all_signals_s4(record, INPUT_SIGNAL_DIR) # Usa nuova funzione di caricamento
        if signals is None: continue

        results = create_rtm_windows_noise_target(signals, WINDOW_SIZE_RTM)
        if results[0] is None:
             print(f"⚠️ Skip record {record} (errore creazione finestre).")
             continue
        X_w, y_noise, y_clean_c, y_noisy_c = results

        if X_w.shape[0] > 0:
            all_X_windows_num.append(X_w)
            all_y_noise_num.append(y_noise)
            all_y_clean_center_num.append(y_clean_c)
            all_y_noisy_center_num.append(y_noisy_c)

    if not all_X_windows_num: print("❌ Nessuna finestra estratta."); return

    print("\n🔄 Concatenamento...")
    X_full_num = np.concatenate(all_X_windows_num); del all_X_windows_num
    y_full_noise_num = np.concatenate(all_y_noise_num); del all_y_noise_num
    y_full_clean_num = np.concatenate(all_y_clean_center_num); del all_y_clean_center_num
    noisy_full_center_num = np.concatenate(all_y_noisy_center_num); del all_y_noisy_center_num
    print(f"  Shape X numerico (finestre rumorose): {X_full_num.shape}")
    print(f"  Shape y numerico (rumore target): {y_full_noise_num.shape}")
    print(f"  Shape y numerico (pulito originale): {y_full_clean_num.shape}")
    print(f"  Shape noisy numerico (centro originale): {noisy_full_center_num.shape}")

    # --- Divisione Train/Test ---
    print("\n Splits Dati Train/Test (80/20)...")
    indices = np.arange(X_full_num.shape[0])
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )
    X_train_num = X_full_num[train_indices]; X_test_num = X_full_num[test_indices]; del X_full_num
    y_train_noise_num = y_full_noise_num[train_indices]; y_test_noise_num = y_full_noise_num[test_indices]; del y_full_noise_num
    y_train_clean_num = y_full_clean_num[train_indices]; y_test_clean_num = y_full_clean_num[test_indices]; del y_full_clean_num
    y_test_noisy_center_num = noisy_full_center_num[test_indices]; del noisy_full_center_num
    print(f"  Train: X={X_train_num.shape}, y_noise={y_train_noise_num.shape}")
    print(f"  Test:  X={X_test_num.shape}, y_noise={y_test_noise_num.shape}, y_clean={y_test_clean_num.shape}, y_noisy={y_test_noisy_center_num.shape}")

    # --- Scaling Input X a Interi ---
    print(f"\n🔄 Scaling Input X a interi [{X_SCALING_MIN}, {X_SCALING_MAX}]...")
    x_scaler = MinMaxScaler(feature_range=(X_SCALING_MIN, X_SCALING_MAX))
    X_train_int = x_scaler.fit_transform(X_train_num).astype(X_INTEGER_DTYPE); del X_train_num
    X_test_int = x_scaler.transform(X_test_num)
    X_test_int = np.clip(X_test_int, X_SCALING_MIN, X_SCALING_MAX).astype(X_INTEGER_DTYPE); del X_test_num
    print(f"  Range X_train_int: min={X_train_int.min()}, max={X_train_int.max()}")
    print(f"  Range X_test_int (dopo clip): min={X_test_int.min()}, max={X_test_int.max()}")

    # --- NON Scaliamo il Target Y (Rumore) ---

    # --- Salvataggio Dati Processati per RTM S4 ---
    print(f"\n💾 Salvataggio dati RTM S4 in '{RTM_S4_DATA_DIR}'...")
    try:
        np.save(os.path.join(RTM_S4_DATA_DIR, "X_train_rtm_s4_int.npy"), X_train_int); del X_train_int
        np.save(os.path.join(RTM_S4_DATA_DIR, "X_test_rtm_s4_int.npy"), X_test_int); del X_test_int
        # Target Rumore (numerico float)
        np.save(os.path.join(RTM_S4_DATA_DIR, "y_train_rtm_s4_noise_num.npy"), y_train_noise_num); del y_train_noise_num
        np.save(os.path.join(RTM_S4_DATA_DIR, "y_test_rtm_s4_noise_num.npy"), y_test_noise_num); del y_test_noise_num
        # Target Pulito originale (per valutazione)
        np.save(os.path.join(RTM_S4_DATA_DIR, "y_train_rtm_s4_clean_num.npy"), y_train_clean_num); del y_train_clean_num
        np.save(os.path.join(RTM_S4_DATA_DIR, "y_test_rtm_s4_clean_num.npy"), y_test_clean_num); del y_test_clean_num
        # Target Rumoroso originale (per baseline e ricostruzione)
        np.save(os.path.join(RTM_S4_DATA_DIR, "y_test_noisy_center_num.npy"), y_test_noisy_center_num); del y_test_noisy_center_num
        # Scaler Input X
        with open(os.path.join(RTM_S4_DATA_DIR, "rtm_s4_x_scaler.pkl"), 'wb') as f: pickle.dump(x_scaler, f)
        print("✅ Salvataggio completato.")
    except Exception as e: print(f"❌ Errore salvataggio file RTM S4: {e}")

if __name__ == "__main__":
    preprocess_rtm_s4_data()