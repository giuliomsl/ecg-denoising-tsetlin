# -*- coding: utf-8 -*-
import os
import numpy as np
import wfdb # wfdb serve ancora per caricare i record originali
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import sys
import errno
from tqdm import tqdm
import traceback # Per debug errori

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = False
# -------------------------

# --- Definizione Percorsi ---
if RUNNING_ON_COLAB:
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data"); MODEL_DIR = os.path.join(GDRIVE_BASE, "models")
else:
    # ... (definizioni Locali come prima) ...
    PROJECT_ROOT = "."; DATA_DIR = os.path.join(PROJECT_ROOT, "data"); MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/") # Directory dove generate_noisy_ecg salva i file completi
RTM_DATA_DIR = os.path.join(DATA_DIR, "rtm_processed") # Output di questo script

print(f"INFO: Leggendo segnali completi da: {NOISY_ECG_DIR}") # Modificato per chiarezza
print(f"INFO: Output RTM DATA DIR: {RTM_DATA_DIR}")
try: os.makedirs(RTM_DATA_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {RTM_DATA_DIR}.")
except NameError: pass

# --- Configurazioni ---
WINDOW_SIZE_RTM = 31 # Dimensione finestra input RTM (dispari!) - IPERPARAMETRO DA OTTIMIZZARE
# Scaling Input X (Rumoroso)
X_SCALING_MIN = 0
X_SCALING_MAX = 1023
X_INTEGER_DTYPE = np.uint16
# Scaling Target Y (Pulito) - Per S3
Y_SCALING_MIN = 0
Y_SCALING_MAX = 1000
Y_INTEGER_DTYPE = np.uint16

# Directory di INPUT (dove generate_noisy_ecg ha salvato i segnali completi)
INPUT_SIGNAL_DIR = os.path.join(DATA_DIR, "noisy_ecg/") # Legge da qui
# Directory di OUTPUT per RTM
RTM_DATA_DIR = os.path.join(DATA_DIR, "rtm_processed")

# START e DURATION non servono più per caricare segmenti, ma possono servire
# per identificare i file generati da generate_noisy_ecg.py se quel
# nome file è ancora usato. Assumiamo che generate_noisy_ecg salvi ora
# file come NOME_RECORD_clean.npy, NOME_RECORD_noisy.npy etc.
# Se generate_noisy_ecg usa ancora START/DURATION nel nome file, mantienili qui.
# START = 30
# DURATION = 10

# --- Funzioni Helper ---

def load_full_signals(record_name, input_dir):
    """Carica tutti i segnali numerici salvati per un record."""
    signals = {}
    # Nomi file attesi (senza start/duration)
    expected_suffixes = ["_clean.npy", "_noisy.npy", "_noise_bw.npy", "_noise_ma.npy", "_noise_pli.npy"]
    base_path = os.path.join(input_dir, record_name)
    sig_len = -1 # Per controllare la lunghezza

    try:
        for suffix in expected_suffixes:
            key = suffix.replace(".npy", "").replace("_", "")
            path = f"{base_path}{suffix}" # Costruisci percorso completo
            if not os.path.exists(path):
                if "noise" in key:
                     print(f"⚠️ File componente rumore non trovato: {path}. Verrà generato array di zeri.")
                     # Determina lunghezza necessaria da clean o noisy
                     if sig_len < 0: # Se non ancora determinata
                         clean_path = f"{base_path}_clean.npy"
                         noisy_path = f"{base_path}_noisy.npy"
                         if os.path.exists(clean_path): sig_len = len(np.load(clean_path))
                         elif os.path.exists(noisy_path): sig_len = len(np.load(noisy_path))
                         else: raise FileNotFoundError(f"Impossibile determinare lunghezza per {key}, file base mancanti.")
                     if sig_len > 0: signals[key] = np.zeros(sig_len)
                     else: signals[key] = None # Segnala errore
                else:
                     raise FileNotFoundError(f"File base non trovato: {path}")
            else:
                signals[key] = np.load(path)
                if sig_len < 0: sig_len = len(signals[key]) # Imposta lunghezza dal primo file trovato

        # Verifica lunghezze e tipi
        valid_signals = {}
        if sig_len <= 0: raise ValueError("Lunghezza segnale non valida.")

        all_lengths_match = True
        for k, v in signals.items():
            if v is None: # Se un rumore non è stato caricato/generato
                 print(f"⚠️ Segnale '{k}' mancante o non generato, verrà riempito con zeri.")
                 valid_signals[k] = np.zeros(sig_len) # Crea array di zeri
            elif len(v) != sig_len:
                 print(f"❌ Errore: Lunghezza segnale '{k}' ({len(v)}) != lunghezza base ({sig_len}) per record {record_name}.")
                 all_lengths_match = False
                 break
            else:
                 valid_signals[k] = v # Segnale valido

        if not all_lengths_match: return None # Salta record se le lunghezze non corrispondono

        # Assicurati che tutti i segnali attesi siano presenti (anche come zeri)
        for suffix in expected_suffixes:
             key = suffix.replace(".npy", "").replace("_", "")
             if key not in valid_signals:
                  print(f"⚠️ Chiave '{key}' mancante dopo caricamento, verrà riempita con zeri.")
                  valid_signals[key] = np.zeros(sig_len)

        return valid_signals

    except Exception as e:
        print(f"❌ Errore caricamento segnali per {record_name}: {e}")
        traceback.print_exc() # Stampa traceback per debug
        return None

def create_rtm_windows_multi_target(signals, window_size):
    """Crea finestre da input rumoroso e target multipli."""
    # ... (come prima, ma aggiungi controlli se una chiave manca in signals) ...
    noisy_signal = signals.get('noisy')
    clean_signal = signals.get('clean')
    bw_signal = signals.get('noisebw', np.zeros_like(clean_signal) if clean_signal is not None else None)
    ma_signal = signals.get('noisema', np.zeros_like(clean_signal) if clean_signal is not None else None)
    pli_signal = signals.get('noisepli', np.zeros_like(clean_signal) if clean_signal is not None else None)

    # Verifica che i segnali base esistano
    if noisy_signal is None or clean_signal is None:
        print("ERRORE: Segnale noisy o clean mancante in create_rtm_windows.")
        return None, None, None, None, None, None # Restituisci None per tutti

    if window_size % 2 == 0: window_size += 1
    half_window = window_size // 2
    n_total_samples = len(noisy_signal)
    if n_total_samples < window_size: return None, None, None, None, None, None # Segnale troppo corto

    X_windows = []; y_clean = []; noisy_centers = []
    y_bw = []; y_ma = []; y_pli = []

    for i in range(half_window, n_total_samples - half_window):
        X_windows.append(noisy_signal[i - half_window : i + half_window + 1])
        y_clean.append(clean_signal[i])
        noisy_centers.append(noisy_signal[i])
        # Aggiungi controllo esistenza rumori (anche se load_full_signals dovrebbe averli riempiti)
        y_bw.append(bw_signal[i] if bw_signal is not None else 0)
        y_ma.append(ma_signal[i] if ma_signal is not None else 0)
        y_pli.append(pli_signal[i] if pli_signal is not None else 0)

    return (np.array(X_windows), np.array(y_clean), np.array(noisy_centers),
            np.array(y_bw), np.array(y_ma), np.array(y_pli))


# --- Funzione Principale di Preprocessing per RTM ---
def preprocess_rtm_data_full_signal():
    print(f"INFO: Cerco file _clean.npy in: {INPUT_SIGNAL_DIR}")
    try:
        clean_files = sorted([f for f in os.listdir(INPUT_SIGNAL_DIR) if f.endswith("_clean.npy")])
        record_ids = sorted(list(set(f.replace("_clean.npy", "") for f in clean_files)))
        print(f"INFO: Trovati {len(clean_files)} file _clean.npy per {len(record_ids)} record.")
    except Exception as e: print(f"❌ ERRORE leggendo '{INPUT_SIGNAL_DIR}': {e}"); return
    if not record_ids: print("❌ ERRORE: Nessun record trovato."); return

    # Liste per accumulare dati da tutti i record
    all_X_windows_num = []
    all_y_clean_num = []
    all_y_noisy_center_num = []
    all_y_bw_num = []
    all_y_ma_num = []
    all_y_pli_num = []

    print(f"▶️ Estrazione finestre numeriche da segnali completi (window_size={WINDOW_SIZE_RTM})...")
    for record in tqdm(record_ids, desc="Processing Records"):
        signals = load_full_signals(record, INPUT_SIGNAL_DIR)
        if signals is None: continue # Errore caricamento

        # Estrai finestre e target
        results = create_rtm_windows_multi_target(signals, WINDOW_SIZE_RTM)
        if results[0] is None: # Se create_rtm_windows fallisce
             print(f"⚠️ Skip record {record} (errore creazione finestre).")
             continue
        X_w, y_c, y_n_c, y_bw, y_ma, y_pli = results

        if X_w.shape[0] > 0:
            all_X_windows_num.append(X_w)
            all_y_clean_num.append(y_c)
            all_y_noisy_center_num.append(y_n_c)
            all_y_bw_num.append(y_bw)
            all_y_ma_num.append(y_ma)
            all_y_pli_num.append(y_pli)

    if not all_X_windows_num: print("❌ Nessuna finestra estratta."); return

    print("\n🔄 Concatenamento...")
    X_full_num = np.concatenate(all_X_windows_num); del all_X_windows_num # Libera memoria
    y_full_clean_num = np.concatenate(all_y_clean_num); del all_y_clean_num
    noisy_full_center_num = np.concatenate(all_y_noisy_center_num); del all_y_noisy_center_num
    y_full_bw_num = np.concatenate(all_y_bw_num); del all_y_bw_num
    y_full_ma_num = np.concatenate(all_y_ma_num); del all_y_ma_num
    y_full_pli_num = np.concatenate(all_y_pli_num); del all_y_pli_num
    print(f"  Shape X numerico (finestre rumorose): {X_full_num.shape}")
    # ... (altre stampe shape) ...

    # --- Divisione Train/Test ---
    print("\n Splits Dati Train/Test (80/20)...")
    indices = np.arange(X_full_num.shape[0])
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )
    # Libera memoria array completi dopo split
    X_train_num = X_full_num[train_indices]; X_test_num = X_full_num[test_indices]; del X_full_num
    y_train_clean_num = y_full_clean_num[train_indices]; y_test_clean_num = y_full_clean_num[test_indices]; del y_full_clean_num
    y_train_bw_num = y_full_bw_num[train_indices]; y_test_bw_num = y_full_bw_num[test_indices]; del y_full_bw_num
    y_train_ma_num = y_full_ma_num[train_indices]; y_test_ma_num = y_full_ma_num[test_indices]; del y_full_ma_num
    y_train_pli_num = y_full_pli_num[train_indices]; y_test_pli_num = y_full_pli_num[test_indices]; del y_full_pli_num
    y_test_noisy_center_num = noisy_full_center_num[test_indices]; del noisy_full_center_num
    print(f"  Train: X={X_train_num.shape}, y_clean={y_train_clean_num.shape}...")
    print(f"  Test:  X={X_test_num.shape}, y_clean={y_test_clean_num.shape}...")

    # --- Scaling Input X a Interi ---
    print(f"\n🔄 Scaling Input X a interi [{X_SCALING_MIN}, {X_SCALING_MAX}]...")
    x_scaler = MinMaxScaler(feature_range=(X_SCALING_MIN, X_SCALING_MAX))
    # Fitta e trasforma train, poi libera X_train_num
    X_train_int = x_scaler.fit_transform(X_train_num).astype(X_INTEGER_DTYPE); del X_train_num
    # Trasforma test e applica clip, poi libera X_test_num
    X_test_int = x_scaler.transform(X_test_num)
    X_test_int = np.clip(X_test_int, X_SCALING_MIN, X_SCALING_MAX).astype(X_INTEGER_DTYPE); del X_test_num
    print(f"  Range X_train_int: min={X_train_int.min()}, max={X_train_int.max()}")
    print(f"  Range X_test_int (dopo clip): min={X_test_int.min()}, max={X_test_int.max()}")

    # --- Scaling Target Y (Pulito) a Interi (SOLO per S3) ---
    print(f"\n🔄 Scaling Target Y (pulito) a interi [{Y_SCALING_MIN}, {Y_SCALING_MAX}] (per S3)...")
    y_scaler_clean = MinMaxScaler(feature_range=(Y_SCALING_MIN, Y_SCALING_MAX))
    y_scaler_clean.fit(y_train_clean_num.reshape(-1, 1))
    y_train_clean_int = y_scaler_clean.transform(y_train_clean_num.reshape(-1, 1)).flatten().astype(Y_INTEGER_DTYPE)
    print(f"  Range y_train_clean_int: min={y_train_clean_int.min()}, max={y_train_clean_int.max()}")

    # --- Salvataggio Dati Processati per RTM ---
    print(f"\n💾 Salvataggio dati RTM in '{RTM_DATA_DIR}'...")
    try:
        np.save(os.path.join(RTM_DATA_DIR, "X_train_rtm_int.npy"), X_train_int); del X_train_int
        np.save(os.path.join(RTM_DATA_DIR, "X_test_rtm_int.npy"), X_test_int); del X_test_int
        np.save(os.path.join(RTM_DATA_DIR, "y_train_rtm_clean_int.npy"), y_train_clean_int); del y_train_clean_int
        np.save(os.path.join(RTM_DATA_DIR, "y_train_rtm_clean_num.npy"), y_train_clean_num); del y_train_clean_num
        np.save(os.path.join(RTM_DATA_DIR, "y_test_rtm_clean_num.npy"), y_test_clean_num); del y_test_clean_num
        np.save(os.path.join(RTM_DATA_DIR, "y_train_rtm_bw_num.npy"), y_train_bw_num); del y_train_bw_num
        np.save(os.path.join(RTM_DATA_DIR, "y_test_rtm_bw_num.npy"), y_test_bw_num); del y_test_bw_num
        np.save(os.path.join(RTM_DATA_DIR, "y_train_rtm_ma_num.npy"), y_train_ma_num); del y_train_ma_num
        np.save(os.path.join(RTM_DATA_DIR, "y_test_rtm_ma_num.npy"), y_test_ma_num); del y_test_ma_num
        np.save(os.path.join(RTM_DATA_DIR, "y_train_rtm_pli_num.npy"), y_train_pli_num); del y_train_pli_num
        np.save(os.path.join(RTM_DATA_DIR, "y_test_rtm_pli_num.npy"), y_test_pli_num); del y_test_pli_num
        np.save(os.path.join(RTM_DATA_DIR, "y_test_noisy_center_num.npy"), y_test_noisy_center_num); del y_test_noisy_center_num
        with open(os.path.join(RTM_DATA_DIR, "rtm_x_scaler.pkl"), 'wb') as f: pickle.dump(x_scaler, f)
        with open(os.path.join(RTM_DATA_DIR, "rtm_y_clean_scaler.pkl"), 'wb') as f: pickle.dump(y_scaler_clean, f)
        print("✅ Salvataggio completato.")
    except Exception as e: print(f"❌ Errore salvataggio file RTM: {e}")

if __name__ == "__main__":
    preprocess_rtm_data_full_signal()