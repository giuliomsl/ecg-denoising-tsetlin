# -*- coding: utf-8 -*-
import os
import numpy as np
import wfdb
from ml_binarization import binarize, save_binarization_info # Assicurati che save_binarization_info esista
import pickle
import sys
import errno
from collections import defaultdict # Utile per calcolare le medie
import traceback

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = False # <--- MODIFICA QUESTA RIGA MANUALMENTE
# -------------------------

# --- Definizione Percorsi basata sul Flag ---
# (Blocco if/else per definire DATA_DIR, MODEL_DIR, ECG_DIR, NOISE_DIR,
#  NOISY_ECG_DIR, SAMPLE_DATA_DIR, MODEL_OUTPUT_DIR, BIN_INFO_PATH come prima)
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
    except ImportError:
         print("ERRORE: Impossibile importare google.colab. Assicurati di essere in Colab.")
         exit()
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

ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samplewise/")
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "multi_tm_denoiser/")
BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

print(f"INFO: Usando DATA_DIR: {DATA_DIR}")
print(f"INFO: Usando MODEL_DIR: {MODEL_DIR}")

try:
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOISY_ECG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {e.filename}.")
except NameError: pass


# --- Configurazioni ---
WINDOW_SIZE = 1024
N_BINS_QUANT = 13 # <--- IMPOSTA IL NUMERO DI BIN DESIDERATO QUI
CONTEXT_K = 10
BINARIZATION_METHOD = "combined"
# Aggiorna i parametri passati a binarize
BINARIZATION_PARAMS = {
    "quant_n_bins": N_BINS_QUANT,
    "mean_filt_window": 50, # Mantieni o modifica questi se vuoi
    "grad_window": 10
}
START = 30
DURATION = 10

# --- Funzioni Caricamento/Segmentazione (invariate) ---
def load_clean_ecg_segment(record_name):
    # ... (come prima) ...
    record_path = os.path.join(ECG_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    start_sample = int(START * fs)
    end_sample = int((START + DURATION) * fs)
    signal_len = record.p_signal.shape[0]
    end_sample = min(end_sample, signal_len)
    start_sample = min(start_sample, end_sample)
    return record.p_signal[start_sample:end_sample, 0]

def load_noisy_ecg(record_name):
    # ... (come prima) ...
    path = os.path.join(NOISY_ECG_DIR, f"{record_name}_noisy_{START}-{START+DURATION}s.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File rumoroso non trovato: {path}")
    return np.load(path)

def segment_signal(signal, window_size=WINDOW_SIZE):
    # ... (come prima) ...
    if len(signal) < window_size:
        return np.empty((0, window_size), dtype=signal.dtype)
    num_segments = len(signal) // window_size
    return np.array([signal[i * window_size:(i + 1) * window_size] for i in range(num_segments)])

# --- Funzione Estrazione Campioni (invariata) ---
def extract_samples(clean_segments_num, noisy_segments_num,
                    clean_segments_bin, noisy_segments_bin, k=CONTEXT_K):
    # ... (come prima, restituisce X_bin, y_bin, y_clean_num, y_noisy_num) ...
    X_samples_bin_list = []
    y_samples_bin_list = []
    y_clean_numeric_samples_list = []
    y_noisy_numeric_samples_list = []
    num_segments = noisy_segments_bin.shape[0]
    window_len = noisy_segments_bin.shape[1]
    for seg_idx in range(num_segments):
        if noisy_segments_bin[seg_idx].shape[0] != window_len or \
           clean_segments_bin[seg_idx].shape[0] != window_len or \
           noisy_segments_num[seg_idx].shape[0] != window_len or \
           clean_segments_num[seg_idx].shape[0] != window_len:
            print(f"⚠️ Attenzione: Segmento {seg_idx} ha lunghezza inattesa. Saltato.")
            continue
        for t in range(k, window_len - k):
            context_bin = noisy_segments_bin[seg_idx, t - k : t + k + 1, :]
            X_samples_bin_list.append(context_bin.flatten())
            target_bin = clean_segments_bin[seg_idx, t, :]
            y_samples_bin_list.append(target_bin)
            target_clean_num = clean_segments_num[seg_idx, t]
            y_clean_numeric_samples_list.append(target_clean_num)
            target_noisy_num = noisy_segments_num[seg_idx, t]
            y_noisy_numeric_samples_list.append(target_noisy_num)
    X_samples_bin = np.array(X_samples_bin_list)
    y_samples_bin = np.array(y_samples_bin_list)
    y_clean_numeric_samples = np.array(y_clean_numeric_samples_list)
    y_noisy_numeric_samples = np.array(y_noisy_numeric_samples_list)
    return X_samples_bin, y_samples_bin, y_clean_numeric_samples, y_noisy_numeric_samples


# --- Funzione Principale di Preprocessing (Modificata per Mappa Inversione) ---
def preprocess_and_save_all():
    print(f"INFO (Preprocessing): Cerco file rumorosi in: {NOISY_ECG_DIR}")
    try:
        noisy_files = sorted([
            f for f in os.listdir(NOISY_ECG_DIR)
            if f.endswith(f"_{START}-{START+DURATION}s.npy")
        ])
        record_ids = sorted(list(set(f.split("_")[0] for f in noisy_files)))
        print(f"INFO (Preprocessing): Trovati {len(noisy_files)} file .npy rumorosi corrispondenti a {len(record_ids)} record.")
    except FileNotFoundError:
         print(f"❌ ERRORE (Preprocessing): La directory dei file rumorosi '{NOISY_ECG_DIR}' non esiste!")
         return
    except Exception as e:
         print(f"❌ ERRORE (Preprocessing): Errore leggendo '{NOISY_ECG_DIR}': {e}")
         return

    if not record_ids:
        print("❌ ERRORE (Preprocessing): Nessun record trovato con file rumorosi corrispondenti. Interruzione.")
        return

    all_X_bin = []
    all_y_bin = []
    all_y_clean_num = []
    all_y_noisy_num = []
    first_record_processed = False
    binarization_info_accumulated = {} # Per salvare bin_edges ecc. dal primo record

    # Dizionario per accumulare statistiche per la mappa di inversione
    # Struttura: { tupla_bit_pattern: {'sum': float, 'count': int} }
    pattern_stats_accumulator = defaultdict(lambda: {'sum': 0.0, 'count': 0})

    for record in record_ids:
        print(f"🔹 Processing {record}...")
        try:
            clean_signal_num = load_clean_ecg_segment(record)
            noisy_signal_num = load_noisy_ecg(record)
            if len(clean_signal_num) < WINDOW_SIZE or len(noisy_signal_num) < WINDOW_SIZE:
                 print(f"⚠️ Segnale troppo corto per {record}. Skip.")
                 continue

            clean_segments_num = segment_signal(clean_signal_num)
            noisy_segments_num = segment_signal(noisy_signal_num)
            n_segments = min(len(clean_segments_num), len(noisy_segments_num))
            if n_segments == 0: continue
            clean_segments_num = clean_segments_num[:n_segments]
            noisy_segments_num = noisy_segments_num[:n_segments]

            # Binarizza e ottieni info ausiliarie
            binarized_clean_data = [binarize(seg, method=BINARIZATION_METHOD, **BINARIZATION_PARAMS) for seg in clean_segments_num]
            binarized_noisy_data = [binarize(seg, method=BINARIZATION_METHOD, **BINARIZATION_PARAMS) for seg in noisy_segments_num]
            clean_segments_bin = np.array([item[0] for item in binarized_clean_data])
            noisy_segments_bin = np.array([item[0] for item in binarized_noisy_data])
            current_aux_info = binarized_clean_data[0][1]

            # Salva info binarizzazione (solo dal primo record)
            if not first_record_processed and current_aux_info:
                binarization_info_accumulated = current_aux_info.copy() # Copia le info base
                first_record_processed = True

            # Estrai campioni
            X_bin, y_bin, y_clean_num, y_noisy_num = extract_samples(
                clean_segments_num, noisy_segments_num,
                clean_segments_bin, noisy_segments_bin, k=CONTEXT_K
            )

            # Aggiungi alle liste globali e accumula statistiche per inversione
            if X_bin.shape[0] > 0:
                all_X_bin.append(X_bin)
                all_y_bin.append(y_bin)
                all_y_clean_num.append(y_clean_num)
                all_y_noisy_num.append(y_noisy_num)

                # --- INIZIO CALCOLO STATISTICHE INVERSIONE ---
                # Accumula somma e conteggio per ogni pattern di bit in y_bin
                # usando i valori corrispondenti in y_clean_num
                for idx in range(y_bin.shape[0]):
                    pattern_tuple = tuple(y_bin[idx]) # Chiave hashable
                    value = y_clean_num[idx]
                    pattern_stats_accumulator[pattern_tuple]['sum'] += value
                    pattern_stats_accumulator[pattern_tuple]['count'] += 1
                # --- FINE CALCOLO STATISTICHE INVERSIONE ---

                print(f"  Estratti {X_bin.shape[0]} campioni da {record}.")
            else:
                print(f"  Nessun campione estratto da {record}.")

        except FileNotFoundError as e: print(f"⚠️ {e}. Skip record {record}.")
        except Exception as e: print(f"❌ Errore processando {record}: {e}"); traceback.print_exc()

    # --- Concatenamento e Salvataggio Finale ---
    if not all_X_bin:
        print("\n❌ Nessun campione estratto. Controlla dati e config.")
        return

    print("\n🔄 Concatenamento di tutti i campioni...")
    X_total_bin = np.concatenate(all_X_bin)
    y_total_bin = np.concatenate(all_y_bin)
    y_total_clean_num = np.concatenate(all_y_clean_num)
    y_total_noisy_num = np.concatenate(all_y_noisy_num)

    # --- Calcolo Mappa Media Finale ---
    print("📊 Calcolo mappa Pattern -> Media Valore Pulito...")
    pattern_to_mean_map = {
        pattern: stats['sum'] / stats['count']
        for pattern, stats in pattern_stats_accumulator.items()
        if stats['count'] > 0 # Evita divisione per zero (improbabile)
    }
    print(f"  Calcolate medie per {len(pattern_to_mean_map)} pattern unici.")
    # Aggiungi la mappa alle informazioni di binarizzazione
    binarization_info_accumulated['pattern_to_mean_map'] = pattern_to_mean_map
    # Calcola e aggiungi la media globale come fallback (opzionale)
    global_mean_clean = y_total_clean_num.mean()
    binarization_info_accumulated['global_mean_clean'] = global_mean_clean
    print(f"  Media globale segnale pulito (fallback): {global_mean_clean:.4f}")
    # --- Fine Calcolo Mappa ---

    # --- Salvataggio ---
    print("💾 Salvataggio finale dei file campionati e info binarizzazione...")
    np.save(os.path.join(SAMPLE_DATA_DIR, "X_train_samples.npy"), X_total_bin)
    np.save(os.path.join(SAMPLE_DATA_DIR, "y_train_samples.npy"), y_total_bin)
    np.save(os.path.join(SAMPLE_DATA_DIR, "y_clean_numeric_samples.npy"), y_total_clean_num)
    np.save(os.path.join(SAMPLE_DATA_DIR, "y_noisy_numeric_samples.npy"), y_total_noisy_num)
    # Salva le informazioni aggiornate (include mappa e media globale)
    save_binarization_info(binarization_info_accumulated, BIN_INFO_PATH)

    print(f"\n✅ Salvataggio completato in '{SAMPLE_DATA_DIR}':")
    # ... (stampe finali come prima, ma ora BIN_INFO_PATH contiene la mappa) ...
    print(f"   X_train_samples.npy: {X_total_bin.shape}")
    print(f"   y_train_samples.npy: {y_total_bin.shape}")
    print(f"   y_clean_numeric_samples.npy: {y_total_clean_num.shape}")
    print(f"   y_noisy_numeric_samples.npy: {y_total_noisy_num.shape}")
    if binarization_info_accumulated:
         print(f"   binarization_info.pkl: Contiene chiavi: {list(binarization_info_accumulated.keys())}")
    else:
         print(f"   ⚠️ Attenzione: binarization_info.pkl non salvato.")


if __name__ == "__main__":
    preprocess_and_save_all()