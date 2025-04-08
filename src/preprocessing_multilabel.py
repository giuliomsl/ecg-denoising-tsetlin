# -*- coding: utf-8 -*-
import os
import numpy as np
import wfdb
# Assicurati che binarization.py sia nella stessa directory o nel PYTHONPATH
from binarization import binarize, save_binarization_info
import pickle # Necessario se save_binarization_info usa pickle

# === Configurazione ===
ECG_DIR = "data/mit-bih/"
NOISY_DIR = "data/noisy_ecg/"
# PROCESSED_DIR = "data/processed_multilabel/" # Non più usato direttamente qui
SAMPLE_DIR = "data/samplewise/"
BIN_INFO_PATH = os.path.join(SAMPLE_DIR, "binarization_info.pkl") # Percorso per info decodifica
os.makedirs(SAMPLE_DIR, exist_ok=True)

WINDOW_SIZE = 1024
CONTEXT_K = 10  # finestra di contesto ±k → 2k+1
""" CONTEXT_K è cruciale, deve essere ottimizzato """
BINARIZATION_METHOD = "combined" # Assicurati sia coerente con train_tm_multilabel.py
BINARIZATION_PARAMS = {"quant_n_bins": 4} # Parametri per binarize()

START = 30
DURATION = 10

# --- Funzioni di Caricamento e Segmentazione (invariate) ---
def load_clean_ecg_segment(record_name):
    record_path = os.path.join(ECG_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    start_sample = int(START * fs)
    end_sample = int((START + DURATION) * fs)
    # Gestione possibile lunghezza segnale inferiore a end_sample
    signal_len = record.p_signal.shape[0]
    end_sample = min(end_sample, signal_len)
    start_sample = min(start_sample, end_sample) # Assicura start <= end
    return record.p_signal[start_sample:end_sample, 0]

def load_noisy_ecg(record_name):
    path = os.path.join(NOISY_DIR, f"{record_name}_noisy_{START}-{START+DURATION}s.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File rumoroso non trovato: {path}")
    return np.load(path)

def segment_signal(signal, window_size=WINDOW_SIZE):
    if len(signal) < window_size:
        return np.empty((0, window_size), dtype=signal.dtype) # Nessun segmento completo
    num_segments = len(signal) // window_size
    # Ignora l'ultimo pezzo se non forma un segmento completo
    return np.array([signal[i * window_size:(i + 1) * window_size] for i in range(num_segments)])

# --- Funzione Modificata per Estrarre Campioni (Binari e Numerici) ---
def extract_samples(clean_segments_num, noisy_segments_num,
                    clean_segments_bin, noisy_segments_bin, k=CONTEXT_K):
    """
    Estrae campioni per il training/testing, includendo sia i dati
    binarizzati che i corrispondenti valori numerici originali centrali.

    Args:
        clean_segments_num (np.ndarray): Segmenti numerici puliti.
        noisy_segments_num (np.ndarray): Segmenti numerici rumorosi.
        clean_segments_bin (np.ndarray): Segmenti binarizzati puliti.
        noisy_segments_bin (np.ndarray): Segmenti binarizzati rumorosi.
        k (int): Semidimensione della finestra di contesto.

    Returns:
        tuple: (X_samples_bin, y_samples_bin, y_clean_numeric_samples, y_noisy_numeric_samples)
    """
    X_samples_bin_list = []
    y_samples_bin_list = []
    y_clean_numeric_samples_list = []
    y_noisy_numeric_samples_list = [] # Per il campione centrale rumoroso

    num_segments = noisy_segments_bin.shape[0]
    window_len = noisy_segments_bin.shape[1] # Dovrebbe essere WINDOW_SIZE

    for seg_idx in range(num_segments):
        # Assicurati che le forme siano coerenti per questo segmento
        if noisy_segments_bin[seg_idx].shape[0] != window_len or \
           clean_segments_bin[seg_idx].shape[0] != window_len or \
           noisy_segments_num[seg_idx].shape[0] != window_len or \
           clean_segments_num[seg_idx].shape[0] != window_len:
            print(f"⚠️ Attenzione: Segmento {seg_idx} ha lunghezza inattesa. Saltato.")
            continue

        for t in range(k, window_len - k):
            # Contesto rumoroso binarizzato
            context_bin = noisy_segments_bin[seg_idx, t - k : t + k + 1, :]
            X_samples_bin_list.append(context_bin.flatten()) # Input per la TM

            # Target pulito binarizzato
            target_bin = clean_segments_bin[seg_idx, t, :]
            y_samples_bin_list.append(target_bin) # Target binario per la TM

            # Target pulito numerico (per valutazione denoising)
            target_clean_num = clean_segments_num[seg_idx, t]
            y_clean_numeric_samples_list.append(target_clean_num)

            # (Opzionale) Campione centrale rumoroso numerico (per SNR input)
            target_noisy_num = noisy_segments_num[seg_idx, t]
            y_noisy_numeric_samples_list.append(target_noisy_num)

    # Converti liste in array NumPy
    X_samples_bin = np.array(X_samples_bin_list)
    y_samples_bin = np.array(y_samples_bin_list)
    y_clean_numeric_samples = np.array(y_clean_numeric_samples_list)
    y_noisy_numeric_samples = np.array(y_noisy_numeric_samples_list)

    return X_samples_bin, y_samples_bin, y_clean_numeric_samples, y_noisy_numeric_samples

# --- Funzione Principale di Preprocessing (Modificata) ---
def preprocess_and_save_all():
    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(f"_{START}-{START+DURATION}s.npy")])
    record_ids = sorted(list(set(f.split("_")[0] for f in noisy_files))) # Usa list()

    all_X_bin = []
    all_y_bin = []
    all_y_clean_num = []
    all_y_noisy_num = []
    first_record_processed = False
    saved_binarization_info = None

    print(f"Trovati {len(record_ids)} record con file rumorosi corrispondenti.")

    for record in record_ids:
        print(f"🔹 Processing {record}...")
        try:
            # Carica segnali numerici
            clean_signal_num = load_clean_ecg_segment(record)
            noisy_signal_num = load_noisy_ecg(record)

            # Verifica lunghezza minima
            if len(clean_signal_num) < WINDOW_SIZE or len(noisy_signal_num) < WINDOW_SIZE:
                 print(f"⚠️ Segnale troppo corto per {record} dopo caricamento ({len(clean_signal_num)} campioni). Skip.")
                 continue

            # Segmenta segnali numerici
            clean_segments_num = segment_signal(clean_signal_num)
            noisy_segments_num = segment_signal(noisy_signal_num)

            # Allinea numero segmenti (basato su numerici)
            n_segments = min(len(clean_segments_num), len(noisy_segments_num))
            if n_segments == 0:
                print(f"⚠️ Nessun segmento completo valido per {record}. Skip.")
                continue
            clean_segments_num = clean_segments_num[:n_segments]
            noisy_segments_num = noisy_segments_num[:n_segments]

            # Binarizza i segmenti numerici
            # Usiamo list comprehension per gestire l'output tupla da binarize
            binarized_clean_data = [binarize(seg, method=BINARIZATION_METHOD, **BINARIZATION_PARAMS) for seg in clean_segments_num]
            binarized_noisy_data = [binarize(seg, method=BINARIZATION_METHOD, **BINARIZATION_PARAMS) for seg in noisy_segments_num]

            # Estrai i dati binari e le info ausiliarie
            # Nota: Assumiamo che aux_info sia simile per tutti i segmenti di un record
            #       e salviamo solo quella del primo segmento del primo record.
            clean_segments_bin = np.array([item[0] for item in binarized_clean_data])
            noisy_segments_bin = np.array([item[0] for item in binarized_noisy_data])
            current_aux_info = binarized_clean_data[0][1] # Prendi aux_info dal primo segmento

            # Salva aux_info solo dal primo record processato con successo
            if not first_record_processed and current_aux_info:
                saved_binarization_info = current_aux_info
                save_binarization_info(saved_binarization_info, BIN_INFO_PATH)
                print(f"  💾 Informazioni di binarizzazione salvate da {record} in {BIN_INFO_PATH}")
                first_record_processed = True

            # Estrai campioni (binari e numerici)
            X_bin, y_bin, y_clean_num, y_noisy_num = extract_samples(
                clean_segments_num, noisy_segments_num,
                clean_segments_bin, noisy_segments_bin,
                k=CONTEXT_K
            )

            # Aggiungi alle liste globali
            if X_bin.shape[0] > 0: # Assicurati ci siano campioni
                all_X_bin.append(X_bin)
                all_y_bin.append(y_bin)
                all_y_clean_num.append(y_clean_num)
                all_y_noisy_num.append(y_noisy_num)
                print(f"  Estratti {X_bin.shape[0]} campioni da {record}.")
            else:
                print(f"  Nessun campione estratto da {record} (probabilmente a causa di k).")

        except FileNotFoundError as e:
            print(f"⚠️ {e}. Skip record {record}.")
        except Exception as e:
            print(f"❌ Errore inaspettato processando {record}: {e}")
            # Potresti voler aggiungere traceback per debug: import traceback; traceback.print_exc()
            continue # Salta al prossimo record

    # --- Concatenamento e Salvataggio Finale ---
    if not all_X_bin:
        print("\n❌ Nessun campione estratto da nessun record. Controlla i dati e la configurazione (WINDOW_SIZE, k).")
        return

    print("\n🔄 Concatenamento di tutti i campioni...")
    X_total_bin = np.concatenate(all_X_bin)
    y_total_bin = np.concatenate(all_y_bin)
    y_total_clean_num = np.concatenate(all_y_clean_num)
    y_total_noisy_num = np.concatenate(all_y_noisy_num) # Opzionale

    print("💾 Salvataggio finale dei file campionati...")
    np.save(os.path.join(SAMPLE_DIR, "X_train_samples.npy"), X_total_bin)
    np.save(os.path.join(SAMPLE_DIR, "y_train_samples.npy"), y_total_bin)
    np.save(os.path.join(SAMPLE_DIR, "y_clean_numeric_samples.npy"), y_total_clean_num)
    np.save(os.path.join(SAMPLE_DIR, "y_noisy_numeric_samples.npy"), y_total_noisy_num) # Opzionale

    print(f"\n✅ Salvataggio completato in '{SAMPLE_DIR}':")
    print(f"   X_train_samples.npy (binario, input TM): {X_total_bin.shape}")
    print(f"   y_train_samples.npy (binario, target TM): {y_total_bin.shape}")
    print(f"   y_clean_numeric_samples.npy (numerico, target valutazione): {y_total_clean_num.shape}")
    print(f"   y_noisy_numeric_samples.npy (numerico, baseline valutazione): {y_total_noisy_num.shape}")
    if saved_binarization_info:
         print(f"   binarization_info.pkl (per decodifica): Contiene chiavi: {list(saved_binarization_info.keys())}")
    else:
         print(f"   ⚠️ Attenzione: binarization_info.pkl non salvato (nessun record processato con successo?).")


if __name__ == "__main__":
    preprocess_and_save_all()