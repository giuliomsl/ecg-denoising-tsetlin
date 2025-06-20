# -*- coding: utf-8 -*-
# File: generate_noisy_ecg.py (Aggiornato)
import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import errno
from tqdm import tqdm

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = False # <--- MODIFICA QUESTA RIGA MANUALMENTE
# -------------------------

# --- Configurazioni Generali ---
SEGMENT_LENGTH = 1024
OVERLAP_LENGTH = 0
# -----------------------------

# --- Definizione Percorsi ---
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
         print("ERRORE: Impossibile importare google.colab."); exit()
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data")
else:
    print("INFO: Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    PROJECT_ROOT = "."
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
SEGMENTED_SIGNAL_DIR = os.path.join(DATA_DIR, "segmented_signals") # Output per i segmenti

print(f"INFO: Usando ECG_DIR: {ECG_DIR}")
print(f"INFO: Usando NOISE_DIR: {NOISE_DIR}")
print(f"INFO: Output Segmenti Generati in: {SEGMENTED_SIGNAL_DIR}")
os.makedirs(SEGMENTED_SIGNAL_DIR, exist_ok=True)

# --- Configurazioni Generazione Rumore ---
RANDOMIZE_NOISE_LEVELS = True
LEVEL_MIN_WEIGHT = 0.1
LEVEL_MAX_WEIGHT = 0.6
RANDOMIZE_PLI_AMPLITUDE = True
PLI_AMP_MIN = 0.05
PLI_AMP_MAX = 0.25
ADD_PLI_HARMONICS_PROB = 0.15
RECORDS_TO_PROCESS = [] # Vuoto per tutti
ENABLE_PLOTTING_DEBUG = False

# --- Funzioni Helper ---
def load_ecg_full(record_name):
    record_path = os.path.join(ECG_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        ecg_signal_full = record.p_signal[:, 0]
        fs = record.fs
        if len(ecg_signal_full) == 0: raise ValueError("Segnale ECG vuoto.")
        return ecg_signal_full, fs
    except Exception as e:
        print(f"❌ Errore caricamento record {record_name}: {e}"); return None, None

def load_noise_realistically(noise_name, target_length):
    noise_path = os.path.join(NOISE_DIR, noise_name)
    try:
        noise_record = wfdb.rdrecord(noise_path)
        full_noise_signal = noise_record.p_signal[:, 0]
    except Exception as e:
        print(f"❌ Errore caricamento rumore {noise_name}: {e}"); return np.zeros(target_length)
    current_len = len(full_noise_signal)
    if current_len == 0: return np.zeros(target_length)
    if current_len < target_length:
        return np.tile(full_noise_signal, int(np.ceil(target_length / current_len)))[:target_length]
    elif current_len > target_length:
        start_index = random.randint(0, current_len - target_length)
        return full_noise_signal[start_index : start_index + target_length]
    return full_noise_signal

def generate_pli(length, fs=360, base_freq=50):
    t = np.arange(length) / fs
    amplitude = random.uniform(PLI_AMP_MIN, PLI_AMP_MAX) if RANDOMIZE_PLI_AMPLITUDE else 0.1
    deviation_factor = 1 + random.uniform(0, 0.08)
    effective_freq = base_freq * deviation_factor
    pli_signal = amplitude * np.sin(2 * np.pi * effective_freq * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        pli_signal += amplitude * random.uniform(0.1, 0.4) * np.sin(2 * np.pi * (3 * effective_freq) * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        pli_signal += amplitude * random.uniform(0.05, 0.3) * np.sin(2 * np.pi * (5 * effective_freq) * t)
    return pli_signal

def generate_noise_weights_gancitano(n_weights=3, min_w=0.1, max_w=0.6, total_sum=1.0):
    # Tentativo di generare pesi che sommano a 1 e rispettano min/max
    # Questo è un problema non banale. Un approccio iterativo:
    weights = np.random.dirichlet(np.ones(n_weights)) * total_sum # Sommano a total_sum
    for _ in range(10): # Iterazioni per aggiustare
        if np.all(weights >= min_w) and np.all(weights <= max_w): break
        weights = np.clip(weights, min_w, max_w)
        current_sum = np.sum(weights)
        if current_sum > 0: weights = weights * (total_sum / current_sum)
        else: weights = np.ones(n_weights) * (total_sum / n_weights) # Fallback
    # Ultimo clip per sicurezza, anche se potrebbe violare la somma esatta
    weights = np.clip(weights, min_w, max_w)
    if not np.isclose(np.sum(weights), total_sum): # Ri-normalizza se necessario
        weights = weights * (total_sum / np.sum(weights))
    return weights

def generate_noisy_ecg_components(ecg_signal, bw_template_signal, ma_template_signal, pli_template_signal):
    if RANDOMIZE_NOISE_LEVELS:
        bw_weight, ma_weight, pli_weight = generate_noise_weights_gancitano(
            min_w=LEVEL_MIN_WEIGHT, max_w=LEVEL_MAX_WEIGHT
        )
    else:
        bw_weight, ma_weight, pli_weight = (0.33, 0.33, 0.34) # Esempio

    # Applica i pesi direttamente ai template di rumore (NON normalizzare i rumori qui)
    bw_component = bw_template_signal * bw_weight
    ma_component = ma_template_signal * ma_weight
    pli_component = pli_template_signal * pli_weight
    noisy_ecg = ecg_signal + bw_component + ma_component + pli_component
    return noisy_ecg, bw_component, ma_component, pli_component

def segment_signal(signal, segment_len, overlap_len):
    segments = []
    step = segment_len - overlap_len
    for i in range(0, len(signal) - segment_len + 1, step):
        segments.append(signal[i : i + segment_len])
    return np.array(segments) if segments else np.empty((0, segment_len))

def save_segmented_signals(output_dir, base_record_name, segment_idx, signals_dict_for_segment):
    try:
        for signal_type, signal_data in signals_dict_for_segment.items():
            file_name = f"{base_record_name}_segment_{segment_idx:03d}_{signal_type}.npy"
            file_path = os.path.join(output_dir, file_name)
            np.save(file_path, signal_data)
        return True
    except Exception as e:
        print(f"❌ Errore salvataggio segmenti {base_record_name}_{segment_idx}: {e}"); return False

def process_record_and_save_segments(ecg_record_name, output_dir, seg_len, overlap_len):
    ecg_signal_full, fs = load_ecg_full(ecg_record_name)
    if ecg_signal_full is None: return 0
    full_length = len(ecg_signal_full)
    bw_noise_template = load_noise_realistically("bw", full_length)
    ma_noise_template = load_noise_realistically("ma", full_length)
    pli_noise_template = generate_pli(full_length, fs)
    noisy_ecg_full, bw_comp_full, ma_comp_full, pli_comp_full = generate_noisy_ecg_components(
        ecg_signal_full, bw_noise_template, ma_noise_template, pli_noise_template
    )
    all_signals_to_segment = {
        "clean": ecg_signal_full, "noisy": noisy_ecg_full,
        "noise_bw": bw_comp_full, "noise_ma": ma_comp_full, "noise_pli": pli_comp_full
    }
    segmented_data = {}
    num_segments = -1
    for sig_type, sig_full in all_signals_to_segment.items():
        segments = segment_signal(sig_full, seg_len, overlap_len)
        if num_segments == -1: num_segments = len(segments)
        elif len(segments) != num_segments:
            print(f"⚠️ Discrepanza segmenti per {ecg_record_name} ({sig_type}). Salto."); return 0
        segmented_data[sig_type] = segments

    if num_segments == 0: return 0
    segments_processed_count = 0
    for i in range(num_segments):
        signals_for_this_segment = {st: segmented_data[st][i] for st in segmented_data}
        if save_segmented_signals(output_dir, ecg_record_name, i, signals_for_this_segment):
            segments_processed_count += 1
    # ... (plotting debug opzionale) ...
    return segments_processed_count

if __name__ == "__main__":
    print(f"--- Generazione Dataset ECG Segmentato Rumoroso (Lunghezza: {SEGMENT_LENGTH}) ---")
    try:
        all_ecg_files = sorted([f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")])
    except FileNotFoundError: print(f"❌ ERRORE: Cartella ECG '{ECG_DIR}' non trovata."); exit()
    if not all_ecg_files: print(f"❌ ERRORE: Nessun file .dat trovato in '{ECG_DIR}'."); exit()

    records_to_run = all_ecg_files if not RECORDS_TO_PROCESS else [r for r in RECORDS_TO_PROCESS if r in all_ecg_files]
    if not records_to_run: print("ℹ️ Nessun record da processare."); exit()
    print(f"▶️ Processando {len(records_to_run)} record ECG...")
    total_segments_generated = 0
    for ecg_name in tqdm(records_to_run, desc="Processing Records"):
        total_segments_generated += process_record_and_save_segments(ecg_name, SEGMENTED_SIGNAL_DIR, SEGMENT_LENGTH, OVERLAP_LENGTH)
    print(f"\n✅ Generazione completata. Segmenti totali: {total_segments_generated}")
    print(f"   File salvati in: '{SEGMENTED_SIGNAL_DIR}'")