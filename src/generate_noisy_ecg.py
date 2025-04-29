# -*- coding: utf-8 -*-
# File: generate_noisy_ecg.py (Aggiornato per S4)
import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import errno
from tqdm import tqdm

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

ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
# --- MODIFICA: Directory di OUTPUT per tutti i segnali generati ---
GENERATED_SIGNAL_DIR = os.path.join(DATA_DIR, "generated_signals") # Nuova cartella o usa noisy_ecg/
# --------------------------------------------------------------------

print(f"INFO: Usando ECG_DIR: {ECG_DIR}")
print(f"INFO: Usando NOISE_DIR: {NOISE_DIR}")
print(f"INFO: Output Segnali Generati in: {GENERATED_SIGNAL_DIR}")
try: os.makedirs(GENERATED_SIGNAL_DIR, exist_ok=True) # Crea la nuova directory
except OSError as e:
     if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {GENERATED_SIGNAL_DIR}.")
except NameError: pass

# --- Configurazioni Generazione Rumore ---
RANDOMIZE_NOISE_LEVELS = True
LEVEL_MIN = 0.1; LEVEL_MAX = 0.6
RANDOMIZE_PLI_AMPLITUDE = True
PLI_AMP_MIN = 0.05; PLI_AMP_MAX = 0.25
ADD_PLI_HARMONICS_PROB = 0.15
RECORDS_TO_PROCESS = []
ENABLE_PLOTTING = False

# --- Funzioni Helper ---
def load_ecg_full(record_name):
    """Carica l'intero segnale ECG dal MIT-BIH."""
    record_path = os.path.join(ECG_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        ecg_signal_full = record.p_signal[:, 0]
        fs = record.fs
        if len(ecg_signal_full) == 0: raise ValueError("Segnale ECG vuoto.")
        return ecg_signal_full, fs
    except Exception as e:
        print(f"❌ Errore caricamento record completo {record_name}: {e}")
        return None, None

def load_noise_realistically(noise_name, length):
    # ... (Funzione invariata) ...
    noise_path = os.path.join(NOISE_DIR, noise_name)
    try:
        noise_record = wfdb.rdrecord(noise_path)
        full_noise_signal = noise_record.p_signal[:, 0]
    except Exception as e: print(f"❌ Errore caricamento rumore {noise_name}: {e}"); return np.zeros(length)
    current_len = len(full_noise_signal)
    if current_len == 0: print(f"⚠️ Rumore {noise_name} ha lunghezza 0."); return np.zeros(length)
    if current_len < length:
        repeats = int(np.ceil(length / current_len)); full_noise_signal = np.tile(full_noise_signal, repeats); current_len = len(full_noise_signal)
    if current_len > length:
        start_index = random.randint(0, current_len - length); noise_segment = full_noise_signal[start_index : start_index + length]
    else: noise_segment = full_noise_signal
    return noise_segment

def generate_pli(length, fs=360, base_freq=50):
    # ... (Funzione invariata) ...
    t = np.arange(length) / fs
    amplitude = random.uniform(PLI_AMP_MIN, PLI_AMP_MAX) if RANDOMIZE_PLI_AMPLITUDE else 0.1
    pli_signal = amplitude * np.sin(2 * np.pi * base_freq * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        amp_3 = amplitude * random.uniform(0.1, 0.4); pli_signal += amp_3 * np.sin(2 * np.pi * (3 * base_freq) * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        amp_5 = amplitude * random.uniform(0.05, 0.3); pli_signal += amp_5 * np.sin(2 * np.pi * (5 * base_freq) * t)
    return pli_signal

def generate_noisy_ecg_randomized(ecg_signal, bw_signal, ma_signal, pli_signal):
    # ... (Logica pesi invariata) ...
    if RANDOMIZE_NOISE_LEVELS:
        weights = np.random.uniform(0.1, 1.0, 3); weights /= np.sum(weights)
        weights = np.maximum(weights, LEVEL_MIN); weights /= np.sum(weights)
        bw_weight, ma_weight, pli_weight = weights
    else: bw_weight, ma_weight, pli_weight = (0.3, 0.2, 0.05)

    # Normalizza rumori
    max_abs_bw = np.max(np.abs(bw_signal)); max_abs_ma = np.max(np.abs(ma_signal)); max_abs_pli = np.max(np.abs(pli_signal))
    bw_norm = bw_signal / max_abs_bw if max_abs_bw > 1e-9 else bw_signal
    ma_norm = ma_signal / max_abs_ma if max_abs_ma > 1e-9 else ma_signal
    pli_norm = pli_signal / max_abs_pli if max_abs_pli > 1e-9 else pli_signal

    # Calcola componenti pesati
    bw_weighted = bw_norm * bw_weight
    ma_weighted = ma_norm * ma_weight
    pli_weighted = pli_norm * pli_weight

    # Calcola segnale rumoroso
    noisy_ecg = ecg_signal + bw_weighted + ma_weighted + pli_weighted

    # Restituisce anche i componenti individuali pesati
    return noisy_ecg, weights, bw_weighted, ma_weighted, pli_weighted

def calculate_snr(original_signal, noisy_signal):
    # ... (Funzione invariata) ...
    power_signal = np.mean(original_signal ** 2);
    if power_signal < 1e-15: return -np.inf
    noise = original_signal - noisy_signal; power_noise = np.mean(noise ** 2)
    if power_noise < 1e-15: return np.inf
    return 10 * np.log10(power_signal / power_noise)

# --- Funzione di Salvataggio Aggiornata ---
def save_all_signals(output_dir, ecg_name, signals_dict):
    """Salva tutti i segnali forniti in un dizionario in formato .npy"""
    base_path = os.path.join(output_dir, ecg_name)
    try:
        for key, signal_data in signals_dict.items():
            np.save(f"{base_path}_{key}.npy", signal_data)
        return True
    except Exception as e:
        print(f"❌ Errore salvataggio segnali per {ecg_name} in {output_dir}: {e}")
        return False

# --- Funzione Principale Aggiornata ---
def process_and_save_record(ecg_name, output_dir):
    """Carica l'intero record ECG, aggiunge rumore e salva tutti i segnali."""
    # print(f"🔹 Processando record completo: {ecg_name}...") # Riduci verbosità
    ecg_signal_full, fs = load_ecg_full(ecg_name)
    if ecg_signal_full is None: return

    length = len(ecg_signal_full)

    bw_signal = load_noise_realistically("bw", length)
    ma_signal = load_noise_realistically("ma", length)
    pli_signal = generate_pli(length, fs)

    noisy_signal, weights, bw_weighted, ma_weighted, pli_weighted = generate_noisy_ecg_randomized(
        ecg_signal_full, bw_signal, ma_signal, pli_signal
    )

    # Crea dizionario con tutti i segnali da salvare
    signals_to_save = {
        "clean": ecg_signal_full,
        "noisy": noisy_signal,
        "noise_bw": bw_weighted,
        "noise_ma": ma_weighted,
        "noise_pli": pli_weighted
    }

    # Salva tutti i segnali
    saved = save_all_signals(output_dir, ecg_name, signals_to_save)

    # Plotting (invariato, ma ora legge da signals_to_save se necessario)
    if saved and ENABLE_PLOTTING:
        # ... (codice plotting come prima, usando ecg_signal_full e noisy_signal) ...
        pass # Mantieni commentato per esecuzione batch

# --- Blocco Principale ---
if __name__ == "__main__":
    print("--- Generazione Dataset ECG Rumoroso (Segnali Completi per S4) ---")
    try:
        all_ecg_files = sorted([f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")])
    except FileNotFoundError: print(f"❌ Errore: Cartella ECG '{ECG_DIR}' non trovata."); exit()
    if not all_ecg_files: print(f"❌ Errore: Nessun file .dat trovato in '{ECG_DIR}'."); exit()

    if not RECORDS_TO_PROCESS: records_to_run = all_ecg_files
    else:
        records_to_run = [rec for rec in RECORDS_TO_PROCESS if rec in all_ecg_files]
        if len(records_to_run) != len(RECORDS_TO_PROCESS): print("⚠️ Attenzione: Record specificati non trovati.")

    print(f"▶️ Processando {len(records_to_run)} record completi...")
    # Usa la directory di output corretta
    output_directory = GENERATED_SIGNAL_DIR
    for ecg_name in tqdm(records_to_run, desc="Generating Signals"):
        process_and_save_record(ecg_name, output_directory) # Passa output_dir

    print(f"\n✅ Generazione completata per {len(records_to_run)} record.")
    print(f"   File salvati in: '{output_directory}'")
    print("   (Contengono: _clean.npy, _noisy.npy, _noise_bw.npy, _noise_ma.npy, _noise_pli.npy)")