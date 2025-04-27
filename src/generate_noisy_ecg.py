# -*- coding: utf-8 -*-
import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import errno
from tqdm import tqdm # Aggiungi tqdm per il loop principale

# --- FLAG PER AMBIENTE ---
RUNNING_ON_COLAB = False
# -------------------------

# --- Definizione Percorsi ---
# (Blocco if/else invariato)
# ... (definisce DATA_DIR, MODEL_DIR, ECG_DIR, NOISE_DIR, NOISY_ECG_DIR) ...
if RUNNING_ON_COLAB:
    print("INFO: Flag RUNNING_ON_COLAB impostato a True.")
    try: from google.colab import drive; drive.mount('/content/drive', force_remount=True); import time; time.sleep(5)
    except ImportError: print("ERRORE: Impossibile importare google.colab."); exit()
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data"); MODEL_DIR = os.path.join(GDRIVE_BASE, "models")
else:
    print("INFO: Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    PROJECT_ROOT = "."; DATA_DIR = os.path.join(PROJECT_ROOT, "data"); MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/") # Output directory

print(f"INFO: Usando ECG_DIR: {ECG_DIR}")
print(f"INFO: Usando NOISE_DIR: {NOISE_DIR}")
print(f"INFO: Output ECG Rumorosi in: {NOISY_ECG_DIR}")
try: os.makedirs(NOISY_ECG_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST: print(f"⚠️ Attenzione: impossibile creare directory {NOISY_ECG_DIR}.")
except NameError: pass

# --- Configurazioni Generazione Rumore ---
# (Parametri di randomizzazione come prima)
RANDOMIZE_NOISE_LEVELS = True
LEVEL_MIN = 0.1; LEVEL_MAX = 0.6
RANDOMIZE_PLI_AMPLITUDE = True
PLI_AMP_MIN = 0.05; PLI_AMP_MAX = 0.25
ADD_PLI_HARMONICS_PROB = 0.15

RECORDS_TO_PROCESS = [] # Processa tutti i record trovati se vuoto
ENABLE_PLOTTING = False # Disabilitato per esecuzione batch

# --- Funzioni (load_noise_realistically, generate_pli, generate_noisy_ecg_randomized, calculate_snr - INVARIATE) ---
# ... (copia qui le definizioni di queste funzioni dalla versione precedente) ...
def load_noise_realistically(noise_name, length):
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
    t = np.arange(length) / fs
    amplitude = random.uniform(PLI_AMP_MIN, PLI_AMP_MAX) if RANDOMIZE_PLI_AMPLITUDE else 0.1
    pli_signal = amplitude * np.sin(2 * np.pi * base_freq * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        amp_3 = amplitude * random.uniform(0.1, 0.4); pli_signal += amp_3 * np.sin(2 * np.pi * (3 * base_freq) * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        amp_5 = amplitude * random.uniform(0.05, 0.3); pli_signal += amp_5 * np.sin(2 * np.pi * (5 * base_freq) * t)
    return pli_signal

def generate_noisy_ecg_randomized(ecg_signal, bw_signal, ma_signal, pli_signal):
    if RANDOMIZE_NOISE_LEVELS:
        weights = np.random.uniform(0.1, 1.0, 3); weights /= np.sum(weights)
        weights = np.maximum(weights, LEVEL_MIN); weights /= np.sum(weights)
        bw_weight, ma_weight, pli_weight = weights
    else: bw_weight, ma_weight, pli_weight = (0.3, 0.2, 0.05)
    max_abs_bw = np.max(np.abs(bw_signal)); max_abs_ma = np.max(np.abs(ma_signal)); max_abs_pli = np.max(np.abs(pli_signal))
    bw_norm = bw_signal / max_abs_bw if max_abs_bw > 1e-9 else bw_signal
    ma_norm = ma_signal / max_abs_ma if max_abs_ma > 1e-9 else ma_signal
    pli_norm = pli_signal / max_abs_pli if max_abs_pli > 1e-9 else pli_signal
    # --- MODIFICA: Restituisce anche i componenti pesati ---
    bw_weighted = bw_norm * bw_weight
    ma_weighted = ma_norm * ma_weight
    pli_weighted = pli_norm * pli_weight
    noisy_ecg = ecg_signal + bw_weighted + ma_weighted + pli_weighted
    # Restituisce noisy, pesi, e componenti individuali pesati
    return noisy_ecg, weights, bw_weighted, ma_weighted, pli_weighted
# ---------------------------------------------------------

def calculate_snr(original_signal, noisy_signal):
    power_signal = np.mean(original_signal ** 2);
    if power_signal < 1e-15: return -np.inf
    noise = original_signal - noisy_signal; power_noise = np.mean(noise ** 2)
    if power_noise < 1e-15: return np.inf
    return 10 * np.log10(power_signal / power_noise)

# --- Funzione di Salvataggio (Modificata per salvare anche componenti rumore) ---
def save_signals(output_dir, ecg_name, noisy_signal, clean_signal, bw_noise, ma_noise, pli_noise):
    """Salva l'ECG rumoroso, pulito e i componenti di rumore in formato .npy"""
    base_path = os.path.join(output_dir, ecg_name)
    try:
        np.save(f"{base_path}_noisy.npy", noisy_signal)
        np.save(f"{base_path}_clean.npy", clean_signal) # Salva anche il pulito corrispondente
        np.save(f"{base_path}_noise_bw.npy", bw_noise) # Salva componente BW
        np.save(f"{base_path}_noise_ma.npy", ma_noise) # Salva componente MA
        np.save(f"{base_path}_noise_pli.npy", pli_noise) # Salva componente PLI
        return True
    except Exception as e:
        print(f"❌ Errore salvataggio segnali per {ecg_name} in {output_dir}: {e}")
        return False

# --- Funzione Principale (Modificata per processare l'intero record) ---
def process_and_save_record(ecg_name):
    """Carica l'intero record ECG, aggiunge rumore e salva i segnali."""
    print(f"🔹 Processando record completo: {ecg_name}...")
    record_path = os.path.join(ECG_DIR, ecg_name)
    try:
        record = wfdb.rdrecord(record_path)
        ecg_signal_full = record.p_signal[:, 0] # Prendi il primo canale
        fs = record.fs
        length = len(ecg_signal_full)
        if length == 0: raise ValueError("Segnale ECG vuoto.")
    except Exception as e:
        print(f"❌ Errore caricamento record completo {ecg_name}: {e}")
        return

    # Carica/Genera rumori della stessa lunghezza del segnale ECG
    bw_signal = load_noise_realistically("bw", length)
    ma_signal = load_noise_realistically("ma", length)
    pli_signal = generate_pli(length, fs)

    # Genera l'ECG rumoroso e ottieni i componenti di rumore pesati
    noisy_signal, weights, bw_weighted, ma_weighted, pli_weighted = generate_noisy_ecg_randomized(
        ecg_signal_full, bw_signal, ma_signal, pli_signal
    )

    # Salva tutti i segnali necessari (rumoroso, pulito, e componenti rumore)
    # Salva nella directory NOISY_ECG_DIR per coerenza con preprocessing
    saved = save_signals(NOISY_ECG_DIR, ecg_name,
                         noisy_signal, ecg_signal_full, # Salva rumoroso e pulito
                         bw_weighted, ma_weighted, pli_weighted) # Salva componenti

    if saved and ENABLE_PLOTTING: # Plotta solo un piccolo segmento per verifica
        plot_start_sec = 30
        plot_duration_sec = 5
        start_idx = int(plot_start_sec * fs)
        end_idx = int((plot_start_sec + plot_duration_sec) * fs)
        end_idx = min(end_idx, length)
        start_idx = min(start_idx, end_idx)

        if end_idx > start_idx:
            plt.figure(figsize=(15, 6))
            time_axis = np.arange(start_idx, end_idx) / fs
            plt.plot(time_axis, ecg_signal_full[start_idx:end_idx], label="Pulito", color='green')
            plt.plot(time_axis, noisy_signal[start_idx:end_idx], label="Rumoroso", color='red', alpha=0.7)
            # Potresti plottare anche i rumori individuali qui per debug
            # plt.plot(time_axis, bw_weighted[start_idx:end_idx], label="BW", alpha=0.5)
            # plt.plot(time_axis, ma_weighted[start_idx:end_idx], label="MA", alpha=0.5)
            # plt.plot(time_axis, pli_weighted[start_idx:end_idx], label="PLI", alpha=0.5)
            plt.xlabel("Tempo (s)")
            plt.ylabel("Ampiezza")
            plt.legend()
            snr_segment = calculate_snr(ecg_signal_full[start_idx:end_idx], noisy_signal[start_idx:end_idx])
            plt.title(f"Record {ecg_name} (Segmento {plot_start_sec}-{plot_start_sec+plot_duration_sec}s) - SNR Segmento: {snr_segment:.2f} dB")
            plt.show()

# --- Blocco Principale (Modificato) ---
if __name__ == "__main__":
    print("--- Generazione Dataset ECG Rumoroso (Segnali Completi) ---")
    try:
        all_ecg_files = sorted([f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")])
    except FileNotFoundError: print(f"❌ Errore: Cartella ECG '{ECG_DIR}' non trovata."); exit()
    if not all_ecg_files: print(f"❌ Errore: Nessun file .dat trovato in '{ECG_DIR}'."); exit()

    if not RECORDS_TO_PROCESS: records_to_run = all_ecg_files
    else:
        records_to_run = [rec for rec in RECORDS_TO_PROCESS if rec in all_ecg_files]
        if len(records_to_run) != len(RECORDS_TO_PROCESS): print("⚠️ Attenzione: Record specificati non trovati.")

    print(f"▶️ Processando {len(records_to_run)} record completi...")
    # Usa tqdm per la barra di progresso sul loop principale
    for ecg_name in tqdm(records_to_run, desc="Generating Noisy Records"):
        process_and_save_record(ecg_name) # Chiama la nuova funzione

    print(f"\n✅ Generazione completata per {len(records_to_run)} record.")
    print(f"   File salvati in: '{NOISY_ECG_DIR}'")
    print("   (Contengono: _noisy.npy, _clean.npy, _noise_bw.npy, _noise_ma.npy, _noise_pli.npy)")