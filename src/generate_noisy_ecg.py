# -*- coding: utf-8 -*-
import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import random # Per randomizzare
import sys
import errno

# --- FLAG PER AMBIENTE ---
# Imposta a True se stai eseguendo su Google Colab, False se in locale
RUNNING_ON_COLAB = False # <--- MODIFICA QUESTA RIGA MANUALMENTE
# -------------------------

# --- Definizione Percorsi basata sul Flag ---
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
         # Potresti voler uscire qui o impostare percorsi di default
         exit()


    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/" # <--- Modifica se necessario
    REPO_NAME = "denoising_ecg"  # <--- Modifica col nome del tuo repo clonato
    PROJECT_ROOT = f"/content/{REPO_NAME}/" # Percorso del progetto clonato in Colab

    DATA_DIR = os.path.join(GDRIVE_BASE, "data")
    MODEL_DIR = os.path.join(GDRIVE_BASE, "models")

else:
    print("INFO: Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    # Definisci i percorsi relativi o assoluti per l'ambiente locale
    # Assumiamo esecuzione dalla root del progetto per semplicità
    PROJECT_ROOT = "."
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Costruisci percorsi specifici che verranno usati nel resto dello script
ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samplewise/")
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "multi_tm_denoiser/")
BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

print(f"INFO: Usando DATA_DIR: {DATA_DIR}")
print(f"INFO: Usando MODEL_DIR: {MODEL_DIR}")

# Assicurati che le directory di output esistano (necessario crearle
# sia in locale che su Drive se non esistono già)
try:
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOISY_ECG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST:
         print(f"⚠️ Attenzione: impossibile creare directory {e.filename}. Assicurati che il percorso base esista.")
except NameError: # Se errno non è importato
     pass # Ignora l'errore se non possiamo controllare errno


# Coerenza con preprocessing_multilabel.py
START = 30
DURATION = 10

# Opzioni per randomizzazione rumore
RANDOMIZE_NOISE_LEVELS = True
LEVEL_MIN = 0.1
LEVEL_MAX = 0.6 # Limite superiore per singolo noise prima di normalizzare somma a 1
RANDOMIZE_PLI_AMPLITUDE = True
PLI_AMP_MIN = 0.05 # Esempio
PLI_AMP_MAX = 0.25 # Esempio
ADD_PLI_HARMONICS_PROB = 0.15 # Probabilità di aggiungere armoniche

# Record da processare (lista vuota = tutti quelli trovati)
# Esempio: RECORDS_TO_PROCESS = ["100", "101", "103", "105", "106", "108", "109", "111"]
RECORDS_TO_PROCESS = [] # Lascia vuoto per processare tutti i file .dat in ECG_DIR

# Disabilita plotting di massa
ENABLE_PLOTTING = False # Metti a True solo per debug di pochi file

# === Codice ===

def load_ecg(record_name, start=START, duration=DURATION):
    """Carica un segnale ECG dal MIT-BIH e ne estrae una finestra temporale."""
    record_path = os.path.join(ECG_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"❌ Errore caricamento record {record_name}: {e}")
        return None, None
    fs = record.fs

    start_sample = int(start * fs)
    end_sample = int((start + duration) * fs)

    # Gestione lunghezza segnale
    signal_len = record.p_signal.shape[0]
    end_sample = min(end_sample, signal_len)
    start_sample = min(start_sample, end_sample)

    if end_sample <= start_sample:
        print(f"⚠️ Finestra non valida per {record_name} ({start}-{start+duration}s). Skip.")
        return None, None

    # Estrarre solo il segmento richiesto (canale 0)
    ecg_signal = record.p_signal[start_sample:end_sample, 0]
    return ecg_signal, fs

def load_noise_realistically(noise_name, length):
    """
    Carica un segnale di rumore dal Noise Stress Test Database e ne estrae
    un segmento casuale della lunghezza richiesta, piastrellando se necessario.
    """
    noise_path = os.path.join(NOISE_DIR, noise_name)
    try:
        noise_record = wfdb.rdrecord(noise_path)
        # Usa il primo canale se ce ne sono multipli (es. em)
        full_noise_signal = noise_record.p_signal[:, 0]
    except Exception as e:
        print(f"❌ Errore caricamento rumore {noise_name}: {e}")
        return np.zeros(length) # Restituisce zero se non caricabile

    current_len = len(full_noise_signal)

    if current_len == 0:
         print(f"⚠️ Rumore {noise_name} ha lunghezza 0. Restituisco zeri.")
         return np.zeros(length)

    # Piastrella se il rumore è più corto della lunghezza richiesta
    if current_len < length:
        repeats = int(np.ceil(length / current_len))
        full_noise_signal = np.tile(full_noise_signal, repeats)
        current_len = len(full_noise_signal) # Lunghezza aggiornata

    # Seleziona un punto di inizio casuale
    if current_len > length:
        start_index = random.randint(0, current_len - length)
        noise_segment = full_noise_signal[start_index : start_index + length]
    else: # Se la lunghezza è esattamente quella giusta dopo il tiling
        noise_segment = full_noise_signal

    return noise_segment

def generate_pli(length, fs=360, base_freq=50):
    """Genera un segnale di Powerline Interference (PLI) sintetico."""
    t = np.arange(length) / fs  # Tempo in secondi

    # Ampiezza base
    amplitude = random.uniform(PLI_AMP_MIN, PLI_AMP_MAX) if RANDOMIZE_PLI_AMPLITUDE else 0.1

    pli_signal = amplitude * np.sin(2 * np.pi * base_freq * t)

    # Aggiungi armoniche (opzionale)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        # 3a armonica (ampiezza ridotta)
        amp_3 = amplitude * random.uniform(0.1, 0.4) # Esempio: 10-40% dell'amp base
        pli_signal += amp_3 * np.sin(2 * np.pi * (3 * base_freq) * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
         # 5a armonica (ampiezza ridotta)
        amp_5 = amplitude * random.uniform(0.05, 0.3) # Esempio: 5-30% dell'amp base
        pli_signal += amp_5 * np.sin(2 * np.pi * (5 * base_freq) * t)

    return pli_signal

def generate_noisy_ecg_randomized(ecg_signal, bw_signal, ma_signal, pli_signal):
    """Combina ECG pulito con i rumori usando pesi randomizzati."""

    if RANDOMIZE_NOISE_LEVELS:
        # Campiona pesi casuali e normalizza la somma a 1
        weights = np.random.uniform(0.1, 1.0, 3) # Campiona da 0.1 a 1.0
        # Applica limite massimo se necessario (opzionale, dipende se vuoi rispettare 0.6 max *prima* della normalizzazione)
        # weights = np.clip(weights, None, LEVEL_MAX)
        weights /= np.sum(weights) # Normalizza somma a 1
        # Assicura che nessun peso sia sotto il minimo dopo normalizzazione (raro ma possibile)
        weights = np.maximum(weights, LEVEL_MIN)
        # Rinormalizza se necessario dopo aver applicato il minimo
        weights /= np.sum(weights)
        bw_weight, ma_weight, pli_weight = weights
    else:
        # Usa pesi fissi (come prima, ma non ideale)
        bw_weight, ma_weight, pli_weight = (0.3, 0.2, 0.05) # Esempio fisso

    # Normalizza ampiezza rumori individualmente prima di applicare i pesi
    # Evita divisione per zero se il rumore è piatto
    max_abs_bw = np.max(np.abs(bw_signal))
    max_abs_ma = np.max(np.abs(ma_signal))
    max_abs_pli = np.max(np.abs(pli_signal))

    bw_norm = bw_signal / max_abs_bw if max_abs_bw > 1e-9 else bw_signal
    ma_norm = ma_signal / max_abs_ma if max_abs_ma > 1e-9 else ma_signal
    pli_norm = pli_signal / max_abs_pli if max_abs_pli > 1e-9 else pli_signal

    # Combina con i pesi
    # Nota: Questo approccio somma i rumori pesati all'ECG originale.
    # Se l'ECG ha ampiezze molto diverse, l'SNR risultante varierà.
    # Un approccio alternativo è definire l'SNR target e scalare il rumore *combinato*.
    noisy_ecg = ecg_signal + (bw_norm * bw_weight) + (ma_norm * ma_weight) + (pli_norm * pli_weight)

    return noisy_ecg, (bw_weight, ma_weight, pli_weight) # Restituisce anche i pesi usati

def calculate_snr(original_signal, noisy_signal):
    """Calcola il rapporto segnale-rumore (SNR) in decibel."""
    power_signal = np.mean(original_signal ** 2)
    # Evita errore se segnale è zero
    if power_signal < 1e-15: return -np.inf
    noise = original_signal - noisy_signal
    power_noise = np.mean(noise ** 2)
    # Evita divisione per zero o log(0)
    if power_noise < 1e-15: return np.inf
    return 10 * np.log10(power_signal / power_noise)

def save_noisy_ecg(ecg_name, noisy_signal, start, duration):
    """Salva l'ECG rumoroso in formato .npy"""
    output_path = os.path.join(NOISY_ECG_DIR, f"{ecg_name}_noisy_{start}-{start+duration}s.npy")
    try:
        np.save(output_path, noisy_signal)
        # print(f"✅ Salvato: {output_path}") # Riduci verbosità
        return True
    except Exception as e:
        print(f"❌ Errore salvataggio {output_path}: {e}")
        return False

def process_and_save_noisy_ecg(ecg_name, start=START, duration=DURATION):
    """Genera e salva un segnale ECG rumoroso su una finestra temporale selezionata."""
    # print(f"🔹 Processando {ecg_name} (finestra {start}-{start+duration}s)...") # Riduci verbosità

    # Carica il segmento ECG
    ecg_signal, fs = load_ecg(ecg_name, start, duration)
    if ecg_signal is None:
        return # Errore caricamento o finestra non valida

    length = len(ecg_signal)
    if length < duration * fs * 0.9: # Controllo lunghezza minima ragionevole
         print(f"⚠️ Segmento ECG per {ecg_name} troppo corto ({length} campioni). Skip.")
         return

    # Carica i rumori (BW, MA) realisticamente
    bw_signal = load_noise_realistically("bw", length)
    ma_signal = load_noise_realistically("ma", length)
    # Genera PLI (con possibile randomizzazione)
    pli_signal = generate_pli(length, fs)

    # Genera l'ECG rumoroso (con pesi randomizzati)
    noisy_signal, weights = generate_noisy_ecg_randomized(ecg_signal, bw_signal, ma_signal, pli_signal)

    # Calcola il SNR (opzionale, può rallentare)
    # snr_value = calculate_snr(ecg_signal, noisy_signal)
    # print(f"📊 SNR: {snr_value:.2f} dB | Pesi (BW,MA,PLI): ({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})")

    # Salva il segnale rumoroso
    saved = save_noisy_ecg(ecg_name, noisy_signal, start, duration)

    # Plot per verifica (solo se abilitato)
    if saved and ENABLE_PLOTTING:
        plt.figure(figsize=(12, 5))
        plt.plot(ecg_signal, label="ECG Pulito", color='blue')
        plt.plot(noisy_signal, label="ECG Rumoroso", alpha=0.7, color='orange')
        plt.xlabel("Campioni")
        plt.ylabel("Ampiezza")
        plt.legend()
        snr_value = calculate_snr(ecg_signal, noisy_signal) # Ricalcola se serve per titolo
        plt.title(f"ECG Pulito vs Rumoroso - {ecg_name} ({start}-{start+duration}s) - SNR: {snr_value:.2f} dB")
        plt.show()

# --- Blocco Principale ---
if __name__ == "__main__":
    print("--- Generazione Dataset ECG Rumoroso ---")
    # Lista dei file ECG disponibili
    try:
        all_ecg_files = sorted([f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")])
    except FileNotFoundError:
        print(f"❌ Errore: Cartella ECG '{ECG_DIR}' non trovata.")
        exit()

    if not all_ecg_files:
        print(f"❌ Errore: Nessun file .dat trovato in '{ECG_DIR}'.")
        exit()

    if not RECORDS_TO_PROCESS: # Se la lista è vuota, usa tutti i file trovati
        records_to_run = all_ecg_files
    else:
        records_to_run = [rec for rec in RECORDS_TO_PROCESS if rec in all_ecg_files]
        if len(records_to_run) != len(RECORDS_TO_PROCESS):
             print("⚠️ Attenzione: Alcuni record specificati in RECORDS_TO_PROCESS non sono stati trovati.")

    print(f"▶️ Processando {len(records_to_run)} record per la finestra {START}-{START+DURATION}s...")
    processed_count = 0
    for ecg_name in records_to_run:
        process_and_save_noisy_ecg(ecg_name, start=START, duration=DURATION)
        processed_count += 1
        if processed_count % 10 == 0: # Aggiornamento ogni 10 record
             print(f"  {processed_count}/{len(records_to_run)} record processati...")

    print(f"\n✅ Generazione completata per {processed_count} record.")
    print(f"   File rumorosi salvati in: '{NOISY_ECG_DIR}'")
    print(f"   Finestra temporale: {START}s - {START+DURATION}s")