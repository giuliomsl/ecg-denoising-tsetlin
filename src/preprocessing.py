
import os
import numpy as np
import wfdb
from binarization import binarize

# === Configurazione ===
ECG_DIR = "data/mit-bih/"
NOISY_DIR = "data/noisy_ecg/"
OUTPUT_DIR = "data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 1024  # campioni per finestra
START = 30          # in secondi
DURATION = 10       # in secondi

def load_clean_ecg_segment(record_name, start=START, duration=DURATION):
    """Carica una finestra (segmento) di ECG pulito dal MIT-BIH."""
    record_path = os.path.join(ECG_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    start_sample = int(start * fs)
    end_sample = int((start + duration) * fs)
    return record.p_signal[start_sample:end_sample, 0]  # Primo canale

def load_noisy_ecg(record_name, start=START, duration=DURATION):
    """Carica un ECG rumoroso salvato in formato .npy."""
    noisy_path = os.path.join(NOISY_DIR, f"{record_name}_noisy_{start}-{start+duration}s.npy")
    return np.load(noisy_path)

def segment_signal(signal, window_size=WINDOW_SIZE):
    """Segmenta il segnale in finestre di dimensione fissa."""
    num_segments = len(signal) // window_size
    return np.array([signal[i * window_size:(i + 1) * window_size] for i in range(num_segments)])

def preprocess_and_save(record_name):
    print(f"🔹 Processando {record_name}...")

    try:
        clean_signal = load_clean_ecg_segment(record_name)
        noisy_signal = load_noisy_ecg(record_name)
    except FileNotFoundError:
        print(f"⚠️ Segnali mancanti per {record_name}, skip.")
        return

    # Segmentazione
    clean_segments = segment_signal(clean_signal)
    noisy_segments = segment_signal(noisy_signal)

    # Allineamento numero finestre
    n_segments = min(len(clean_segments), len(noisy_segments))
    if n_segments == 0:
        print(f"⚠️ Nessuna finestra valida per {record_name}, skip.")
        return

    clean_segments = clean_segments[:n_segments]
    noisy_segments = noisy_segments[:n_segments]

    # Binarizzazione
    clean_bin = np.array([binarize(seg, method="combined") for seg in clean_segments])
    noisy_bin = np.array([binarize(seg, method="combined") for seg in noisy_segments])

    # Salvataggio
    np.save(os.path.join(OUTPUT_DIR, f"{record_name}_clean.npy"), clean_bin)
    np.save(os.path.join(OUTPUT_DIR, f"{record_name}_noisy.npy"), noisy_bin)
    print(f"✅ Salvati {n_segments} segmenti per {record_name}")

if __name__ == "__main__":
    # Preprocessa solo record per cui esiste il file noisy corrispondente
    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(f"_{START}-{START+DURATION}s.npy")])
    record_ids = sorted(set(f.split("_")[0] for f in noisy_files))

    for record in record_ids:
        preprocess_and_save(record)
