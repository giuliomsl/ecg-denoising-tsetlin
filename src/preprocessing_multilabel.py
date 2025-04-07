import os
import numpy as np
import wfdb
from binarization import binarize

# === Configurazione ===
ECG_DIR = "data/mit-bih/"
NOISY_DIR = "data/noisy_ecg/"
PROCESSED_DIR = "data/processed_multilabel/"
SAMPLE_DIR = "data/samplewise/"
os.makedirs(SAMPLE_DIR, exist_ok=True)

WINDOW_SIZE = 1024
CONTEXT_K = 10  # finestra di contesto ±k → 2k+1
""" CONTEXT_K è cruciale, deve essere ottimizzato """
START = 30
DURATION = 10

def load_clean_ecg_segment(record_name):
    record_path = os.path.join(ECG_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    start_sample = int(START * fs)
    end_sample = int((START + DURATION) * fs)
    return record.p_signal[start_sample:end_sample, 0]

def load_noisy_ecg(record_name):
    path = os.path.join(NOISY_DIR, f"{record_name}_noisy_{START}-{START+DURATION}s.npy")
    return np.load(path)

def segment_signal(signal, window_size=WINDOW_SIZE):
    num_segments = len(signal) // window_size
    return np.array([signal[i * window_size:(i + 1) * window_size] for i in range(num_segments)])

def extract_samples(clean_bin, noisy_bin, k=CONTEXT_K):
    X_samples = []
    y_samples = []
    for seg_idx in range(noisy_bin.shape[0]):
        for t in range(k, WINDOW_SIZE - k):
            context = noisy_bin[seg_idx, t - k : t + k + 1, :]
            target = clean_bin[seg_idx, t, :]
            X_samples.append(context.flatten())      # shape: ( (2k+1)*features, )
            y_samples.append(target)                 # shape: (features,)
    return np.array(X_samples), np.array(y_samples)

def preprocess_and_save_all():
    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(f"_{START}-{START+DURATION}s.npy")])
    record_ids = sorted(set(f.split("_")[0] for f in noisy_files))

    all_X = []
    all_y = []

    for record in record_ids:
        print(f"🔹 Processing {record}...")
        try:
            clean_signal = load_clean_ecg_segment(record)
            noisy_signal = load_noisy_ecg(record)
        except FileNotFoundError:
            print(f"⚠️ File mancante per {record}")
            continue

        clean_segments = segment_signal(clean_signal)
        noisy_segments = segment_signal(noisy_signal)

        n_segments = min(len(clean_segments), len(noisy_segments))
        clean_segments = clean_segments[:n_segments]
        noisy_segments = noisy_segments[:n_segments]

        clean_bin = np.array([binarize(seg, method="combined") for seg in clean_segments])
        noisy_bin = np.array([binarize(seg, method="combined") for seg in noisy_segments])

        X, y = extract_samples(clean_bin, noisy_bin, k=CONTEXT_K)
        all_X.append(X)
        all_y.append(y)

    X_total = np.concatenate(all_X)
    y_total = np.concatenate(all_y)

    np.save(os.path.join(SAMPLE_DIR, "X_train_samples.npy"), X_total)
    np.save(os.path.join(SAMPLE_DIR, "y_train_samples.npy"), y_total)

    print(f"\n✅ Salvataggio completato:")
    print(f"   X shape: {X_total.shape}")
    print(f"   y shape: {y_total.shape}")

if __name__ == "__main__":
    preprocess_and_save_all()
