# File: src/classifier/preprocess_ctm_multichannel.py
#!/usr/bin/env python3
"""
Preprocessing Multi-Canale per CTM Classifier.

Crea un dataset HDF5 dove ogni campione è un'immagine multi-canale,
combinando diverse "viste" del segnale ECG per massimizzare l'informazione
fornita al modello CTM.
"""
import os
import sys
import numpy as np
import h5py
import json
import argparse
from tqdm import tqdm
from scipy.signal import butter, sosfilt

# --- Import Path Setup ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

# Directory e costanti locali (evita dipendenze da loader)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SAMPLE_RATE = 360.0  # Hz, coerente con generator/classifier
from src.config import CLASSIFIER_PREPROC_PARAMS as PARAMS

# --- Configurazioni ---
INPUT_DIR = os.path.join(DATA_DIR, 'consolidated_data', 'classifier_data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'preprocessed_ctm_multichannel')

# --- Funzioni Helper Vettorizzate ---
def zscore_normalize(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + 1e-8)

def moving_rms(x, window_size):
    square = np.square(x)
    conv = np.convolve(square, np.ones(window_size)/window_size, mode='same')
    return np.sqrt(conv)

def binarize_channel(data, thresholds):
    n_bins = len(thresholds) + 1
    # Usa searchsorted per una binarizzazione vettorizzata veloce
    binned_data = np.searchsorted(thresholds, data, side='right')
    # Crea canali one-hot
    one_hot = np.eye(n_bins, dtype=np.uint8)[binned_data]
    # Trasponi per avere (N, Bins, Lunghezza)
    return np.transpose(one_hot, (0, 2, 1))

# --- Classe Preprocessor ---
class MultichannelPreprocessor:
    def __init__(self, config):
        self.config = config
        self.is_fitted = False
        self.thresholds = {}

    def fit(self, signals_chunk):
        print("🔧 Fitting preprocessor (calcolo quantili per ogni canale)...")
        # 1) Ampiezza (z-score), soglie per quantili
        amp_data = zscore_normalize(signals_chunk)
        self.thresholds['amplitude'] = np.percentile(
            amp_data,
            np.linspace(0, 100, self.config['amplitude_bins'] + 1)[1:-1]
        )

        # 2) Derivata prima
        deriv_data = zscore_normalize(
            np.diff(signals_chunk, axis=1, prepend=signals_chunk[:, :1])
        )
        self.thresholds['derivative'] = np.percentile(
            deriv_data,
            np.linspace(0, 100, self.config.get('derivative_bins', 32) + 1)[1:-1]
        )

        # 3) Seconda derivata
        deriv2 = zscore_normalize(
            np.diff(deriv_data, axis=1, prepend=deriv_data[:, :1])
        )
        self.thresholds['second_derivative'] = np.percentile(
            deriv2,
            np.linspace(0, 100, self.config.get('second_derivative_bins', 16) + 1)[1:-1]
        )

        # 4) RMS locale
        win = max(3, int((self.config.get('rms_win_ms', 40.0) / 1000.0) * SAMPLE_RATE))
        rms = np.vstack([moving_rms(x, win) for x in signals_chunk])
        rms = zscore_normalize(rms)
        self.thresholds['rms'] = np.percentile(
            rms,
            np.linspace(0, 100, self.config.get('rms_bins', 16) + 1)[1:-1]
        )

        # 5) Soglie spettrali per ciascuna banda
        bands = self.config.get('spectral_bands', [])
        self.thresholds['spectral'] = []
        if bands:
            try:
                from scipy.signal import stft
                nperseg = max(64, int(0.256 * SAMPLE_RATE))
                noverlap = int(0.75 * nperseg)
                for b_lo, b_hi in bands:
                    env_list = []
                    for x in signals_chunk:
                        f, t, Z = stft(
                            x,
                            fs=SAMPLE_RATE,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            window='hann',
                            padded=True,
                            boundary='zeros'
                        )
                        S = np.abs(Z)
                        mask = (f >= b_lo) & (f < b_hi)
                        env = S[mask].mean(axis=0) if np.any(mask) else np.zeros_like(t)
                        env_up = np.interp(
                            np.linspace(0, len(env) - 1, num=x.shape[-1]),
                            np.arange(len(env)),
                            env
                        )
                        env_list.append(env_up)
                    env_arr = zscore_normalize(np.vstack(env_list))
                    thr = np.percentile(
                        env_arr,
                        np.linspace(0, 100, self.config.get('spectral_bins', 8) + 1)[1:-1]
                    )
                    self.thresholds['spectral'].append(thr)
            except Exception as e:
                print(f"[WARN] STFT non disponibile o fallita durante fit: {e}")

        self.is_fitted = True
        print("✅ Preprocessor fittato.")

    def transform(self, signals_chunk):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        
        all_channels = []
        
        # Canale Ampiezza
        amp_data = zscore_normalize(signals_chunk)
        all_channels.append(binarize_channel(amp_data, self.thresholds['amplitude']))
        
        # Canale Derivata
        deriv_data = zscore_normalize(np.diff(signals_chunk, axis=1, prepend=signals_chunk[:, :1]))
        all_channels.append(binarize_channel(deriv_data, self.thresholds['derivative']))

        # Seconda Derivata
        deriv2 = zscore_normalize(np.diff(deriv_data, axis=1, prepend=deriv_data[:, :1]))
        all_channels.append(binarize_channel(deriv2, self.thresholds['second_derivative']))

        # RMS
        win = max(3, int((self.config.get('rms_win_ms', 40.0) / 1000.0) * SAMPLE_RATE))
        rms = np.vstack([moving_rms(x, win) for x in signals_chunk])
        rms = zscore_normalize(rms)
        all_channels.append(binarize_channel(rms, self.thresholds['rms']))

        # Spettrale per bande definite
        bands = self.config.get('spectral_bands', [])
        if bands:
            try:
                from scipy.signal import stft
                nperseg = max(64, int(0.256 * SAMPLE_RATE))
                noverlap = int(0.75 * nperseg)
                # STFT per batch (una per segnale per semplicità)
                for (b_lo, b_hi), thr in zip(bands, self.thresholds.get('spectral', [])):
                    env_list = []
                    for x in signals_chunk:
                        f, t, Z = stft(
                            x,
                            fs=SAMPLE_RATE,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            window='hann',
                            padded=True,
                            boundary='zeros'
                        )
                        S = np.abs(Z)
                        mask = (f >= b_lo) & (f < b_hi)
                        env = S[mask].mean(axis=0) if np.any(mask) else np.zeros_like(t)
                        # upsample a L
                        env_up = np.interp(
                            np.linspace(0, len(env) - 1, num=x.shape[-1]),
                            np.arange(len(env)),
                            env
                        )
                        env_list.append(env_up)
                    env_arr = zscore_normalize(np.vstack(env_list))
                    use_thr = thr if thr is not None else np.percentile(
                        env_arr,
                        np.linspace(0, 100, self.config.get('spectral_bins', 8) + 1)[1:-1]
                    )
                    all_channels.append(binarize_channel(env_arr, use_thr))
            except Exception as e:
                print(f"[WARN] STFT non disponibile o fallita: {e}")
        
        # Concatena tutti i canali
        X = np.concatenate(all_channels, axis=1).astype(np.uint8)
        # Pooling temporale opzionale per test rapidi
        pool = int(self.config.get('temporal_pool', 1))
        if pool > 1 and X.shape[-1] // pool >= 4:
            L = X.shape[-1]
            newL = (L // pool) * pool
            if newL > 0:
                X = X[:, :, :newL].reshape(X.shape[0], X.shape[1], newL // pool, pool).max(axis=-1)
        return X

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    preprocessor = MultichannelPreprocessor(PARAMS)

    # --- Fase 1: Fit su dati di training ---
    print("--- Fase 1: Fitting del Preprocessor ---")
    train_input_path = os.path.join(INPUT_DIR, "train_classifier_data.npz")
    with np.load(train_input_path) as data:
        X_train_noisy = data['X_noisy']
    preprocessor.fit(X_train_noisy)

    # Salva il preprocessor fittato
    preprocessor_path = os.path.join(OUTPUT_DIR, "ctm_multichannel_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"💾 Preprocessor salvato in: {preprocessor_path}")

    # --- Fase 2: Trasformazione e Salvataggio HDF5 ---
    output_h5_path = os.path.join(OUTPUT_DIR, "classifier_multichannel_data.h5")
    with h5py.File(output_h5_path, 'w') as hf:
        print(f"\n--- Fase 2: Creazione file HDF5: {output_h5_path} ---")
        
        for split in ['train', 'validation', 'test']:
            print(f"\n>> Processando split: {split}")
            input_path = os.path.join(INPUT_DIR, f"{split}_classifier_data.npz")
            with np.load(input_path) as data:
                X_noisy, y_class = data['X_noisy'], data['y_class']
            N = X_noisy.shape[0]
            bs = 512
            # Primo batch per inferire H,L
            head = min(bs, N)
            X_head = preprocessor.transform(X_noisy[:head])
            H, L = X_head.shape[1], X_head.shape[2]

            grp = hf.create_group(split)
            dX = grp.create_dataset('X', shape=(N, H, L), dtype=np.uint8, compression='gzip')
            dy = grp.create_dataset('y', data=y_class, dtype=np.uint32)
            dX[:head] = X_head
            wrote = head
            for i in range(head, N, bs):
                xb = X_noisy[i:i+bs]
                Xt = preprocessor.transform(xb)
                dX[i:i+Xt.shape[0]] = Xt
                wrote += Xt.shape[0]
            # meta
            grp.attrs['H'] = H
            grp.attrs['L'] = L
            grp.attrs['bins_amplitude'] = int(preprocessor.config.get('amplitude_bins', 0))
            grp.attrs['bins_derivative'] = int(preprocessor.config.get('derivative_bins', 0))
            grp.attrs['bins_second_derivative'] = int(preprocessor.config.get('second_derivative_bins', 0))
            grp.attrs['bins_rms'] = int(preprocessor.config.get('rms_bins', 0))
            grp.attrs['spectral_bands'] = json.dumps(preprocessor.config.get('spectral_bands', []))
            grp.attrs['spectral_bins'] = int(preprocessor.config.get('spectral_bins', 0))
            
            print(f"   ✅ Dati per '{split}' salvati: X={(N,H,L)}, y={y_class.shape}")

    print("\n🎉 Preprocessing Multi-Canale completato con successo!")

if __name__ == "__main__":
    # Importa le dipendenze necessarie solo se lo script è eseguito direttamente
    import joblib
    from src.config import CLASSIFIER_PREPROC_PARAMS
    main()