#!/usr/bin/env python3
"""
Preprocess CTM Contrastivo: 2 viste per campione
- Vista 1: waveform z-score binarizzata (feature learner principale)
- Vista 2: magnitudo spettrale compatta (aiuta a distinguere rumore vs segnale)

Input: NPZ consolidati in data/consolidated_data/classifier_data/*_classifier_data.npz
Output: HDF5 con X shape (N, H, L) dove H = bins_wave + bins_spec
"""
import os, sys, json
import numpy as np
import h5py
from tqdm import tqdm

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
except Exception:
    PROJECT_ROOT = os.getcwd()

from src.config import CLASSIFIER_PREPROC_PARAMS as PARAMS

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'consolidated_data', 'classifier_data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'preprocessed_ctm_contrastive')

def zscore(x):
    m = x.mean(axis=1, keepdims=True); s = x.std(axis=1, keepdims=True) + 1e-8
    return (x - m) / s

def binarize_quantiles(X, n_bins):
    # Calcola soglie su X (batch intero) per stabilità; usa percentili equispaziati
    thr = np.percentile(X, np.linspace(0, 100, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    # Assegna bin
    idx = np.searchsorted(thr, X, side='right')
    one_hot = np.eye(n_bins, dtype=np.uint8)[idx]
    return np.transpose(one_hot, (0, 2, 1)), thr

def spectral_magnitude(x, n_fft=256, hop=128):
    # STFT semplificata usando numpy.fft (finestra Hann implicita via smoothing opzionale)
    # Segmenta in frame con hop e calcola |FFT| per ciascun frame, poi interpola a L
    import numpy.fft as nfft
    N = x.shape[-1]
    frames = []
    for start in range(0, N - n_fft + 1, hop):
        seg = x[..., start:start+n_fft]
        win = np.hanning(n_fft).astype(np.float32)
        mag = np.abs(nfft.rfft(seg * win, axis=-1))
        frames.append(mag)
    if not frames:
        # pad minimo
        pad = np.pad(x, ((0,0),(0, max(0, n_fft - N))), mode='reflect')[:, :n_fft]
        frames = [np.abs(nfft.rfft(pad * np.hanning(n_fft), axis=-1))]
    S = np.stack(frames, axis=1)  # (B, T, F)
    # Media sulle frequenze per una vista compatta; possiamo anche prendere banda ECG vs PLI, ma qui compattiamo
    env = S.mean(axis=-1)  # (B, T)
    # Interpola a lunghezza originale
    B, T = env.shape
    L = x.shape[-1]
    t = np.linspace(0, T - 1, num=L)
    env_up = np.stack([np.interp(t, np.arange(T), env[i]) for i in range(B)], axis=0)
    return env_up.astype(np.float32)

def process_split(hf, split, bins_wave=48, bins_spec=16):
    inp = os.path.join(INPUT_DIR, f"{split}_classifier_data.npz")
    with np.load(inp) as data:
        X_noisy = data['X_noisy']
        y = data['y_class']
    # Vista 1: waveform
    Xw = zscore(X_noisy)
    Xw_bin, thr_w = binarize_quantiles(Xw, bins_wave)
    # Vista 2: spettrale compatta
    Xs_env = zscore(spectral_magnitude(X_noisy))
    Xs_bin, thr_s = binarize_quantiles(Xs_env, bins_spec)
    X = np.concatenate([Xw_bin, Xs_bin], axis=1).astype(np.uint8)
    H, L = X.shape[1], X.shape[2]

    grp = hf.create_group(split)
    grp.create_dataset('X', data=X, dtype=np.uint8, compression='gzip')
    grp.create_dataset('y', data=y, dtype=np.uint32)
    grp.attrs['H'] = int(H); grp.attrs['L'] = int(L)
    grp.attrs['bins_wave'] = int(bins_wave)
    grp.attrs['bins_spec'] = int(bins_spec)
    grp.attrs['thr_wave'] = json.dumps([float(v) for v in thr_w.tolist()]) if thr_w.size else json.dumps([])
    grp.attrs['thr_spec'] = json.dumps([float(v) for v in thr_s.tolist()]) if thr_s.size else json.dumps([])
    print(f"   ✅ {split}: X={X.shape} y={y.shape}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_h5 = os.path.join(OUTPUT_DIR, 'classifier_contrastive_data.h5')
    bins_wave = int(PARAMS.get('amplitude_bins', 48))
    # Usa meno bin per vista spettrale per restare compatto
    bins_spec = max(6, int(PARAMS.get('spectral_bins', 8)))
    with h5py.File(out_h5, 'w') as hf:
        for split in ('train','validation','test'):
            print(f"[Preprocess Contrastivo] split={split}")
            process_split(hf, split, bins_wave=bins_wave, bins_spec=bins_spec)
    print("🎉 Contrastive preprocessing completato:", out_h5)

if __name__ == '__main__':
    main()
