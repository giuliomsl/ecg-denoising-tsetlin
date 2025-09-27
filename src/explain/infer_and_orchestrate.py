#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferenza explainability TM + denoise orchestrato (robusto a set_state).
- Carica modelli RTM (bw/emg/pli) da .state.npz/.meta.json (fallback .joblib).
- Inizializza RTM con un fit "dummy" per allocare strutture interne (evita segfault).
- Canonizza i dtypes dello stato (int32/float32, contigui).
- Ricostruisce mappe di intensità (overlap-add), calibra + smooth, applica filtri (opz).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import h5py
import scipy.ndimage as ndi

# -------------------- feature encoder --------------------
# Se usi il tuo modulo, importa da lì:
import sys
sys.path.append('src')
try:
    from explain.features import build_features
except Exception:
    # fallback minimale: errore chiaro
    raise SystemExit("Non trovo 'explain.features.build_features'. Assicurati che sia nel PYTHONPATH.")

# -------------------- DSP filters --------------------
from scipy.signal import iirnotch, filtfilt, butter

def notch_filter(x, fs=360.0, f0=50.0, Q=30.0, strength=1.0):
    if strength <= 1e-3: return x
    w0 = f0/(fs/2.0)
    b,a = iirnotch(w0, Q=max(5.0, Q*(1.0+2.0*strength)))
    return filtfilt(b,a,x).astype(np.float32)

def highpass_butter(x, fs=360.0, cutoff=0.5, order=2, strength=1.0):
    if strength <= 1e-3: return x
    co = cutoff*(1.0 + 0.5*strength)
    co = min(co, 1.5)  # non esagerare
    b,a = butter(int(order), co/(fs/2.0), btype='highpass')
    return filtfilt(b,a,x).astype(np.float32)

def wavelet_denoise(x, strength=1.0, wavelet="bior2.6", mode="symmetric"):
    if strength <= 1e-3: return x.astype(np.float32)
    import pywt
    coeffs = pywt.wavedec(x.astype(np.float32), wavelet=wavelet, mode=mode)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745 + 1e-9
    thr = sigma * np.sqrt(2*np.log(len(x))) * (0.5 + 1.5*strength)
    coeffs[1:] = [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
    y = pywt.waverec(coeffs, wavelet=wavelet, mode=mode)
    return y[:len(x)].astype(np.float32)

# -------------------- helpers --------------------
def overlap_add(center_vals, L, window, stride):
    out = np.zeros(L, dtype=np.float32); wsum = np.zeros(L, dtype=np.float32)
    half = window//2
    for i, y in enumerate(center_vals):
        c = i*stride + half
        if 0 <= c < L: out[c] += float(y); wsum[c] += 1.0
    nz = wsum > 0; out[nz] /= wsum[nz]
    if np.any(~nz): out = np.interp(np.arange(L), np.where(nz)[0], out[nz])
    return out

def calib_strength(i, kind="gamma", param=0.85):
    i = np.clip(i.astype(np.float32), 0.0, 1.0)
    if kind == "gamma":
        return i ** float(param)       # <1 → più aggressivo a basse intensità
    if kind == "logistic":
        k = float(param)
        return 1.0 / (1.0 + np.exp(-k * (i - 0.3)))
    return i

def smooth_med(x, size=5):
    return ndi.median_filter(x.astype(np.float32), size=size, mode="nearest")

def binarize_u8_to_i32(Xu8):
    return (Xu8 != 0).astype(np.int32, copy=False)

# -------------------- robust loader --------------------
def _canon_array(a):
    """Canonizza dtype/contiguità per lo stato interno TM."""
    if isinstance(a, np.ndarray):
        if a.dtype == np.bool_:
            a = a.astype(np.uint32)
        elif a.dtype.kind in ('i','u'):
            # TM si aspetta uint32 per gli stati
            a = a.astype(np.uint32)
        elif a.dtype.kind == 'f':
            a = a.astype(np.float32)
        return np.ascontiguousarray(a)
    return a

def _canon_state(obj):
    if isinstance(obj, dict):
        return {k: _canon_state(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return tuple(_canon_state(v) for v in obj)
    return _canon_array(obj)

def _safe_init_model(n_features, clauses, T, s):
    """Crea RTM e fa un fit 'dummy' per allocare strutture interne coerenti con n_features."""
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
    model = RegressionTsetlinMachine(int(clauses), int(T), float(s))
    Xd = np.zeros((1, n_features), dtype=np.int32)  # binario
    yd = np.zeros(1, dtype=np.float32)
    # epochs=1 è sufficiente per inizializzare
    model.fit(Xd, yd, epochs=1, incremental=True)
    return model

def load_rtm_bundle(stem: Path, n_features: int):
    """
    Carica un modello RTM portabile:
      - meta (.meta.json) con clauses/T/s (obbligatorio)
      - stato (.state.npz | .state.npy) con canonizzazione dtypes
      - fallback .joblib (fragile, stessa env)
    Inizializza SEMPRE con fit dummy prima di set_state (evita segfault).
    """
    meta_p = stem.with_suffix(".meta.json")
    if not meta_p.exists():
        # fallback diretto joblib
        pkl = stem.with_suffix(".joblib")
        if pkl.exists():
            import joblib
            print(f"[WARN] Meta non trovata: carico pickle (fragile): {pkl.name}")
            return joblib.load(pkl)
        raise FileNotFoundError(f"META non trovata: {meta_p}")

    meta = json.loads(meta_p.read_text())
    clauses, T, s = int(meta["clauses"]), int(meta["T"]), float(meta["s"])
    model = _safe_init_model(n_features, clauses, T, s)

    # prova .state.npz
    npz = stem.with_suffix(".state.npz")
    npy = stem.with_suffix(".state.npy")
    if npz.exists():
        data = np.load(str(npz), allow_pickle=True)
        fmt = meta.get("state_format", "mapping")
        if fmt == "mapping":
            keys = meta.get("state_keys", list(data.files))
            state = {k: data[k] for k in keys if k in data}
        elif fmt == "sequence":
            L = int(meta.get("state_len", len(data.files)))
            state = tuple(data[f"s{i}"] for i in range(L) if f"s{i}" in data)
        else:
            # prova generica: mappa tutti i campi
            state = {k: data[k] for k in data.files}
        state = _canon_state(state)
        # set_state può ancora fallire se la struttura non matcha la versione binaria
        model.set_state(state)
        return model

    if npy.exists():
        state = np.load(str(npy), allow_pickle=True)
        try:
            state = state.item()
        except Exception:
            pass
        state = _canon_state(state)
        model.set_state(state)
        return model

    # fallback joblib
    pkl = stem.with_suffix(".joblib")
    if pkl.exists():
        import joblib
        print(f"[WARN] Stato non trovato: carico pickle (fragile): {pkl.name}")
        return joblib.load(pkl)

    raise FileNotFoundError(f"Bundle stato non trovato per {stem}")

# -------------------- orchestrator --------------------
def orchestrated_denoise(x, fs, i_bw, i_emg, i_pli):
    s_bw  = float(np.median(i_bw[np.isfinite(i_bw)])) if np.any(np.isfinite(i_bw)) else 0.0
    s_emg = float(np.median(i_emg[np.isfinite(i_emg)])) if np.any(np.isfinite(i_emg)) else 0.0
    s_pli = float(np.median(i_pli[np.isfinite(i_pli)])) if np.any(np.isfinite(i_pli)) else 0.0
    y = x.astype(np.float32)
    y = notch_filter(y, fs=fs, f0=50.0, Q=30.0, strength=s_pli)
    y = highpass_butter(y, fs=fs, cutoff=0.5, order=2, strength=s_bw)
    y = wavelet_denoise(y, strength=s_emg)
    return y

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, help="cartella con rtm_intensity_{bw,emg,pli}*.state.*")
    ap.add_argument("--input-h5", required=True, help="con *_noisy (e opz. *_clean) e fs")
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--split", choices=["auto","train","val","test"], default="auto")
    ap.add_argument("--encoder", choices=["bitplanes","thermometer","onehot2d"], default="bitplanes")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--no-deriv", action="store_true")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--do-denoise", action="store_true")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)

    # --- carica dataset e determina fs/split ---
    with h5py.File(str(Path(args.input_h5)), "r") as f:
        fs = float(f.attrs.get("fs", 360.0))
        split = args.split
        if split == "auto":
            split = "test" if f.get("test_noisy") is not None else ("val" if f.get("val_noisy") is not None else "train")
        noisy = f[f"{split}_noisy"][:]
        clean = f.get(f"{split}_clean")
        clean = None if clean is None else clean[:]

    # --- calcola n_features dalla PRIMA traccia (serve per init sicuro RTM) ---
    x0 = noisy[0].astype(np.float32)
    X0 = build_features(x0, window=args.window, stride=args.stride,
                        encoder=args.encoder, bits=args.bits, levels=args.levels,
                        include_deriv=(not args.no_deriv))
    n_features = X0.shape[1] if X0.ndim == 2 else X0.size
    print(f"[INFO] n_features={n_features}")

    # --- carica modelli con init sicuro ---
    m_bw  = load_rtm_bundle(models_dir / "rtm_intensity_bw",  n_features)
    m_emg = load_rtm_bundle(models_dir / "rtm_intensity_emg", n_features)
    m_pli = load_rtm_bundle(models_dir / "rtm_intensity_pli", n_features)

    preds = []
    dens = []
    for k in range(noisy.shape[0]):
        x = noisy[k].astype(np.float32)
        X = build_features(x, window=args.window, stride=args.stride,
                           encoder=args.encoder, bits=args.bits, levels=args.levels,
                           include_deriv=(not args.no_deriv))
        X = X.reshape(X.shape[0], -1).astype(np.uint8, copy=False)
        Xbin = binarize_u8_to_i32(X)

        pbw  = m_bw.predict(Xbin).astype(np.float32)
        pemg = m_emg.predict(Xbin).astype(np.float32)
        ppli = m_pli.predict(Xbin).astype(np.float32)

        ibw  = overlap_add(pbw,  L=len(x), window=args.window, stride=args.stride)
        iemg = overlap_add(pemg, L=len(x), window=args.window, stride=args.stride)
        ipli = overlap_add(ppli, L=len(x), window=args.window, stride=args.stride)

        # calibrazione + smoothing per le forze
        ibw_c  = smooth_med(calib_strength(ibw,  "gamma",    0.9), size=5)
        iemg_c = smooth_med(calib_strength(iemg, "gamma",    0.8), size=5)
        ipli_c = smooth_med(calib_strength(ipli, "logistic", 4.0), size=5)

        preds.append(np.stack([ibw, iemg, ipli], axis=0))

        if args.do_denoise:
            y = orchestrated_denoise(x, fs, ibw_c, iemg_c, ipli_c)
            dens.append(y)

    preds = np.stack(preds, axis=0)  # (N,3,L)
    outp = Path(args.out_h5); outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(outp), "w") as h:
        h.create_dataset("intensities", data=preds, compression="gzip", compression_opts=4)
        if dens:
            h.create_dataset("denoised", data=np.stack(dens, axis=0), compression="gzip", compression_opts=4)
        if clean is not None:
            h.create_dataset("clean", data=clean, compression="gzip", compression_opts=4)
        h.attrs["window"] = args.window; h.attrs["stride"] = args.stride
        h.attrs["encoder"] = args.encoder; h.attrs["bits"] = args.bits; h.attrs["levels"] = args.levels
        h.attrs["fs"] = fs
    print(f"[OK] {outp}  preds:{preds.shape}  denoised:{len(dens)}")

if __name__ == "__main__":
    main()
