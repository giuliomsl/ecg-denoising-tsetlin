# ============================================
# File: src/classifier/preprocess_ctm_classifier.py
# ============================================
"""
Preprocess avanzato per CTM/TMU.
Converte:
  - (A) directory NPZ:  in_dir/{train,val,test}.npz  con X_noisy, y_class[, y_multi]
  - (B) HDF5 "raw":    src_h5 con gruppi {train,val,test}/sig(float32), y(uint32)

In output: HDF5 binarizzato per TM:
  - {split}/X : uint8 {0,1} shape (N, C, Lp)   (C = somma canali di tutte le viste)
  - {split}/y : uint32 shape (N,)
  - attrs vari con meta e thresholds

Feature (abilitabili a flag):
  - Amplitude bins (quantili da train)                   --amp_bins
  - Delta bins (primo differenziale)                     --delta_bins
  - Envelope RMS (moving RMS)                            --env_bins --env_win
  - STFT bands (tempo-risolte, no SciPy)                 --spec_bins --spec_win --spec_hop --bands
  - PLI detector (50/60Hz + 2f)                          --pli_bins --pli_freq --pli_win --pli_harm

Tutte le feature sono z-score / [0,1] normalizzate su base finestra e poi binarizzate a soglie uniformi (multi-hot cumulativo).
"""

from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import h5py


# ----------------------------- util base -----------------------------

def _ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _zscore_last(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score lungo l'ultima dimensione."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (x - mu) / sd

def _to01(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    z = _zscore_last(x)
    z = np.clip(z, -clip, clip)
    return (z + clip) / (2.0 * clip)

def _quantile_thresholds(train_vec: np.ndarray, n_bins: int) -> np.ndarray:
    # train_vec: (N, L) -> thresholds globali su train, esclusi 0 e 1
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    thr = np.quantile(train_vec.reshape(-1), qs, method="linear")
    return thr.astype(np.float32)

def _level_encode(sig_like: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """
    Binarizzazione multi-hot cumulativa su soglie 'thr'.
    - sig_like: (N, L) oppure (N, C, L) con valori in [0,1] (o z-scored/clipped)
    - thr: (B-1,) soglie crescenti
    Ritorna:
      - se sig_like è (N, L):      (N, B, L)
      - se sig_like è (N, C, L):   (N, C*B, L) concatenando i B canali per ciascun C
    """
    if sig_like.ndim == 2:
        N, L = sig_like.shape
        B = thr.shape[0] + 1
        idx = np.searchsorted(thr, sig_like, side="right")  # (N, L)
        out = np.empty((N, B, L), dtype=np.uint8)
        for k in range(B):
            out[:, k, :] = (idx >= k).astype(np.uint8)
        return out
    elif sig_like.ndim == 3:
        N, C, L = sig_like.shape
        B = thr.shape[0] + 1
        x2 = sig_like.reshape(N * C, L)
        idx = np.searchsorted(thr, x2, side="right")  # (N*C, L)
        out2 = np.empty((N * C, B, L), dtype=np.uint8)
        for k in range(B):
            out2[:, k, :] = (idx >= k).astype(np.uint8)
        # rimappa canali: (N*C, B, L) -> (N, C, B, L) -> (N, C*B, L)
        out2 = out2.reshape(N, C, B, L)
        out = out2.transpose(0, 2, 1, 3).reshape(N, C * B, L)
        return out
    else:
        raise ValueError("sig_like deve essere 2D (N,L) o 3D (N,C,L)")

def _temporal_pool(X: np.ndarray, pool: int) -> np.ndarray:
    if pool <= 1:
        return X
    N, C, L = X.shape
    newL = (L // pool) * pool
    if newL == 0:
        return X
    Xc = X[:, :, :newL].reshape(N, C, newL // pool, pool).mean(axis=-1)
    return (Xc >= 0.5).astype(np.uint8)


# ----------------------- feature: delta & envelope -------------------

def _first_diff(x: np.ndarray) -> np.ndarray:
    # x: (N, L) -> (N, L) differenziale con padding iniziale
    return np.diff(x, axis=1, prepend=x[:, :1])

def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """RMS a finestra scorrevole, normalizzata a [0,1] per finestra globale."""
    if win <= 1:
        return np.sqrt(np.maximum(x * x, 0.0))
    N, L = x.shape
    # moving mean su x^2 via integrale cumulativo per efficienza
    xx = x.astype(np.float32) ** 2
    csum = np.cumsum(xx, axis=1)
    # media su [i, i+win)
    out = np.empty_like(xx)
    for i in range(L):
        j0 = max(0, i - win // 2)
        j1 = min(L, j0 + win)
        j0 = j1 - win
        if j0 < 0:
            j0 = 0
        # somma in [j0,j1)
        seg = csum[:, j1 - 1] - (csum[:, j0 - 1] if j0 > 0 else 0.0)
        m = seg / float(j1 - j0)
        out[:, i] = np.sqrt(np.maximum(m, 0.0))
    # normalizza [0,1] per finestra completa (per istanza)
    mn = out.min(axis=1, keepdims=True)
    mx = out.max(axis=1, keepdims=True)
    denom = np.where((mx - mn) < 1e-8, 1.0, (mx - mn))
    out01 = (out - mn) / denom
    return out01.astype(np.float32)


# ------------------------ feature: STFT bands ------------------------

def _frame_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """
    x: (N, L) -> (N, T, win) con finestre sovrapposte, zero-padding riflessivo ai bordi.
    """
    N, L = x.shape
    if win <= 1:
        return x[:, :, None]
    if hop <= 0:
        hop = win // 4
    T = 1 + int(np.ceil((L - win) / float(hop))) if L > win else 1
    # estrazione frame con padding riflessivo per semplicità/robustezza
    frames = np.empty((N, T, win), dtype=np.float32)
    for t in range(T):
        i0 = t * hop
        i1 = i0 + win
        if i1 <= L:
            frames[:, t, :] = x[:, i0:i1]
        else:
            # pad riflessivo in coda
            need = i1 - L
            tail = np.pad(x[:, i0:L], ((0,0),(0,need)), mode='reflect')
            frames[:, t, :] = tail
    return frames

def _stft_band_energy(x: np.ndarray, fs: int, win: int, hop: int,
                      bands: List[Tuple[float, float]]) -> np.ndarray:
    """
    x: (N, L) float32
    bands: lista di bande [(f0,f1),...]
    return: (N, B, T) con energia normalizzata per frame (somma sulle bande = 1)
    """
    frames = _frame_signal(x, win=win, hop=hop)  # (N,T,win)
    N, T, W = frames.shape
    window = np.hanning(W).astype(np.float32)
    # FFT rfft su asse -1
    Xf = np.fft.rfft(frames * window[None, None, :], axis=-1)
    power = (np.abs(Xf) ** 2).astype(np.float32)  # (N,T,F)
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)
    B = len(bands)
    out = np.empty((N, B, T), dtype=np.float32)
    total = power.sum(axis=-1, keepdims=True) + 1e-12
    for bi, (f0, f1) in enumerate(bands):
        mask = (freqs >= f0) & (freqs < f1)
        band_e = power[:, :, mask].sum(axis=-1, keepdims=False)  # (N,T,1)
        out[:, bi, :] = (band_e / total.squeeze(-1))
    return out  # non ancora upsample su L

def _upsample_frames_to_L(band_e: np.ndarray, L: int, win: int, hop: int) -> np.ndarray:
    """
    band_e: (N, B, T) -> (N, B, L) replicando ciascun frame su intervallo [t*hop, t*hop+win)
    con sovrapposizione gestita come media.
    """
    N, B, T = band_e.shape
    out = np.zeros((N, B, L), dtype=np.float32)
    cnt = np.zeros((N, 1, L), dtype=np.float32)
    for t in range(T):
        i0 = t * hop
        i1 = min(L, i0 + win)
        if i0 >= L:
            break
        out[:, :, i0:i1] += band_e[:, :, t][:, :, None]
        cnt[:, :, i0:i1] += 1.0
    out = out / np.maximum(cnt, 1.0)
    # normalizza per istanza-canale a [0,1] (stabilizza soglie)
    mn = out.min(axis=-1, keepdims=True)
    mx = out.max(axis=-1, keepdims=True)
    out = (out - mn) / np.maximum(mx - mn, 1e-8)
    return out.astype(np.float32)


# ------------------------ feature: PLI detector ----------------------

def _pli_envelope(x: np.ndarray, fs: int, f0: float, win: int) -> np.ndarray:
    """
    Stima ampiezza locale di una sinusoide a f0 con proiezione sin/cos in finestra scorrevole.
    Ritorna (N, L) normalizzato [0,1] per istanza.
    """
    N, L = x.shape
    if win <= 1:
        win = int(0.2 * fs)  # default 200ms
    t = np.arange(win, dtype=np.float32) / float(fs)
    s = np.sin(2.0 * np.pi * f0 * t).astype(np.float32)
    c = np.cos(2.0 * np.pi * f0 * t).astype(np.float32)
    # proiezione per ogni posizione (con riflessione bordo)
    env = np.empty((N, L), dtype=np.float32)
    s_norm = np.sum(s * s); c_norm = np.sum(c * c)
    for i in range(L):
        j0 = max(0, i - win // 2)
        j1 = min(L, j0 + win)
        j0 = j1 - win
        seg = x[:, j0:j1]
        if seg.shape[1] < win:
            # pad riflessivo
            need = win - seg.shape[1]
            seg = np.pad(seg, ((0,0),(0,need)), mode='reflect')
        a = seg @ s / (s_norm + 1e-8)
        b = seg @ c / (c_norm + 1e-8)
        env[:, i] = np.sqrt(a * a + b * b)
    # normalizza [0,1] per istanza
    mn = env.min(axis=1, keepdims=True); mx = env.max(axis=1, keepdims=True)
    env01 = (env - mn) / np.maximum(mx - mn, 1e-8)
    return env01

def _binarize_uniform01(x01: np.ndarray, n_bins: int) -> np.ndarray:
    """Binarizza in [0,1] con soglie uniformi: restituisce (N, B, L)."""
    if n_bins <= 0:
        raise ValueError("n_bins deve essere >=1")
    if x01.ndim != 2:
        raise ValueError("atteso (N,L)")
    taus = np.linspace(0.0, 1.0, n_bins + 2, dtype=np.float32)[1:-1]
    N, L = x01.shape
    out = np.empty((N, n_bins + 1, L), dtype=np.uint8)
    # multi-hot cumulativo su soglie uniformi
    for k in range(n_bins + 1):
        thr = taus[k - 1] if k > 0 else -1e9
        out[:, k, :] = (x01 >= thr).astype(np.uint8)
    return out


# ----------------------------- IO loaders ----------------------------

def _load_from_npz(in_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    p = os.path.join(in_dir, f"{split}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    data = np.load(p, allow_pickle=True)
    X = data["X_noisy"].astype(np.float32)
    if "y_class" in data:
        y = data["y_class"].astype(np.uint32)
    elif "y" in data:
        y = data["y"].astype(np.uint32)
    else:
        raise KeyError("y_class/y non trovato in npz")
    return X, y

def _load_from_h5_raw(src_h5: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(src_h5, "r") as h5:
        X = h5[f"{split}/sig"][:].astype(np.float32, copy=False)
        y = h5[f"{split}/y"][:].astype(np.uint32, copy=False)
    return X, y


# ----------------------------- main core -----------------------------

def _build_features_for_split(
    Xsig: np.ndarray,
    cfg: argparse.Namespace,
    thr_amp: Optional[np.ndarray],
    thr_delta: Optional[np.ndarray],
    thr_env: Optional[np.ndarray],
    thr_spec: Optional[np.ndarray],
    thr_pli: Optional[np.ndarray],
    fs: int,
) -> np.ndarray:
    """
    Genera tutte le viste richieste e concatena lungo C.
    Xsig: (N, L) float32
    ritorna: Xbin (N, C_tot, Lp) uint8
    """
    N, L = Xsig.shape
    feats: List[np.ndarray] = []

    # --- amplitude bins (quantili) ---
    if cfg.amp_bins > 0:
        x01 = _to01(Xsig, clip=cfg.clip_z)
        feats.append(_level_encode(x01, thr_amp))  # (N,B_amp,L)

    # --- delta bins ---
    if cfg.delta_bins > 0:
        dx = _first_diff(_zscore_last(Xsig))
        dx01 = _to01(dx, clip=cfg.clip_z)
        feats.append(_level_encode(dx01, thr_delta))  # (N,B_delta,L)

    # --- envelope RMS ---
    if cfg.env_bins > 0:
        env01 = _moving_rms(Xsig, win=cfg.env_win)
        feats.append(_level_encode(env01, thr_env))   # (N,B_env,L)

    # --- STFT bands tempo-risolte ---
    if cfg.spec_bins > 0 and cfg.bands:
        be = _stft_band_energy(Xsig, fs=fs, win=cfg.spec_win, hop=cfg.spec_hop, bands=cfg.bands)
        beL = _upsample_frames_to_L(be, L=L, win=cfg.spec_win, hop=cfg.spec_hop)  # (N,B,T)->(N,B,L)
        feats.append(_level_encode(beL, thr_spec))  # (N,B_spec,L)

    # --- PLI detector (f0 e 2*f0 opzionali) ---
    if cfg.pli_bins > 0:
        env_f0 = _pli_envelope(Xsig, fs=fs, f0=cfg.pli_freq, win=cfg.pli_win)  # (N,L)
        feats.append(_level_encode(env_f0, thr_pli))  # (N,B_pli,L)
        if cfg.pli_harm:
            env_2f = _pli_envelope(Xsig, fs=fs, f0=2.0 * cfg.pli_freq, win=cfg.pli_win)
            feats.append(_level_encode(env_2f, thr_pli))  # riuso stesse soglie

    # concat su C
    Xcat = np.concatenate(feats, axis=1).astype(np.uint8) if feats else np.zeros((N, 0, L), dtype=np.uint8)
    # pooling finale
    Xcat = _temporal_pool(Xcat, cfg.pool)
    return Xcat


def run_preprocess(
    in_dir: Optional[str],
    src_h5: Optional[str],
    dst_h5: str,
    fs: int,
    cfg: argparse.Namespace
):
    _ensure_dir(os.path.dirname(dst_h5))

    # ---- carica split ----
    if src_h5:
        loader = lambda s: _load_from_h5_raw(src_h5, s)
    else:
        loader = lambda s: _load_from_npz(in_dir, s)

    Xtr, ytr = loader("train")
    Xva, yva = loader("val")
    Xte, yte = loader("test")

    # sicurezza tipi
    ytr = ytr.astype(np.uint32, copy=False)
    yva = yva.astype(np.uint32, copy=False)
    yte = yte.astype(np.uint32, copy=False)

    # ---- thresholds solo da train ----
    # Amplitude
    thr_amp = _quantile_thresholds(_to01(Xtr, clip=cfg.clip_z), cfg.amp_bins) if cfg.amp_bins > 0 else None
    # Delta
    thr_delta = _quantile_thresholds(_to01(_first_diff(_zscore_last(Xtr)), clip=cfg.clip_z), cfg.delta_bins) if cfg.delta_bins > 0 else None
    # Envelope
    thr_env = _quantile_thresholds(_moving_rms(Xtr, win=cfg.env_win), cfg.env_bins) if cfg.env_bins > 0 else None
    # Spectral
    thr_spec = None
    if cfg.spec_bins > 0 and cfg.bands:
        be_tr = _stft_band_energy(Xtr, fs=fs, win=cfg.spec_win, hop=cfg.spec_hop, bands=cfg.bands)
        be_trL = _upsample_frames_to_L(be_tr, L=Xtr.shape[1], win=cfg.spec_win, hop=cfg.spec_hop)
        thr_spec = _quantile_thresholds(be_trL, cfg.spec_bins)
        del be_tr, be_trL
    # PLI
    thr_pli = _quantile_thresholds(_pli_envelope(Xtr, fs=fs, f0=cfg.pli_freq, win=cfg.pli_win), cfg.pli_bins) if cfg.pli_bins > 0 else None

    # ---- build features per split ----
    print(f"[INFO] Building features | amp={cfg.amp_bins} delta={cfg.delta_bins} env={cfg.env_bins} "
          f"spec={cfg.spec_bins} pli={cfg.pli_bins} pool={cfg.pool} bands={len(cfg.bands) if cfg.bands else 0}")

    Xtr_bin = _build_features_for_split(Xtr, cfg, thr_amp, thr_delta, thr_env, thr_spec, thr_pli, fs)
    print(f"[train] -> X {Xtr_bin.shape}")
    Xva_bin = _build_features_for_split(Xva, cfg, thr_amp, thr_delta, thr_env, thr_spec, thr_pli, fs)
    print(f"[val]   -> X {Xva_bin.shape}")
    Xte_bin = _build_features_for_split(Xte, cfg, thr_amp, thr_delta, thr_env, thr_spec, thr_pli, fs)
    print(f"[test]  -> X {Xte_bin.shape}")

    # ---- salva HDF5 ----
    with h5py.File(dst_h5, "w") as h5:
        for name, X, y in (("train", Xtr_bin, ytr), ("val", Xva_bin, yva), ("test", Xte_bin, yte)):
            g = h5.create_group(name)
            g.create_dataset("X", data=X, dtype=np.uint8, chunks=True, compression="gzip", compression_opts=4, shuffle=True)
            g.create_dataset("y", data=y, dtype=np.uint32, chunks=True, compression="gzip", compression_opts=4, shuffle=True)
            g.attrs["fs"] = fs

        meta = {
            "amp_bins": int(cfg.amp_bins),
            "delta_bins": int(cfg.delta_bins),
            "env_bins": int(cfg.env_bins),
            "env_win": int(cfg.env_win),
            "spec_bins": int(cfg.spec_bins),
            "spec_win": int(cfg.spec_win),
            "spec_hop": int(cfg.spec_hop),
            "bands": cfg.bands,
            "pli_bins": int(cfg.pli_bins),
            "pli_freq": float(cfg.pli_freq),
            "pli_win": int(cfg.pli_win),
            "pli_harm": bool(cfg.pli_harm),
            "pool": int(cfg.pool),
            "clip_z": float(cfg.clip_z),
            "C_total": int(Xtr_bin.shape[1]),
            "L": int(Xtr_bin.shape[2]),
            "thresholds": {
                "amp": (thr_amp.tolist() if thr_amp is not None else None),
                "delta": (thr_delta.tolist() if thr_delta is not None else None),
                "env": (thr_env.tolist() if thr_env is not None else None),
                "spec": (thr_spec.tolist() if thr_spec is not None else None),
                "pli": (thr_pli.tolist() if thr_pli is not None else None),
            }
        }
        h5.attrs["meta_json"] = json.dumps(meta)
    print(f"[OK] Wrote -> {dst_h5} | C={Xtr_bin.shape[1]} L={Xtr_bin.shape[2]}")


# ----------------------------- CLI -----------------------------------

def _parse_bands(s: str) -> List[Tuple[float,float]]:
    """
    Converte stringa tipo: "0-0.5,0.5-5,5-15,15-40,45-55,55-70"
    in lista di tuple float.
    """
    if not s:
        return []
    out = []
    for chunk in s.split(","):
        f0, f1 = chunk.split("-")
        out.append((float(f0), float(f1)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Advanced preprocess for CTM/TMU classifier")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in_dir", type=str, help="Directory con train/val/test.npz (X_noisy,y_class)")
    src.add_argument("--src_h5", type=str, help="HDF5 raw con {split}/sig,float32 e y,uint32")
    ap.add_argument("--dst_h5", type=str, default="./data/bin/classifier_patches.h5")
    ap.add_argument("--fs", type=int, default=360)

    # feature flags
    ap.add_argument("--amp_bins", type=int, default=44, help="bin amplitude (quantili)")
    ap.add_argument("--delta_bins", type=int, default=16, help="bin su primo differenziale")
    ap.add_argument("--env_bins", type=int, default=8, help="bin su envelope RMS")
    ap.add_argument("--env_win", type=int, default=64, help="finestra RMS (campioni)")

    ap.add_argument("--spec_bins", type=int, default=6, help="bin su energia di banda STFT")
    ap.add_argument("--spec_win", type=int, default=128, help="finestra STFT")
    ap.add_argument("--spec_hop", type=int, default=32, help="hop STFT")
    ap.add_argument("--bands", type=_parse_bands,
                    default=_parse_bands("0-0.5,0.5-5,5-15,15-40,45-55,55-70"),
                    help="bande Hz per STFT, formato 'f0-f1,f0-f1,...'")

    ap.add_argument("--pli_bins", type=int, default=6, help="bin su ampiezza PLI (f0 e opz. 2f)")
    ap.add_argument("--pli_freq", type=float, default=50.0, help="frequenza rete (50/60)")
    ap.add_argument("--pli_win", type=int, default=96, help="finestra per proiezione PLI")
    ap.add_argument("--pli_harm", action="store_true", help="aggiungi canali per armonica 2*f0")

    ap.add_argument("--pool", type=int, default=2, help="temporal pooling finale")
    ap.add_argument("--clip_z", type=float, default=3.0, help="clipping z-score -> [0,1]")

    args = ap.parse_args()

    run_preprocess(
        in_dir=args.in_dir,
        src_h5=args.src_h5,
        dst_h5=args.dst_h5,
        fs=args.fs,
        cfg=args
    )


if __name__ == "__main__":
    main()
