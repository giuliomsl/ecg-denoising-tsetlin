# ============================================
# File: src/estimator/preprocess_estimator.py
# Task: Preprocess per Noise Estimation (presence + SNR)
# Input atteso (da data_generator avanzato):
#   {in_npz_dir}/train.npz, val.npz, test.npz con chiavi:
#     - X_noisy: (N, L) oppure (N, C_leads, L) float32
#     - y_multi: (N, 3) int32 -> presenza [BW, MA, PLI] in {0,1}
#     - snr_db : (N, 3) float32 -> SNR in dB per [BW, MA, PLI] (NaN se assente)
#     - meta   : json string (opzionale)
#
# Output HDF5 (out_h5):
#   /{split}/X                      -> (N, C_tot, Lp) uint8 {0,1}
#   /{split}/targets/present/bw|ma|pli -> (N,) uint8
#   /{split}/targets/ordinal/bw|ma|pli -> (N, K_ord) uint8
#   /{split}/targets/binlabel/bw|ma|pli-> (N,) uint8 in [0..K_bins-1]
#   /{split}/snr_db_raw             -> (N, 3) float32
#   Attributi HDF5:
#     - meta globale: thresholds ampiezza, config bande/soglie, var windows, snr_edges/taus, pool, ecc.
#
# Feature:
#   - Amplitude level-encoding via quantili su train (B_amp = amp_bins).
#   - Spettrali: potenza di banda normalizzata (somma=1), sogliate in K_spec
#       * mode 'fixed' => soglie uniformi in [0,1]
#       * mode 'quantile' => soglie per banda via quantili su train
#   - Varianza locale (rolling std) a più scale; normalizzata con percentili (robusta)
#   - Temporal pooling (media+0.5) per ridurre L -> Lp
#
# Note:
#   - Tutte le statistiche (quantili ampiezza, soglie spettrali per banda se quantile,
#     scale varianza e relativi percentili) sono calcolate SOLO sullo split train.
#   - Batch processing per memoria.
# ============================================

from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import h5py


NOISES = ["BW", "MA", "PLI"]             # ordine fisso per y_multi e snr_db
NOISE_TO_IDX = {n: i for i, n in enumerate(NOISES)}


# ------------------------- util base -------------------------

def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def _zscore_last(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # z-score lungo l'asse finale (tempo)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd

def _clip01_from_z(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = np.clip(x, -clip, clip)
    return (x + clip) / (2.0 * clip)

def _parse_float_list(s: str) -> List[float]:
    return [float(t) for t in s.split(",") if t.strip() != ""]

def _parse_int_list(s: str) -> List[int]:
    return [int(t) for t in s.split(",") if t.strip() != ""]


# ------------------------- thresholds (train) -------------------------

def _compute_amp_thresholds_train(Xtr: np.ndarray, amp_bins: int, sample_cap: int = 2_000_000) -> np.ndarray:
    """
    Xtr: (N,L) o (N,C,L) già z-scored.
    Stima quantili globali lungo tempo (e leads) su un campionamento casuale.
    Ritorna thr_amp shape (B_amp-1,).
    """
    rng = np.random.default_rng(123)
    if Xtr.ndim == 3:
        N, C, L = Xtr.shape
        total = N * C * L
        view = Xtr.reshape(N * C, L)
    else:
        N, L = Xtr.shape
        C = 1
        total = N * L
        view = Xtr  # (N, L)

    take = min(sample_cap, total)
    if take == total:
        sample = view.reshape(-1)
    else:
        idx = rng.integers(0, total, size=take)
        sample = view.reshape(-1)[idx]
    thr = np.quantile(sample, np.linspace(0, 1, amp_bins + 1)[1:-1]).astype(np.float32)
    return thr


def _band_masks(L: int, fs: float, bands: Tuple[Tuple[float, float], ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Precalcolo maschere di banda su asse frequenze per rFFT (L)->(L//2+1)."""
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    masks = []
    for f0, f1 in bands:
        masks.append((freqs >= f0) & (freqs < f1))
    return freqs, np.stack(masks, axis=0)


def _sample_band_powers(X: np.ndarray, fs: float, bands: Tuple[Tuple[float, float], ...],
                        sample_n: int = 2048) -> np.ndarray:
    """
    Campiona alcune finestre da X (z-scored), calcola power per banda normalizzata (somma=1).
    Ritorna (S, B) float32.
    """
    rng = np.random.default_rng(123)
    if X.ndim == 3:
        N, C, L = X.shape
        flat = X.reshape(N * C, L)
    else:
        N, L = X.shape
        flat = X
    S = min(sample_n, flat.shape[0])
    idx = rng.choice(flat.shape[0], size=S, replace=False)
    pick = flat[idx]  # (S, L)

    win = np.hamming(L).astype(np.float32)
    Xf = np.fft.rfft(pick * win, axis=-1)
    power = (np.abs(Xf) ** 2).astype(np.float32)  # (S, F)

    freqs, masks = _band_masks(L, fs, bands)     # masks: (B, F)
    B = masks.shape[0]
    out = np.empty((S, B), dtype=np.float32)
    denom = power.sum(axis=-1, keepdims=True) + 1e-12

    # somma per banda
    for b in range(B):
        e = power[:, masks[b]].sum(axis=-1, keepdims=False)  # (S,)
        out[:, b] = (e / denom[:, 0]).astype(np.float32)
    return out


def _compute_spec_thresholds_train(Xtr: np.ndarray, fs: float,
                                   bands: Tuple[Tuple[float, float], ...],
                                   k_spec: int,
                                   mode: str = "fixed",
                                   sample_n: int = 2048) -> np.ndarray:
    """
    Ritorna soglie per banda:
      - mode='fixed': stesse soglie per tutte le bande, uniformi in [0,1]
      - mode='quantile': per ciascuna banda quantili su train
    Shape: (B, k_spec-1)
    """
    B = len(bands)
    if k_spec <= 0:
        return np.zeros((B, 0), dtype=np.float32)

    if mode == "fixed":
        base = np.linspace(0, 1, k_spec + 1)[1:-1].astype(np.float32)
        return np.tile(base[None, :], (B, 1))

    # quantile per banda
    P = _sample_band_powers(Xtr, fs, bands, sample_n=sample_n)  # (S, B)
    thr = []
    qs = np.linspace(0, 1, k_spec + 1)[1:-1]
    for b in range(B):
        thr.append(np.quantile(P[:, b], qs).astype(np.float32))
    return np.stack(thr, axis=0)  # (B, k_spec-1)


def _rolling_std_batch(X: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling std 1D per batch:
      X: (B, L) -> std su finestre di lunghezza w (valid) -> (B, L - w + 1)
    Implementazione via cumsum per efficienza e assenza di dipendenze.
    """
    B, L = X.shape
    if w <= 1 or w > L:
        return np.zeros((B, max(1, L - w + 1)), dtype=np.float32)

    csum = np.cumsum(X, axis=1, dtype=np.float64)
    csum2 = np.cumsum(X * X, axis=1, dtype=np.float64)
    # somma su intervalli [i, i+w)
    s = csum[:, w - 1:] - np.concatenate([np.zeros((B, 1)), csum[:, :-w]], axis=1)
    s2 = csum2[:, w - 1:] - np.concatenate([np.zeros((B, 1)), csum2[:, :-w]], axis=1)
    mean = s / w
    var = np.maximum(s2 / w - mean * mean, 0.0)
    std = np.sqrt(var, dtype=np.float64)
    return std.astype(np.float32)  # (B, L-w+1)


def _pad_to_length(arr: np.ndarray, L: int, w: int) -> np.ndarray:
    """
    Arr 'valid' length L-w+1 -> ricampiona/pad per tornare a L (bordo con replicate).
    """
    B, Lv = arr.shape
    if Lv == L:
        return arr
    # centro: posizioniamo la finestra in "valid" con offset
    left_pad = (w - 1) // 2
    right_pad = L - (Lv + left_pad)
    left = np.repeat(arr[:, :1], repeats=left_pad, axis=1) if left_pad > 0 else np.empty((B, 0), arr.dtype)
    right = np.repeat(arr[:, -1:], repeats=max(0, right_pad), axis=1) if right_pad > 0 else np.empty((B, 0), arr.dtype)
    return np.concatenate([left, arr, right], axis=1)


def _compute_var_stats_train(Xtr: np.ndarray, var_windows: List[int],
                             p_lo: float, p_hi: float, sample_n: int = 1024) -> Dict[int, Tuple[float, float]]:
    """
    Stima (low, high) percentili per normalizzare rolling std per ogni finestra.
    """
    rng = np.random.default_rng(123)
    if Xtr.ndim == 3:
        N, C, L = Xtr.shape
        flat = Xtr.reshape(N * C, L)
    else:
        N, L = Xtr.shape
        flat = Xtr

    S = min(sample_n, flat.shape[0])
    idx = rng.choice(flat.shape[0], size=S, replace=False)
    pick = flat[idx]  # (S, L)

    stats = {}
    for w in var_windows:
        v = _rolling_std_batch(pick, w)  # (S, L-w+1)
        v = _pad_to_length(v, L, w)      # (S, L)
        lo = float(np.quantile(v, p_lo / 100.0))
        hi = float(np.quantile(v, p_hi / 100.0))
        if hi <= lo:
            hi = lo + 1e-6
        stats[w] = (lo, hi)
    return stats


# ------------------------- feature builders -------------------------

def _encode_amplitude(sig_z: np.ndarray, thr_amp: np.ndarray) -> np.ndarray:
    """
    sig_z: (B, L) z-scored -> level-encoding cumulativo (B_amp channels, L)
    """
    Bn, L = sig_z.shape
    B_amp = thr_amp.shape[0] + 1
    idx = np.searchsorted(thr_amp, sig_z, side='right')  # (B, L) in 0..B_amp-1
    out = np.empty((Bn, B_amp, L), dtype=np.uint8)
    # canali cumulativi
    for k in range(B_amp):
        out[:, k, :] = (idx >= k).astype(np.uint8)
    return out  # (B, B_amp, L)


def _encode_spectral(sig_z: np.ndarray, fs: float,
                     bands: Tuple[Tuple[float, float], ...],
                     thr_spec: np.ndarray) -> np.ndarray:
    """
    sig_z: (B, L) -> calcola potenza normalizzata per banda; per ogni banda soglia in K_spec-1 -> K_spec canali cumulativi.
    thr_spec: (Bbands, K_spec-1) oppure (Bbands, 0) se K_spec==0
    """
    Bn, L = sig_z.shape
    Bbands = len(bands)
    if thr_spec.shape[1] == 0:
        return np.zeros((Bn, 0, L), dtype=np.uint8)

    win = np.hamming(L).astype(np.float32)
    Xf = np.fft.rfft(sig_z * win, axis=-1)
    power = (np.abs(Xf) ** 2).astype(np.float32)  # (B, F)

    _, masks = _band_masks(L, fs, bands)
    # energia per banda normalizzata
    denom = power.sum(axis=-1, keepdims=True) + 1e-12
    band_e = np.empty((Bn, Bbands), dtype=np.float32)
    for b in range(Bbands):
        band_e[:, b] = (power[:, masks[b]].sum(axis=-1) / denom[:, 0]).astype(np.float32)

    # per banda, level encoding sui K_spec
    K_spec = thr_spec.shape[1] + 1
    rows = []
    for b in range(Bbands):
        idx = np.searchsorted(thr_spec[b], band_e[:, b], side='right')  # (B,)
        # cumulativo su L con repliche costanti lungo il tempo
        # costruiamo K_spec righe costanti e le replichiamo su L
        band_bin = np.empty((Bn, K_spec), dtype=np.uint8)
        for k in range(K_spec):
            band_bin[:, k] = (idx >= k).astype(np.uint8)
        band_bin = np.repeat(band_bin[:, :, None], repeats=L, axis=2)  # (B, K_spec, L)
        rows.append(band_bin)
    return np.concatenate(rows, axis=1)  # (B, Bbands*K_spec, L)


def _encode_var(sig_z: np.ndarray, var_windows: List[int],
                var_stats: Dict[int, Tuple[float, float]],
                k_var: int) -> np.ndarray:
    """
    sig_z: (B, L) -> rolling std a più scale, normalizza su [0,1] con (lo,hi),
    poi multi-soglia in k_var (cumulativo).
    """
    Bn, L = sig_z.shape
    rows = []
    taus = np.linspace(0, 1, k_var + 1, dtype=np.float32)[1:-1] if k_var > 0 else np.array([], dtype=np.float32)

    for w in var_windows:
        stdv = _rolling_std_batch(sig_z, w)      # (B, L-w+1)
        stdv = _pad_to_length(stdv, L, w)        # (B, L)
        lo, hi = var_stats[w]
        x01 = np.clip((stdv - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        if k_var <= 0:
            continue
        # multi soglia cumulativa
        out = np.empty((Bn, k_var, L), dtype=np.uint8)
        for i, t in enumerate(taus):
            out[:, i, :] = (x01 >= t).astype(np.uint8)
        rows.append(out)
    if not rows:
        return np.zeros((Bn, 0, L), dtype=np.uint8)
    return np.concatenate(rows, axis=1)  # (B, len(var_windows)*k_var, L)


def _temporal_pool(X: np.ndarray, pool: int) -> np.ndarray:
    if pool <= 1:
        return X
    N, C, L = X.shape
    newL = (L // pool) * pool
    if newL == 0:
        return X
    Xc = X[:, :, :newL].reshape(N, C, newL // pool, pool).mean(axis=-1)
    return (Xc >= 0.5).astype(np.uint8)


# ------------------------- targets SNR -------------------------

def _snr_bins_and_ordinal(snr_db: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    snr_db: (N,) float32 (può contenere NaN per assenti).
    edges:  (K+1,) es. [0,3,6,9,12,15]
    Ritorna:
      binlabel: (N,) in [0..K-1]  (NaN -> 0 di default)
      ordinal:  (N, K) con y_ge[k] = 1{ snr >= tau_k }, tau_k midpoint tra edges
    """
    K = edges.shape[0] - 1
    taus = (edges[:-1] + edges[1:]) / 2.0  # (K,)
    s = snr_db.copy()
    s = np.where(np.isnan(s), edges[0], s)                       # NaN -> bin più basso
    s = np.clip(s, edges[0], edges[-1] - 1e-6)
    binlab = np.digitize(s, edges[1:], right=False).astype(np.uint8)  # 0..K-1
    # ordinal
    ords = np.empty((s.shape[0], K), dtype=np.uint8)
    for k in range(K):
        ords[:, k] = (s >= taus[k]).astype(np.uint8)
    return binlab, ords


# ------------------------- split writer -------------------------

def _write_split(h5: h5py.File, split: str, npz_path: str,
                 stats: Dict, cfg: argparse.Namespace) -> Dict:
    data = np.load(npz_path, allow_pickle=True)
    X_noisy = data["X_noisy"].astype(np.float32)     # (N, L) o (N, C_leads, L)
    y_multi = data["y_multi"].astype(np.int32)       # (N, 3)
    snr_db  = data["snr_db"].astype(np.float32)      # (N, 3)

    # normalizza per-finestra (z-score)
    if X_noisy.ndim == 3:
        N, C_leads, L = X_noisy.shape
        Xz = _zscore_last(X_noisy)
    else:
        N, L = X_noisy.shape
        C_leads = 1
        Xz = _zscore_last(X_noisy[:, None, :])       # (N, 1, L)

    # feature per lead -> concat sui canali
    B_amp = stats["amp_thr"].shape[0] + 1
    B_spec = len(stats["spec_bands"]) * (stats["spec_thr"].shape[1] + 1)
    B_var  = len(stats["var_windows"]) * cfg.k_var
    C_tot_per_lead = B_amp + B_spec + B_var
    C_tot = C_tot_per_lead * C_leads

    # prealloc dataset con chunks/compression
    g = h5.create_group(split)
    Xd = g.create_dataset(
        "X",
        shape=(N, C_tot, L), dtype=np.uint8,
        chunks=(min(256, N), min(128, C_tot), min(256, L)),
        compression="gzip", compression_opts=4, shuffle=True,
    )
    g.create_dataset("snr_db_raw", data=snr_db, dtype=np.float32, chunks=True)

    # targets: presence + ordinal + binlabel
    tg = g.create_group("targets")
    pg = tg.create_group("present")
    og = tg.create_group("ordinal")
    bg = tg.create_group("binlabel")
    for n in NOISES:
        pg.create_dataset(n.lower(), data=y_multi[:, NOISE_TO_IDX[n]].astype(np.uint8), dtype=np.uint8, chunks=True)

    # SNR: per rumore, bins + ordinal
    edges = stats["snr_edges"]
    for ni, n in enumerate(NOISES):
        binlab, ords = _snr_bins_and_ordinal(snr_db[:, ni], edges)
        og.create_dataset(n.lower(), data=ords, dtype=np.uint8, chunks=True)         # (N, K_ord)
        bg.create_dataset(n.lower(), data=binlab, dtype=np.uint8, chunks=True)       # (N,)

    # costruzione feature in batch
    BATCH = cfg.batch_size
    thr_amp = stats["amp_thr"]                      # (B_amp-1,)
    bands = stats["spec_bands"]                     # tuple di tuple
    thr_spec = stats["spec_thr"]                    # (Bbands, K_spec-1)
    var_windows = stats["var_windows"]              # list
    var_stats = stats["var_stats"]                  # dict {w: (lo,hi)}
    fs = float(cfg.fs)

    # calcolo dimensioni di posizionamento canali
    for i0 in range(0, N, BATCH):
        i1 = min(N, i0 + BATCH)
        block = Xz[i0:i1]                           # (b, C_leads, L)
        b = block.shape[0]
        # per ogni lead computa feature e concat
        rows_per_lead = []
        for c in range(C_leads):
            sig = block[:, c, :]                    # (b, L)
            # amp
            z = sig
            amp = _encode_amplitude(z, thr_amp)     # (b, B_amp, L)
            # spectral
            spec = _encode_spectral(z, fs, bands, thr_spec)  # (b, Bbands*K_spec, L)
            # var
            varr = _encode_var(z, var_windows, var_stats, cfg.k_var)  # (b, len(w)*k_var, L)
            rows_per_lead.append(np.concatenate([amp, spec, varr], axis=1))  # (b, C_lead, L)

        feat = np.concatenate(rows_per_lead, axis=1)  # (b, C_tot, L)
        Xd[i0:i1] = feat

        if (i0 // BATCH) % max(1, (N // BATCH) // 10 + 1) == 0:
            print(f"[{split}] {i1}/{N} ({int(100*i1/N)}%)")

    # pooling temporale
    if cfg.pool > 1:
        Xd_pool = (Xd[:, :, : (L // cfg.pool) * cfg.pool].reshape(N, C_tot, -1, cfg.pool).mean(axis=-1) >= 0.5).astype(np.uint8)
        del h5[f"{split}/X"]
        g.create_dataset("X", data=Xd_pool, dtype=np.uint8, chunks=True, compression="gzip", compression_opts=4, shuffle=True)

    # meta split
    smeta = {
        "N": int(N),
        "C_leads": int(C_leads),
        "L_in": int(L),
        "L": int(h5[f"{split}/X"].shape[2]),
        "C": int(h5[f"{split}/X"].shape[1]),
    }
    g.attrs["meta_json"] = json.dumps(smeta)
    return smeta


def _write_split_from_h5(h5_out: h5py.File, split: str, h5_in: h5py.File,
                         stats: Dict, cfg: argparse.Namespace) -> Dict:
    """Versione di _write_split che legge da un gruppo HDF5 di input
    generato da data_generator_estimator.py: datasets 'sig', 'y_present', 'y_snr_db'.
    """
    g_in = h5_in[split]
    X_noisy = g_in["sig"][...]            # (N, L)
    y_multi = g_in["y_present"][...]      # (N, 3)
    snr_db  = g_in["y_snr_db"][...]       # (N, 3)

    # normalizza per-finestra (z-score)
    if X_noisy.ndim == 3:
        N, C_leads, L = X_noisy.shape
        Xz = _zscore_last(X_noisy.astype(np.float32))
    else:
        N, L = X_noisy.shape
        C_leads = 1
        Xz = _zscore_last(X_noisy.astype(np.float32)[:, None, :])  # (N,1,L)

    # feature per lead -> concat sui canali
    B_amp = stats["amp_thr"].shape[0] + 1
    B_spec = len(stats["spec_bands"]) * (stats["spec_thr"].shape[1] + 1)
    B_var  = len(stats["var_windows"]) * cfg.k_var
    C_tot_per_lead = B_amp + B_spec + B_var
    C_tot = C_tot_per_lead * C_leads

    # prealloc dataset con chunks/compression
    g = h5_out.create_group(split)
    Xd = g.create_dataset(
        "X",
        shape=(N, C_tot, L), dtype=np.uint8,
        chunks=(min(256, N), min(128, C_tot), min(256, L)),
        compression="gzip", compression_opts=4, shuffle=True,
    )
    g.create_dataset("snr_db_raw", data=snr_db.astype(np.float32), dtype=np.float32, chunks=True)

    # targets: presence + ordinal + binlabel
    tg = g.create_group("targets")
    pg = tg.create_group("present")
    og = tg.create_group("ordinal")
    bg = tg.create_group("binlabel")
    for n in NOISES:
        pg.create_dataset(n.lower(), data=y_multi[:, NOISE_TO_IDX[n]].astype(np.uint8), dtype=np.uint8, chunks=True)

    # SNR: per rumore, bins + ordinal
    edges = stats["snr_edges"]
    for ni, n in enumerate(NOISES):
        binlab, ords = _snr_bins_and_ordinal(snr_db[:, ni].astype(np.float32), edges)
        og.create_dataset(n.lower(), data=ords, dtype=np.uint8, chunks=True)         # (N, K_ord)
        bg.create_dataset(n.lower(), data=binlab, dtype=np.uint8, chunks=True)       # (N,)

    # costruzione feature in batch
    BATCH = cfg.batch_size
    thr_amp = stats["amp_thr"]                      # (B_amp-1,)
    bands = stats["spec_bands"]                     # tuple di tuple
    thr_spec = stats["spec_thr"]                    # (Bbands, K_spec-1)
    var_windows = stats["var_windows"]              # list
    var_stats = stats["var_stats"]                  # dict {w: (lo,hi)}
    fs = float(cfg.fs)

    for i0 in range(0, N, BATCH):
        i1 = min(N, i0 + BATCH)
        block = Xz[i0:i1]                           # (b, C_leads, L)
        rows_per_lead = []
        for c in range(C_leads):
            sig = block[:, c, :]                    # (b, L)
            amp = _encode_amplitude(sig, thr_amp)   # (b, B_amp, L)
            spec = _encode_spectral(sig, fs, bands, thr_spec) if thr_spec.shape[1] > 0 else np.zeros((sig.shape[0], 0, sig.shape[1]), dtype=np.uint8)
            varr = _encode_var(sig, var_windows, var_stats, cfg.k_var) if cfg.k_var > 0 and len(var_windows) > 0 else np.zeros((sig.shape[0], 0, sig.shape[1]), dtype=np.uint8)
            rows_per_lead.append(np.concatenate([amp, spec, varr], axis=1))

        feat = np.concatenate(rows_per_lead, axis=1)  # (b, C_tot, L)
        Xd[i0:i1] = feat

        if (i0 // BATCH) % max(1, (N // BATCH) // 10 + 1) == 0:
            print(f"[{split}] {i1}/{N} ({int(100*i1/N)}%)")

    # pooling temporale
    if cfg.pool > 1:
        Xd_pool = (Xd[:, :, : (L // cfg.pool) * cfg.pool].reshape(N, C_tot, -1, cfg.pool).mean(axis=-1) >= 0.5).astype(np.uint8)
        del h5_out[f"{split}/X"]
        h5_out.create_dataset("/".join([split, "X"]), data=Xd_pool, dtype=np.uint8, chunks=True, compression="gzip", compression_opts=4, shuffle=True)

    # meta split
    smeta = {
        "N": int(N),
        "C_leads": int(C_leads),
        "L_in": int(L),
        "L": int(h5_out[f"{split}/X"].shape[2]),
        "C": int(h5_out[f"{split}/X"].shape[1]),
    }
    h5_out[split].attrs["meta_json"] = json.dumps(smeta)
    return smeta


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Preprocess Estimator (presence + SNR) -> HDF5")
    ap.add_argument("--in_npz_dir", type=str, default="./data/consolidated/classifier", help="Directory con train/val/test .npz (X_noisy,y_multi,snr_db)")
    ap.add_argument("--in_h5", type=str, default=None, help="HDF5 generato dal data_generator_estimator (usa gruppi train/val/test)")
    ap.add_argument("--out_h5", type=str, default="./data/bin/estimator_patches.h5")

    # amplitude encoding
    ap.add_argument("--amp_bins", type=int, default=64, help="Numero di bin ampiezza (=> B_amp canali cumulativi)")
    ap.add_argument("--amp_sample_cap", type=int, default=2_000_000, help="Campioni max per stimare quantili ampiezza")

    # spectral features
    ap.add_argument("--add_spectral", action="store_true", help="Abilita righe spettrali")
    ap.add_argument("--spec_thresholds", type=int, default=6, help="K_spec (canali cumulativi per banda)")
    # alias di qualità della vita
    ap.add_argument("--spec_bins", type=int, default=None, help="Alias di --spec_thresholds")
    ap.add_argument("--spec_mode", choices=["fixed", "quantile"], default="fixed")
    ap.add_argument("--spec_sample_n", type=int, default=2048, help="Campioni per quantili banda (spec_mode=quantile)")
    ap.add_argument("--spec_bands", type=str,
                    default="0.0-0.5,0.5-2.0,2.0-8.0,8.0-20.0,45.0-55.0,55.0-70.0,95.0-105.0,100.0-120.0",
                    help="Bande in Hz separate da virgole, es. '0.0-0.5,45.0-55.0,100.0-120.0'")
    ap.add_argument("--bands", type=str, default=None, help="Alias di --spec_bands")
    # local variance
    ap.add_argument("--var_windows", type=str, default="16,64,256", help="Finestre per rolling std (campioni)")
    ap.add_argument("--k_var", type=int, default=4, help="Soglie cumulative per varianza locale")
    ap.add_argument("--var_p_lo", type=float, default=5.0, help="Percentile basso per normalizzare rolling std")
    ap.add_argument("--var_p_hi", type=float, default=95.0, help="Percentile alto per normalizzare rolling std")
    ap.add_argument("--var_sample_n", type=int, default=1024, help="Campioni per stimare percentili varianza")

    # pooling & fs
    ap.add_argument("--pool", type=int, default=2, help="Temporal pooling (>=1)")
    ap.add_argument("--fs", type=float, default=360.0, help="Sampling rate (Hz)")

    # SNR labelization
    ap.add_argument("--snr_edges", type=str, default="0,3,6,9,12,15", help="Edge dei bin SNR (K+1 valori)")

    # performance
    ap.add_argument("--batch_size", type=int, default=1024)

    args = ap.parse_args()

    # alias handling
    if args.spec_bins is not None:
        args.spec_thresholds = int(args.spec_bins)
    if args.bands is not None and args.bands.strip() != "":
        args.spec_bands = args.bands

    # parse bands
    bands_list = []
    for token in args.spec_bands.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split("-")
        bands_list.append((float(a), float(b)))
    bands = tuple(bands_list)
    var_windows = _parse_int_list(args.var_windows)
    # snr_edges: se input HDF5 presente e non è stato sovrascritto dall'utente (diverso dal default),
    # prova a ereditare dai metadati dell'HDF5 (train.attrs['snr_bin_edges']).
    user_snr_edges = np.array(_parse_float_list(args.snr_edges), dtype=np.float32)
    snr_edges: np.ndarray
    if args.in_h5 and os.path.exists(args.in_h5) and args.snr_edges == "0,3,6,9,12,15":
        with h5py.File(args.in_h5, "r") as h_in:
            try:
                snr_edges = np.array(h_in["train"].attrs.get("snr_bin_edges", user_snr_edges), dtype=np.float32)
            except Exception:
                snr_edges = user_snr_edges
    else:
        snr_edges = user_snr_edges
    if snr_edges.ndim != 1 or snr_edges.shape[0] < 2:
        raise ValueError("snr_edges deve contenere almeno due valori (K+1).")

    # auto-abilita spettrali se esplicitate soglie/bande
    if not args.add_spectral and (args.spec_thresholds > 0 or (args.bands is not None)):
        args.add_spectral = True

    # ---- carica TRAIN per stimare statistiche ----
    # Carica TRAIN per stimare statistiche
    Xtr: np.ndarray
    if args.in_h5:
        if not os.path.exists(args.in_h5):
            raise FileNotFoundError(args.in_h5)
        with h5py.File(args.in_h5, "r") as h_in:
            Xtr = h_in["train"]["sig"][...].astype(np.float32)
    else:
        train_npz = os.path.join(args.in_npz_dir, "train.npz")
        if not os.path.exists(train_npz):
            raise FileNotFoundError(train_npz)
        dtr = np.load(train_npz, allow_pickle=True)
        Xtr = dtr["X_noisy"].astype(np.float32)  # (N, L) o (N, C, L)
    # z-score per-finestra/lead prima della stima
    if Xtr.ndim == 2:
        Xtr_z = _zscore_last(Xtr[:, None, :])
    else:
        Xtr_z = _zscore_last(Xtr)

    # amp thresholds
    amp_thr = _compute_amp_thresholds_train(Xtr_z, args.amp_bins, sample_cap=args.amp_sample_cap)

    # spec thresholds
    if args.add_spectral and args.spec_thresholds > 0 and len(bands) > 0:
        spec_thr = _compute_spec_thresholds_train(
            Xtr_z, args.fs, bands, args.spec_thresholds,
            mode=args.spec_mode, sample_n=args.spec_sample_n
        )
    else:
        spec_thr = np.zeros((len(bands), 0), dtype=np.float32)

    # var stats
    var_stats = _compute_var_stats_train(
        Xtr_z, var_windows, args.var_p_lo, args.var_p_hi,
        sample_n=args.var_sample_n
    )

    # pack stats
    stats = {
        "amp_thr": amp_thr,                # (B_amp-1,)
        "spec_thr": spec_thr,              # (Bbands, K_spec-1)
        "spec_bands": bands,               # tuple di tuple
        "var_windows": var_windows,        # list
        "var_stats": var_stats,            # dict {w:(lo,hi)}
        "snr_edges": snr_edges,            # (K+1,)
    }

    # ---- scrittura HDF5 ----
    _ensure_dir(os.path.dirname(args.out_h5))
    with h5py.File(args.out_h5, "w") as h5:
        metas = {}
        if args.in_h5:
            with h5py.File(args.in_h5, "r") as h_in:
                for split in ("train", "val", "test"):
                    if split not in h_in:
                        raise FileNotFoundError(f"Split '{split}' mancante in {args.in_h5}")
                    metas[split] = _write_split_from_h5(h5, split, h_in, stats, args)
        else:
            for split in ("train", "val", "test"):
                npz_path = os.path.join(args.in_npz_dir, f"{split}.npz")
                if not os.path.exists(npz_path):
                    raise FileNotFoundError(npz_path)
                metas[split] = _write_split(h5, split, npz_path, stats, args)

        # meta globale
        gmeta = {
            "amp_bins": int(args.amp_bins),
            "amp_thr": stats["amp_thr"].tolist(),
            "add_spectral": bool(args.add_spectral),
            "spec_thresholds": int(args.spec_thresholds),
            "spec_mode": args.spec_mode,
            "spec_bands": [list(b) for b in stats["spec_bands"]],
            "spec_thr": stats["spec_thr"].tolist(),
            "var_windows": [int(w) for w in var_windows],
            "k_var": int(args.k_var),
            "var_stats": {str(k): [float(v[0]), float(v[1])] for k, v in stats["var_stats"].items()},
            "pool": int(args.pool),
            "fs": float(args.fs),
            "snr_edges": stats["snr_edges"].tolist(),
            "snr_taus": ((stats["snr_edges"][:-1] + stats["snr_edges"][1:]) / 2.0).astype(float).tolist(),
        }
        h5.attrs["meta_json"] = json.dumps(gmeta)

    print(f"Wrote HDF5 -> {args.out_h5}")
    print("Done.")


if __name__ == "__main__":
    main()
