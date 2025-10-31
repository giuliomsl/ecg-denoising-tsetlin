#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMU ECG Denoising - Dataset Preparation V2
==========================================

VERSIONE MIGLIORATA con features HF per EMG (Punto 5):
- Band energy (10-30 Hz, 30-50 Hz) → 2 features
- Zero Crossing Rate (ZCR) → 1 feature
- Teager-Kaiser Energy Operator (TKEO) → 1 feature
Totale: 4 features aggiuntive → +12 bits (3 livelli thermometer)

Aggiunte rispetto a V1:
1. compute_hf_features(): estrae band energy, ZCR, TKEO
2. Concatenazione features HF a bitplane standard
3. Output dataset compatibile con train_tmu_v2.py

Uso:
python src/explain/prepare_and_build_explain_dataset_v2.py \
  --input data/explain_input_dataset.h5 \
  --output data/explain_features_dataset_v2.h5 \
  --encoder thermometer --bits 32 --levels 3 \
  --window 360 --stride 120 --limit 3000
"""
from __future__ import annotations
import argparse
import time
import warnings
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import h5py
from scipy.signal import welch, butter, sosfiltfilt

warnings.filterwarnings("ignore", message="invalid value encountered")


class RunningStats:
    """Utility to keep streaming statistics without storing all values."""

    __slots__ = ("count", "sum", "sumsq", "minimum", "maximum")

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float, weight: int = 1) -> None:
        if weight <= 0:
            return
        val = float(value)
        self.count += weight
        self.sum += val * weight
        self.sumsq += (val * val) * weight
        if val < self.minimum:
            self.minimum = val
        if val > self.maximum:
            self.maximum = val

    def extend(self, values: Iterable[float]) -> None:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0:
            return
        self.count += arr.size
        self.sum += float(arr.sum())
        self.sumsq += float(np.dot(arr, arr))
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_min < self.minimum:
            self.minimum = arr_min
        if arr_max > self.maximum:
            self.maximum = arr_max

    def to_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": std,
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


@lru_cache(maxsize=None)
def _hf_filter_bank(fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns cached SOS coefficients for HF feature extraction."""
    sos_10_30 = butter(4, [10, 30], btype="band", fs=fs, output="sos")
    sos_30_50 = butter(4, [30, 50], btype="band", fs=fs, output="sos")
    sos_hp = butter(4, 5, btype="high", fs=fs, output="sos")
    return sos_10_30, sos_30_50, sos_hp


@lru_cache(maxsize=None)
def _lowpass_filter(fs: float) -> np.ndarray:
    """Returns cached SOS coefficients for the optional low-pass blend."""
    return butter(4, 0.7, btype="low", fs=fs, output="sos")

# ============================================================================
#                           FEATURE EXTRACTION
# ============================================================================

def compute_intensities(n: np.ndarray, fs: float = 360.0) -> dict[str, float]:
    """
    Calcola intensità BW/EMG/PLI per un segnale di rumore (risoluzione corretta).
    
    Args:
        n: Noise signal (1024 samples expected)
        fs: Sampling frequency
    
    Returns:
        dict: {'bw': float, 'emg': float, 'pli': float}
    """
    # Per BW (0.05-0.7 Hz) serve risoluzione < 0.05 Hz
    # Con fs=360 Hz: Δf = fs/nfft → per Δf < 0.05 serve nfft > 360/0.05 = 7200
    # Usiamo nfft=8192 per avere Δf ≈ 0.044 Hz con zero-padding
    nperseg = len(n)  # Usa tutto il segnale
    nfft = 8192  # Zero-padding per alta risoluzione frequenziale
    freqs, psd = welch(n, fs=fs, nperseg=nperseg, noverlap=0, nfft=nfft)
    
    # Bande di frequenza
    bw_mask = (freqs >= 0.05) & (freqs <= 0.7)     # Baseline Wander
    emg_mask = (freqs >= 10.0) & (freqs <= 100.0)  # EMG
    pli_mask = (freqs >= 48.0) & (freqs <= 52.0)   # Power-Line Interference
    
    def _band_power(mask, fmin, fmax):
        if not np.any(mask):
            return 0.0
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        if band_freqs.size >= 2:
            return float(np.trapz(band_psd, band_freqs))
        # Single bin -> approssima con PSD*bandwidth
        bandwidth = max(fmax - fmin, freqs[1] - freqs[0] if len(freqs) > 1 else fs / len(n))
        return float(band_psd[0] * bandwidth)

    bw_power = _band_power(bw_mask, 0.05, 0.7)
    emg_power = _band_power(emg_mask, 10.0, 100.0)
    pli_power = _band_power(pli_mask, 48.0, 52.0)
    
    # Normalizza a [0,1] con robust method
    total_power = np.trapz(psd, freqs)
    if total_power > 1e-10:
        bw_int = min(bw_power / total_power, 1.0)
        emg_int = min(emg_power / total_power, 1.0)
        pli_int = min(pli_power / total_power, 1.0)
    else:
        bw_int = emg_int = pli_int = 0.0
    
    return {'bw': bw_int, 'emg': emg_int, 'pli': pli_int}


def compute_hf_features(x: np.ndarray, fs: float = 360.0) -> np.ndarray:
    """
    Estrae 4 features ad alta frequenza per EMG detection (Punto 5).
    
    Args:
        x: Signal window (360 samples)
        fs: Sampling frequency
    
    Returns:
        np.ndarray: [band_10_30, band_30_50, zcr, tkeo] shape=(4,)
    """
    sos_10_30, sos_30_50, sos_hp = _hf_filter_bank(float(fs))

    # 1. Band energy 10-30 Hz
    y_10_30 = sosfiltfilt(sos_10_30, x)
    energy_10_30 = float(np.mean(y_10_30**2))
    
    # 2. Band energy 30-50 Hz
    y_30_50 = sosfiltfilt(sos_30_50, x)
    energy_30_50 = float(np.mean(y_30_50**2))
    
    # 3. Zero Crossing Rate (ZCR) dopo rimozione componente DC
    x_hp = sosfiltfilt(sos_hp, x)
    signs = np.signbit(x_hp)
    zero_crossings = np.count_nonzero(signs[:-1] != signs[1:]) / max(len(x_hp) - 1, 1)
    
    # 4. Teager-Kaiser Energy Operator (TKEO): TKEO[n] = x[n]^2 - x[n-1]*x[n+1]
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
    tkeo_mean = float(np.mean(np.abs(tkeo))) if tkeo.size else 0.0
    
    # Normalizza features a [0,1] con robust scaling (clip outliers)
    energy_10_30_norm = np.clip(energy_10_30 / 0.5, 0, 1)  # Empirical scale
    energy_30_50_norm = np.clip(energy_30_50 / 0.3, 0, 1)
    zcr_norm = np.clip(zero_crossings / 0.3, 0, 1)  # ZCR tipicamente < 0.3
    tkeo_norm = np.clip(tkeo_mean / 1.0, 0, 1)
    
    return np.array([energy_10_30_norm, energy_30_50_norm, zcr_norm, tkeo_norm], dtype=np.float32)


def thermometer_encode(val: float, levels: int = 3) -> np.ndarray:
    """
    Thermometer encoding: val → [1,1,...,1,0,0,...,0]
    
    Args:
        val: Value in [0,1]
        levels: Number of thermometer levels
    
    Returns:
        np.ndarray: Binary array of shape (levels,)
    """
    return thermometer_encode_vector(np.array([val], dtype=np.float32), levels=levels)[0]


def thermometer_encode_vector(vals: np.ndarray, levels: int) -> np.ndarray:
    """Vectorised thermometer encoding for 1-D arrays."""
    vals = np.asarray(vals, dtype=np.float32)
    vals = np.clip(vals, 0.0, 1.0)
    thresholds = vals[:, None] * levels  # shape (n, 1)
    level_ids = np.arange(levels, dtype=np.float32)
    encoded = thresholds > level_ids  # broadcast to (n, levels)
    return encoded.astype(np.uint8)


def bitplane_encode(x: np.ndarray, bits: int = 8, signed: bool = True) -> np.ndarray:
    """
    Bitplane encoding: converte segnale a uint e estrae bitplanes.
    
    Args:
        x: Signal (n_samples,)
        bits: Number of bits
        signed: If True, handle negative values
    
    Returns:
        np.ndarray: shape=(n_samples, bits)
    """
    x = np.asarray(x, dtype=np.float32)
    
    if signed:
        # Range simmetrico
        xmin, xmax = x.min(), x.max()
        amax = max(abs(xmin), abs(xmax))
        if amax > 1e-9:
            x_norm = x / amax  # [-1, 1]
        else:
            x_norm = x
        x_uint = ((x_norm + 1.0) * 0.5 * ((1 << bits) - 1)).astype(np.int32)
    else:
        # Range [0,1]
        xmin, xmax = x.min(), x.max()
        if xmax - xmin > 1e-9:
            x_norm = (x - xmin) / (xmax - xmin)
        else:
            x_norm = x - xmin
        x_uint = (x_norm * ((1 << bits) - 1)).astype(np.int32)
    
    x_uint = np.clip(x_uint, 0, (1 << bits) - 1)
    
    # Extract bitplanes
    bitplanes = np.zeros((x.shape[0], bits), dtype=np.uint8)
    for b in range(bits):
        bitplanes[:, b] = (x_uint >> b) & 1
    
    return bitplanes


def build_features(x: np.ndarray, window: int, stride: int,
                   encoder: str, bits: int, levels: int,
                   include_deriv: bool, lp_weight: float,
                   fs: float) -> np.ndarray:
    """
    Estrae features bitplane + HF per tutte le finestre di un segnale.
    
    Args:
        x: Noisy signal (1024 samples)
        window: Window size (360)
        stride: Stride (120)
        encoder: 'thermometer' or 'bitplane'
        bits: Number of bits/levels
        levels: Thermometer levels
        include_deriv: Add derivative features
        lp_weight: Low-pass weight for BW enhancement
    
    Returns:
        np.ndarray: shape=(n_windows, feature_dim)
    """
    # Low-pass filter per BW enhancement
    fs = float(fs)
    x = np.asarray(x, dtype=np.float32)
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive integers")
    if x.size < window:
        raise ValueError(f"Signal length ({x.size}) is shorter than window ({window}).")

    if lp_weight > 0:
        sos_lp = _lowpass_filter(fs)
        x_lp = sosfiltfilt(sos_lp, x)
        x = (1 - lp_weight) * x + lp_weight * x_lp
    
    # Windowing
    n_windows = (len(x) - window) // stride + 1
    if n_windows <= 0:
        return np.empty((0, 0), dtype=np.uint8)

    features = None
    
    for i in range(n_windows):
        start = i * stride
        end = start + window
        win = x[start:end]
        
        # 1. Bitplane encoding
        if encoder == "thermometer":
            bp = thermometer_encode_vector(win, levels)
        else:  # bitplane
            bp = bitplane_encode(win, bits, signed=True)
        
        # 2. Derivative (opzionale)
        if include_deriv:
            deriv = np.diff(win, prepend=win[0])
            if encoder == "thermometer":
                bp_deriv = thermometer_encode_vector(deriv, levels)
            else:
                bp_deriv = bitplane_encode(deriv, bits, signed=True)
            bp = np.hstack([bp, bp_deriv])
        
        # 3. HF features (Punto 5) - 4 valori scalari
        hf_feat = compute_hf_features(win, fs=fs)  # shape=(4,)
        
        # 4. Thermometer encode HF features
        hf_thermo = thermometer_encode_vector(hf_feat, levels).reshape(-1)
        
        # 5. Concatena bitplane + HF
        # Flatten bitplane to 1D
        bp_flat = bp.flatten()  # (window * n_channels,)
        
        # Replicate HF features to match bitplane length (broadcast)
        # Oppure: append alla fine (più semplice)
        feat_combined = np.concatenate([bp_flat, hf_thermo])
        
        if features is None:
            feature_dim = feat_combined.size
            features = np.empty((n_windows, feature_dim), dtype=np.uint8)
        features[i] = feat_combined.astype(np.uint8, copy=False)
    
    return features if features is not None else np.empty((0, 0), dtype=np.uint8)


# ============================================================================
#                           DATASET BUILDING
# ============================================================================

class H5AppendVec:
    def __init__(self, h: h5py.File, name: str, comp=4, dtype=np.float32):
        self.ds = h.create_dataset(
            name, shape=(0,), maxshape=(None,),
            chunks=(65536,), dtype=dtype, compression="gzip", compression_opts=int(comp)
        )
    def append(self, vec: np.ndarray):
        vec = np.asarray(vec).reshape(-1)
        n0 = self.ds.shape[0]
        n1 = n0 + vec.shape[0]
        self.ds.resize((n1,))
        self.ds[n0:n1] = vec


class H5AppendMat:
    def __init__(self, h: h5py.File, name: str, feat_dim: int, comp=4, dtype=np.uint8):
        self.ds = h.create_dataset(
            name, shape=(0, feat_dim), maxshape=(None, feat_dim),
            chunks=(4096, feat_dim), dtype=dtype, compression="gzip", compression_opts=int(comp)
        )
    def append(self, mat: np.ndarray):
        mat = np.asarray(mat)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        n0 = self.ds.shape[0]
        n1 = n0 + mat.shape[0]
        self.ds.resize((n1, self.ds.shape[1]))
        self.ds[n0:n1] = mat


def build_split(hin: h5py.File, hout: h5py.File, split: str, fs: float,
                encoder: str, bits: int, levels: int, include_deriv: bool,
                window: int, stride: int, lp_weight: float, limit: int | None
                ) -> Dict[str, Any]:
    print(f"\n=== ELABORAZIONE SPLIT '{split.upper()}' ===")
    
    noisy = hin.get(f"{split}_noisy")
    clean = hin.get(f"{split}_clean")
    if noisy is None or clean is None:
        print(f"[WARN] split '{split}' mancante (noisy/clean). Skippato.")
        return {
            "segments": 0,
            "windows": 0,
            "duration_sec": 0.0,
            "intensity_stats": {},
        }

    total_available = noisy.shape[0]
    N = total_available if limit is None else min(limit, total_available)
    signal_length = noisy.shape[1]
    segment_duration = signal_length / fs if fs > 0 else None
    
    print(f"📊 Dataset info:")
    print(f"  • Campioni: {N:,} (shape: {N} x {signal_length})")
    print(f"  • Window/Stride: {window}/{stride} → ~{(signal_length-window)//stride + 1} finestre/campione")
    print(f"  • FS: {fs} Hz, Encoder: {encoder}, LP weight: {lp_weight}")
    print(f"  • ✨ Features HF attivate: Band Energy (10-30, 30-50 Hz), ZCR, TKEO → +12 bits")

    if N == 0:
        print("  • Nessun campione da processare (limit o dati mancanti).")
        return {
            "segments": 0,
            "windows": 0,
            "duration_sec": 0.0,
            "intensity_stats": {},
        }
    
    start_time = time.time()

    # Sonda per determinare feature_dim
    print("🔬 Analizzando primo campione per determinare dimensioni features...")
    x0 = np.asarray(noisy[0], np.float32)
    F0 = build_features(
        x0, window=window, stride=stride,
        encoder=encoder, bits=bits, levels=levels,
        include_deriv=include_deriv, lp_weight=lp_weight, fs=fs
    )
    n_wins = F0.shape[0]
    feat_dim = F0.shape[1]
    print(f"  • Feature dim: {feat_dim} (bitplane + HF)")
    print(f"  • Windows per signal: {n_wins}")
    
    # Create appenders
    app_X = H5AppendMat(hout, f"{split}_X", feat_dim, comp=4, dtype=np.uint8)
    app_y_bw = H5AppendVec(hout, f"{split}_y_bw", comp=4, dtype=np.float32)
    app_y_emg = H5AppendVec(hout, f"{split}_y_emg", comp=4, dtype=np.float32)
    app_y_pli = H5AppendVec(hout, f"{split}_y_pli", comp=4, dtype=np.float32)

    target_stats = {
        "bw": RunningStats(),
        "emg": RunningStats(),
        "pli": RunningStats(),
    }
    total_windows = 0
    processed_segments = 0
    
    # Process first sample
    n = np.asarray(noisy[0], np.float32) - np.asarray(clean[0], np.float32)
    intensities = compute_intensities(n, fs)
    y_bw_rep = np.full(n_wins, intensities['bw'], dtype=np.float32)
    y_emg_rep = np.full(n_wins, intensities['emg'], dtype=np.float32)
    y_pli_rep = np.full(n_wins, intensities['pli'], dtype=np.float32)
    
    app_X.append(F0)
    app_y_bw.append(y_bw_rep)
    app_y_emg.append(y_emg_rep)
    app_y_pli.append(y_pli_rep)

    for key, stat in target_stats.items():
        stat.update(intensities[key], n_wins)
    total_windows += n_wins
    processed_segments += 1
    
    # Process remaining samples
    print(f"🔄 Processing {N-1} rimanenti...")
    for i in range(1, N):
        if (i+1) % 500 == 0 or i == N-1:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed
            eta = (N - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:6d}/{N}] {elapsed/60:5.1f}m | {rate:5.1f} sig/s | ETA: {eta/60:4.1f}m")
        
        x = np.asarray(noisy[i], np.float32)
        c = np.asarray(clean[i], np.float32)
        n = x - c
        
        F = build_features(
            x, window=window, stride=stride,
            encoder=encoder, bits=bits, levels=levels,
            include_deriv=include_deriv, lp_weight=lp_weight, fs=fs
        )
        
        intensities = compute_intensities(n, fs)
        y_bw_rep = np.full(F.shape[0], intensities['bw'], dtype=np.float32)
        y_emg_rep = np.full(F.shape[0], intensities['emg'], dtype=np.float32)
        y_pli_rep = np.full(F.shape[0], intensities['pli'], dtype=np.float32)
        
        app_X.append(F)
        app_y_bw.append(y_bw_rep)
        app_y_emg.append(y_emg_rep)
        app_y_pli.append(y_pli_rep)

        for key, stat in target_stats.items():
            stat.update(intensities[key], F.shape[0])
        total_windows += F.shape[0]
        processed_segments += 1
    
    elapsed = time.time() - start_time
    summary = {
        "segments": int(processed_segments),
        "segments_available": int(total_available),
        "windows": int(total_windows),
        "signal_length": int(signal_length),
        "window": int(window),
        "stride": int(stride),
        "feature_dim": int(feat_dim),
        "windows_per_segment": int(n_wins),
        "fs": float(fs),
    }
    if limit is not None:
        summary["limit"] = int(limit)
    if segment_duration is not None:
        duration_sec = segment_duration * processed_segments
        summary["duration_sec"] = float(duration_sec)
        summary["duration_minutes"] = float(duration_sec / 60.0)
        summary["duration_hours"] = float(duration_sec / 3600.0)
    summary["intensity_stats"] = {k: stat.to_dict() for k, stat in target_stats.items()}

    print(f"✅ Split '{split}' completato:")
    print(f"  • Total windows: {total_windows:,}")
    rate = (total_windows / elapsed) if elapsed > 0 else 0.0
    print(f"  • Tempo: {elapsed/60:.1f} min ({rate:.1f} win/s)")
    
    for key, label in (("bw", "BW"), ("emg", "EMG"), ("pli", "PLI")):
        stats_dict = summary["intensity_stats"][key]
        if stats_dict["count"] == 0 or stats_dict["mean"] is None:
            print(f"  • {label}: nessun dato")
        else:
            print(
                f"  • {label}: mean={stats_dict['mean']:.3f}, std={stats_dict['std']:.3f}, "
                f"range=[{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build TMU dataset V2 with HF features")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 (noisy/clean)")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 (features)")
    parser.add_argument("--encoder", type=str, default="thermometer", choices=["thermometer", "bitplane"])
    parser.add_argument("--bits", type=int, default=32, help="Bits for bitplane (o levels per thermometer)")
    parser.add_argument("--levels", type=int, default=3, help="Thermometer levels")
    parser.add_argument("--include-deriv", action="store_true", help="Include derivative features")
    parser.add_argument("--window", type=int, default=360, help="Window size (samples)")
    parser.add_argument("--stride", type=int, default=120, help="Stride (samples)")
    parser.add_argument("--lp-weight", type=float, default=0.0, help="Low-pass weight for BW")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per split")
    parser.add_argument("--splits", nargs="+", default=None, help="Lista di split da processare (default: tutti disponibili)")
    parser.add_argument("--fs", type=float, default=360.0, help="Frequenza di campionamento utilizzata per le features")
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 TMU ECG Denoising - Dataset Preparation V2 (with HF features)")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Encoder: {args.encoder}, Bits/Levels: {args.bits}/{args.levels}")
    print(f"FS: {args.fs} Hz")
    print(f"Window: {args.window}, Stride: {args.stride}")
    print(f"LP weight: {args.lp_weight}")
    print(f"Derivative: {args.include_deriv}")
    print(f"✨ HF Features: Band Energy (10-30, 30-50 Hz), ZCR, TKEO → +12 bits")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_summaries: Dict[str, Any] = {}

    with h5py.File(input_path, "r") as hin, h5py.File(output_path, "w") as hout:
        detected = {key.split("_")[0].lower() for key in hin.keys() if key.endswith("_noisy")}
        if not detected:
            raise RuntimeError("Nessuno split trovato nell'HDF5 di input (attesi *_noisy).")

        preferred_order = ["train", "validation", "test"]
        available_splits = [s for s in preferred_order if s in detected]
        available_splits.extend(sorted(detected - set(available_splits)))

        if args.splits is None:
            splits_to_process = available_splits
        else:
            lower_available = {s.lower() for s in available_splits}
            splits_to_process = []
            seen: set[str] = set()
            for requested in args.splits:
                req = requested.lower()
                if req not in lower_available:
                    raise ValueError(f"Split '{requested}' non presente nell'input. Disponibili: {available_splits}")
                if req not in seen:
                    splits_to_process.append(req)
                    seen.add(req)

        print(f"Splits da processare: {splits_to_process}")

        for split in splits_to_process:
            summary = build_split(
                hin, hout, split, fs=args.fs,
                encoder=args.encoder, bits=args.bits, levels=args.levels,
                include_deriv=args.include_deriv,
                window=args.window, stride=args.stride,
                lp_weight=args.lp_weight, limit=args.limit
            )
            split_summaries[split] = summary

        # Save metadata into HDF5
        hout.attrs["encoder"] = args.encoder
        hout.attrs["bits"] = args.bits
        hout.attrs["levels"] = args.levels
        hout.attrs["window"] = args.window
        hout.attrs["stride"] = args.stride
        hout.attrs["lp_weight"] = args.lp_weight
        hout.attrs["include_deriv"] = args.include_deriv
        hout.attrs["version"] = "v2_with_hf_features"
        hout.attrs["fs"] = args.fs
        hout.attrs["created_at"] = datetime.now().isoformat()
        hout.attrs["splits"] = json.dumps(splits_to_process)
        if split_summaries:
            first_stats = next((s for s in split_summaries.values() if s.get("feature_dim")), None)
            if first_stats:
                hout.attrs["feature_dim"] = first_stats.get("feature_dim")
                hout.attrs["windows_per_segment"] = first_stats.get("windows_per_segment")
        hout.attrs["summary_json"] = json.dumps(split_summaries)

    duration_values = [
        s["duration_sec"] for s in split_summaries.values()
        if s.get("duration_sec") is not None
    ]
    totals: Dict[str, Any] = {
        "segments": int(sum(s.get("segments", 0) for s in split_summaries.values())),
        "windows": int(sum(s.get("windows", 0) for s in split_summaries.values())),
        "duration_sec": float(sum(duration_values)) if duration_values else None,
    }
    if totals["duration_sec"] is not None:
        totals["duration_minutes"] = totals["duration_sec"] / 60.0
        totals["duration_hours"] = totals["duration_sec"] / 3600.0

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "config": {
            "encoder": args.encoder,
            "bits": args.bits,
            "levels": args.levels,
            "include_deriv": args.include_deriv,
            "window": args.window,
            "stride": args.stride,
            "lp_weight": args.lp_weight,
            "fs": args.fs,
            "limit": args.limit,
            "splits": splits_to_process,
        },
        "splits": split_summaries,
        "totals": totals,
    }

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"📝 Summary salvato in: {summary_path}")

    print("Totali aggregati:")
    print(f"  • Segmenti: {totals['segments']:,}")
    print(f"  • Finestre: {totals['windows']:,}")
    if totals["duration_sec"] is not None:
        print(f"  • Durata: {totals['duration_hours']:.2f} h ({totals['duration_minutes']:.1f} min)")
    
    print("\n" + "="*80)
    print("✅ Dataset V2 preparato con successo!")
    print("="*80)


if __name__ == "__main__":
    main()
