#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare + Build Explainability Dataset (features + fractional intensities in [0,1])

- INPUT H5 atteso con: {split}_noisy, {split}_clean (float, shape [N, L]) e attr 'fs'.
- OUTPUT H5 con: {split}_X (uint8), {split}_y_bw|_y_emg|_y_pli (float32), attrs coerenti.

Caratteristiche:
- Feature binarie pronte per TM (via features.build_features).
- Intensità BW/EMG/PLI come FRAZIONI p_band / p_tot ∈ [0,1].
  * BW robusta: Welch [0.05, bw_hi] + time-domain low-pass a 0.7 Hz (50/50).
- Scrittura incrementale su H5 (chunked, gzip) → gestisce dataset grandi senza esplodere RAM.
- Attributi H5 con config (encoder, window, stride, welch, ecc.) per garantire coerenza end-to-end.

Esempio:
python -m explain.prepare_and_build_explain_dataset \
  --input-h5 data/explain_input_dataset.h5 \
  --out-h5   data/explain_features_dataset.h5 \
  --encoder bitplanes --bits 8 --window 512 --stride 256 --include-deriv
"""

from __future__ import annotations
import argparse, json, warnings, time
from pathlib import Path

import numpy as np
import h5py

try:
    # repo monolitica
    from features import build_features
except Exception:
    # pacchetto installabile
    from explain.features import build_features

from scipy.signal import welch, butter, filtfilt

# ----------------------------- Welch & BW robusta -----------------------------

def welch_band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float, nperseg: int) -> float:
    """PSD integrata su [f_lo, f_hi] con Welch ad alta risoluzione e detrend per togliere DC."""
    x = np.asarray(x, np.float32)
    noverlap = nperseg // 2
    nfft = 1 << int(np.ceil(np.log2(max(nperseg, 256))))
    f, Pxx = welch(
        x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, detrend='constant', return_onesided=True, scaling='density'
    )
    m = (f >= f_lo) & (f <= f_hi)
    return float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0

def lowpass_power_time(x: np.ndarray, fs: float, cutoff: float = 0.7, order: int = 2) -> float:
    """Potenza time-domain sotto cutoff Hz; robusta al problema del DC e della risoluzione Welch."""
    x = np.asarray(x, np.float32)
    b, a = butter(int(order), cutoff/(fs/2.0), btype='lowpass')
    xl = filtfilt(b, a, x).astype(np.float32)
    return float(np.mean(xl*xl) + 1e-12)

def bw_power_hybrid(n: np.ndarray, fs: float, nperseg: int, lp_weight: float = 0.5) -> float:
    """
    BW ibrida: Welch su [0.05, bw_hi] + time-domain lowpass @0.7 Hz.
    bw_hi include almeno il primo bin non-DC (f1 ~ fs/nperseg), ma non oltre 1.5 Hz.
    """
    f1 = float(fs) / float(nperseg)  # prima bin non-DC (≈0.703 Hz con fs=360, win=512)
    bw_hi = max(0.7, min(1.5, 1.1 * f1))
    p_welch = welch_band_power(n, fs, 0.05, bw_hi, nperseg=nperseg)
    p_time  = lowpass_power_time(n, fs, cutoff=0.7, order=2)
    lp_weight = float(np.clip(lp_weight, 0.0, 1.0))
    return (1.0 - lp_weight) * p_welch + lp_weight * p_time

def compute_intensities(noisy_seg: np.ndarray, clean_seg: np.ndarray, fs: float, win_len: int,
                        lp_weight: float = 0.5) -> tuple[float,float,float]:
    """
    Intensità frazionarie ∈[0,1] per BW/EMG/PLI su una finestra.
    n = noisy - clean; p_tot = Welch [0, hi_lim]; BW ibrida; EMG 20..hi_lim; PLI 45..55.
    """
    n = (np.asarray(noisy_seg, np.float32) - np.asarray(clean_seg, np.float32))
    nperseg = int(win_len)
    hi_lim = min(150.0, fs/2.0)

    p_tot = welch_band_power(n, fs, 0.0, hi_lim, nperseg=nperseg) + 1e-12
    p_bw  = bw_power_hybrid(n, fs, nperseg=nperseg, lp_weight=lp_weight)
    p_emg = welch_band_power(n, fs, 20.0, hi_lim, nperseg=nperseg)
    p_pli = welch_band_power(n, fs, 45.0, 55.0, nperseg=nperseg)

    i_bw  = np.clip(p_bw  / p_tot, 0.0, 1.0)
    i_emg = np.clip(p_emg / p_tot, 0.0, 1.0)
    i_pli = np.clip(p_pli / p_tot, 0.0, 1.0)
    return i_bw, i_emg, i_pli

# ------------------------------- Sliding windows ------------------------------

def iter_windows(sig: np.ndarray, window: int, stride: int):
    L = len(sig)
    for i in range(0, L - window + 1, stride):
        yield i, sig[i:i+window]

# ------------------------------- H5 Append utils ------------------------------

class H5Append:
    """Dset estendibili su H5 (chunked + gzip) per gestire dataset grandi senza esplodere RAM."""
    def __init__(self, h: h5py.File, name: str, dtype, feat_dim: int, comp=4):
        self.ds = h.create_dataset(
            name, shape=(0, feat_dim), maxshape=(None, feat_dim),
            chunks=(max(1, 32768 // max(1, feat_dim)), feat_dim),
            dtype=dtype, compression="gzip", compression_opts=int(comp)
        )
    def append(self, arr: np.ndarray):
        arr = np.asarray(arr)
        n0 = self.ds.shape[0]
        n1 = n0 + arr.shape[0]
        self.ds.resize((n1, self.ds.shape[1]))
        self.ds[n0:n1, :] = arr

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

# ------------------------------- Build per split ------------------------------

def build_split(hin: h5py.File, hout: h5py.File, split: str, fs: float,
                encoder: str, bits: int, levels: int, include_deriv: bool,
                window: int, stride: int, lp_weight: float, limit: int | None):
    print(f"\n=== ELABORAZIONE SPLIT '{split.upper()}' ===")
    
    noisy = hin.get(f"{split}_noisy")
    clean = hin.get(f"{split}_clean")
    if noisy is None or clean is None:
        print(f"[WARN] split '{split}' mancante (noisy/clean). Skippato.")
        return

    N = noisy.shape[0] if limit is None else min(limit, noisy.shape[0])
    signal_length = noisy.shape[1]
    
    print(f"📊 Dataset info:")
    print(f"  • Campioni: {N:,} (shape: {N} x {signal_length})")
    print(f"  • Window/Stride: {window}/{stride} → ~{(signal_length-window)//stride + 1} finestre/campione")
    print(f"  • FS: {fs} Hz, Encoder: {encoder}, BW LP weight: {lp_weight}")
    
    start_time = time.time()

    # Sonda per determinare feature_dim
    print("🔬 Analizzando primo campione per determinare dimensioni features...")
    x0 = np.asarray(noisy[0], np.float32)
    F0 = build_features(
        x0, window=window, stride=stride,
        encoder=encoder, bits=bits, levels=levels,
        include_deriv=include_deriv
    )
    F0 = F0.reshape(F0.shape[0], -1).astype(np.uint8, copy=False)
    feat_dim = F0.shape[1]
    n_windows = F0.shape[0]
    
    print(f"  • Features per finestra: {feat_dim:,}")
    print(f"  • Finestre per campione: {n_windows}")
    print(f"  • Finestre totali stimate: {N * n_windows:,}")
    
    # Dset estendibili
    print("📝 Creando dataset H5 estendibili...")
    dX   = H5Append(hout, f"{split}_X", dtype=np.uint8, feat_dim=feat_dim, comp=4)
    dYbw = H5AppendVec(hout, f"{split}_y_bw", comp=4, dtype=np.float32)
    dYem = H5AppendVec(hout, f"{split}_y_emg", comp=4, dtype=np.float32)
    dYpl = H5AppendVec(hout, f"{split}_y_pli", comp=4, dtype=np.float32)

    # Primo record
    print("⚡ Processando primo campione...")
    dX.append(F0)
    bw_list, emg_list, pli_list = [], [], []
    for i, _ in iter_windows(x0, window, stride):
        seg_n = np.asarray(noisy[0][i:i+window],  np.float32)
        seg_c = np.asarray(clean[0][i:i+window],  np.float32)
        bw, emg, pli = compute_intensities(seg_n, seg_c, fs, win_len=window, lp_weight=lp_weight)
        bw_list.append(bw); emg_list.append(emg); pli_list.append(pli)
    dYbw.append(np.asarray(bw_list,  np.float32))
    dYem.append(np.asarray(emg_list, np.float32))
    dYpl.append(np.asarray(pli_list, np.float32))
    
    # Statistiche primo campione per controllo qualità
    bw_arr, emg_arr, pli_arr = np.array(bw_list), np.array(emg_list), np.array(pli_list)
    print(f"  • BW:  min={bw_arr.min():.3f}, max={bw_arr.max():.3f}, mean={bw_arr.mean():.3f}")
    print(f"  • EMG: min={emg_arr.min():.3f}, max={emg_arr.max():.3f}, mean={emg_arr.mean():.3f}")
    print(f"  • PLI: min={pli_arr.min():.3f}, max={pli_arr.max():.3f}, mean={pli_arr.mean():.3f}")

    # Restanti campioni con progress monitoring
    print(f"\n🔄 Elaborando campioni rimanenti (2/{N})...")
    
    progress_interval = max(1, N // 20)  # 20 aggiornamenti max
    checkpoint_interval = max(1, N // 10)  # 10 checkpoint
    
    for k in range(1, N):
        # Progress update
        if k % progress_interval == 0 or k == N-1:
            elapsed = time.time() - start_time
            progress_pct = (k / (N-1)) * 100
            rate = k / elapsed if elapsed > 0 else 0
            eta = (N - k - 1) / rate if rate > 0 else 0
            print(f"  [{progress_pct:5.1f}%] Campione {k+1:,}/{N:,} | "
                  f"Rate: {rate:.1f} campioni/sec | ETA: {eta:.0f}s")
        
        # Processing
        x = np.asarray(noisy[k], np.float32)
        F = build_features(
            x, window=window, stride=stride,
            encoder=encoder, bits=bits, levels=levels,
            include_deriv=include_deriv
        )
        F = F.reshape(F.shape[0], -1).astype(np.uint8, copy=False)
        dX.append(F)

        bw_list, emg_list, pli_list = [], [], []
        for i, _ in iter_windows(x, window, stride):
            seg_n = np.asarray(noisy[k][i:i+window], np.float32)
            seg_c = np.asarray(clean[k][i:i+window], np.float32)
            bw, emg, pli = compute_intensities(seg_n, seg_c, fs, win_len=window, lp_weight=lp_weight)
            bw_list.append(bw); emg_list.append(emg); pli_list.append(pli)
        dYbw.append(np.asarray(bw_list,  np.float32))
        dYem.append(np.asarray(emg_list, np.float32))
        dYpl.append(np.asarray(pli_list, np.float32))
        
        # Checkpoint intermedio con statistiche
        if k % checkpoint_interval == 0 and k > 0:
            print(f"    💾 Checkpoint {k}: Dataset shape X={dX.ds.shape}, Y_bw={dYbw.ds.shape}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Split '{split}' completato in {total_time:.1f}s ({N/total_time:.1f} campioni/sec)")

    # Statistiche finali complete
    print(f"\n📈 Generazione statistiche finali split '{split}'...")
    
    for name in (f"{split}_y_bw", f"{split}_y_emg", f"{split}_y_pli"):
        print(f"  🔍 Analizzando {name}...")
        y = hout[name][:]
        
        # Statistiche base
        y_min, y_max = y.min(), y.max()
        y_mean, y_std = y.mean(), y.std()
        
        # Percentili
        q = np.percentile(y, [0, 5, 25, 50, 75, 95, 100]).astype(np.float32)
        
        # Distribuzione
        n_zeros = (y == 0).sum()
        n_small = (y <= 0.05).sum()
        n_large = (y >= 0.5).sum()
        
        # Salva attributi
        hout[name].attrs["quantiles_0_5_25_50_75_95_100"] = q
        hout[name].attrs["mean"] = float(y_mean)
        hout[name].attrs["std"] = float(y_std)
        hout[name].attrs["n_zeros"] = int(n_zeros)
        hout[name].attrs["n_samples"] = int(len(y))
        
        # Report dettagliato
        noise_type = name.split('_')[-1].upper()
        print(f"    📊 {noise_type}: min/max={y_min:.4f}/{y_max:.4f}")
        print(f"        • Media: {y_mean:.4f} ± {y_std:.4f}")
        print(f"        • Percentili: P5={q[1]:.3f}, P50={q[3]:.3f}, P95={q[5]:.3f}")
        print(f"        • Distribuzione: zeros={n_zeros/len(y)*100:.1f}%, "
              f"≤0.05={n_small/len(y)*100:.1f}%, ≥0.5={n_large/len(y)*100:.1f}%")
        
        # Controllo qualità
        if y_max > 1.0:
            print(f"        ⚠️  ATTENZIONE: valori > 1.0 trovati!")
        if y_mean == 0.0:
            print(f"        ⚠️  ATTENZIONE: tutti i valori sono zero!")
        
    print(f"📏 Dimensioni finali: X={dX.ds.shape}, Y_bw={dYbw.ds.shape}")

# ------------------------------------- Main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-h5", required=True, help="H5 con {split}_noisy/{split}_clean e attr 'fs'")
    ap.add_argument("--out-h5",   required=True, help="H5 di output con features+intensities")
    ap.add_argument("--splits", default="auto", help="Comma-list (train,val,test) oppure 'auto'")
    ap.add_argument("--encoder", choices=["bitplanes","thermometer","onehot2d"], default="bitplanes")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--include-deriv", action="store_true", help="aggiunge canale derivata/variazione dove previsto")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--bw-lp-weight", type=float, default=0.5, help="peso della componente time-domain nella BW [0..1]")
    ap.add_argument("--limit-train", type=int, default=None, help="limita N record nel train (debug)")
    ap.add_argument("--limit-val",   type=int, default=None, help="limita N record nel val (debug)")
    ap.add_argument("--limit-test",  type=int, default=None, help="limita N record nel test (debug)")
    args = ap.parse_args()

    in_p  = Path(args.input_h5)
    out_p = Path(args.out_h5)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(in_p), "r") as fin, h5py.File(str(out_p), "w") as fout:
        # fs
        fs = float(fin.attrs.get("fs", 360.0))
        fout.attrs["fs"] = fs

        # quali split?
        if args.splits == "auto":
            splits = [s for s in ("train","val","test") if fin.get(f"{s}_noisy") is not None and fin.get(f"{s}_clean") is not None]
            if not splits:
                raise SystemExit("[ERR] Nessuno split trovato (mancano *_noisy/*_clean).")
        else:
            splits = [s.strip() for s in args.splits.split(",") if s.strip()]

        # metadati globali
        print(f"\n⚙️  Configurazione globale:")
        print(f"  • FS: {fs} Hz")
        print(f"  • Encoder: {args.encoder} (bits={args.bits}, levels={args.levels})")
        print(f"  • Window/Stride: {args.window}/{args.stride}")
        print(f"  • Include derivative: {args.include_deriv}")
        print(f"  • BW LP weight: {args.bw_lp_weight}")
        
        fout.attrs.update({
            "created_by": "prepare_and_build_explain_dataset.py",
            "encoder": args.encoder,
            "bits": int(args.bits),
            "levels": int(args.levels),
            "include_deriv": int(bool(args.include_deriv)),
            "window": int(args.window),
            "stride": int(args.stride),
            "intensity_scale": "fractional_0_1",
            "bw_method": "hybrid(welch[0.05,bw_hi]+lowpass@0.7Hz)",
            "bw_lp_weight": float(args.bw_lp_weight),
            "welch_nperseg": int(args.window),
            "welch_noverlap": int(args.window // 2),
            "welch_detrend": "constant",
            "hi_lim_note": "min(150Hz, fs/2)",
            "version": "1.0.0",
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        print(f"📝 Splits da processare: {splits}")

        # build per split con timing globale
        total_start = time.time()
        
        for i, sp in enumerate(splits, 1):
            print(f"\n{'='*60}")
            print(f"🔄 SPLIT {i}/{len(splits)}: {sp.upper()}")
            print(f"{'='*60}")
            
            limit = None
            if sp == "train": limit = args.limit_train
            elif sp == "val": limit = args.limit_val
            elif sp == "test": limit = args.limit_test
            
            if limit:
                print(f"⚠️  Limite applicato: {limit:,} campioni")

            build_split(
                fin, fout, sp, fs,
                encoder=args.encoder, bits=args.bits, levels=args.levels, include_deriv=args.include_deriv,
                window=args.window, stride=args.stride,
                lp_weight=args.bw_lp_weight,
                limit=limit
            )
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"🎉 PROCESSO COMPLETATO!")
        print(f"{'='*60}")
        print(f"⏱️  Tempo totale: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"📁 File output: {out_p}")
        print(f"📊 Splits elaborati: {len(splits)}")
        
        # Riepilogo finale dimensioni
        print(f"\n📏 RIEPILOGO DIMENSIONI FINALI:")
        for sp in splits:
            if f"{sp}_X" in fout:
                x_shape = fout[f"{sp}_X"].shape
                y_shape = fout[f"{sp}_y_bw"].shape
                print(f"  • {sp:5}: X={x_shape[0]:,} x {x_shape[1]:,}, Y={y_shape[0]:,}")

    print(f"\n✅ [SUCCESS] Dataset salvato: {out_p}")
    print(f"💾 Dimensioni file: {out_p.stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
