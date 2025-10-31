#!/usr/bin/env python3
"""
Script SOLO INFERENZA per modelli TMU già trainati.
Carica modelli da pickle e fa inferenza su segnali test.
"""

import numpy as np
import h5py
import pickle
import argparse
import json
from pathlib import Path
import sys
from scipy.ndimage import median_filter

# Funzioni helper inline (invece di import che potrebbero fallire)
def binarize_uint8_to_i32(Xu8):
    """Binarize uint8 features to uint32 for TMU compatibility"""
    Xb = (Xu8 != 0).astype(np.uint32, copy=False)
    return np.ascontiguousarray(Xb)

def overlap_add(pred_windows, L_target, window, stride):
    """Ricostruisce serie temporale da predizioni sliding windows con overlap-add"""
    n_windows = len(pred_windows)
    out = np.zeros(L_target, dtype=np.float32)
    count = np.zeros(L_target, dtype=np.float32)
    
    for i in range(n_windows):
        start = i * stride
        end = min(start + window, L_target)
        out[start:end] += pred_windows[i]
        count[start:end] += 1.0
    
    # Normalizza per numero di overlap
    count = np.maximum(count, 1.0)
    out /= count
    return out

def smooth_med(serie, size=3):
    """Median filter smoothing"""
    from scipy.ndimage import median_filter
    return median_filter(serie, size=size, mode='nearest')

# Import features builder
sys.path.insert(0, str(Path(__file__).parent.parent))
from explain.features import build_features

def load_model(path):
    """Carica modello TMU da pickle."""
    print(f"📦 Loading model: {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✅ Model loaded: {model}")
    return model

def apply_calibration(pred_raw: np.ndarray, calib_meta: dict) -> np.ndarray:
    """
    Apply calibration (linear or isotonic) to predictions.
    
    Args:
        pred_raw: Raw predictions (already normalized to [0,1])
        calib_meta: Calibration metadata dict
    
    Returns:
        Calibrated predictions in [0, 1]
    """
    calib_type = calib_meta.get("type", "isotonic")
    
    if calib_type == "linear":
        # Simple linear: pred_cal = scale * pred + offset
        params = calib_meta["params"]
        pred_cal = params["scale"] * pred_raw + params["offset"]
        pred_cal = np.clip(pred_cal, 0.0, 1.0)
        return pred_cal
    
    elif calib_type == "isotonic":
        # Isotonic regression: P95 scaling + interpolation
        p95_scale = calib_meta["p95_scale"]
        iso_calib = calib_meta["isotonic"]
        
        # Step 1: P95 robust scaling
        pred_scaled = np.clip(pred_raw / p95_scale, 0.0, 1.0)
        
        # Step 2: Isotonic regression
        xs = np.asarray(iso_calib["x"])
        ys = np.asarray(iso_calib["y"])
        pred_calibrated = np.interp(pred_scaled, xs, ys)
        
        return pred_calibrated
    
    else:
        # Unknown type, return as-is
        return pred_raw

def infer_intensities(models, signals, window=512, stride=256, encoder="bitplanes", bits=8, 
                      calibration=None, save_per_window=False):
    """
    Inferenza intensities per BW/EMG/PLI su batch di segnali.
    
    Args:
        models: dict con chiavi "bw", "emg", "pli" → TMRegressor
        signals: array (N, L) segnali
        window, stride, encoder, bits: config feature extraction
        calibration: dict con calibration metadata per head (opzionale)
        save_per_window: se True, salva anche predizioni per-window (per diagnosi)
    
    Returns:
        intensities: array (N, 3, L) con [bw, emg, pli]
        per_window_data: dict (se save_per_window=True) con 'predictions' e 'metadata'
    """
    N = len(signals)
    L = signals.shape[1]
    intensities = np.zeros((N, 3, L), dtype=np.float32)
    
    # Per-window predictions storage (per diagnosi)
    per_window_preds = {head: [] for head in ["bw", "emg", "pli"]} if save_per_window else None
    per_window_meta = [] if save_per_window else None
    
    print(f"\n🔮 Inferenza su {N} segnali...")
    
    for i, sig in enumerate(signals):
        if i % 171 == 0 and i > 0:
            pct = 100.0 * i / N
            print(f"   📊 Progress: {i}/{N} ({pct:.1f}%)")
        
        # Feature extraction (sliding windows)
        X = build_features(
            sig,
            window=window,
            stride=stride,
            encoder=encoder,
            bits=bits,
            levels=64,
            include_deriv=True
        )
        
        X_flat = X.reshape(X.shape[0], -1).astype(np.uint8)
        X_bin = binarize_uint8_to_i32(X_flat)
        
        # Debug primo segnale
        if i == 0:
            print(f"\n[DEBUG] First signal:")
            print(f"   Windows: {X.shape[0]}, Features: {X_flat.shape[1]}")
            print(f"   Expected features: 8192 (bitplanes 8-bit + deriv)")
        
        # Predizioni per ogni head
        for j, head in enumerate(["bw", "emg", "pli"]):
            model = models[head]
            pred = model.predict(X_bin).astype(np.float32)
            
            # Debug RAW predictions (in sqrt space)
            if i == 0:
                print(f"   {head.upper()} pred_RAW (sqrt): min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
            
            # ✅ APPLY CALIBRATION SCALING (if available) - in sqrt space
            if calibration and head in calibration:
                calib_data = calibration[head]
                if 'scale_factor' in calib_data:
                    # P95 scaling method
                    scale = calib_data['scale_factor']
                    pred = pred / scale
                    
                    if i == 0:
                        print(f"   {head.upper()} after_P95_scaling (/{scale:.4f}): min={pred.min():.4f}, max={pred.max():.4f}")
            
            # ✅ INVERSE TRANSFORM: sqrt → square
            pred = pred ** 2
            
            # ✅ CLIP to [0, 1]
            pred = np.clip(pred, 0.0, 1.0)
            
            if i == 0:
                print(f"   {head.upper()} pred_FINAL (after square+clip): min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
            
            # Save per-window predictions (per diagnosi)
            if save_per_window:
                per_window_preds[head].extend(pred.tolist())
                # Salva metadata solo una volta per segnale
                if j == 0:  # Solo per primo head
                    for win_idx in range(len(pred)):
                        start_sample = win_idx * stride
                        end_sample = min(start_sample + window, L)
                        per_window_meta.append({
                            'signal_id': i,
                            'window_idx': win_idx,
                            'start': start_sample,
                            'end': end_sample
                        })
            
            # Overlap-add per ricostruire serie temporale
            serie = overlap_add(pred, L, window, stride)
            
            if i == 0:
                print(f"   {head.upper()} after_overlap_add: min={serie.min():.4f}, max={serie.max():.4f}, mean={serie.mean():.4f}")
            
            # Smoothing leggero
            serie = smooth_med(serie, size=3)
            
            if i == 0:
                print(f"   {head.upper()} after_smooth: min={serie.min():.4f}, max={serie.max():.4f}, mean={serie.mean():.4f}")
            
            intensities[i, j, :] = serie
    
    print(f"\n✅ Inferenza completata!")
    for j, head in enumerate(["bw", "emg", "pli"]):
        int_head = intensities[:, j, :]
        print(f"   {head.upper()}: range=[{int_head.min():.3f}, {int_head.max():.3f}], mean={int_head.mean():.3f}")
    
    if save_per_window:
        per_window_data = {
            'predictions': {head: np.array(per_window_preds[head], dtype=np.float32) for head in ["bw", "emg", "pli"]},
            'metadata': per_window_meta
        }
        print(f"\n📊 Per-window data saved:")
        print(f"   Total windows: {len(per_window_meta)}")
        for head in ["bw", "emg", "pli"]:
            pw = per_window_data['predictions'][head]
            print(f"   {head.upper()}: {pw.shape}, range=[{pw.min():.3f}, {pw.max():.3f}], mean={pw.mean():.3f}")
        return intensities, per_window_data
    
    return intensities, None

def denoise_with_intensities(signals, intensities):
    """
    Denoising usando intensities stimate.
    
    Args:
        signals: (N, L) segnali noisy
        intensities: (N, 3, L) intensities [bw, emg, pli]
    
    Returns:
        denoised: (N, L) segnali denoised
    """
    N, L = signals.shape
    denoised = np.zeros_like(signals)
    
    print(f"\n🧹 Denoising {N} segnali...")
    
    for i in range(N):
        if i % 342 == 0 and i > 0:
            print(f"   Denoising {i}/{N}...")
        
        sig = signals[i]
        bw = intensities[i, 0, :]
        emg = intensities[i, 1, :]
        pli = intensities[i, 2, :]
        
        # Total noise intensity
        total = np.clip(bw + emg + pli, 0.0, 1.0)
        
        # Adaptive smoothing based on total noise
        # Low noise → preserve, high noise → smooth more
        sig_smooth = np.copy(sig)
        
        for t in range(L):
            if total[t] > 0.3:  # High noise
                # Apply median filter
                left = max(0, t-2)
                right = min(L, t+3)
                sig_smooth[t] = np.median(sig[left:right])
        
        denoised[i] = sig_smooth
    
    print(f"✅ Denoising completato!")
    return denoised

def main():
    parser = argparse.ArgumentParser(description="Inferenza SOLO con modelli TMU già trainati")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory con i modelli pickle (bw.pkl, emg.pkl, pli.pkl)")
    parser.add_argument("--signals", type=str, required=True, help="Path dataset segnali .h5")
    parser.add_argument("--output", type=str, required=True, help="Path output .h5")
    parser.add_argument("--do-denoise", action="store_true", help="Esegui anche denoising")
    parser.add_argument("--save-per-window", action="store_true", help="Salva anche predizioni per-window (diagnosi)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔮 INFERENZA ONLY - Modelli TMU Pre-trainati")
    print("="*80)
    
    # 1. Carica modelli
    models_dir = Path(args.models_dir)
    models = {}
    for head in ["bw", "emg", "pli"]:
        model_path = models_dir / f"{head}.pkl"
        if not model_path.exists():
            print(f"❌ Modello non trovato: {model_path}")
            sys.exit(1)
        models[head] = load_model(model_path)
    
    print(f"\n✅ Tutti i modelli caricati!")
    
    # 2. Load calibration (if requested)
    calibration = None
    meta_path = models_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        # Check for simple P95 scaling calibration
        if "calibration_simple" in metadata:
            calib_data = metadata["calibration_simple"]
            if calib_data.get("type") == "p95_scaling":
                calibration = calib_data["scaling"]
                print(f"\n✅ P95 Calibration loaded from {meta_path}")
                for head in ["bw", "emg", "pli"]:
                    if head in calibration:
                        scale = calibration[head]["scale_factor"]
                        print(f"   {head.upper()}: scale_factor={scale:.4f}")
        else:
            print(f"\n⚠️  No calibration_simple found in {meta_path}")
    else:
        print(f"\n⚠️  No metadata.json found in {models_dir}")
    
    # 3. Carica segnali
    print(f"\n📥 Caricando segnali: {args.signals}")
    with h5py.File(args.signals, "r") as f:
        # Try different possible keys
        if "signals" in f:
            signals = f["signals"][:]
            fs = f["signals"].attrs.get("fs", 360.0)
        elif "test_noisy" in f:
            signals = f["test_noisy"][:]
            fs = 360.0
        elif "test_clean" in f:
            signals = f["test_clean"][:]
            fs = 360.0
        else:
            available = list(f.keys())
            raise KeyError(f"Nessun dataset trovato. Keys disponibili: {available}")
    
    print(f"✅ Signals: shape={signals.shape}, fs={fs} Hz")
    
    # 3. Inferenza intensities
    intensities, per_window_data = infer_intensities(
        models=models,
        signals=signals,
        window=512,
        stride=256,
        encoder="bitplanes",
        bits=8,
        calibration=calibration,
        save_per_window=args.save_per_window
    )
    
    # 4. Denoising (opzionale)
    denoised = None
    if args.do_denoise:
        denoised = denoise_with_intensities(signals, intensities)
    
    # 5. Salva risultati
    print(f"\n💾 Salvando risultati: {args.output}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("intensities", data=intensities, compression="gzip")
        if denoised is not None:
            f.create_dataset("denoised", data=denoised, compression="gzip")
        f["intensities"].attrs["shape_description"] = "(N_signals, 3_heads, L_samples)"
        f["intensities"].attrs["heads"] = "bw,emg,pli"
        
        # Salva per-window data se richiesto
        if per_window_data is not None:
            grp = f.create_group("per_window")
            for head in ["bw", "emg", "pli"]:
                grp.create_dataset(f"pred_{head}", data=per_window_data['predictions'][head], compression="gzip")
            
            # Salva metadata come dataset strutturato
            meta_arr = np.array([(m['signal_id'], m['window_idx'], m['start'], m['end']) 
                                 for m in per_window_data['metadata']],
                                dtype=[('signal_id', 'i4'), ('window_idx', 'i4'), ('start', 'i4'), ('end', 'i4')])
            grp.create_dataset("metadata", data=meta_arr, compression="gzip")
            print(f"   ✅ Per-window data saved: {len(per_window_data['metadata'])} windows")
    
    print(f"✅ Salvato!")
    print(f"   Intensities: {intensities.shape}")
    if denoised is not None:
        print(f"   Denoised: {denoised.shape}")
    if per_window_data is not None:
        print(f"   Per-window predictions: {per_window_data['predictions']['bw'].shape}")
    
    print("\n" + "="*80)
    print("🎉 COMPLETATO!")
    print("="*80)

if __name__ == "__main__":
    main()
