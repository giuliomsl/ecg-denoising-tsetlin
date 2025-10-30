#!/usr/bin/env python3
"""
Simple calibration: Linear scaling to match validation mean/std.
Much more robust than isotonic on compressed predictions.

Usage:
    python calibrate_heads_simple.py \
      --features data/explain_features_dataset.h5 \
      --models data/models/tmu_production_6000c \
      --output data/models/tmu_production_6000c_calibrated_simple
"""

import argparse
import json
import pickle
import os
import shutil
from pathlib import Path
import numpy as np
import h5py
from sklearn.isotonic import IsotonicRegression

HEADS = ["bw", "emg", "pli"]

def binarize_uint8_to_i32(X: np.ndarray) -> np.ndarray:
    """Convert uint8 bitplanes to int32 {0, 1}."""
    if X.dtype == np.uint8:
        return X.astype(np.int32)
    return X

def load_validation_data(features_path: str):
    """Load X_val, y_val for all heads from features dataset."""
    print(f"📥 Loading validation data from {features_path}")
    
    with h5py.File(features_path, "r") as f:
        X_val = binarize_uint8_to_i32(f["val_X"][:])
        y_val = {
            "bw": f["val_y_bw"][:],
            "emg": f["val_y_emg"][:],
            "pli": f["val_y_pli"][:]
        }
    
    print(f"   X_val: {X_val.shape}, dtype={X_val.dtype}")
    for head in HEADS:
        print(f"   y_val_{head}: {y_val[head].shape}, range=[{y_val[head].min():.3f}, {y_val[head].max():.3f}]")
    
    return X_val, y_val

def predict_normalized(model, X: np.ndarray) -> np.ndarray:
    """Get normalized predictions [0,1] like in inference."""
    pred_sqrt = model.predict(X)
    
    # Normalize in sqrt space FIRST (before squaring)
    EMPIRICAL_MAX = 10.0
    pred_sqrt = pred_sqrt / EMPIRICAL_MAX
    pred_sqrt = np.clip(pred_sqrt, 0.0, 1.0)
    
    # Then inverse sqrt transform
    pred_nat = pred_sqrt ** 2
    
    return pred_nat

def fit_calibrator(y_true: np.ndarray, y_pred: np.ndarray, method: str = "isotonic", auto_invert: bool = True) -> dict:
    """
    Fit calibrator con auto-inversione per correlazioni negative.
    
    Args:
        y_true: Ground truth values [0, 1]
        y_pred: Raw predictions [0, 1]
        method: 'isotonic' o 'linear'
        auto_invert: Se True, inverte predizioni quando r < -0.05
    
    Returns:
        Calibration metadata dict
    """
    # Misura correlazione
    r = np.corrcoef(y_true, y_pred)[0, 1]
    invert = auto_invert and (r < -0.05)
    
    x = -y_pred if invert else y_pred
    cal = {"method": method, "invert": bool(invert), "pearson_r": float(r)}
    
    if method == "isotonic":
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        ir.fit(x, y_true)
        cal["kind"] = "isotonic"
        # Serializzazione: salviamo i punti del PAV (Pooled Adjacent Violators)
        cal["X_thresholds"] = ir.X_thresholds_.tolist()
        cal["y_thresholds"] = ir.y_thresholds_.tolist()
    else:  # fallback lineare
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
        cal["kind"] = "linear"
        cal["a"] = float(a)
        cal["b"] = float(b)
    
    return cal

def apply_calibrator(y_pred: np.ndarray, cal: dict) -> np.ndarray:
    """
    Applica calibratore (con eventuale inversione).
    
    Args:
        y_pred: Raw predictions [0, 1]
        cal: Calibration metadata from fit_calibrator()
    
    Returns:
        Calibrated predictions [0, 1]
    """
    x = -y_pred if cal.get("invert", False) else y_pred
    
    if cal["kind"] == "isotonic":
        # Valutazione isotonica veloce via interp su PAV
        X = np.array(cal["X_thresholds"], dtype=float)
        Y = np.array(cal["y_thresholds"], dtype=float)
        out = np.interp(x, X, Y)
    else:  # linear
        out = cal["a"] * x + cal["b"]
    
    return np.clip(out, 0.0, 1.0)

def compute_mae_by_range(pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute MAE for Low (<0.05), Medium (0.05-0.2), High (≥0.2) ranges."""
    low_mask = y_true < 0.05
    med_mask = (y_true >= 0.05) & (y_true < 0.2)
    high_mask = y_true >= 0.2
    
    results = {}
    
    if low_mask.sum() > 0:
        results["low"] = {
            "mae": float(np.mean(np.abs(pred[low_mask] - y_true[low_mask]))),
            "n": int(low_mask.sum())
        }
    
    if med_mask.sum() > 0:
        results["medium"] = {
            "mae": float(np.mean(np.abs(pred[med_mask] - y_true[med_mask]))),
            "n": int(med_mask.sum())
        }
    
    if high_mask.sum() > 0:
        results["high"] = {
            "mae": float(np.mean(np.abs(pred[high_mask] - y_true[high_mask]))),
            "n": int(high_mask.sum())
        }
    
    results["overall"] = {
        "mae": float(np.mean(np.abs(pred - y_true))),
        "n": int(len(y_true))
    }
    
    return results

def calibrate_head(head: str, model, X_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic") -> dict:
    """Calibrate single head with isotonic regression and auto-inversion."""
    print(f"\n🔧 Calibrating {head.upper()} head (method={method})...")
    
    # Get normalized predictions
    pred = predict_normalized(model, X_val)
    print(f"   Predictions (normalized): mean={pred.mean():.4f}, std={pred.std():.4f}")
    print(f"   Ground truth:             mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    print(f"   Bias (before):            {pred.mean() - y_val.mean():.4f}")
    
    # Check correlation
    r = np.corrcoef(pred, y_val)[0, 1]
    print(f"   Pearson r (before):       {r:.4f}")
    if r < -0.05:
        print(f"   ⚠️  NEGATIVE CORRELATION detected! Auto-inversion will be applied.")
    
    # Fit calibrator (with auto-inversion if needed)
    calib = fit_calibrator(y_val, pred, method=method, auto_invert=True)
    
    if calib.get("invert", False):
        print(f"   ✅ Auto-inversion ENABLED (r={calib['pearson_r']:.4f} < -0.05)")
    else:
        print(f"   ℹ️  Auto-inversion NOT needed (r={calib['pearson_r']:.4f})")
    
    # Apply calibration
    pred_cal = apply_calibrator(pred, calib)
    print(f"   Calibrated: mean={pred_cal.mean():.4f}, std={pred_cal.std():.4f}")
    print(f"   Bias (after): {pred_cal.mean() - y_val.mean():.4f}")
    
    # Compute MAE by range
    mae_before = compute_mae_by_range(pred, y_val)
    mae_after = compute_mae_by_range(pred_cal, y_val)
    
    # Correlation after calibration
    r_after = np.corrcoef(pred_cal, y_val)[0, 1]
    print(f"   Pearson r (after):        {r_after:.4f} (Δ={r_after - r:+.4f})")
    
    print(f"\n   📊 MAE by Range (BEFORE calibration):")
    for range_name, stats in mae_before.items():
        if range_name != "overall":
            pct = 100 * stats["n"] / len(y_val)
            print(f"      {range_name.capitalize():8s}: MAE={stats['mae']:.4f}, N={stats['n']:5d} ({pct:5.1f}%)")
    print(f"      Overall:  MAE={mae_before['overall']['mae']:.4f}")
    
    print(f"\n   📊 MAE by Range (AFTER calibration):")
    for range_name, stats in mae_after.items():
        if range_name != "overall":
            improvement = mae_before.get(range_name, {}).get("mae", 0) - stats["mae"]
            pct = 100 * stats["n"] / len(y_val)
            print(f"      {range_name.capitalize():8s}: MAE={stats['mae']:.4f}, N={stats['n']:5d} ({pct:5.1f}%) [Δ {improvement:+.4f}]")
    
    overall_improvement = mae_before['overall']['mae'] - mae_after['overall']['mae']
    improvement_pct = 100 * overall_improvement / mae_before['overall']['mae'] if mae_before['overall']['mae'] > 0 else 0
    print(f"      Overall:  MAE={mae_after['overall']['mae']:.4f} [Δ {overall_improvement:+.4f}, {improvement_pct:+.1f}%]")
    
    # Package calibration metadata
    calibration_meta = {
        "type": method,
        "calibrator": calib,
        "validation_stats": {
            "mae_before": mae_before,
            "mae_after": mae_after,
            "bias_before": float(pred.mean() - y_val.mean()),
            "bias_after": float(pred_cal.mean() - y_val.mean()),
            "pearson_before": float(r),
            "pearson_after": float(r_after),
        }
    }
    
    return calibration_meta

def main():
    parser = argparse.ArgumentParser(description="Calibrate TMU intensity estimator heads with isotonic regression.")
    parser.add_argument("--features", required=True, help="Path to explain_features_dataset_v2.h5")
    parser.add_argument("--models", required=True, help="Directory containing trained models")
    parser.add_argument("--output", required=True, help="Output directory for calibrated models")
    parser.add_argument("--heads", nargs="+", default=["bw", "emg", "pli"], 
                        help="Heads to calibrate (default: bw emg pli)")
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "linear"],
                        help="Calibration method: isotonic (default) or linear")
    args = parser.parse_args()
    
    print(f"🚀 TMU Head Calibration (method={args.method})")
    print(f"   Features:    {args.features}")
    print(f"   Models dir:  {args.models}")
    print(f"   Output dir:  {args.output}")
    print(f"   Heads:       {args.heads}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load validation data
    print(f"\n📥 Loading validation data...")
    with h5py.File(args.features, "r") as f:
        X_val = f["validation_X"][:]
        y_bw_val = f["validation_y_bw"][:]
        y_emg_val = f["validation_y_emg"][:]
        y_pli_val = f["validation_y_pli"][:]
    
    print(f"   Loaded {len(X_val)} validation samples")
    print(f"   Features shape: {X_val.shape}, dtype: {X_val.dtype}")
    
    # Calibrate each head
    results = {}
    for head in args.heads:
        # Load model
        model_path = os.path.join(args.models, f"rtm_intensity_{head}.pkl")
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}, skipping {head}")
            continue
        
        print(f"\n📦 Loading {head.upper()} model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Get corresponding y_val
        y_val_map = {"bw": y_bw_val, "emg": y_emg_val, "pli": y_pli_val}
        y_val = y_val_map[head]
        
        # Calibrate
        calib_meta = calibrate_head(head, model, X_val, y_val, method=args.method)
        
        # Save calibration
        calib_path = os.path.join(args.output, f"{head}_calibration.json")
        with open(calib_path, "w") as f:
            json.dump(calib_meta, f, indent=2)
        print(f"   ✅ Saved calibration to {calib_path}")
        
        results[head] = calib_meta
    
    # Summary
    print(f"\n" + "="*80)
    print(f"📊 CALIBRATION SUMMARY (method={args.method}):")
    print(f"="*80)
    for head, meta in results.items():
        stats = meta["validation_stats"]
        mae_before = stats["mae_before"]["overall"]["mae"]
        mae_after = stats["mae_after"]["overall"]["mae"]
        improvement = mae_before - mae_after
        improvement_pct = 100 * improvement / mae_before if mae_before > 0 else 0
        
        r_before = stats.get("pearson_before", 0.0)
        r_after = stats.get("pearson_after", 0.0)
        
        inverted = meta["calibrator"].get("invert", False)
        invert_str = " [INVERTED]" if inverted else ""
        
        print(f"\n{head.upper()}{invert_str}:")
        print(f"   MAE:      {mae_before:.4f} → {mae_after:.4f} (Δ {improvement:+.4f}, {improvement_pct:+.1f}%)")
        print(f"   Pearson:  {r_before:.4f} → {r_after:.4f} (Δ {r_after - r_before:+.4f})")
    
    print(f"\n✅ Calibration complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
