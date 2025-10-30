#!/usr/bin/env python3
"""
TMU Unified Calibration + Inference
===================================

Two modes:
1. Calibration mode (--calibrate): Fit calibrator on validation set
2. Inference mode (default): Apply calibrator on test/validation/train set

Calibration Usage:
    python src/explain/infer_tmu_calibrated.py --calibrate \
      --features data/explain_features_dataset_v2.h5 \
      --models models/tmu_v2_bw_retrain \
      --output models/tmu_v2_bw_retrain_calibrated \
      --heads bw \
      --method isotonic

Inference Usage:
    python src/explain/infer_tmu_calibrated.py \
      --features data/explain_features_dataset_v2.h5 \
      --models models/tmu_v2_bw_retrain \
      --calibrators models/tmu_v2_bw_retrain_calibrated \
      --output results/predictions_test.h5 \
      --heads bw \
      --split test
"""

import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error


def binarize_uint8_to_i32(X: np.ndarray) -> np.ndarray:
    """Convert uint8 bitplanes to int32 {0, 1}."""
    if X.dtype == np.uint8:
        return X.astype(np.int32)
    return X


def predict_normalized(model, X: np.ndarray) -> np.ndarray:
    """
    Get normalized predictions [0,1] from TMU model.
    
    Applies same normalization as training:
    1. Predict in sqrt space
    2. Normalize by EMPIRICAL_MAX
    3. Clip to [0, 1]
    4. Inverse sqrt transform (square)
    """
    pred_sqrt = model.predict(X)
    
    # Normalize in sqrt space FIRST (before squaring)
    EMPIRICAL_MAX = 10.0
    pred_sqrt = pred_sqrt / EMPIRICAL_MAX
    pred_sqrt = np.clip(pred_sqrt, 0.0, 1.0)
    
    # Then inverse sqrt transform
    pred_nat = pred_sqrt ** 2
    
    return pred_nat


def load_model(model_path):
    """Load TMU model."""
    print(f"📦 Loading model: {model_path.name}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_calibrator(calib_path):
    """Load calibration parameters."""
    print(f"📦 Loading calibrator: {calib_path.name}")
    with open(calib_path, "r") as f:
        calib = json.load(f)
    return calib


def fit_calibrator(y_true: np.ndarray, y_pred: np.ndarray, method: str = "isotonic", auto_invert: bool = True) -> dict:
    """
    Fit calibrator with auto-inversion for negative correlations.
    
    Args:
        y_true: Ground truth values [0, 1]
        y_pred: Raw predictions [0, 1]
        method: 'isotonic' or 'linear'
        auto_invert: If True, invert predictions when r < -0.05
    
    Returns:
        Calibration metadata dict
    """
    # Measure correlation
    r = np.corrcoef(y_true, y_pred)[0, 1]
    invert = auto_invert and (r < -0.05)
    
    x = -y_pred if invert else y_pred
    cal = {"method": method, "invert": bool(invert), "pearson_r": float(r)}
    
    if method == "isotonic":
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        ir.fit(x, y_true)
        cal["kind"] = "isotonic"
        # Serialize PAV (Pooled Adjacent Violators) points
        cal["X_thresholds"] = ir.X_thresholds_.tolist()
        cal["y_thresholds"] = ir.y_thresholds_.tolist()
    else:  # linear fallback
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
        cal["kind"] = "linear"
        cal["a"] = float(a)
        cal["b"] = float(b)
    
    return cal


def apply_calibrator(y_pred, calib):
    """
    Apply calibration to raw predictions (with optional inversion).
    
    Supports:
    - isotonic: PAV interpolation
    - linear: a * pred + b
    
    Args:
        y_pred: Raw predictions [0, 1]
        calib: Calibration metadata from fit_calibrator()
    
    Returns:
        Calibrated predictions [0, 1]
    """
    method = calib.get("type", calib.get("method", "isotonic"))
    
    # Handle nested structure (calibrator key)
    calibrator_data = calib.get("calibrator", calib)
    
    # Check for inversion
    invert = calibrator_data.get("invert", False)
    x = -y_pred if invert else y_pred
    
    if method == "isotonic":
        # Apply isotonic via interpolation on PAV points
        X_thresh = np.array(calibrator_data["X_thresholds"])
        y_thresh = np.array(calibrator_data["y_thresholds"])
        
        y_cal = np.interp(x, X_thresh, y_thresh)
    elif method == "linear":
        # Linear: a * x + b
        if "a" in calibrator_data:
            a = calibrator_data["a"]
            b = calibrator_data["b"]
        else:
            params = calibrator_data.get("params", calibrator_data)
            a = params.get("scale", params.get("a", 1.0))
            b = params.get("offset", params.get("b", 0.0))
        
        y_cal = a * x + b
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return np.clip(y_cal, 0.0, 1.0)


def compute_mae_by_range(pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute MAE for Low (<0.2), Medium (0.2-0.5), High (≥0.5) ranges."""
    low_mask = y_true < 0.2
    med_mask = (y_true >= 0.2) & (y_true < 0.5)
    high_mask = y_true >= 0.5
    
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


def batch_predict(model, X, batch_size=10000):
    """Predict in batches to avoid memory issues."""
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    predictions = []
    
    for i in tqdm(range(n_batches), desc="Predicting", leave=False):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        
        batch_pred = predict_normalized(model, X[start:end])
        predictions.append(batch_pred)
    
    return np.concatenate(predictions)


def calibrate_head(head: str, model, X_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic") -> dict:
    """
    Calibrate single head with isotonic regression and auto-inversion.
    
    Args:
        head: Head name (bw, emg, pli)
        model: Trained TMU model
        X_val: Validation features
        y_val: Validation ground truth
        method: Calibration method (isotonic or linear)
    
    Returns:
        Calibration metadata dict
    """
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


def infer_head(head, models_dir, calibrators_dir, X, y_true, batch_size=10000):
    """
    Inference pipeline for one head.
    
    Returns:
        pred_raw: Raw model predictions
        pred_cal: Calibrated predictions
        metrics: Dict with MAE, Pearson r, etc.
    """
    print(f"\n{'='*80}")
    print(f"🔮 INFERRING {head.upper()} HEAD")
    print(f"{'='*80}")
    
    # Load model
    model_path = Path(models_dir) / f"rtm_intensity_{head}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path)
    
    # Raw predictions (using predict_normalized)
    print(f"\n📊 Generating raw predictions...")
    pred_raw = batch_predict(model, X, batch_size=batch_size)
    
    print(f"   Raw predictions: min={pred_raw.min():.4f}, max={pred_raw.max():.4f}, mean={pred_raw.mean():.4f}")
    
    # Load calibrator
    calib_path = Path(calibrators_dir) / f"{head}_calibration.json"
    if not calib_path.exists():
        print(f"   ⚠️  No calibrator found at {calib_path}, using raw predictions")
        pred_cal = pred_raw
    else:
        calib = load_calibrator(calib_path)
        
        # Apply calibration
        print(f"\n🎯 Applying calibration...")
        pred_cal = apply_calibrator(pred_raw, calib)
        
        if calib.get("calibrator", {}).get("invert", False) or calib.get("invert", False):
            print(f"   ✅ Auto-inversion applied (negative correlation detected)")
        
        print(f"   Calibrated predictions: min={pred_cal.min():.4f}, max={pred_cal.max():.4f}, mean={pred_cal.mean():.4f}")
    
    # Compute metrics
    mae_raw = mean_absolute_error(y_true, pred_raw)
    mae_cal = mean_absolute_error(y_true, pred_cal)
    
    r_raw = np.corrcoef(y_true, pred_raw)[0, 1]
    r_cal = np.corrcoef(y_true, pred_cal)[0, 1]
    
    print(f"\n📊 Metrics:")
    print(f"   MAE (raw): {mae_raw:.4f}")
    print(f"   MAE (cal): {mae_cal:.4f}  (Δ {mae_cal - mae_raw:+.4f})")
    print(f"   Pearson r (raw): {r_raw:.4f}")
    print(f"   Pearson r (cal): {r_cal:.4f}  (Δ {r_cal - r_raw:+.4f})")
    
    # Stratified metrics (using unified compute_mae_by_range)
    stratified = compute_mae_by_range(pred_cal, y_true)
    
    print(f"\n📊 Stratified MAE (calibrated):")
    for name, stats in stratified.items():
        if name != "overall":
            pct = stats.get("pct", 100 * stats["n"] / len(y_true))
            print(f"   {name.capitalize():8s}: MAE={stats['mae']:.4f}, N={stats['n']:6d} ({pct:5.1f}%)")
    
    # Remove redundant 'overall' from stratified for cleaner output
    stratified_clean = {k: v for k, v in stratified.items() if k != "overall"}
    
    metrics = {
        "mae_raw": float(mae_raw),
        "mae_cal": float(mae_cal),
        "pearson_raw": float(r_raw),
        "pearson_cal": float(r_cal),
        "stratified": stratified_clean
    }
    
    return pred_raw, pred_cal, metrics


def run_calibration(args):
    """Run calibration mode: fit calibrators on validation set."""
    print("="*80)
    print("🔧 CALIBRATION MODE")
    print("="*80)
    print(f"Features:    {args.features}")
    print(f"Models:      {args.models}")
    print(f"Output:      {args.output}")
    print(f"Heads:       {args.heads}")
    print(f"Method:      {args.method}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation data
    print(f"\n📥 Loading validation data...")
    with h5py.File(args.features, "r") as f:
        X_val = binarize_uint8_to_i32(f["validation_X"][:])
        y_bw_val = f["validation_y_bw"][:]
        y_emg_val = f["validation_y_emg"][:]
        y_pli_val = f["validation_y_pli"][:]
    
    print(f"   Loaded {len(X_val)} validation samples")
    print(f"   Features shape: {X_val.shape}, dtype: {X_val.dtype}")
    
    # Map heads to ground truth
    y_val_map = {"bw": y_bw_val, "emg": y_emg_val, "pli": y_pli_val}
    
    # Calibrate each head
    results = {}
    for head in args.heads:
        # Load model
        model_path = Path(args.models) / f"rtm_intensity_{head}.pkl"
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {model_path}, skipping {head}")
            continue
        
        print(f"\n📦 Loading {head.upper()} model from {model_path}")
        model = load_model(model_path)
        
        # Get corresponding y_val
        y_val = y_val_map[head]
        
        # Calibrate
        calib_meta = calibrate_head(head, model, X_val, y_val, method=args.method)
        
        # Save calibration
        calib_path = output_dir / f"{head}_calibration.json"
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


def run_inference(args):
    """Run inference mode: apply calibrators on test/validation/train set."""
    print("="*80)
    print("🔮 INFERENCE MODE")
    print("="*80)
    print(f"Features:    {args.features}")
    print(f"Models:      {args.models}")
    print(f"Calibrators: {args.calibrators}")
    print(f"Output:      {args.output}")
    print(f"Heads:       {args.heads}")
    print(f"Split:       {args.split}")
    
    # Load dataset
    print(f"\n📥 Loading {args.split} data...")
    with h5py.File(args.features, "r") as f:
        # Check keys (V2 uses 'test_X', V1 uses different naming)
        if f"{args.split}_X" in f:
            X = binarize_uint8_to_i32(f[f"{args.split}_X"][:])
            y_bw = f[f"{args.split}_y_bw"][:]
            y_emg = f[f"{args.split}_y_emg"][:]
            y_pli = f[f"{args.split}_y_pli"][:]
        elif args.split == "validation" and "validation_X" in f:
            X = binarize_uint8_to_i32(f["validation_X"][:])
            y_bw = f["validation_y_bw"][:]
            y_emg = f["validation_y_emg"][:]
            y_pli = f["validation_y_pli"][:]
        else:
            raise KeyError(f"Split '{args.split}' not found in dataset. Available keys: {list(f.keys())}")
    
    print(f"   Loaded {len(X)} samples")
    print(f"   Features shape: {X.shape}, dtype: {X.dtype}")
    
    # Map heads to ground truth
    y_map = {"bw": y_bw, "emg": y_emg, "pli": y_pli}
    
    # Infer each head
    all_predictions = {}
    all_metrics = {}
    
    for head in args.heads:
        if head not in y_map:
            print(f"\n⚠️  Skipping unknown head: {head}")
            continue
        
        try:
            y_true = y_map[head]
            pred_raw, pred_cal, metrics = infer_head(
                head, args.models, args.calibrators, X, y_true, 
                batch_size=args.batch_size
            )
            
            all_predictions[head] = {
                "raw": pred_raw,
                "calibrated": pred_cal,
                "ground_truth": y_true
            }
            all_metrics[head] = metrics
            
        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping {head}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error inferring {head}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save predictions
    print(f"\n💾 Saving predictions to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        # Save predictions per head
        for head, preds in all_predictions.items():
            f.create_dataset(f"{head}_pred_raw", data=preds["raw"], compression="gzip")
            f.create_dataset(f"{head}_pred_calibrated", data=preds["calibrated"], compression="gzip")
            f.create_dataset(f"{head}_ground_truth", data=preds["ground_truth"], compression="gzip")
        
        # Save features for reference
        f.create_dataset("features", data=X, compression="gzip")
        
        # Save metadata as attributes
        f.attrs["split"] = args.split
        f.attrs["n_samples"] = len(X)
        f.attrs["heads"] = ",".join(args.heads)
    
    print(f"   ✅ Predictions saved")
    
    # Save metrics JSON
    metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"   ✅ Metrics saved to {metrics_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 INFERENCE SUMMARY")
    print(f"{'='*80}")
    
    for head, metrics in all_metrics.items():
        print(f"\n{head.upper()}:")
        print(f"   MAE:      {metrics['mae_raw']:.4f} → {metrics['mae_cal']:.4f} (Δ {metrics['mae_cal'] - metrics['mae_raw']:+.4f})")
        print(f"   Pearson:  {metrics['pearson_raw']:.4f} → {metrics['pearson_cal']:.4f} (Δ {metrics['pearson_cal'] - metrics['pearson_raw']:+.4f})")
        
        if 'stratified' in metrics:
            print(f"   Stratified MAE (calibrated):")
            for bin_name, bin_stats in metrics['stratified'].items():
                print(f"      {bin_name.capitalize():8s}: {bin_stats['mae']:.4f}")
    
    print(f"\n✅ INFERENCE COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="TMU Unified Calibration + Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibration mode (fit on validation set)
  python src/explain/infer_tmu_calibrated.py --calibrate \\
    --features data/explain_features_dataset_v2.h5 \\
    --models models/tmu_v2_bw_retrain \\
    --output models/tmu_v2_bw_retrain_calibrated \\
    --heads bw \\
    --method isotonic

  # Inference mode (apply on test set)
  python src/explain/infer_tmu_calibrated.py \\
    --features data/explain_features_dataset_v2.h5 \\
    --models models/tmu_v2_bw_retrain \\
    --calibrators models/tmu_v2_bw_retrain_calibrated \\
    --output results/predictions_test.h5 \\
    --heads bw \\
    --split test
        """
    )
    
    # Mode selection
    parser.add_argument("--calibrate", action="store_true",
                        help="Run in calibration mode (fit calibrators on validation set)")
    
    # Common arguments
    parser.add_argument("--features", required=True, 
                        help="Path to features dataset (H5)")
    parser.add_argument("--models", required=True, 
                        help="Directory containing trained models")
    parser.add_argument("--output", required=True,
                        help="Output path (directory for calibration, H5 file for inference)")
    parser.add_argument("--heads", nargs="+", default=["bw", "emg", "pli"],
                        help="Heads to process (default: bw emg pli)")
    
    # Calibration-specific arguments
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "linear"],
                        help="Calibration method (default: isotonic, only used with --calibrate)")
    
    # Inference-specific arguments
    parser.add_argument("--calibrators", 
                        help="Directory containing calibration JSON files (required for inference mode)")
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"],
                        help="Dataset split to use (default: test, only used for inference)")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Batch size for inference (default: 10000)")
    
    args = parser.parse_args()
    
    # Dispatch to appropriate mode
    if args.calibrate:
        run_calibration(args)
    else:
        # Inference mode requires calibrators
        if not args.calibrators:
            parser.error("--calibrators is required for inference mode (or use --calibrate for calibration mode)")
        run_inference(args)


if __name__ == "__main__":
    main()
