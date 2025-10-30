#!/usr/bin/env python3
"""
Inference V7: Generate predictions on test set
==============================================
"""

import argparse
import h5py
import numpy as np
import pickle
from pathlib import Path
from tmu.models.regression.vanilla_regressor import TMRegressor
from scipy.stats import pearsonr
import json

def load_models(models_dir):
    """Load TMU models and calibrators"""
    models_dir = Path(models_dir)
    
    print("Loading models...")
    
    # BW
    with open(models_dir / "tmu_bw.pkl", 'rb') as f:
        tm_bw = pickle.load(f)
    
    with open(models_dir / "isotonic_bw.pkl", 'rb') as f:
        iso_bw = pickle.load(f)
    
    # EMG
    with open(models_dir / "tmu_emg.pkl", 'rb') as f:
        tm_emg = pickle.load(f)
    
    with open(models_dir / "isotonic_emg.pkl", 'rb') as f:
        iso_emg = pickle.load(f)
    
    print("  ✓ Models loaded")
    
    return (tm_bw, iso_bw), (tm_emg, iso_emg)


def run_inference(tm, iso, X_test, y_test, noise_type):
    """Run inference and compute metrics"""
    
    print(f"\n{noise_type.upper()} inference:")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Raw predictions
    y_pred_raw = tm.predict(X_test)
    
    # Calibrated predictions
    y_pred_cal = iso.transform(y_pred_raw)
    
    # Metrics
    r_raw, _ = pearsonr(y_test, y_pred_raw)
    r_cal, _ = pearsonr(y_test, y_pred_cal)
    mae_raw = np.mean(np.abs(y_test - y_pred_raw))
    mae_cal = np.mean(np.abs(y_test - y_pred_cal))
    
    print(f"  Raw:        r={r_raw:.4f}, MAE={mae_raw:.4f}")
    print(f"  Calibrated: r={r_cal:.4f}, MAE={mae_cal:.4f}")
    
    return {
        'y_pred_raw': y_pred_raw,
        'y_pred_cal': y_pred_cal,
        'metrics': {
            'r_raw': float(r_raw),
            'r_cal': float(r_cal),
            'mae_raw': float(mae_raw),
            'mae_cal': float(mae_cal)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="V7 Inference")
    parser.add_argument("--models", type=str, 
                        default="models/tmu_v7_selected",
                        help="Models directory")
    parser.add_argument("--features", type=str,
                        default="data/explain_features_dataset_v7_th0.0005.h5",
                        help="V7 dataset")
    parser.add_argument("--output", type=str,
                        default="results/v7_predictions",
                        help="Output directory")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("V7 INFERENCE")
    print("="*80)
    
    # Load models
    (tm_bw, iso_bw), (tm_emg, iso_emg) = load_models(args.models)
    
    # Load test data
    print(f"\nLoading test data: {args.features}")
    with h5py.File(args.features, 'r') as f:
        test_X = f['test_X'][:]
        test_y_bw = f['test_y_bw'][:]
        test_y_emg = f['test_y_emg'][:]
    
    print(f"  Test: {test_X.shape[0]} samples, {test_X.shape[1]} features")
    
    # Run inference
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    
    results_bw = run_inference(tm_bw, iso_bw, test_X, test_y_bw, 'bw')
    results_emg = run_inference(tm_emg, iso_emg, test_X, test_y_emg, 'emg')
    
    # Save predictions
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING PREDICTIONS")
    print("="*80)
    
    # Save as NPZ
    npz_path = output_dir / "v7_test_predictions.npz"
    np.savez_compressed(
        npz_path,
        y_true_bw=test_y_bw,
        y_pred_bw_raw=results_bw['y_pred_raw'],
        y_pred_bw_cal=results_bw['y_pred_cal'],
        y_true_emg=test_y_emg,
        y_pred_emg_raw=results_emg['y_pred_raw'],
        y_pred_emg_cal=results_emg['y_pred_cal']
    )
    print(f"  ✓ {npz_path}")
    
    # Save metrics
    metrics = {
        'bw': results_bw['metrics'],
        'emg': results_emg['metrics']
    }
    
    metrics_path = output_dir / "v7_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {metrics_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ INFERENCE COMPLETE")
    print("="*80)
    print(f"\nTest Performance (Calibrated):")
    print(f"  BW:  r={results_bw['metrics']['r_cal']:.4f}, MAE={results_bw['metrics']['mae_cal']:.4f}")
    print(f"  EMG: r={results_emg['metrics']['r_cal']:.4f}, MAE={results_emg['metrics']['mae_cal']:.4f}")
    print(f"\nPredictions saved to: {output_dir}")
    print("\nNext: python evaluate_v7.py --predictions " + str(npz_path))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
