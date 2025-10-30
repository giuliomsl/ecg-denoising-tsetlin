#!/usr/bin/env python3
"""
Test V7 Optimized Model
========================

Test the optimized V7 model on held-out test set and compare with baseline.

Author: Giulio
Date: 2025-10-30
"""

import numpy as np
import h5py
import pickle
from pathlib import Path
from scipy.stats import pearsonr
import json

def load_model(model_dir):
    """Load TMU model and calibrator."""
    model_dir = Path(model_dir)
    
    with open(model_dir / 'tmu_bw.pkl', 'rb') as f:
        tmu_bw = pickle.load(f)
    
    with open(model_dir / 'tmu_emg.pkl', 'rb') as f:
        tmu_emg = pickle.load(f)
    
    with open(model_dir / 'isotonic_bw.pkl', 'rb') as f:
        iso_bw = pickle.load(f)
    
    with open(model_dir / 'isotonic_emg.pkl', 'rb') as f:
        iso_emg = pickle.load(f)
    
    return tmu_bw, tmu_emg, iso_bw, iso_emg

def load_test_data(dataset_path):
    """Load test data."""
    with h5py.File(dataset_path, 'r') as f:
        X_test = f['test_X'][:]
        y_test_bw = f['test_y_bw'][:]
        y_test_emg = f['test_y_emg'][:]
    
    return X_test, y_test_bw, y_test_emg

def evaluate(y_true, y_pred, task_name):
    """Evaluate predictions."""
    r, _ = pearsonr(y_pred, y_true)
    mae = np.mean(np.abs(y_pred - y_true))
    
    print(f"\n{task_name}:")
    print(f"  r = {r:.4f}")
    print(f"  MAE = {mae:.4f}")
    
    return {'r': float(r), 'mae': float(mae)}

def main():
    print("="*80)
    print("V7 OPTIMIZED MODEL TEST")
    print("="*80)
    
    # Load model
    print("\nLoading V7 optimized model...")
    model_dir = Path('models/tmu_v7_optimized')
    tmu_bw, tmu_emg, iso_bw, iso_emg = load_model(model_dir)
    print("✅ Models loaded")
    
    # Load test data
    print("\nLoading test data...")
    dataset_path = 'data/explain_features_dataset_v7_th0.0005.h5'
    X_test, y_test_bw, y_test_emg = load_test_data(dataset_path)
    print(f"✅ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # BW predictions
    print("\n" + "="*80)
    print("BW (Baseline Wander)")
    print("="*80)
    
    print("\nPredicting...")
    y_test_pred_bw_raw = tmu_bw.predict(X_test)
    y_test_pred_bw_cal = iso_bw.transform(y_test_pred_bw_raw)
    
    print("\nRaw predictions:")
    results_bw_raw = evaluate(y_test_bw, y_test_pred_bw_raw, "BW (raw)")
    
    print("\nCalibrated predictions:")
    results_bw_cal = evaluate(y_test_bw, y_test_pred_bw_cal, "BW (calibrated)")
    
    # EMG predictions
    print("\n" + "="*80)
    print("EMG (Muscle Artifacts)")
    print("="*80)
    
    print("\nPredicting...")
    y_test_pred_emg_raw = tmu_emg.predict(X_test)
    y_test_pred_emg_cal = iso_emg.transform(y_test_pred_emg_raw)
    
    print("\nRaw predictions:")
    results_emg_raw = evaluate(y_test_emg, y_test_pred_emg_raw, "EMG (raw)")
    
    print("\nCalibrated predictions:")
    results_emg_cal = evaluate(y_test_emg, y_test_pred_emg_cal, "EMG (calibrated)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: V7 OPTIMIZED TEST RESULTS")
    print("="*80)
    
    print(f"\nBW:  r = {results_bw_cal['r']:.4f}, MAE = {results_bw_cal['mae']:.4f}")
    print(f"EMG: r = {results_emg_cal['r']:.4f}, MAE = {results_emg_cal['mae']:.4f}")
    print(f"Avg: r = {(results_bw_cal['r'] + results_emg_cal['r'])/2:.4f}")
    
    # Save results
    output_dir = Path('results/v7_optimized_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    np.savez(output_dir / 'predictions_bw.npz',
             y_test=y_test_bw,
             y_test_pred_raw=y_test_pred_bw_raw,
             y_test_pred_cal=y_test_pred_bw_cal)
    
    np.savez(output_dir / 'predictions_emg.npz',
             y_test=y_test_emg,
             y_test_pred_raw=y_test_pred_emg_raw,
             y_test_pred_cal=y_test_pred_emg_cal)
    
    # Save metrics
    results = {
        'bw': {
            'raw': results_bw_raw,
            'calibrated': results_bw_cal
        },
        'emg': {
            'raw': results_emg_raw,
            'calibrated': results_emg_cal
        },
        'avg_r': float((results_bw_cal['r'] + results_emg_cal['r'])/2)
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/")
    
    # Comparison with baselines
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)
    
    # V4 baseline
    print("\nV4 (1092 features, 10k clauses, T=700, s=3.5):")
    print("  BW:  r = 0.6231, MAE = 0.2171")
    print("  EMG: r = 0.6797, MAE = 0.2447")
    print("  Avg: r = 0.6514")
    
    # V7 baseline
    print("\nV7 baseline (57 features, 10k clauses, T=700, s=3.5):")
    print("  BW:  r = 0.5906, MAE = 0.2360")
    print("  EMG: r = 0.6731, MAE = 0.2469")
    print("  Avg: r = 0.6319")
    
    # V7 optimized (current)
    print("\nV7 optimized (57 features, 3k clauses, T=700, s=7.0):")
    print(f"  BW:  r = {results_bw_cal['r']:.4f}, MAE = {results_bw_cal['mae']:.4f}")
    print(f"  EMG: r = {results_emg_cal['r']:.4f}, MAE = {results_emg_cal['mae']:.4f}")
    print(f"  Avg: r = {results['avg_r']:.4f}")
    
    # Deltas
    print("\nΔ vs V4:")
    delta_bw_v4 = results_bw_cal['r'] - 0.6231
    delta_emg_v4 = results_emg_cal['r'] - 0.6797
    delta_avg_v4 = results['avg_r'] - 0.6514
    print(f"  BW:  Δr = {delta_bw_v4:+.4f} ({100*delta_bw_v4/0.6231:+.1f}%)")
    print(f"  EMG: Δr = {delta_emg_v4:+.4f} ({100*delta_emg_v4/0.6797:+.1f}%)")
    print(f"  Avg: Δr = {delta_avg_v4:+.4f} ({100*delta_avg_v4/0.6514:+.1f}%)")
    
    print("\nΔ vs V7 baseline:")
    delta_bw_v7 = results_bw_cal['r'] - 0.5906
    delta_emg_v7 = results_emg_cal['r'] - 0.6731
    delta_avg_v7 = results['avg_r'] - 0.6319
    print(f"  BW:  Δr = {delta_bw_v7:+.4f} ({100*delta_bw_v7/0.5906:+.1f}%)")
    print(f"  EMG: Δr = {delta_emg_v7:+.4f} ({100*delta_emg_v7/0.6731:+.1f}%)")
    print(f"  Avg: Δr = {delta_avg_v7:+.4f} ({100*delta_avg_v7/0.6319:+.1f}%)")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
