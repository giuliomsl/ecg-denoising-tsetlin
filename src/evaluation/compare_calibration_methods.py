#!/usr/bin/env python3
"""
Compare Calibration Methods: Isotonic vs Linear Regression
==========================================================

Validates the necessity of IsotonicRegression for TMU calibration
by comparing it with simple linear regression baseline.

Methodology:
- Load V7 trained model predictions (uncalibrated)
- Apply both isotonic and linear calibration on validation set
- Compare performance on test set
- Report Δr and ΔMAE

Expected outcome: Isotonic should outperform linear, validating choice.

Author: Giulio
Date: 2025-10-30
"""

import numpy as np
import h5py
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import json

def load_predictions(predictions_dir):
    """Load raw predictions from V7 model."""
    predictions_dir = Path(predictions_dir)
    
    print(f"Loading predictions from {predictions_dir}...")
    
    data = {}
    for task in ['bw', 'emg']:
        task_upper = task.upper()
        
        # Load predictions
        pred_path = predictions_dir / f'predictions_{task}.npz'
        preds = np.load(pred_path)
        
        data[f'{task}_val_pred'] = preds['y_val_pred']
        data[f'{task}_val_true'] = preds['y_val']
        data[f'{task}_test_pred'] = preds['y_test_pred']
        data[f'{task}_test_true'] = preds['y_test']
    
    print("✅ Predictions loaded")
    return data

def evaluate_calibration(y_true, y_pred, method_name):
    """Evaluate calibration quality."""
    r, _ = pearsonr(y_pred, y_true)
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Compute residuals statistics
    residuals = y_pred - y_true
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    return {
        'method': method_name,
        'r': float(r),
        'mae': float(mae),
        'mean_residual': float(mean_residual),
        'std_residual': float(std_residual)
    }

def calibrate_isotonic(y_val_pred, y_val_true, y_test_pred):
    """Apply isotonic regression calibration."""
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_val_pred, y_val_true)
    y_test_cal = iso.transform(y_test_pred)
    return y_test_cal

def calibrate_linear(y_val_pred, y_val_true, y_test_pred):
    """Apply linear regression calibration."""
    lr = LinearRegression()
    lr.fit(y_val_pred.reshape(-1, 1), y_val_true)
    y_test_cal = lr.predict(y_test_pred.reshape(-1, 1))
    return y_test_cal

def compare_calibration_methods(predictions_dir, output_dir=None):
    """
    Compare isotonic vs linear calibration.
    
    Returns:
        results: Dictionary with comparison results
    """
    print("="*80)
    print("CALIBRATION METHODS COMPARISON")
    print("="*80)
    
    # Load predictions
    data = load_predictions(predictions_dir)
    
    results = {}
    
    for task in ['bw', 'emg']:
        print(f"\n{'='*80}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*80}")
        
        task_key = task
        y_val_pred = data[f'{task}_val_pred']
        y_val_true = data[f'{task}_val_true']
        y_test_pred = data[f'{task}_test_pred']
        y_test_true = data[f'{task}_test_true']
        
        # 1. Uncalibrated (baseline)
        print("\n1. UNCALIBRATED (Raw predictions)")
        uncal_metrics = evaluate_calibration(y_test_true, y_test_pred, 'uncalibrated')
        print(f"   r = {uncal_metrics['r']:.4f}")
        print(f"   MAE = {uncal_metrics['mae']:.4f}")
        print(f"   Mean residual = {uncal_metrics['mean_residual']:.4f}")
        
        # 2. Linear Regression Calibration
        print("\n2. LINEAR REGRESSION CALIBRATION")
        y_test_linear = calibrate_linear(y_val_pred, y_val_true, y_test_pred)
        linear_metrics = evaluate_calibration(y_test_true, y_test_linear, 'linear')
        print(f"   r = {linear_metrics['r']:.4f} (Δ = {linear_metrics['r'] - uncal_metrics['r']:+.4f})")
        print(f"   MAE = {linear_metrics['mae']:.4f} (Δ = {linear_metrics['mae'] - uncal_metrics['mae']:+.4f})")
        print(f"   Mean residual = {linear_metrics['mean_residual']:.4f}")
        
        # 3. Isotonic Regression Calibration
        print("\n3. ISOTONIC REGRESSION CALIBRATION")
        y_test_isotonic = calibrate_isotonic(y_val_pred, y_val_true, y_test_pred)
        isotonic_metrics = evaluate_calibration(y_test_true, y_test_isotonic, 'isotonic')
        print(f"   r = {isotonic_metrics['r']:.4f} (Δ = {isotonic_metrics['r'] - uncal_metrics['r']:+.4f})")
        print(f"   MAE = {isotonic_metrics['mae']:.4f} (Δ = {isotonic_metrics['mae'] - uncal_metrics['mae']:+.4f})")
        print(f"   Mean residual = {isotonic_metrics['mean_residual']:.4f}")
        
        # Comparison: Isotonic vs Linear
        print("\n" + "-"*80)
        print("ISOTONIC vs LINEAR:")
        delta_r = isotonic_metrics['r'] - linear_metrics['r']
        delta_mae = isotonic_metrics['mae'] - linear_metrics['mae']
        print(f"   Δr = {delta_r:+.4f}")
        print(f"   ΔMAE = {delta_mae:+.4f}")
        
        if delta_r > 0.001:
            print(f"   ✅ Isotonic BETTER (+{delta_r:.4f} correlation)")
        elif abs(delta_r) <= 0.001:
            print(f"   ≈ Isotonic EQUIVALENT (Δr ~ {delta_r:+.4f})")
        else:
            print(f"   ⚠️  Linear BETTER (+{-delta_r:.4f} correlation) - unexpected!")
        
        # Store results
        results[task] = {
            'uncalibrated': uncal_metrics,
            'linear': linear_metrics,
            'isotonic': isotonic_metrics,
            'delta_isotonic_vs_linear': {
                'delta_r': float(delta_r),
                'delta_mae': float(delta_mae)
            }
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for task in ['bw', 'emg']:
        task_results = results[task]
        delta_r = task_results['delta_isotonic_vs_linear']['delta_r']
        delta_mae = task_results['delta_isotonic_vs_linear']['delta_mae']
        
        print(f"\n{task.upper()}:")
        print(f"  Isotonic vs Linear: Δr = {delta_r:+.4f}, ΔMAE = {delta_mae:+.4f}")
        
        if delta_r > 0.001:
            print(f"  ✅ Isotonic justified")
        elif abs(delta_r) <= 0.001:
            print(f"  ≈ Methods equivalent (could use simpler linear)")
        else:
            print(f"  ⚠️ Linear better - investigate!")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'calibration_comparison.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare isotonic vs linear calibration')
    parser.add_argument('--predictions', type=str,
                       default='results/v7_predictions',
                       help='Directory with raw predictions')
    parser.add_argument('--output', type=str,
                       default='results/v7_calibration_comparison',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results = compare_calibration_methods(args.predictions, args.output)
    
    print("\n✅ Calibration comparison complete!")
    print("\nConclusion:")
    print("If isotonic Δr > +0.001, the added complexity is justified.")
    print("If Δr ≈ 0, could simplify to linear regression for faster inference.")
