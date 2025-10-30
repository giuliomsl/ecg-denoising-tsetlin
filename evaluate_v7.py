#!/usr/bin/env python3
"""
Evaluate V7: Compare with V4 and V6
===================================

Compare performance, efficiency, and interpretability.
"""

import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_predictions(path):
    """Load predictions from NPZ"""
    data = np.load(path)
    return {
        'bw': {
            'y_true': data['y_true_bw'],
            'y_pred_raw': data['y_pred_bw_raw'],
            'y_pred_cal': data['y_pred_bw_cal']
        },
        'emg': {
            'y_true': data['y_true_emg'],
            'y_pred_raw': data['y_pred_emg_raw'],
            'y_pred_cal': data['y_pred_emg_cal']
        }
    }


def compute_metrics(y_true, y_pred):
    """Compute regression metrics"""
    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return {'r': float(r), 'mae': float(mae), 'rmse': float(rmse)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate V7")
    parser.add_argument("--v7-predictions", type=str,
                        default="results/v7_predictions/v7_test_predictions.npz")
    parser.add_argument("--v4-predictions", type=str,
                        default="results/v4_test_predictions.npz",
                        help="V4 predictions for comparison (optional)")
    parser.add_argument("--output", type=str,
                        default="results/v7_evaluation")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("V7 EVALUATION")
    print("="*80)
    
    # Load V7 predictions
    print(f"\nLoading V7 predictions: {args.v7_predictions}")
    v7_data = load_predictions(args.v7_predictions)
    
    # Compute V7 metrics
    v7_metrics = {
        'bw': {
            'raw': compute_metrics(v7_data['bw']['y_true'], v7_data['bw']['y_pred_raw']),
            'cal': compute_metrics(v7_data['bw']['y_true'], v7_data['bw']['y_pred_cal'])
        },
        'emg': {
            'raw': compute_metrics(v7_data['emg']['y_true'], v7_data['emg']['y_pred_raw']),
            'cal': compute_metrics(v7_data['emg']['y_true'], v7_data['emg']['y_pred_cal'])
        }
    }
    
    print("\n" + "="*80)
    print("V7 TEST PERFORMANCE")
    print("="*80)
    
    print(f"\nBW Noise:")
    print(f"  Raw:        r={v7_metrics['bw']['raw']['r']:.4f}, MAE={v7_metrics['bw']['raw']['mae']:.4f}")
    print(f"  Calibrated: r={v7_metrics['bw']['cal']['r']:.4f}, MAE={v7_metrics['bw']['cal']['mae']:.4f}")
    
    print(f"\nEMG Noise:")
    print(f"  Raw:        r={v7_metrics['emg']['raw']['r']:.4f}, MAE={v7_metrics['emg']['raw']['mae']:.4f}")
    print(f"  Calibrated: r={v7_metrics['emg']['cal']['r']:.4f}, MAE={v7_metrics['emg']['cal']['mae']:.4f}")
    
    # Compare with V4 if available
    if Path(args.v4_predictions).exists():
        print("\n" + "="*80)
        print("COMPARISON: V4 vs V7")
        print("="*80)
        
        print(f"\nLoading V4 predictions: {args.v4_predictions}")
        v4_data = load_predictions(args.v4_predictions)
        
        v4_metrics = {
            'bw': {
                'cal': compute_metrics(v4_data['bw']['y_true'], v4_data['bw']['y_pred_cal'])
            },
            'emg': {
                'cal': compute_metrics(v4_data['emg']['y_true'], v4_data['emg']['y_pred_cal'])
            }
        }
        
        print(f"\n{'Noise Type':<12} {'Metric':<8} {'V4 (1092)':<12} {'V7 (57)':<12} {'Delta':<12}")
        print("-"*80)
        
        # BW comparison
        bw_r_delta = v7_metrics['bw']['cal']['r'] - v4_metrics['bw']['cal']['r']
        bw_mae_delta = v7_metrics['bw']['cal']['mae'] - v4_metrics['bw']['cal']['mae']
        
        print(f"{'BW':<12} {'r':<8} {v4_metrics['bw']['cal']['r']:<12.4f} {v7_metrics['bw']['cal']['r']:<12.4f} {bw_r_delta:<+12.4f}")
        print(f"{'BW':<12} {'MAE':<8} {v4_metrics['bw']['cal']['mae']:<12.4f} {v7_metrics['bw']['cal']['mae']:<12.4f} {bw_mae_delta:<+12.4f}")
        
        # EMG comparison
        emg_r_delta = v7_metrics['emg']['cal']['r'] - v4_metrics['emg']['cal']['r']
        emg_mae_delta = v7_metrics['emg']['cal']['mae'] - v4_metrics['emg']['cal']['mae']
        
        print(f"{'EMG':<12} {'r':<8} {v4_metrics['emg']['cal']['r']:<12.4f} {v7_metrics['emg']['cal']['r']:<12.4f} {emg_r_delta:<+12.4f}")
        print(f"{'EMG':<12} {'MAE':<8} {v4_metrics['emg']['cal']['mae']:<12.4f} {v7_metrics['emg']['cal']['mae']:<12.4f} {emg_mae_delta:<+12.4f}")
        
        avg_r_delta = (bw_r_delta + emg_r_delta) / 2
        avg_r_v4 = (v4_metrics['bw']['cal']['r'] + v4_metrics['emg']['cal']['r']) / 2
        avg_r_pct = (avg_r_delta / avg_r_v4) * 100
        
        print(f"\nAverage r delta: {avg_r_delta:+.4f} ({avg_r_pct:+.1f}%)")
        
        # Efficiency
        print("\n" + "-"*80)
        print("EFFICIENCY GAINS")
        print("-"*80)
        print(f"\nFeatures: 1092 → 57 (95% reduction)")
        print(f"Expected training speedup: ~10-15x")
        print(f"Expected memory reduction: ~95%")
        
        if abs(avg_r_pct) < 1:
            print(f"\n✅ EXCELLENT: V7 matches V4 ({avg_r_pct:+.1f}%) with 95% fewer features!")
        elif abs(avg_r_pct) < 2:
            print(f"\n✅ GOOD: V7 nearly matches V4 ({avg_r_pct:+.1f}%) with 95% fewer features")
        else:
            print(f"\n🤔 ACCEPTABLE: V7 {avg_r_pct:+.1f}% vs V4, but efficiency gain is significant")
    
    # Save evaluation
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_results = {
        'v7': v7_metrics,
        'summary': {
            'bw_r_test': v7_metrics['bw']['cal']['r'],
            'emg_r_test': v7_metrics['emg']['cal']['r'],
            'bw_mae_test': v7_metrics['bw']['cal']['mae'],
            'emg_mae_test': v7_metrics['emg']['cal']['mae']
        }
    }
    
    results_path = output_dir / "v7_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  ✓ {results_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
