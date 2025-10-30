#!/usr/bin/env python3
"""
Train TMU V7: Selected Features (45 bitplanes + 12 HF = 57 features)
====================================================================

Same hyperparameters as V4, but with 95% fewer features (1092 → 57).
Expected: Similar performance, ~10x faster training.

IMPROVEMENTS:
- ✅ Reproducible seeds (np, random, PYTHONHASHSEED)
- ✅ Early stopping on r_val (patience=3)
- ✅ Metadata completi (selection strategy, indices)
- ✅ Calibration leak-free (fit on val, apply on test)
"""

import argparse
import h5py
import numpy as np
import json
import random
import os
from pathlib import Path
from tmu.models.regression.vanilla_regressor import TMRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr
import time


def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🌱 Seeds set to {seed} (numpy, random, PYTHONHASHSEED)")

def load_dataset(path):
    """Load V7 dataset with full metadata"""
    with h5py.File(path, 'r') as f:
        train_X = f['train_X'][:]
        train_y_bw = f['train_y_bw'][:]
        train_y_emg = f['train_y_emg'][:]
        val_X = f['validation_X'][:]
        val_y_bw = f['validation_y_bw'][:]
        val_y_emg = f['validation_y_emg'][:]
        test_X = f['test_X'][:]
        test_y_bw = f['test_y_bw'][:]
        test_y_emg = f['test_y_emg'][:]
        
        # Get metadata
        n_features = f.attrs.get('n_features', train_X.shape[1])
        n_bitplanes = f.attrs.get('n_bitplanes_selected', 0)
        strategy = f.attrs.get('selection_strategy', 'unknown')
        threshold = f.attrs.get('threshold', None)
        
        # Try to load selected indices if available
        selected_indices = None
        if 'selected_bitplane_indices' in f:
            selected_indices = f['selected_bitplane_indices'][:]
    
    return {
        'train': (train_X, train_y_bw, train_y_emg),
        'val': (val_X, val_y_bw, val_y_emg),
        'test': (test_X, test_y_bw, test_y_emg),
        'meta': {
            'n_features': n_features,
            'n_bitplanes': n_bitplanes,
            'strategy': strategy,
            'threshold': threshold,
            'selected_indices': selected_indices
        }
    }


def train_model(X_train, y_train, X_val, y_val, noise_type, output_dir, 
                n_clauses=10000, T=700, s=3.5, n_epochs=10, patience=3):
    """Train TMU model with early stopping"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING TMU V7: {noise_type.upper()}")
    print(f"{'='*80}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Hyperparameters: clauses={n_clauses}, T={T}, s={s}")
    print(f"Early stopping: patience={patience}")
    
    # Initialize model (same as V4)
    tm = TMRegressor(
        number_of_clauses=n_clauses,
        T=T,
        s=s,
        platform='CPU',
        weighted_clauses=True
    )
    
    # Training with early stopping
    train_times = []
    val_metrics = []
    best_r_val = -np.inf
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train
        t_start = time.time()
        tm.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        train_times.append(t_elapsed)
        
        # Validation predictions (raw)
        y_val_pred = tm.predict(X_val)
        
        # Metrics
        r_val, _ = pearsonr(y_val, y_val_pred)
        mae_val = np.mean(np.abs(y_val - y_val_pred))
        
        val_metrics.append({
            'epoch': epoch + 1,
            'r_val': float(r_val),
            'mae_val': float(mae_val),
            'train_time': float(t_elapsed)
        })
        
        print(f"  Train time: {t_elapsed:.1f}s")
        print(f"  r_val (raw): {r_val:.4f}")
        print(f"  MAE: {mae_val:.4f}")
        
        # Early stopping check
        if r_val > best_r_val:
            best_r_val = r_val
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  ✅ NEW BEST (r_val={r_val:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch+1}")
                print(f"   Best r_val: {best_r_val:.4f} (epoch {best_epoch})")
                break
    
    # Final predictions for calibration (LEAK-FREE: use validation set)
    print("\nGenerating final predictions for calibration...")
    y_train_pred = tm.predict(X_train)
    y_val_pred = tm.predict(X_val)
    
    # Calibration (fit on VALIDATION only - leak-free!)
    print("Training calibration (IsotonicRegression on validation)...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_val_pred, y_val)
    
    # Calibrated metrics on validation
    y_val_cal = iso.transform(y_val_pred)
    r_val_cal, _ = pearsonr(y_val, y_val_cal)
    mae_val_cal = np.mean(np.abs(y_val - y_val_cal))
    
    # Compute slope/intercept for diagnostics
    from scipy.stats import linregress
    slope_raw, intercept_raw, _, _, _ = linregress(y_val_pred, y_val)
    slope_cal, intercept_cal, _, _, _ = linregress(y_val_cal, y_val)
    
    print(f"\nCalibrated validation:")
    print(f"  r_val: {r_val_cal:.4f} (raw: {best_r_val:.4f})")
    print(f"  MAE:   {mae_val_cal:.4f} (raw: {mae_val:.4f})")
    print(f"\nRegression diagnostics:")
    print(f"  Raw:  slope={slope_raw:.4f}, intercept={intercept_raw:.4f}")
    print(f"  Cal:  slope={slope_cal:.4f}, intercept={intercept_cal:.4f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"tmu_{noise_type}.pkl"
    iso_path = output_dir / f"isotonic_{noise_type}.pkl"
    
    print(f"\nSaving model...")
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(tm, f)
    
    with open(iso_path, 'wb') as f:
        pickle.dump(iso, f)
    
    print(f"  ✓ {model_path}")
    print(f"  ✓ {iso_path}")
    
    # Save training metrics
    metrics_path = output_dir / f"training_metrics_{noise_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'epochs': val_metrics,
            'best_epoch': best_epoch,
            'best_r_val_raw': float(best_r_val),
            'final_r_val_cal': float(r_val_cal),
            'final_mae_cal': float(mae_val_cal),
            'total_train_time': sum(train_times),
            'avg_epoch_time': np.mean(train_times),
            'early_stopped': patience_counter >= patience,
            'calibration_diagnostics': {
                'slope_raw': float(slope_raw),
                'intercept_raw': float(intercept_raw),
                'slope_cal': float(slope_cal),
                'intercept_cal': float(intercept_cal)
            }
        }, f, indent=2)
    
    print(f"  ✓ {metrics_path}")
    
    return tm, iso, val_metrics


def main():
    parser = argparse.ArgumentParser(description="Train TMU V7")
    parser.add_argument("--features", type=str, 
                        default="data/explain_features_dataset_v7_th0.0005.h5",
                        help="V7 dataset path")
    parser.add_argument("--output", type=str, 
                        default="models/tmu_v7_selected",
                        help="Output directory")
    parser.add_argument("--clauses", type=int, default=10000, help="Number of clauses")
    parser.add_argument("--T", type=int, default=700, help="T parameter")
    parser.add_argument("--s", type=float, default=3.5, help="s parameter")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TMU V7 TRAINING: SELECTED FEATURES")
    print("="*80)
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Load data
    print(f"\nLoading dataset: {args.features}")
    data = load_dataset(args.features)
    
    print(f"\nDataset info:")
    print(f"  Features: {data['meta']['n_features']} ({data['meta']['n_bitplanes']} bitplanes + 12 HF)")
    print(f"  Strategy: {data['meta']['strategy']}")
    if data['meta']['threshold'] is not None:
        print(f"  Threshold: {data['meta']['threshold']}")
    print(f"  Train: {data['train'][0].shape[0]} samples")
    print(f"  Val:   {data['val'][0].shape[0]} samples")
    print(f"  Test:  {data['test'][0].shape[0]} samples")
    
    output_dir = Path(args.output)
    
    # Train BW model
    print("\n" + "="*80)
    print("TRAINING BW MODEL")
    print("="*80)
    
    tm_bw, iso_bw, metrics_bw = train_model(
        data['train'][0], data['train'][1],
        data['val'][0], data['val'][1],
        'bw', output_dir,
        n_clauses=args.clauses, T=args.T, s=args.s, 
        n_epochs=args.epochs, patience=args.patience
    )
    
    # Train EMG model
    print("\n" + "="*80)
    print("TRAINING EMG MODEL")
    print("="*80)
    
    tm_emg, iso_emg, metrics_emg = train_model(
        data['train'][0], data['train'][2],
        data['val'][0], data['val'][2],
        'emg', output_dir,
        n_clauses=args.clauses, T=args.T, s=args.s, 
        n_epochs=args.epochs, patience=args.patience
    )
    
    # Save config with full metadata
    config = {
        'version': 'V7',
        'dataset': args.features,
        'n_features': int(data['meta']['n_features']),
        'n_bitplanes': int(data['meta']['n_bitplanes']),
        'selection_strategy': data['meta']['strategy'],
        'selection_threshold': float(data['meta']['threshold']) if data['meta']['threshold'] else None,
        'hyperparameters': {
            'n_clauses': int(args.clauses),
            'T': int(args.T),
            's': float(args.s),
            'n_epochs': int(args.epochs),
            'patience': int(args.patience),
            'weighted_clauses': True
        },
        'seed': int(args.seed),
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'improvements': [
            'reproducible_seeds',
            'early_stopping',
            'leak_free_calibration',
            'regression_diagnostics'
        ]
    }
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {output_dir}")
    
    # Find best epochs
    best_bw = max(metrics_bw, key=lambda x: x['r_val'])
    best_emg = max(metrics_emg, key=lambda x: x['r_val'])
    
    print(f"\nBW:  r_val = {best_bw['r_val']:.4f} (epoch {best_bw['epoch']})")
    print(f"EMG: r_val = {best_emg['r_val']:.4f} (epoch {best_emg['epoch']})")
    print(f"\nTotal training time:")
    print(f"  BW:  {sum(m['train_time'] for m in metrics_bw):.1f}s")
    print(f"  EMG: {sum(m['train_time'] for m in metrics_emg):.1f}s")
    print("\nNext steps:")
    print(f"  1. Run inference:")
    print(f"     python inference_v7.py --models {args.output}")
    print(f"  2. Evaluate:")
    print(f"     python evaluate_v7.py --models {args.output}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
