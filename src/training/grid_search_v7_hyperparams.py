#!/usr/bin/env python3
"""
Grid Search V7 Hyperparameters
================================

Quick grid search to find optimal hyperparameters for reduced feature space (57 features).

HYPOTHESIS: With 57 features (vs 1092), we need fewer clauses and possibly different T/s.

Grid:
- clauses ∈ {3000, 6000, 10000}
- T ∈ {300, 500, 700}
- s ∈ {3.5, 5.0, 7.0}

Total: 27 configurations per head (BW, EMG) = 54 runs
Expected time: ~2-3 hours (each config ~2-4 min with early stopping)

USAGE:
------
python grid_search_v7_hyperparams.py \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output results/v7_grid_search \
  --seed 42 \
  --patience 2 \
  --max-epochs 10
"""

import argparse
import h5py
import numpy as np
import json
import pandas as pd
import random
import os
import time
from pathlib import Path
from tmu.models.regression.vanilla_regressor import TMRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr
from itertools import product


def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset(path):
    """Load V7 dataset."""
    with h5py.File(path, 'r') as f:
        train_X = f['train_X'][:]
        train_y_bw = f['train_y_bw'][:]
        train_y_emg = f['train_y_emg'][:]
        val_X = f['validation_X'][:]
        val_y_bw = f['validation_y_bw'][:]
        val_y_emg = f['validation_y_emg'][:]
    
    return {
        'train_X': train_X,
        'train_y_bw': train_y_bw,
        'train_y_emg': train_y_emg,
        'val_X': val_X,
        'val_y_bw': val_y_bw,
        'val_y_emg': val_y_emg
    }


def train_and_evaluate(X_train, y_train, X_val, y_val, 
                       clauses, T, s, max_epochs=10, patience=2):
    """
    Train TMU with given hyperparams and return validation metrics.
    
    Returns:
        dict with r_val_raw, r_val_cal, mae_cal, train_time, best_epoch
    """
    # Initialize
    tm = TMRegressor(
        number_of_clauses=clauses,
        T=T,
        s=s,
        platform='CPU',
        weighted_clauses=True
    )
    
    # Training with early stopping
    best_r = -np.inf
    best_epoch = 0
    patience_counter = 0
    total_time = 0
    
    for epoch in range(max_epochs):
        t_start = time.time()
        tm.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        total_time += t_elapsed
        
        # Validate
        y_val_pred = tm.predict(X_val)
        r_val, _ = pearsonr(y_val, y_val_pred)
        
        # Early stopping
        if r_val > best_r:
            best_r = r_val
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Final predictions
    y_val_pred = tm.predict(X_val)
    
    # Calibration (isotonic on validation)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_val_pred, y_val)
    y_val_cal = iso.transform(y_val_pred)
    
    # Metrics
    r_val_cal, _ = pearsonr(y_val, y_val_cal)
    mae_cal = float(np.mean(np.abs(y_val - y_val_cal)))
    
    return {
        'r_val_raw': float(best_r),
        'r_val_cal': float(r_val_cal),
        'mae_cal': mae_cal,
        'train_time_sec': float(total_time),
        'best_epoch': best_epoch,
        'epochs_ran': epoch + 1
    }


def main():
    parser = argparse.ArgumentParser(description="Grid Search V7 Hyperparameters")
    parser.add_argument("--features", default="data/explain_features_dataset_v7_th0.0005.h5")
    parser.add_argument("--output", default="results/v7_grid_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--max-epochs", type=int, default=10)
    
    # Grid parameters
    parser.add_argument("--clauses-grid", nargs='+', type=int, default=[3000, 6000, 10000])
    parser.add_argument("--T-grid", nargs='+', type=int, default=[300, 500, 700])
    parser.add_argument("--s-grid", nargs='+', type=float, default=[3.5, 5.0, 7.0])
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GRID SEARCH V7 HYPERPARAMETERS")
    print("="*80)
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    print(f"\nLoading dataset: {args.features}")
    data = load_dataset(args.features)
    print(f"Train: {data['train_X'].shape[0]} samples, {data['train_X'].shape[1]} features")
    print(f"Val:   {data['val_X'].shape[0]} samples")
    
    # Grid
    clauses_grid = args.clauses_grid
    T_grid = args.T_grid
    s_grid = args.s_grid
    
    configs = list(product(clauses_grid, T_grid, s_grid))
    n_configs = len(configs)
    
    print(f"\n📊 Grid Configuration:")
    print(f"  Clauses: {clauses_grid}")
    print(f"  T:       {T_grid}")
    print(f"  s:       {s_grid}")
    print(f"  Total:   {n_configs} configurations per head")
    print(f"  Heads:   BW, EMG")
    print(f"  Total runs: {n_configs * 2}")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results_bw = []
    results_emg = []
    
    # Run grid search
    print(f"\n{'='*80}")
    print("STARTING GRID SEARCH")
    print(f"{'='*80}\n")
    
    global_start = time.time()
    
    for i, (clauses, T, s) in enumerate(configs, 1):
        print(f"\n[{i}/{n_configs}] Testing: clauses={clauses}, T={T}, s={s}")
        print("-" * 60)
        
        # BW
        print("  🔵 BW model...")
        t_start = time.time()
        metrics_bw = train_and_evaluate(
            data['train_X'], data['train_y_bw'],
            data['val_X'], data['val_y_bw'],
            clauses, T, s, args.max_epochs, args.patience
        )
        t_bw = time.time() - t_start
        
        results_bw.append({
            'clauses': clauses,
            'T': T,
            's': s,
            **metrics_bw,
            'wall_time_sec': t_bw
        })
        
        print(f"     r_val_cal={metrics_bw['r_val_cal']:.4f}, "
              f"mae={metrics_bw['mae_cal']:.4f}, "
              f"time={t_bw:.1f}s")
        
        # EMG
        print("  🟢 EMG model...")
        t_start = time.time()
        metrics_emg = train_and_evaluate(
            data['train_X'], data['train_y_emg'],
            data['val_X'], data['val_y_emg'],
            clauses, T, s, args.max_epochs, args.patience
        )
        t_emg = time.time() - t_start
        
        results_emg.append({
            'clauses': clauses,
            'T': T,
            's': s,
            **metrics_emg,
            'wall_time_sec': t_emg
        })
        
        print(f"     r_val_cal={metrics_emg['r_val_cal']:.4f}, "
              f"mae={metrics_emg['mae_cal']:.4f}, "
              f"time={t_emg:.1f}s")
    
    global_time = time.time() - global_start
    
    # Convert to DataFrames
    df_bw = pd.DataFrame(results_bw)
    df_emg = pd.DataFrame(results_emg)
    
    # Save results
    df_bw.to_csv(output_dir / 'grid_results_bw.csv', index=False)
    df_emg.to_csv(output_dir / 'grid_results_emg.csv', index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {global_time/60:.1f} min")
    print(f"\nResults saved to: {output_dir}")
    
    # Best configurations
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*80}")
    
    # BW
    best_bw = df_bw.loc[df_bw['r_val_cal'].idxmax()]
    print(f"\n🔵 BW (best r_val_cal={best_bw['r_val_cal']:.4f}):")
    print(f"   clauses={int(best_bw['clauses'])}, T={int(best_bw['T'])}, s={best_bw['s']:.1f}")
    print(f"   MAE={best_bw['mae_cal']:.4f}, time={best_bw['train_time_sec']:.1f}s")
    
    # EMG
    best_emg = df_emg.loc[df_emg['r_val_cal'].idxmax()]
    print(f"\n🟢 EMG (best r_val_cal={best_emg['r_val_cal']:.4f}):")
    print(f"   clauses={int(best_emg['clauses'])}, T={int(best_emg['T'])}, s={best_emg['s']:.1f}")
    print(f"   MAE={best_emg['mae_cal']:.4f}, time={best_emg['train_time_sec']:.1f}s")
    
    # Top 5 for each
    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURATIONS (by r_val_cal)")
    print(f"{'='*80}")
    
    print("\n🔵 BW Top 5:")
    top5_bw = df_bw.nlargest(5, 'r_val_cal')[['clauses', 'T', 's', 'r_val_cal', 'mae_cal', 'train_time_sec']]
    print(top5_bw.to_string(index=False))
    
    print("\n🟢 EMG Top 5:")
    top5_emg = df_emg.nlargest(5, 'r_val_cal')[['clauses', 'T', 's', 'r_val_cal', 'mae_cal', 'train_time_sec']]
    print(top5_emg.to_string(index=False))
    
    # Save summary
    summary = {
        'grid_params': {
            'clauses': clauses_grid,
            'T': T_grid,
            's': s_grid
        },
        'n_configurations': n_configs,
        'total_runs': n_configs * 2,
        'total_time_sec': float(global_time),
        'best_bw': {
            'clauses': int(best_bw['clauses']),
            'T': int(best_bw['T']),
            's': float(best_bw['s']),
            'r_val_cal': float(best_bw['r_val_cal']),
            'mae_cal': float(best_bw['mae_cal']),
            'train_time_sec': float(best_bw['train_time_sec'])
        },
        'best_emg': {
            'clauses': int(best_emg['clauses']),
            'T': int(best_emg['T']),
            's': float(best_emg['s']),
            'r_val_cal': float(best_emg['r_val_cal']),
            'mae_cal': float(best_emg['mae_cal']),
            'train_time_sec': float(best_emg['train_time_sec'])
        }
    }
    
    with open(output_dir / 'grid_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to: {output_dir / 'grid_summary.json'}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review results in {output_dir}")
    print(f"   2. Retrain with best hyperparameters:")
    print(f"      python train_tmu_v7.py --clauses {int(best_bw['clauses'])} --T {int(best_bw['T'])} --s {best_bw['s']:.1f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
