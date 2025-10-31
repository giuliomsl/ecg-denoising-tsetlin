#!/usr/bin/env python3
"""
Create V7b Dataset: Union-Per-Task Feature Selection
====================================================

Strategy: Instead of selecting features globally (intersection approach),
we select TOP-K features independently for EACH task (BW, EMG), then take
the UNION. This allows task-specific features to be preserved.

Expected outcome:
- More features than V7 (e.g., 80 instead of 57)
- Better task-specific performance
- Still massive reduction vs V4 (1092 features)

Author: Giulio
Date: 2025-10-30
"""

import h5py
import numpy as np
from pathlib import Path
import json

def load_v4_dataset(path):
    """Load V4 dataset with all features."""
    print(f"Loading V4 dataset from {path}...")
    with h5py.File(path, 'r') as f:
        X_train = f['X_train'][:]
        X_val = f['X_val'][:]
        X_test = f['X_test'][:]
        Y_train_BW = f['Y_train_BW'][:]
        Y_val_BW = f['Y_val_BW'][:]
        Y_test_BW = f['Y_test_BW'][:]
        Y_train_EMG = f['Y_train_EMG'][:]
        Y_val_EMG = f['Y_val_EMG'][:]
        Y_test_EMG = f['Y_test_EMG'][:]
        
        meta = {
            'n_features': f.attrs['n_features'],
            'n_bitplanes': f.attrs['n_bitplanes'],
            'n_hf': f.attrs['n_hf']
        }
    
    print(f"Loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test samples")
    print(f"Features: {meta['n_features']} total ({meta['n_bitplanes']} bitplanes + {meta['n_hf']} HF)")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'Y_train_BW': Y_train_BW, 'Y_val_BW': Y_val_BW, 'Y_test_BW': Y_test_BW,
        'Y_train_EMG': Y_train_EMG, 'Y_val_EMG': Y_val_EMG, 'Y_test_EMG': Y_test_EMG,
        'meta': meta
    }

def compute_feature_importance_per_task(X, Y, task_name):
    """Compute correlation-based importance for each feature vs target."""
    print(f"\nComputing importance for {task_name}...")
    n_features = X.shape[1]
    importance = np.zeros(n_features)
    
    for i in range(n_features):
        # Pearson correlation
        corr = np.corrcoef(X[:, i], Y)[0, 1]
        importance[i] = abs(corr)  # Absolute value
    
    print(f"  Mean importance: {importance.mean():.6f}")
    print(f"  Max importance: {importance.max():.6f}")
    print(f"  Top-10 range: {importance[np.argsort(importance)[-10:]].min():.6f} - {importance.max():.6f}")
    
    return importance

def select_top_k_per_task(importance_bw, importance_emg, n_bitplanes, n_hf, top_k=50):
    """
    Select top-K bitplanes independently for each task, then take UNION.
    HF features (last 12) are ALWAYS included.
    
    Args:
        importance_bw: Importance scores for BW task (shape: n_features)
        importance_emg: Importance scores for EMG task (shape: n_features)
        n_bitplanes: Number of bitplane features
        n_hf: Number of HF features (last n_hf features)
        top_k: Number of top bitplanes to select per task
    
    Returns:
        selected_indices: Sorted indices of selected features
        stats: Dictionary with selection statistics
    """
    print(f"\n=== Union-Per-Task Selection (top-{top_k} per task) ===")
    
    # Separate bitplanes and HF features
    bp_importance_bw = importance_bw[:n_bitplanes]
    bp_importance_emg = importance_emg[:n_bitplanes]
    
    # Select top-K bitplanes for each task
    top_k_bw = np.argsort(bp_importance_bw)[-top_k:]
    top_k_emg = np.argsort(bp_importance_emg)[-top_k:]
    
    print(f"BW top-{top_k}: {len(top_k_bw)} bitplanes")
    print(f"EMG top-{top_k}: {len(top_k_emg)} bitplanes")
    
    # UNION of selected bitplanes
    union_bp = np.unique(np.concatenate([top_k_bw, top_k_emg]))
    print(f"Union: {len(union_bp)} unique bitplanes")
    
    # Overlap analysis
    overlap = len(np.intersect1d(top_k_bw, top_k_emg))
    print(f"Overlap: {overlap} bitplanes ({100*overlap/top_k:.1f}%)")
    
    # Add HF features (always included)
    hf_indices = np.arange(n_bitplanes, n_bitplanes + n_hf)
    selected_indices = np.concatenate([union_bp, hf_indices])
    selected_indices = np.sort(selected_indices)
    
    print(f"\nFinal selection: {len(selected_indices)} features ({len(union_bp)} BP + {n_hf} HF)")
    
    # Statistics
    stats = {
        'strategy': 'union_per_task',
        'top_k_per_task': int(top_k),
        'selected_bp_bw': top_k_bw.tolist(),
        'selected_bp_emg': top_k_emg.tolist(),
        'union_bp': union_bp.tolist(),
        'overlap_count': int(overlap),
        'overlap_pct': float(100 * overlap / top_k),
        'total_features': int(len(selected_indices)),
        'n_bitplanes_selected': int(len(union_bp)),
        'n_hf_selected': int(n_hf)
    }
    
    return selected_indices, stats

def create_v7b_dataset(v4_path, output_path, top_k=50):
    """Create V7b dataset with union-per-task feature selection."""
    
    print("="*80)
    print("Creating V7b Dataset: Union-Per-Task Feature Selection")
    print("="*80)
    
    # Load V4 dataset
    data = load_v4_dataset(v4_path)
    
    # Compute importance per task (using training data)
    importance_bw = compute_feature_importance_per_task(
        data['X_train'], data['Y_train_BW'], 'BW'
    )
    importance_emg = compute_feature_importance_per_task(
        data['X_train'], data['Y_train_EMG'], 'EMG'
    )
    
    # Select features using union-per-task strategy
    selected_indices, stats = select_top_k_per_task(
        importance_bw, importance_emg,
        n_bitplanes=data['meta']['n_bitplanes'],
        n_hf=data['meta']['n_hf'],
        top_k=top_k
    )
    
    # Extract selected features
    X_train_selected = data['X_train'][:, selected_indices]
    X_val_selected = data['X_val'][:, selected_indices]
    X_test_selected = data['X_test'][:, selected_indices]
    
    print(f"\nSelected features shape: {X_train_selected.shape}")
    
    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving V7b dataset to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Data
        f.create_dataset('X_train', data=X_train_selected, compression='gzip')
        f.create_dataset('X_val', data=X_val_selected, compression='gzip')
        f.create_dataset('X_test', data=X_test_selected, compression='gzip')
        f.create_dataset('Y_train_BW', data=data['Y_train_BW'], compression='gzip')
        f.create_dataset('Y_val_BW', data=data['Y_val_BW'], compression='gzip')
        f.create_dataset('Y_test_BW', data=data['Y_test_BW'], compression='gzip')
        f.create_dataset('Y_train_EMG', data=data['Y_train_EMG'], compression='gzip')
        f.create_dataset('Y_val_EMG', data=data['Y_val_EMG'], compression='gzip')
        f.create_dataset('Y_test_EMG', data=data['Y_test_EMG'], compression='gzip')
        
        # Metadata
        f.attrs['version'] = 'V7b'
        f.attrs['strategy'] = 'union_per_task'
        f.attrs['top_k_per_task'] = top_k
        f.attrs['n_features'] = len(selected_indices)
        f.attrs['n_bitplanes'] = data['meta']['n_bitplanes']
        f.attrs['n_hf'] = data['meta']['n_hf']
        f.attrs['n_bitplanes_selected'] = stats['n_bitplanes_selected']
        f.attrs['selected_indices'] = selected_indices
        f.attrs['overlap_count'] = stats['overlap_count']
        f.attrs['overlap_pct'] = stats['overlap_pct']
    
    print("✅ V7b dataset saved!")
    
    # Save selection stats to JSON
    stats_path = output_path.parent / f"{output_path.stem}_selection_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Selection stats saved to {stats_path}")
    
    # Summary
    print("\n" + "="*80)
    print("V7b DATASET SUMMARY")
    print("="*80)
    print(f"Strategy: Union-per-task (top-{top_k} bitplanes per task)")
    print(f"Total features: {len(selected_indices)} ({stats['n_bitplanes_selected']} BP + {stats['n_hf_selected']} HF)")
    print(f"Reduction: {100*(1 - len(selected_indices)/data['meta']['n_features']):.1f}% vs V4")
    print(f"BW-specific: {len(stats['selected_bp_bw'])} bitplanes")
    print(f"EMG-specific: {len(stats['selected_bp_emg'])} bitplanes")
    print(f"Overlap: {stats['overlap_count']} bitplanes ({stats['overlap_pct']:.1f}%)")
    print(f"Union: {stats['n_bitplanes_selected']} unique bitplanes")
    print("="*80)
    
    return stats

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create V7b dataset with union-per-task selection')
    parser.add_argument('--input', type=str, 
                       default='data/explain_features_dataset.h5',
                       help='Path to V4 dataset')
    parser.add_argument('--output', type=str,
                       default='data/explain_features_dataset_v7b_union_top50.h5',
                       help='Output path for V7b dataset')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of top bitplanes to select per task')
    
    args = parser.parse_args()
    
    stats = create_v7b_dataset(args.input, args.output, args.top_k)
    
    print("\n✅ V7b dataset creation complete!")
    print(f"\nNext steps:")
    print(f"1. Train V7b model:")
    print(f"   python train_tmu_v7.py --features {args.output} --output models/tmu_v7b_union \\")
    print(f"       --clauses 10000 --T 700 --s 3.5 --seed 42 --patience 3")
    print(f"2. Compare V7 vs V7b performance")
