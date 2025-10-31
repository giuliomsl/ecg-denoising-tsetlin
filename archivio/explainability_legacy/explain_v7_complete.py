#!/usr/bin/env python3
"""
V7 Explainability: Feature Importance Analysis
==============================================

Analyze V7 (57 features) interpretability:
- Feature importance (45 selected bitplanes + 12 HF)
- Compare bitplanes vs HF contribution
- Verify if selected 45 bitplanes are actually used

NOTE: TMRegressor doesn't expose clause weights for analysis.
We use Random Forest feature importance as proxy.
"""

import argparse
import pickle
import h5py
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

def load_v7_models(models_dir):
    """Load V7 TMU models"""
    models_dir = Path(models_dir)
    
    with open(models_dir / "tmu_bw.pkl", 'rb') as f:
        tm_bw = pickle.load(f)
    
    with open(models_dir / "tmu_emg.pkl", 'rb') as f:
        tm_emg = pickle.load(f)
    
    return tm_bw, tm_emg


def analyze_weights(tm, noise_type, output_dir):
    """Analyze clause weights distribution"""
    print(f"\n{'='*80}")
    print(f"WEIGHTS ANALYSIS: {noise_type.upper()}")
    print(f"{'='*80}")
    
    # Get weights
    weights = tm.get_weight()
    n_clauses = len(weights)
    
    print(f"\nTotal clauses: {n_clauses}")
    
    # Statistics
    weights_abs = np.abs(weights)
    mean_w = np.mean(weights_abs)
    max_w = np.max(weights_abs)
    min_w = np.min(weights_abs)
    
    # Gini coefficient (sparsity measure)
    sorted_weights = np.sort(weights_abs)
    n = len(sorted_weights)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    # Effective rank (participation ratio)
    weights_norm = weights_abs / np.sum(weights_abs)
    effective_rank = 1 / np.sum(weights_norm ** 2)
    effective_rank_pct = (effective_rank / n_clauses) * 100
    
    # Pareto analysis (top-k% clauses)
    sorted_indices = np.argsort(weights_abs)[::-1]
    cumsum_weights = np.cumsum(weights_abs[sorted_indices])
    total_weight = cumsum_weights[-1]
    
    # Find how many clauses for 50%, 80%, 90%
    n_50 = np.searchsorted(cumsum_weights, 0.5 * total_weight) + 1
    n_80 = np.searchsorted(cumsum_weights, 0.8 * total_weight) + 1
    n_90 = np.searchsorted(cumsum_weights, 0.9 * total_weight) + 1
    
    print(f"\nWeight Statistics:")
    print(f"  Mean:  {mean_w:.6f}")
    print(f"  Max:   {max_w:.6f}")
    print(f"  Min:   {min_w:.6f}")
    print(f"  Range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    print(f"\nSparsity Metrics:")
    print(f"  Gini coefficient:     {gini:.4f} (0=uniform, 1=sparse)")
    print(f"  Effective rank:       {effective_rank:.1f} / {n_clauses} ({effective_rank_pct:.1f}%)")
    
    print(f"\nPareto Analysis:")
    print(f"  Top {n_50:4d} clauses ({n_50/n_clauses*100:5.1f}%) → 50% weight")
    print(f"  Top {n_80:4d} clauses ({n_80/n_clauses*100:5.1f}%) → 80% weight")
    print(f"  Top {n_90:4d} clauses ({n_90/n_clauses*100:5.1f}%) → 90% weight")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Weight distribution
    axes[0, 0].hist(weights, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 0].set_xlabel('Weight', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'Weight Distribution - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sorted weights (absolute)
    axes[0, 1].plot(sorted_weights[::-1], linewidth=2, color='steelblue')
    axes[0, 1].set_xlabel('Clause Rank', fontsize=12)
    axes[0, 1].set_ylabel('|Weight|', fontsize=12)
    axes[0, 1].set_title(f'Sorted Absolute Weights - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Cumulative weight
    cumsum_pct = cumsum_weights / total_weight * 100
    axes[1, 0].plot(cumsum_pct, linewidth=2, color='forestgreen')
    axes[1, 0].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1, 0].axhline(80, color='orange', linestyle='--', alpha=0.5, label='80%')
    axes[1, 0].axhline(90, color='purple', linestyle='--', alpha=0.5, label='90%')
    axes[1, 0].scatter([n_50, n_80, n_90], [50, 80, 90], s=100, c='red', zorder=5)
    axes[1, 0].set_xlabel('Number of Top Clauses', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Weight (%)', fontsize=12)
    axes[1, 0].set_title(f'Pareto Curve - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top 20 clauses
    top_20_weights = weights_abs[sorted_indices[:20]]
    axes[1, 1].barh(range(20), top_20_weights, color='coral', alpha=0.7)
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([f"C{i}" for i in sorted_indices[:20]])
    axes[1, 1].set_xlabel('|Weight|', fontsize=12)
    axes[1, 1].set_ylabel('Clause ID', fontsize=12)
    axes[1, 1].set_title(f'Top 20 Clauses - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    plot_path = output_dir / f"v7_weights_{noise_type}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {plot_path}")
    
    return {
        'gini': float(gini),
        'effective_rank': float(effective_rank),
        'effective_rank_pct': float(effective_rank_pct),
        'pareto_50': int(n_50),
        'pareto_80': int(n_80),
        'pareto_90': int(n_90),
        'mean_weight': float(mean_w),
        'max_weight': float(max_w)
    }


def analyze_features(X_train, y_train, X_val, y_val, noise_type, output_dir, feature_names):
    """Analyze feature importance with Random Forest"""
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE: {noise_type.upper()}")
    print(f"{'='*80}")
    
    print(f"\nTraining Random Forest on {X_train.shape[0]} samples...")
    
    # Subsample for speed
    n_samples = min(25000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train[indices], y_train[indices])
    
    importance = rf.feature_importances_
    
    # Split bitplanes vs HF
    bitplane_importance = importance[:45]  # First 45 are selected bitplanes
    hf_importance = importance[45:]  # Last 12 are HF features
    
    print(f"\nFeature Groups:")
    print(f"  Bitplanes (45): {bitplane_importance.sum():.4f} total")
    print(f"  HF (12):        {hf_importance.sum():.4f} total")
    
    # Top features
    top_indices = np.argsort(importance)[::-1][:20]
    
    print(f"\nTop 20 Features:")
    for i, idx in enumerate(top_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {i+1:2d}. {name:<30} {importance[idx]:.6f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. All features sorted
    sorted_importance = np.sort(importance)[::-1]
    axes[0, 0].bar(range(len(sorted_importance)), sorted_importance, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Feature Rank', fontsize=12)
    axes[0, 0].set_ylabel('Importance', fontsize=12)
    axes[0, 0].set_title(f'Feature Importance (sorted) - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Bitplanes vs HF comparison
    group_importance = [bitplane_importance.sum(), hf_importance.sum()]
    group_labels = [f'Bitplanes\n(45 features)', f'HF Features\n(12 features)']
    axes[0, 1].bar(group_labels, group_importance, color=['steelblue', 'forestgreen'], alpha=0.7)
    axes[0, 1].set_ylabel('Total Importance', fontsize=12)
    axes[0, 1].set_title(f'Feature Groups - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Top 20 features
    top_20_imp = importance[top_indices[:20]]
    top_20_names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_indices[:20]]
    
    axes[1, 0].barh(range(20), top_20_imp, color='coral', alpha=0.7)
    axes[1, 0].set_yticks(range(20))
    axes[1, 0].set_yticklabels(top_20_names, fontsize=8)
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title(f'Top 20 Features - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # 4. HF features detail
    hf_names = feature_names[45:] if len(feature_names) > 45 else [f"HF_{i}" for i in range(12)]
    axes[1, 1].bar(range(12), hf_importance, color='forestgreen', alpha=0.7)
    axes[1, 1].set_xticks(range(12))
    axes[1, 1].set_xticklabels(hf_names, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Importance', fontsize=12)
    axes[1, 1].set_title(f'HF Features Detail - {noise_type.upper()}', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = output_dir / f"v7_importance_{noise_type}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {plot_path}")
    
    return {
        'bitplane_total': float(bitplane_importance.sum()),
        'hf_total': float(hf_importance.sum()),
        'top_20_features': [
            {'name': feature_names[i] if i < len(feature_names) else f"F{i}", 
             'importance': float(importance[i])}
            for i in top_indices[:20]
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="V7 Explainability Analysis")
    parser.add_argument("--models", type=str, default="models/tmu_v7_selected")
    parser.add_argument("--features", type=str, default="data/explain_features_dataset_v7_th0.0005.h5")
    parser.add_argument("--output", type=str, default="plots/v7_explainability")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("V7 EXPLAINABILITY ANALYSIS")
    print("="*80)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading dataset: {args.features}")
    with h5py.File(args.features, 'r') as f:
        train_X = f['train_X'][:]
        train_y_bw = f['train_y_bw'][:]
        train_y_emg = f['train_y_emg'][:]
        val_X = f['validation_X'][:]
        val_y_bw = f['validation_y_bw'][:]
        val_y_emg = f['validation_y_emg'][:]
    
    # Feature names (45 selected bitplanes + 12 HF)
    feature_names = [f"Bitplane_{i}" for i in range(45)] + [
        "Band_10_30_L0", "Band_10_30_L1", "Band_10_30_L2",
        "Band_30_50_L0", "Band_30_50_L1", "Band_30_50_L2",
        "ZCR_L0", "ZCR_L1", "ZCR_L2",
        "TKEO_L0", "TKEO_L1", "TKEO_L2"
    ]
    
    print(f"  Train: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    print(f"  Val:   {val_X.shape[0]} samples")
    
    # Feature Importance Analysis
    print("\n" + "="*80)
    print("NOTE: TMRegressor doesn't expose clause weights.")
    print("Using Random Forest feature importance as proxy.")
    print("="*80)
    
    features_bw = analyze_features(train_X, train_y_bw, val_X, val_y_bw, 'bw', output_dir, feature_names)
    features_emg = analyze_features(train_X, train_y_emg, val_X, val_y_emg, 'emg', output_dir, feature_names)
    
    # Save results
    results = {
        'features': {
            'bw': features_bw,
            'emg': features_emg
        },
        'note': 'TMRegressor does not expose clause weights for analysis. Feature importance computed with Random Forest.'
    }
    
    results_path = output_dir / "v7_explainability_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ V7 FEATURE IMPORTANCE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  ✓ Feature importance: v7_importance_bw.png, v7_importance_emg.png")
    print(f"  ✓ Summary: v7_explainability_results.json")
    print("\nKey Findings:")
    print(f"  BW  - Bitplanes: {features_bw['bitplane_total']:.2f}, HF: {features_bw['hf_total']:.2f}")
    print(f"  EMG - Bitplanes: {features_emg['bitplane_total']:.2f}, HF: {features_emg['hf_total']:.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
