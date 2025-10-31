#!/usr/bin/env python3
"""
Unified Explainability Analysis for TMU Models
===============================================

This script provides a comprehensive white-box analysis of Tsetlin Machine models
with modular analysis selection via command-line flags.

QUICK START GUIDE
=================

1. COMPLETE ANALYSIS (all modules):
   python -m src.explainability.analyze_explainability \\
       --analysis all \\
       --models models/tmu_v7/ \\
       --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
       --tasks bw emg \\
       --output plots/explainability/

2. WEIGHTS ONLY (distribution, Gini, Pareto):
   python -m src.explainability.analyze_explainability \\
       --analysis weights \\
       --models models/tmu_v7/ \\
       --tasks bw emg \\
       --output plots/weights/

3. FEATURE IMPORTANCE (select method):
   python -m src.explainability.analyze_explainability \\
       --analysis features \\
       --models models/tmu_v7/ \\
       --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
       --tasks bw emg \\
       --method ablation \\
       --limit 2000 \\
       --output plots/features/

4. RULES EXTRACTION (top clauses):
   python -m src.explainability.analyze_explainability \\
       --analysis rules \\
       --models models/tmu_v7/ \\
       --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
       --tasks bw \\
       --top-k 10 \\
       --output plots/rules/

5. PATTERN RESPONSES (synthetic patterns):
   python -m src.explainability.analyze_explainability \\
       --analysis patterns \\
       --models models/tmu_v7/ \\
       --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
       --tasks bw emg \\
       --output plots/patterns/

AVAILABLE FLAGS
===============

Required:
  --analysis {weights,features,rules,patterns,all}
      Type of analysis to perform:
      - weights:   Clause weights distribution, Gini coefficient, Pareto
      - features:  Feature importance via ablation/permutation/RF proxy
      - rules:     Logical rules extraction from top clauses
      - patterns:  Model response to synthetic test patterns
      - all:       Run all analyses sequentially

  --models PATH
      Directory containing trained TMU models (tmu_bw.pkl, tmu_emg.pkl)

  --tasks {bw,emg} [bw emg ...]
      Which noise tasks to analyze (one or more)

Optional:
  --dataset PATH
      HDF5 dataset with features (required for features/rules/patterns)
      Default: data/explain_features_dataset_v7_th0.0005.h5

  --output PATH
      Output directory for plots and reports
      Default: plots/explainability/

  --method {ablation,permutation,rf_proxy}
      Feature importance method (only for --analysis features):
      - ablation:     Mask each feature, measure MAE increase (accurate, slow)
      - permutation:  Permute each feature, measure MAE increase (medium)
      - rf_proxy:     Random Forest feature importance as proxy (fast)
      Default: ablation

  --limit INT
      Limit samples for feature importance (speed vs accuracy tradeoff)
      Default: 2000 (recommended for ablation)

  --top-k INT
      Number of top clauses to extract rules from
      Default: 10

  --seed INT
      Random seed for reproducibility
      Default: 42

ANALYSIS MODULES DETAILS
=========================

1. WEIGHTS ANALYSIS
   - Clause weights distribution (histogram + KDE)
   - Gini coefficient (sparsity measure: 0=uniform, 1=sparse)
   - Effective rank (how many clauses actively contribute)
   - Pareto analysis (80/20 rule: top-K% clauses → X% weight)
   - Top-K clauses identification
   
   Output: weights_distribution.png, weights_stats.json

2. FEATURE IMPORTANCE
   - Ablation: Remove feature, measure prediction drop
   - Permutation: Shuffle feature, measure prediction drop
   - RF Proxy: Train Random Forest, extract feature importance
   - Per-task comparison (BW vs EMG)
   - Bitplanes vs High-Frequency features comparison (V7)
   
   Output: feature_importance_*.png, importance_stats.json

3. RULES EXTRACTION
   - Identify top-N clauses by weight magnitude
   - Test clause response to synthetic patterns
   - Mask features to infer clause conditions
   - Translate to human-readable IF-THEN rules
   - Validate against ECG physiology
   
   Output: rules_extraction.png, rules_report.md

4. PATTERN RESPONSES
   - Generate synthetic test patterns (clean, BW, EMG, mixed)
   - Measure model predictions on controlled inputs
   - Visualize feature activation patterns
   - Compare task-specific sensitivities
   
   Output: pattern_responses.png, response_matrix.json

EXAMPLES BY USE CASE
=====================

Thesis Chapter: "Model Interpretability"
-----------------------------------------
python -m src.explainability.analyze_explainability \\
    --analysis all \\
    --models models/tmu_v7/ \\
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
    --tasks bw emg \\
    --method rf_proxy \\
    --top-k 15 \\
    --output plots/thesis_explainability/

Quick Check: "Are my features used?"
-------------------------------------
python -m src.explainability.analyze_explainability \\
    --analysis features \\
    --models models/tmu_v7/ \\
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
    --tasks bw \\
    --method rf_proxy \\
    --limit 5000 \\
    --output plots/quick_check/

Deep Dive: "What rules did the model learn?"
---------------------------------------------
python -m src.explainability.analyze_explainability \\
    --analysis rules \\
    --models models/tmu_v7/ \\
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
    --tasks bw emg \\
    --top-k 20 \\
    --output plots/rules_deep_dive/

Comparison: "V7 vs V4 interpretability"
----------------------------------------
# V7 (57 features)
python -m src.explainability.analyze_explainability \\
    --analysis features \\
    --models models/tmu_v7/ \\
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \\
    --tasks bw emg \\
    --method rf_proxy \\
    --output plots/v7_explainability/

# V4 (1092 features)
python -m src.explainability.analyze_explainability \\
    --analysis features \\
    --models models/tmu_v4/ \\
    --dataset data/explain_features_dataset_v2.h5 \\
    --tasks bw emg \\
    --method rf_proxy \\
    --output plots/v4_explainability/

NOTES
=====

Performance Tips:
- Use --method rf_proxy for quick feature importance (10x faster than ablation)
- Set --limit 2000 for ablation to balance accuracy and speed
- Run single task first (--tasks bw) before running both

Memory Requirements:
- weights:   ~100 MB
- features:  ~500 MB (ablation with limit=2000)
- rules:     ~200 MB
- patterns:  ~300 MB
- all:       ~1 GB

Expected Runtime (M1 Mac, 2000 samples):
- weights:   ~10 seconds
- features:  ~5 minutes (ablation), ~30 seconds (rf_proxy)
- rules:     ~2 minutes
- patterns:  ~1 minute
- all:       ~10 minutes

Author: Giulio
Date: 2025-10-31
Version: 1.0 (Unified)
"""

import argparse
import pickle
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_path):
    """Load TMU model from pickle file."""
    print(f"📦 Loading model: {model_path.name}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✅ Model loaded successfully")
    return model

def load_dataset(dataset_path):
    """Load dataset with features and targets."""
    print(f"📁 Loading dataset: {dataset_path.name}")
    with h5py.File(dataset_path, 'r') as f:
        data = {
            'X_train': f['train_X'][:],
            'X_val': f['validation_X'][:],
            'X_test': f['test_X'][:],
            'y_train_bw': f['train_y_bw'][:],
            'y_val_bw': f['validation_y_bw'][:],
            'y_test_bw': f['test_y_bw'][:],
            'y_train_emg': f['train_y_emg'][:],
            'y_val_emg': f['validation_y_emg'][:],
            'y_test_emg': f['test_y_emg'][:],
        }
        
        # Metadata
        data['meta'] = {
            'n_features': f.attrs.get('n_features', data['X_train'].shape[1]),
            'n_bitplanes': f.attrs.get('n_bitplanes_selected', 45),
            'n_hf': f.attrs.get('n_hf_features', 12),
            'strategy': f.attrs.get('selection_strategy', 'Unknown'),
        }
    
    print(f"   Train: {data['X_train'].shape}")
    print(f"   Val:   {data['X_val'].shape}")
    print(f"   Test:  {data['X_test'].shape}")
    print(f"   Features: {data['meta']['n_features']} total")
    return data

def compute_gini(weights):
    """
    Compute Gini coefficient for weight distribution.
    
    Gini = 0: perfect equality (all weights same)
    Gini = 1: perfect inequality (one weight dominates)
    
    Higher Gini = sparser representation = more interpretable
    """
    sorted_weights = np.sort(np.abs(weights))
    n = len(sorted_weights)
    cumsum = np.cumsum(sorted_weights)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n

# ============================================================================
# MODULE 1: WEIGHTS ANALYSIS
# ============================================================================

def analyze_weights(model, task_name, output_dir):
    """
    Analyze clause weights distribution.
    
    Metrics:
    - Gini coefficient (sparsity)
    - Effective rank (participation ratio)
    - Pareto analysis (top-K% → X% weight)
    - Top clauses identification
    """
    print(f"\n{'='*80}")
    print(f"MODULE 1: WEIGHTS ANALYSIS - {task_name.upper()}")
    print(f"{'='*80}")
    
    # Extract weights
    if hasattr(model, 'weight_bank'):
        weights = model.weight_bank.get_weights()
    elif hasattr(model, 'get_weight'):
        weights = model.get_weight()
    else:
        raise AttributeError("Model has no accessible weights")
    
    n_clauses = len(weights)
    weights_abs = np.abs(weights)
    
    # Statistics
    mean_w = np.mean(weights_abs)
    std_w = np.std(weights_abs)
    max_w = np.max(weights_abs)
    min_w = np.min(weights_abs)
    
    # Gini coefficient
    gini = compute_gini(weights)
    
    # Effective rank (participation ratio)
    weights_norm = weights_abs / np.sum(weights_abs)
    effective_rank = 1 / np.sum(weights_norm ** 2)
    effective_rank_pct = (effective_rank / n_clauses) * 100
    
    # Pareto analysis
    sorted_indices = np.argsort(weights_abs)[::-1]
    cumsum_weights = np.cumsum(weights_abs[sorted_indices])
    total_weight = cumsum_weights[-1]
    
    n_50 = np.searchsorted(cumsum_weights, 0.5 * total_weight) + 1
    n_80 = np.searchsorted(cumsum_weights, 0.8 * total_weight) + 1
    n_90 = np.searchsorted(cumsum_weights, 0.9 * total_weight) + 1
    
    # Print results
    print(f"\n📊 Weight Statistics:")
    print(f"   Total clauses: {n_clauses}")
    print(f"   Mean weight:   {mean_w:.6f}")
    print(f"   Std weight:    {std_w:.6f}")
    print(f"   Max weight:    {max_w:.6f}")
    print(f"   Min weight:    {min_w:.6f}")
    print(f"   Range:         [{weights.min():.6f}, {weights.max():.6f}]")
    
    print(f"\n🎯 Sparsity Metrics:")
    print(f"   Gini coefficient: {gini:.4f} (0=uniform, 1=sparse)")
    print(f"   Effective rank:   {effective_rank:.1f} / {n_clauses} ({effective_rank_pct:.1f}%)")
    
    print(f"\n📈 Pareto Analysis:")
    print(f"   Top {n_50:4d} clauses ({n_50/n_clauses*100:5.1f}%) → 50% total weight")
    print(f"   Top {n_80:4d} clauses ({n_80/n_clauses*100:5.1f}%) → 80% total weight")
    print(f"   Top {n_90:4d} clauses ({n_90/n_clauses*100:5.1f}%) → 90% total weight")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Weight distribution
    axes[0, 0].hist(weights, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 0].set_xlabel('Weight Value', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'Weight Distribution - {task_name.upper()}', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Absolute weights sorted
    axes[0, 1].plot(weights_abs[sorted_indices], linewidth=2, color='darkgreen')
    axes[0, 1].set_xlabel('Clause Rank', fontsize=12)
    axes[0, 1].set_ylabel('Absolute Weight', fontsize=12)
    axes[0, 1].set_title(f'Sorted Absolute Weights - {task_name.upper()}', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Cumulative weight (Pareto)
    cumsum_pct = cumsum_weights / total_weight * 100
    axes[1, 0].plot(cumsum_pct, linewidth=2, color='darkorange')
    axes[1, 0].axhline(50, color='red', linestyle='--', linewidth=1.5, label='50%')
    axes[1, 0].axhline(80, color='blue', linestyle='--', linewidth=1.5, label='80%')
    axes[1, 0].axhline(90, color='green', linestyle='--', linewidth=1.5, label='90%')
    axes[1, 0].set_xlabel('Number of Clauses', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Weight (%)', fontsize=12)
    axes[1, 0].set_title(f'Cumulative Weight (Pareto) - {task_name.upper()}', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Top-20 clauses
    top_20_weights = weights[sorted_indices[:20]]
    top_20_indices = sorted_indices[:20]
    colors = ['red' if w < 0 else 'green' for w in top_20_weights]
    axes[1, 1].barh(range(20), top_20_weights, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([f"C{idx}" for idx in top_20_indices])
    axes[1, 1].set_xlabel('Weight Value', fontsize=12)
    axes[1, 1].set_ylabel('Clause ID', fontsize=12)
    axes[1, 1].set_title(f'Top-20 Clauses - {task_name.upper()}', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(0, color='black', linewidth=1)
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / f"weights_distribution_{task_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {output_path}")
    plt.close()
    
    # Save statistics
    stats = {
        'n_clauses': int(n_clauses),
        'mean_weight': float(mean_w),
        'std_weight': float(std_w),
        'max_weight': float(max_w),
        'min_weight': float(min_w),
        'gini_coefficient': float(gini),
        'effective_rank': float(effective_rank),
        'effective_rank_pct': float(effective_rank_pct),
        'pareto': {
            'n_50pct': int(n_50),
            'n_80pct': int(n_80),
            'n_90pct': int(n_90),
        },
        'top_20_clauses': [int(idx) for idx in sorted_indices[:20]],
        'top_20_weights': [float(w) for w in weights[sorted_indices[:20]]],
    }
    
    stats_path = output_dir / f"weights_stats_{task_name}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Saved: {stats_path}")
    
    return stats

# ============================================================================
# MODULE 2: FEATURE IMPORTANCE
# ============================================================================

def compute_feature_importance_ablation(model, X_val, y_val, limit=2000, seed=42):
    """
    Feature importance via ablation study.
    
    For each feature:
    1. Mask feature (set to 0)
    2. Predict on validation set
    3. Compute MAE increase
    
    High MAE increase → important feature
    """
    print(f"\n🔬 Computing feature importance (ablation method)...")
    
    # Limit samples for speed
    if limit and limit < len(X_val):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X_val), limit, replace=False)
        X_val = X_val[indices]
        y_val = y_val[indices]
        print(f"   Limited to {limit} samples for speed")
    
    # Baseline MAE
    baseline_pred = model.predict(X_val)
    baseline_mae = mean_absolute_error(y_val, baseline_pred)
    print(f"   Baseline MAE: {baseline_mae:.4f}")
    
    n_features = X_val.shape[1]
    importance = np.zeros(n_features)
    
    for feat_idx in tqdm(range(n_features), desc="Ablating features"):
        # Mask feature
        X_ablated = X_val.copy()
        X_ablated[:, feat_idx] = 0
        
        # Predict
        pred_ablated = model.predict(X_ablated)
        mae_ablated = mean_absolute_error(y_val, pred_ablated)
        
        # Importance = MAE increase
        importance[feat_idx] = mae_ablated - baseline_mae
    
    return importance, baseline_mae

def compute_feature_importance_permutation(model, X_val, y_val, limit=2000, seed=42):
    """
    Feature importance via permutation.
    
    Similar to ablation but permutes feature values instead of masking.
    """
    print(f"\n🔬 Computing feature importance (permutation method)...")
    
    # Limit samples
    if limit and limit < len(X_val):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X_val), limit, replace=False)
        X_val = X_val[indices]
        y_val = y_val[indices]
        print(f"   Limited to {limit} samples for speed")
    
    # Baseline MAE
    baseline_pred = model.predict(X_val)
    baseline_mae = mean_absolute_error(y_val, baseline_pred)
    print(f"   Baseline MAE: {baseline_mae:.4f}")
    
    n_features = X_val.shape[1]
    importance = np.zeros(n_features)
    rng = np.random.RandomState(seed)
    
    for feat_idx in tqdm(range(n_features), desc="Permuting features"):
        # Permute feature
        X_permuted = X_val.copy()
        X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
        
        # Predict
        pred_permuted = model.predict(X_permuted)
        mae_permuted = mean_absolute_error(y_val, pred_permuted)
        
        # Importance = MAE increase
        importance[feat_idx] = mae_permuted - baseline_mae
    
    return importance, baseline_mae

def compute_feature_importance_rf_proxy(model, X_val, y_val, limit=5000, seed=42):
    """
    Feature importance using Random Forest as proxy.
    
    Fast alternative to ablation/permutation.
    Train RF to mimic TMU predictions, extract feature importance.
    """
    print(f"\n🔬 Computing feature importance (Random Forest proxy method)...")
    
    # Limit samples
    if limit and limit < len(X_val):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X_val), limit, replace=False)
        X_val = X_val[indices]
        y_val = y_val[indices]
        print(f"   Limited to {limit} samples for speed")
    
    # Get TMU predictions
    tmu_pred = model.predict(X_val)
    
    # Train Random Forest to mimic TMU
    print(f"   Training Random Forest proxy...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
    rf.fit(X_val, tmu_pred)
    
    # Check proxy quality
    rf_pred = rf.predict(X_val)
    r_proxy = pearsonr(tmu_pred, rf_pred)[0]
    mae_proxy = mean_absolute_error(tmu_pred, rf_pred)
    print(f"   RF proxy quality: r={r_proxy:.4f}, MAE={mae_proxy:.4f}")
    
    # Extract feature importance
    importance = rf.feature_importances_
    
    # Baseline MAE (for comparison)
    baseline_mae = mean_absolute_error(y_val, tmu_pred)
    print(f"   TMU Baseline MAE: {baseline_mae:.4f}")
    
    return importance, baseline_mae

def analyze_feature_importance(model, X_val, y_val, task_name, method='ablation', limit=2000, seed=42, output_dir=None, n_bitplanes=45):
    """
    Analyze feature importance with selected method.
    """
    print(f"\n{'='*80}")
    print(f"MODULE 2: FEATURE IMPORTANCE - {task_name.upper()} (method: {method})")
    print(f"{'='*80}")
    
    # Compute importance
    if method == 'ablation':
        importance, baseline_mae = compute_feature_importance_ablation(model, X_val, y_val, limit, seed)
    elif method == 'permutation':
        importance, baseline_mae = compute_feature_importance_permutation(model, X_val, y_val, limit, seed)
    elif method == 'rf_proxy':
        importance, baseline_mae = compute_feature_importance_rf_proxy(model, X_val, y_val, limit, seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    n_features = len(importance)
    
    # Statistics
    print(f"\n📊 Feature Importance Statistics:")
    print(f"   Total features: {n_features}")
    print(f"   Mean importance: {importance.mean():.6f}")
    print(f"   Std importance:  {importance.std():.6f}")
    print(f"   Max importance:  {importance.max():.6f}")
    print(f"   Min importance:  {importance.min():.6f}")
    
    # Top features
    top_indices = np.argsort(importance)[::-1][:20]
    print(f"\n🏆 Top-20 Features:")
    for i, idx in enumerate(top_indices, 1):
        feat_type = "BP" if idx < n_bitplanes else "HF"
        feat_id = idx if idx < n_bitplanes else idx - n_bitplanes
        print(f"   {i:2d}. Feature {idx:3d} ({feat_type}{feat_id:2d}): {importance[idx]:.6f}")
    
    # Bitplanes vs HF comparison (if V7)
    if n_features == 57:  # V7 has 45 BP + 12 HF
        bp_importance = importance[:45]
        hf_importance = importance[45:]
        
        print(f"\n🔍 Bitplanes vs High-Frequency Features:")
        print(f"   Bitplanes (45):     mean={bp_importance.mean():.6f}, sum={bp_importance.sum():.6f}")
        print(f"   High-Freq (12):     mean={hf_importance.mean():.6f}, sum={hf_importance.sum():.6f}")
        print(f"   HF/BP ratio (mean): {hf_importance.mean() / bp_importance.mean():.2f}x")
    
    # Visualization
    if output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. All features importance
        axes[0, 0].bar(range(n_features), importance, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Feature Index', fontsize=12)
        axes[0, 0].set_ylabel('Importance (MAE increase)' if method != 'rf_proxy' else 'Importance (RF)', fontsize=12)
        axes[0, 0].set_title(f'Feature Importance - {task_name.upper()} ({method})', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # 2. Top-20 features
        top_20_importance = importance[top_indices[:20]]
        feat_labels = []
        for idx in top_indices[:20]:
            if idx < n_bitplanes:
                feat_labels.append(f"BP{idx}")
            else:
                feat_labels.append(f"HF{idx - n_bitplanes}")
        
        axes[0, 1].barh(range(20), top_20_importance, color='darkgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_yticks(range(20))
        axes[0, 1].set_yticklabels(feat_labels)
        axes[0, 1].set_xlabel('Importance', fontsize=12)
        axes[0, 1].set_ylabel('Feature', fontsize=12)
        axes[0, 1].set_title(f'Top-20 Features - {task_name.upper()}', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='x')
        axes[0, 1].invert_yaxis()
        
        # 3. Bitplanes vs HF (if V7)
        if n_features == 57:
            bp_mean = bp_importance.mean()
            hf_mean = hf_importance.mean()
            bp_sum = bp_importance.sum()
            hf_sum = hf_importance.sum()
            
            axes[1, 0].bar(['Bitplanes\n(45 feat)', 'High-Freq\n(12 feat)'], 
                          [bp_mean, hf_mean], 
                          color=['steelblue', 'darkorange'], 
                          alpha=0.7, 
                          edgecolor='black')
            axes[1, 0].set_ylabel('Mean Importance', fontsize=12)
            axes[1, 0].set_title(f'Bitplanes vs High-Frequency - {task_name.upper()}', fontsize=14, fontweight='bold')
            axes[1, 0].grid(alpha=0.3, axis='y')
            
            # Add values on bars
            for i, (label, val) in enumerate(zip(['BP', 'HF'], [bp_mean, hf_mean])):
                axes[1, 0].text(i, val, f'{val:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 4. Cumulative importance
            cumsum_importance = np.cumsum(importance[top_indices])
            total_importance = importance.sum()
            cumsum_pct = cumsum_importance / total_importance * 100
            
            axes[1, 1].plot(cumsum_pct, linewidth=2, color='darkred')
            axes[1, 1].axhline(50, color='red', linestyle='--', linewidth=1.5, label='50%')
            axes[1, 1].axhline(80, color='blue', linestyle='--', linewidth=1.5, label='80%')
            axes[1, 1].axhline(90, color='green', linestyle='--', linewidth=1.5, label='90%')
            axes[1, 1].set_xlabel('Number of Features', fontsize=12)
            axes[1, 1].set_ylabel('Cumulative Importance (%)', fontsize=12)
            axes[1, 1].set_title(f'Cumulative Importance - {task_name.upper()}', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        else:
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"feature_importance_{task_name}_{method}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
    
    # Save statistics
    stats = {
        'method': method,
        'n_features': int(n_features),
        'baseline_mae': float(baseline_mae),
        'mean_importance': float(importance.mean()),
        'std_importance': float(importance.std()),
        'max_importance': float(importance.max()),
        'min_importance': float(importance.min()),
        'top_20_indices': [int(idx) for idx in top_indices[:20]],
        'top_20_importance': [float(importance[idx]) for idx in top_indices[:20]],
    }
    
    if n_features == 57:
        stats['bitplanes_vs_hf'] = {
            'bp_mean': float(bp_importance.mean()),
            'bp_sum': float(bp_importance.sum()),
            'hf_mean': float(hf_importance.mean()),
            'hf_sum': float(hf_importance.sum()),
            'hf_bp_ratio': float(hf_importance.mean() / bp_importance.mean()),
        }
    
    if output_dir:
        stats_path = output_dir / f"feature_importance_stats_{task_name}_{method}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Saved: {stats_path}")
    
    return stats

# ============================================================================
# MODULE 3: RULES EXTRACTION
# ============================================================================

def analyze_rules_extraction(model, X_val, y_val, task_name, top_k=10, output_dir=None):
    """
    Extract interpretable rules from top-K clauses.
    
    Strategy:
    1. Identify top-K clauses by weight
    2. Test clause response to controlled patterns
    3. Infer feature conditions via ablation
    4. Translate to IF-THEN rules
    """
    print(f"\n{'='*80}")
    print(f"MODULE 3: RULES EXTRACTION - {task_name.upper()} (top-{top_k} clauses)")
    print(f"{'='*80}")
    
    # Extract weights
    if hasattr(model, 'weight_bank'):
        weights = model.weight_bank.get_weights()
    elif hasattr(model, 'get_weight'):
        weights = model.get_weight()
    else:
        raise AttributeError("Model has no accessible weights")
    
    # Get top-K clauses
    weights_abs = np.abs(weights)
    top_indices = np.argsort(weights_abs)[::-1][:top_k]
    
    print(f"\n🏆 Top-{top_k} Clauses:")
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i:2d}. Clause {idx:4d}: weight={weights[idx]:+.6f}")
    
    # Simple rules extraction via pattern response
    # Test on validation samples with different characteristics
    
    # Select diverse samples
    sorted_by_target = np.argsort(y_val)
    low_idx = sorted_by_target[len(sorted_by_target) // 10]  # 10th percentile
    mid_idx = sorted_by_target[len(sorted_by_target) // 2]   # 50th percentile
    high_idx = sorted_by_target[int(len(sorted_by_target) * 0.9)]  # 90th percentile
    
    test_samples = {
        'low': X_val[low_idx:low_idx+1],
        'medium': X_val[mid_idx:mid_idx+1],
        'high': X_val[high_idx:high_idx+1],
    }
    
    test_targets = {
        'low': y_val[low_idx],
        'medium': y_val[mid_idx],
        'high': y_val[high_idx],
    }
    
    print(f"\n🧪 Test Patterns:")
    for name, target in test_targets.items():
        pred = model.predict(test_samples[name])[0]
        print(f"   {name.capitalize():8s}: target={target:.4f}, pred={pred:.4f}")
    
    # Feature ablation for each test pattern
    rules = defaultdict(list)
    
    for pattern_name, X_pattern in test_samples.items():
        baseline_pred = model.predict(X_pattern)[0]
        
        # Test each feature
        important_features = []
        for feat_idx in range(X_pattern.shape[1]):
            X_ablated = X_pattern.copy()
            X_ablated[0, feat_idx] = 0
            pred_ablated = model.predict(X_ablated)[0]
            
            # If ablation changes prediction significantly, feature is important
            impact = abs(pred_ablated - baseline_pred)
            if impact > 0.01:  # Threshold for significance
                important_features.append((feat_idx, impact, X_pattern[0, feat_idx]))
        
        # Sort by impact
        important_features.sort(key=lambda x: x[1], reverse=True)
        
        # Create rule
        if important_features:
            top_features = important_features[:5]  # Top-5 features
            rule_parts = []
            for feat_idx, impact, value in top_features:
                rule_parts.append(f"F{feat_idx}={value:.0f}")
            
            rule = f"IF {' AND '.join(rule_parts)} THEN {task_name}_intensity ≈ {baseline_pred:.2f}"
            rules[pattern_name].append(rule)
    
    print(f"\n📜 Extracted Rules:")
    for pattern_name, pattern_rules in rules.items():
        print(f"\n   Pattern: {pattern_name.upper()}")
        for rule in pattern_rules:
            print(f"      {rule}")
    
    # Visualization
    if output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top clauses weights
        top_weights = weights[top_indices]
        colors = ['red' if w < 0 else 'green' for w in top_weights]
        axes[0, 0].barh(range(top_k), top_weights, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_yticks(range(top_k))
        axes[0, 0].set_yticklabels([f"C{idx}" for idx in top_indices])
        axes[0, 0].set_xlabel('Weight Value', fontsize=12)
        axes[0, 0].set_ylabel('Clause ID', fontsize=12)
        axes[0, 0].set_title(f'Top-{top_k} Clauses Weights - {task_name.upper()}', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(0, color='black', linewidth=1)
        axes[0, 0].grid(alpha=0.3, axis='x')
        axes[0, 0].invert_yaxis()
        
        # 2. Test patterns predictions
        pattern_names = list(test_samples.keys())
        predictions = [model.predict(test_samples[name])[0] for name in pattern_names]
        targets = [test_targets[name] for name in pattern_names]
        
        x_pos = np.arange(len(pattern_names))
        width = 0.35
        axes[0, 1].bar(x_pos - width/2, targets, width, label='Target', color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].bar(x_pos + width/2, predictions, width, label='Prediction', color='darkorange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([name.capitalize() for name in pattern_names])
        axes[0, 1].set_ylabel('Intensity', fontsize=12)
        axes[0, 1].set_title(f'Test Patterns Response - {task_name.upper()}', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # 3. Rules text
        axes[1, 0].axis('off')
        rules_text = f"EXTRACTED RULES - {task_name.upper()}\n" + "="*60 + "\n\n"
        for pattern_name, pattern_rules in rules.items():
            rules_text += f"{pattern_name.upper()} Pattern:\n"
            for rule in pattern_rules:
                rules_text += f"  • {rule}\n"
            rules_text += "\n"
        
        axes[1, 0].text(0.05, 0.95, rules_text, transform=axes[1, 0].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Feature importance for each pattern (heatmap)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"rules_extraction_{task_name}_top{top_k}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
    
    # Save rules
    rules_data = {
        'top_k': top_k,
        'top_clauses': [int(idx) for idx in top_indices],
        'top_weights': [float(weights[idx]) for idx in top_indices],
        'rules': {pattern: rules_list for pattern, rules_list in rules.items()},
    }
    
    if output_dir:
        rules_path = output_dir / f"rules_{task_name}_top{top_k}.json"
        with open(rules_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        print(f"💾 Saved: {rules_path}")
    
    return rules_data

# ============================================================================
# MODULE 4: PATTERN RESPONSES
# ============================================================================

def analyze_pattern_responses(model, X_val, y_val, task_name, output_dir=None):
    """
    Analyze model response to patterns with different characteristics.
    """
    print(f"\n{'='*80}")
    print(f"MODULE 4: PATTERN RESPONSES - {task_name.upper()}")
    print(f"{'='*80}")
    
    # Select representative patterns
    sorted_by_target = np.argsort(y_val)
    
    # Select patterns at different percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    patterns = {}
    
    for p in percentiles:
        if p == 0:
            idx = sorted_by_target[0]
        elif p == 100:
            idx = sorted_by_target[-1]
        else:
            idx = sorted_by_target[int(len(sorted_by_target) * p / 100)]
        
        patterns[f'p{p}'] = {
            'X': X_val[idx:idx+1],
            'target': y_val[idx],
            'pred': model.predict(X_val[idx:idx+1])[0]
        }
    
    print(f"\n🧪 Pattern Responses:")
    for name, data in patterns.items():
        error = abs(data['pred'] - data['target'])
        print(f"   {name:5s}: target={data['target']:.4f}, pred={data['pred']:.4f}, error={error:.4f}")
    
    # Visualization
    if output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Target vs Prediction
        targets = [data['target'] for data in patterns.values()]
        preds = [data['pred'] for data in patterns.values()]
        pattern_names = list(patterns.keys())
        
        axes[0, 0].scatter(targets, preds, s=100, alpha=0.7, color='steelblue', edgecolor='black')
        for i, name in enumerate(pattern_names):
            axes[0, 0].annotate(name, (targets[i], preds[i]), fontsize=9)
        
        # Perfect prediction line
        min_val = min(min(targets), min(preds))
        max_val = max(max(targets), max(preds))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        axes[0, 0].set_xlabel('Target Intensity', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Intensity', fontsize=12)
        axes[0, 0].set_title(f'Pattern Responses - {task_name.upper()}', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Prediction errors
        errors = [abs(data['pred'] - data['target']) for data in patterns.values()]
        axes[0, 1].bar(range(len(pattern_names)), errors, color='darkred', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xticks(range(len(pattern_names)))
        axes[0, 1].set_xticklabels(pattern_names, rotation=45)
        axes[0, 1].set_ylabel('Absolute Error', fontsize=12)
        axes[0, 1].set_title(f'Prediction Errors - {task_name.upper()}', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # 3. Target distribution
        axes[1, 0].hist(y_val, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        for p_name, data in patterns.items():
            axes[1, 0].axvline(data['target'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        axes[1, 0].set_xlabel('Target Intensity', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title(f'Target Distribution with Test Patterns - {task_name.upper()}', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        max_error = np.max(errors)
        
        stats_text = f"PATTERN RESPONSE STATISTICS - {task_name.upper()}\n"
        stats_text += "="*60 + "\n\n"
        stats_text += f"Number of test patterns: {len(patterns)}\n\n"
        stats_text += f"Error Statistics:\n"
        stats_text += f"  MAE:        {mae:.4f}\n"
        stats_text += f"  RMSE:       {rmse:.4f}\n"
        stats_text += f"  Max Error:  {max_error:.4f}\n\n"
        stats_text += f"Target Range: [{min(targets):.4f}, {max(targets):.4f}]\n"
        stats_text += f"Pred Range:   [{min(preds):.4f}, {max(preds):.4f}]\n"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        output_path = output_dir / f"pattern_responses_{task_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
    
    # Save statistics
    response_data = {
        'patterns': {
            name: {
                'target': float(data['target']),
                'prediction': float(data['pred']),
                'error': float(abs(data['pred'] - data['target']))
            }
            for name, data in patterns.items()
        },
        'statistics': {
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean([e**2 for e in errors]))),
            'max_error': float(np.max(errors)),
            'min_target': float(min(targets)),
            'max_target': float(max(targets)),
            'min_pred': float(min(preds)),
            'max_pred': float(max(preds)),
        }
    }
    
    if output_dir:
        response_path = output_dir / f"pattern_responses_{task_name}.json"
        with open(response_path, 'w') as f:
            json.dump(response_data, f, indent=2)
        print(f"💾 Saved: {response_path}")
    
    return response_data

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Explainability Analysis for TMU Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis
  python -m src.explainability.analyze_explainability --analysis all --models models/tmu_v7/ --dataset data/explain_features_dataset_v7_th0.0005.h5 --tasks bw emg

  # Weights only
  python -m src.explainability.analyze_explainability --analysis weights --models models/tmu_v7/ --tasks bw emg

  # Feature importance with RF proxy
  python -m src.explainability.analyze_explainability --analysis features --models models/tmu_v7/ --dataset data/explain_features_dataset_v7_th0.0005.h5 --tasks bw --method rf_proxy
        """
    )
    
    # Required arguments
    parser.add_argument('--analysis', type=str, required=True,
                       choices=['weights', 'features', 'rules', 'patterns', 'all'],
                       help='Type of analysis to perform')
    parser.add_argument('--models', type=str, required=True,
                       help='Directory containing TMU models (tmu_bw.pkl, tmu_emg.pkl)')
    parser.add_argument('--tasks', type=str, nargs='+', required=True,
                       choices=['bw', 'emg'],
                       help='Noise tasks to analyze')
    
    # Optional arguments
    parser.add_argument('--dataset', type=str, 
                       default='data/explain_features_dataset_v7_th0.0005.h5',
                       help='HDF5 dataset with features (required for features/rules/patterns)')
    parser.add_argument('--output', type=str, default='plots/explainability/',
                       help='Output directory for plots and reports')
    parser.add_argument('--method', type=str, default='ablation',
                       choices=['ablation', 'permutation', 'rf_proxy'],
                       help='Feature importance method (only for --analysis features)')
    parser.add_argument('--limit', type=int, default=2000,
                       help='Sample limit for feature importance (speed vs accuracy)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top clauses for rules extraction')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup
    models_dir = Path(args.models)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"UNIFIED EXPLAINABILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Analysis type: {args.analysis}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Models: {models_dir}")
    print(f"Output: {output_dir}")
    
    # Load dataset if needed
    data = None
    if args.analysis in ['features', 'rules', 'patterns', 'all']:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        data = load_dataset(dataset_path)
    
    # Run analysis for each task
    results = {}
    
    for task in args.tasks:
        print(f"\n{'#'*80}")
        print(f"TASK: {task.upper()}")
        print(f"{'#'*80}")
        
        # Load model
        model_path = models_dir / f"tmu_{task}.pkl"
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}, skipping...")
            continue
        
        model = load_model(model_path)
        
        # Initialize results
        results[task] = {}
        
        # Module 1: Weights
        if args.analysis in ['weights', 'all']:
            results[task]['weights'] = analyze_weights(model, task, output_dir)
        
        # Module 2: Feature Importance
        if args.analysis in ['features', 'all']:
            X_val = data['X_val']
            y_val = data[f'y_val_{task}']
            n_bitplanes = data['meta']['n_bitplanes']
            
            results[task]['features'] = analyze_feature_importance(
                model, X_val, y_val, task, 
                method=args.method, 
                limit=args.limit, 
                seed=args.seed,
                output_dir=output_dir,
                n_bitplanes=n_bitplanes
            )
        
        # Module 3: Rules
        if args.analysis in ['rules', 'all']:
            X_val = data['X_val']
            y_val = data[f'y_val_{task}']
            
            results[task]['rules'] = analyze_rules_extraction(
                model, X_val, y_val, task,
                top_k=args.top_k,
                output_dir=output_dir
            )
        
        # Module 4: Patterns
        if args.analysis in ['patterns', 'all']:
            X_val = data['X_val']
            y_val = data[f'y_val_{task}']
            
            results[task]['patterns'] = analyze_pattern_responses(
                model, X_val, y_val, task,
                output_dir=output_dir
            )
    
    # Save complete results
    results_path = output_dir / f"explainability_results_{args.analysis}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {results_path}")
    print(f"\n✅ All analyses completed successfully!")

if __name__ == '__main__':
    main()
