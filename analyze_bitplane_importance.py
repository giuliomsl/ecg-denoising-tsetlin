#!/usr/bin/env python3
"""
Analyze Bitplane Feature Importance
====================================

Find which specific bitplanes (if any) have non-negligible importance.
Create V7 dataset with: HF features + top-k important bitplanes.
"""

import h5py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from pathlib import Path

def load_dataset(path):
    """Load V2 dataset"""
    with h5py.File(path, 'r') as f:
        train_X = f['train_X'][:]
        train_y_bw = f['train_y_bw'][:]
        train_y_emg = f['train_y_emg'][:]
        val_X = f['validation_X'][:]
        val_y_bw = f['validation_y_bw'][:]
        val_y_emg = f['validation_y_emg'][:]
    return train_X, train_y_bw, train_y_emg, val_X, val_y_bw, val_y_emg


print("="*80)
print("BITPLANE FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Load data
print("\nLoading V2 dataset...")
train_X, train_y_bw, train_y_emg, val_X, val_y_bw, val_y_emg = load_dataset("data/explain_features_dataset_v2.h5")

print(f"Train: {train_X.shape[0]} samples, {train_X.shape[1]} features")
print(f"  - Bitplanes: {train_X.shape[1] - 12} features (indices 0-1079)")
print(f"  - HF:        12 features (indices 1080-1091)")

# Subsample for speed (10% of data is enough for feature importance)
n_samples = min(25000, len(train_X))
indices = np.random.choice(len(train_X), n_samples, replace=False)
X_sub = train_X[indices]
y_bw_sub = train_y_bw[indices]
y_emg_sub = train_y_emg[indices]

print(f"\nSubsampling {n_samples} samples for speed...")

# Train RF on BW
print("\nTraining RF on BW noise...")
rf_bw = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_bw.fit(X_sub, y_bw_sub)
importance_bw = rf_bw.feature_importances_

# Train RF on EMG
print("Training RF on EMG noise...")
rf_emg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_emg.fit(X_sub, y_emg_sub)
importance_emg = rf_emg.feature_importances_

# Average importance
importance_avg = (importance_bw + importance_emg) / 2

# Split bitplanes vs HF
bitplane_importance = importance_avg[:1080]
hf_importance = importance_avg[1080:]

print("\n" + "="*80)
print("FEATURE IMPORTANCE STATISTICS")
print("="*80)

print(f"\nBitplanes (1080 features):")
print(f"  Total importance:    {bitplane_importance.sum():.6f}")
print(f"  Mean importance:     {bitplane_importance.mean():.6f}")
print(f"  Max importance:      {bitplane_importance.max():.6f}")
print(f"  Median importance:   {np.median(bitplane_importance):.6f}")
print(f"  Std importance:      {bitplane_importance.std():.6f}")

print(f"\nHF features (12 features):")
print(f"  Total importance:    {hf_importance.sum():.6f}")
print(f"  Mean importance:     {hf_importance.mean():.6f}")
print(f"  Max importance:      {hf_importance.max():.6f}")
print(f"  Median importance:   {np.median(hf_importance):.6f}")

# Find top bitplanes
print("\n" + "="*80)
print("TOP BITPLANES (sorted by importance)")
print("="*80)

# Sort bitplanes by importance
bitplane_indices = np.argsort(bitplane_importance)[::-1]

print(f"\n{'Rank':<6} {'Index':<8} {'Importance':<15} {'Cumulative %':<15}")
print("-"*80)

cumsum = 0
for i, idx in enumerate(bitplane_indices[:50]):  # Top 50
    imp = bitplane_importance[idx]
    cumsum += imp
    cumsum_pct = (cumsum / bitplane_importance.sum()) * 100
    print(f"{i+1:<6} {idx:<8} {imp:<15.6f} {cumsum_pct:<15.2f}%")
    
    if cumsum_pct > 90:
        print(f"\n... (top {i+1} bitplanes explain 90% of bitplane importance)")
        break

# Compare with HF
print("\n" + "="*80)
print("COMPARISON: Top Bitplanes vs HF Features")
print("="*80)

hf_indices = np.arange(1080, 1092)
hf_names = [
    "Band_10_30_L0", "Band_10_30_L1", "Band_10_30_L2",
    "Band_30_50_L0", "Band_30_50_L1", "Band_30_50_L2",
    "ZCR_L0", "ZCR_L1", "ZCR_L2",
    "TKEO_L0", "TKEO_L1", "TKEO_L2"
]

print(f"\n{'Feature Type':<30} {'Index':<8} {'Importance':<15}")
print("-"*80)

# Top 10 bitplanes
print("\nTop 10 Bitplanes:")
for i in range(10):
    idx = bitplane_indices[i]
    imp = bitplane_importance[idx]
    print(f"  Bitplane_{idx:<19} {idx:<8} {imp:<15.6f}")

print("\nAll HF Features:")
for i, (idx, name) in enumerate(zip(hf_indices, hf_names)):
    imp = importance_avg[idx]
    print(f"  {name:<26} {idx:<8} {imp:<15.6f}")

# Determine thresholds
print("\n" + "="*80)
print("FEATURE SELECTION RECOMMENDATIONS")
print("="*80)

# Strategy 1: Top-k bitplanes
for k in [10, 20, 50, 100]:
    top_k_importance = bitplane_importance[bitplane_indices[:k]].sum()
    total_importance_k = top_k_importance + hf_importance.sum()
    print(f"\nV7_{k}: Top {k} bitplanes + 12 HF = {k+12} features")
    print(f"  Bitplane importance retained: {top_k_importance:.6f} ({top_k_importance/bitplane_importance.sum()*100:.1f}% of bitplanes)")
    print(f"  Total importance:             {total_importance_k:.6f} ({total_importance_k*100:.1f}% of V4)")
    print(f"  Feature reduction:            1092 → {k+12} ({(1-k+12)/1092*100:.1f}% reduction)")

# Strategy 2: Importance threshold
print("\n" + "-"*80)
print("Alternative: Importance Threshold")
print("-"*80)

for threshold in [0.0001, 0.0005, 0.001, 0.005]:
    n_selected = np.sum(bitplane_importance >= threshold)
    selected_importance = bitplane_importance[bitplane_importance >= threshold].sum()
    total_importance_th = selected_importance + hf_importance.sum()
    print(f"\nThreshold {threshold:.4f}: {n_selected} bitplanes + 12 HF = {n_selected+12} features")
    print(f"  Bitplane importance retained: {selected_importance:.6f} ({selected_importance/bitplane_importance.sum()*100:.1f}% of bitplanes)")
    print(f"  Total importance:             {total_importance_th:.6f} ({total_importance_th*100:.1f}% of V4)")
    print(f"  Feature reduction:            1092 → {n_selected+12} ({(1092-n_selected-12)/1092*100:.1f}% reduction)")

# Visualize
print("\n" + "="*80)
print("GENERATING PLOTS...")
print("="*80)

Path("plots/v7_feature_selection").mkdir(parents=True, exist_ok=True)

# Plot 1: All feature importance (sorted)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Bitplanes
axes[0].bar(range(1080), bitplane_importance[bitplane_indices], color='steelblue', alpha=0.7)
axes[0].axhline(y=0.001, color='red', linestyle='--', label='Threshold 0.001')
axes[0].axhline(y=0.0005, color='orange', linestyle='--', label='Threshold 0.0005')
axes[0].set_xlabel("Bitplane Rank (sorted by importance)", fontsize=12)
axes[0].set_ylabel("Importance", fontsize=12)
axes[0].set_title("Bitplane Feature Importance (sorted)", fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# HF features
axes[1].bar(range(12), hf_importance, color='forestgreen', alpha=0.7)
axes[1].set_xticks(range(12))
axes[1].set_xticklabels(hf_names, rotation=45, ha='right')
axes[1].set_ylabel("Importance", fontsize=12)
axes[1].set_title("HF Feature Importance", fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("plots/v7_feature_selection/feature_importance_comparison.png", dpi=150, bbox_inches='tight')
print("  ✓ plots/v7_feature_selection/feature_importance_comparison.png")

# Plot 2: Cumulative importance
fig, ax = plt.subplots(figsize=(10, 6))
cumulative = np.cumsum(bitplane_importance[bitplane_indices])
cumulative_pct = cumulative / bitplane_importance.sum() * 100

ax.plot(range(1, len(cumulative)+1), cumulative_pct, linewidth=2, color='steelblue')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% importance')
ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% importance')
ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% importance')

# Find where we cross thresholds
idx_50 = np.where(cumulative_pct >= 50)[0][0]
idx_80 = np.where(cumulative_pct >= 80)[0][0]
idx_90 = np.where(cumulative_pct >= 90)[0][0]

ax.scatter([idx_50, idx_80, idx_90], [50, 80, 90], s=100, c='red', zorder=5)
ax.text(idx_50, 50, f'  {idx_50} features', va='center', fontsize=10)
ax.text(idx_80, 80, f'  {idx_80} features', va='center', fontsize=10)
ax.text(idx_90, 90, f'  {idx_90} features', va='center', fontsize=10)

ax.set_xlabel("Number of Top Bitplanes", fontsize=12)
ax.set_ylabel("Cumulative Importance (%)", fontsize=12)
ax.set_title("Cumulative Bitplane Importance", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/v7_feature_selection/cumulative_importance.png", dpi=150, bbox_inches='tight')
print("  ✓ plots/v7_feature_selection/cumulative_importance.png")

# Save selected indices
print("\n" + "="*80)
print("SAVING SELECTED FEATURE INDICES")
print("="*80)

# Save top-k indices for different k
for k in [10, 20, 50, 100]:
    top_k = bitplane_indices[:k]
    np.save(f"data/v7_selected_bitplanes_top{k}.npy", top_k)
    print(f"  ✓ data/v7_selected_bitplanes_top{k}.npy")

# Save threshold-based indices
for threshold in [0.0001, 0.0005, 0.001]:
    selected = np.where(bitplane_importance >= threshold)[0]
    np.save(f"data/v7_selected_bitplanes_th{threshold:.4f}.npy", selected)
    print(f"  ✓ data/v7_selected_bitplanes_th{threshold:.4f}.npy ({len(selected)} bitplanes)")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print("\nRecommendations:")
print("  1. V7_20:  Top 20 bitplanes + 12 HF = 32 features (97% reduction, ~80% bitplane importance)")
print("  2. V7_50:  Top 50 bitplanes + 12 HF = 62 features (94% reduction, ~90% bitplane importance)")
print("  3. V7_100: Top 100 bitplanes + 12 HF = 112 features (90% reduction, ~95% bitplane importance)")
print("\nNext steps:")
print("  python create_v7_dataset.py --strategy top20")
print("  python quick_test_v7.py --strategy top20")
print("="*80 + "\n")
