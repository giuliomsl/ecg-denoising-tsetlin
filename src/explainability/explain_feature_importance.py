#!/usr/bin/env python3
"""
Feature Importance Analysis V2 - Ablation-Based
================================================

FIXED: Non usa TA states (non accessibili in TMU library).
Usa ablation study: rimuove features e misura MAE increase.

Metodi:
1. **Ablation**: Maschera ogni feature, misura drop predizione (lento, accurato)
2. **Permutation**: Permuta feature, misura drop predizione (medio, accurato)
3. **Gradient-free sensitivity**: Noise injection (veloce, approssimato)

Per tesi: usa Ablation su subset (2000 samples) per accuracy/speed tradeoff.

Usage:
    python explain_feature_importance_v2.py \
      --models models/tmu_v2_final \
      --features data/explain_features_dataset_v2.h5 \
      --output plots/feature_importance_v2/ \
      --method ablation \
      --limit 2000 \
      --heads bw emg pli
"""

import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Seaborn styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


def load_model_and_data(models_dir, features_path, head, limit=None):
    """Load TMU model and validation data."""
    print(f"\n📦 Loading {head.upper()} model and data...")
    
    # Load model
    model_path = Path(models_dir) / f"rtm_intensity_{head}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"   ✅ Model loaded: {model_path.name}")
    
    # Load validation data
    with h5py.File(features_path, "r") as f:
        X_val = f["validation_X"][:]
        y_val = f[f"validation_y_{head}"][:]
    
    # Limit for speed
    if limit and limit < len(X_val):
        indices = np.random.RandomState(42).choice(len(X_val), limit, replace=False)
        X_val = X_val[indices]
        y_val = y_val[indices]
        print(f"   ⚠️  Limited to {limit} samples (speed optimization)")
    
    print(f"   Data: {X_val.shape}, dtype={X_val.dtype}")
    print(f"   Target: mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    
    return model, X_val, y_val


def compute_baseline_mae(model, X_val, y_val):
    """Compute baseline MAE (no ablation)."""
    print(f"\n🎯 Computing baseline MAE...")
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    print(f"   Baseline MAE: {mae:.4f}")
    return mae, pred


def ablation_importance(model, X_val, y_val, baseline_mae):
    """
    Ablation-based feature importance.
    
    For each feature:
    1. Mask (set to 0)
    2. Predict
    3. Compute MAE increase
    
    High MAE increase → important feature
    """
    print(f"\n🔬 Running ablation study...")
    n_features = X_val.shape[1]
    importance = np.zeros(n_features)
    
    for feat_idx in tqdm(range(n_features), desc="Ablating features"):
        # Mask feature
        X_masked = X_val.copy()
        X_masked[:, feat_idx] = 0
        
        # Predict with masked feature
        pred_masked = model.predict(X_masked)
        mae_masked = mean_absolute_error(y_val, pred_masked)
        
        # Importance = MAE increase
        importance[feat_idx] = mae_masked - baseline_mae
    
    print(f"   ✅ Ablation complete")
    print(f"   Importance range: [{importance.min():.6f}, {importance.max():.6f}]")
    print(f"   Mean importance: {importance.mean():.6f}")
    
    return importance


def permutation_importance(model, X_val, y_val, baseline_mae, n_repeats=5):
    """
    Permutation-based feature importance (più robusto di ablation).
    
    For each feature:
    1. Shuffle (permute) values across samples
    2. Predict
    3. Compute MAE increase (averaged over n_repeats)
    """
    print(f"\n🔀 Running permutation importance (n_repeats={n_repeats})...")
    n_features = X_val.shape[1]
    importance = np.zeros(n_features)
    
    rng = np.random.RandomState(42)
    
    for feat_idx in tqdm(range(n_features), desc="Permuting features"):
        mae_deltas = []
        
        for _ in range(n_repeats):
            # Permute feature
            X_permuted = X_val.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
            
            # Predict
            pred_permuted = model.predict(X_permuted)
            mae_permuted = mean_absolute_error(y_val, pred_permuted)
            
            mae_deltas.append(mae_permuted - baseline_mae)
        
        # Average importance over repeats
        importance[feat_idx] = np.mean(mae_deltas)
    
    print(f"   ✅ Permutation complete")
    print(f"   Importance range: [{importance.min():.6f}, {importance.max():.6f}]")
    
    return importance


def group_by_bitplane(importance, window=360, n_bits_per_sample=3, n_samples=360):
    """
    Group feature importance by bitplane.
    
    Features organization (V2 dataset):
    - Samples 0-359: thermometer encoded (3 bits each) → 1080 features
    - Samples 360-371: HF features (12 bits total, 3 per feature × 4 features)
    
    Total: 1092 features
    
    Returns:
    - bitplane_imp: [360] array (importance per sample position)
    - hf_imp: [4] array (importance per HF feature: band_10_30, band_30_50, zcr, tkeo)
    """
    # Standard bitplane features (360 samples × 3 bits = 1080)
    n_bitplane_features = n_samples * n_bits_per_sample
    
    if len(importance) < n_bitplane_features:
        raise ValueError(f"Expected ≥{n_bitplane_features} features, got {len(importance)}")
    
    # Group by sample position (average over 3 thermometer bits)
    bitplane_imp = np.zeros(n_samples)
    for sample_idx in range(n_samples):
        start = sample_idx * n_bits_per_sample
        end = start + n_bits_per_sample
        bitplane_imp[sample_idx] = importance[start:end].mean()
    
    # HF features (if present)
    hf_imp = None
    if len(importance) > n_bitplane_features:
        hf_features = importance[n_bitplane_features:]
        n_hf = len(hf_features)
        
        # Each HF feature encoded with 3 thermometer bits
        n_hf_features = n_hf // n_bits_per_sample
        hf_imp = np.zeros(n_hf_features)
        
        for hf_idx in range(n_hf_features):
            start = hf_idx * n_bits_per_sample
            end = start + n_bits_per_sample
            hf_imp[hf_idx] = hf_features[start:end].mean()
    
    return bitplane_imp, hf_imp


def plot_importance(bitplane_imp, hf_imp, head, output_dir):
    """
    Plot feature importance:
    1. Time-domain importance (360 samples)
    2. Frequency bands (derived from sample position)
    3. HF features bar chart
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{head.upper()} Head - Feature Importance Analysis", fontsize=16, fontweight='bold')
    
    # 1. Time-domain importance
    ax = axes[0, 0]
    time_axis = np.arange(len(bitplane_imp)) / 360.0  # seconds (@ 360 Hz)
    ax.plot(time_axis, bitplane_imp, linewidth=1.5, alpha=0.8)
    ax.fill_between(time_axis, 0, bitplane_imp, alpha=0.3)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Importance (MAE Δ)")
    ax.set_title("Importance Over Time Window (1 second)")
    ax.grid(True, alpha=0.3)
    
    # 2. Frequency bands (aggregate)
    ax = axes[0, 1]
    
    # Define frequency bands based on ECG/noise characteristics
    bands = {
        'Very Low\n(0-30 samples)\n0-11 Hz': (0, 30),
        'Low\n(30-90)\n11-33 Hz': (30, 90),
        'Mid\n(90-180)\n33-66 Hz': (90, 180),
        'High\n(180-270)\n66-99 Hz': (180, 270),
        'Very High\n(270-360)\n99-132 Hz': (270, 360)
    }
    
    band_names = list(bands.keys())
    band_importance = []
    
    for (start, end) in bands.values():
        band_importance.append(bitplane_imp[start:end].mean())
    
    colors = sns.color_palette("RdYlGn_r", len(band_names))
    bars = ax.bar(range(len(band_names)), band_importance, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names, fontsize=9)
    ax.set_ylabel("Mean Importance")
    ax.set_title("Importance by Frequency Band")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, band_importance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. HF Features (if available)
    ax = axes[1, 0]
    if hf_imp is not None and len(hf_imp) == 4:
        hf_names = ['Band Energy\n10-30 Hz', 'Band Energy\n30-50 Hz', 'Zero Crossing\nRate (ZCR)', 'Teager-Kaiser\nEnergy (TKEO)']
        colors_hf = sns.color_palette("viridis", 4)
        bars = ax.bar(range(4), hf_imp, color=colors_hf, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(4))
        ax.set_xticklabels(hf_names, fontsize=10)
        ax.set_ylabel("Importance (MAE Δ)")
        ax.set_title("High-Frequency Features Importance")
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, hf_imp):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.5f}',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, "No HF features available", ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 4. Top-20 most important samples
    ax = axes[1, 1]
    top_k = 20
    top_indices = np.argsort(bitplane_imp)[-top_k:][::-1]
    top_values = bitplane_imp[top_indices]
    
    colors_top = sns.color_palette("Reds_r", top_k)
    bars = ax.barh(range(top_k), top_values, color=colors_top, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"Sample {idx}\n({idx/360:.3f}s)" for idx in top_indices], fontsize=8)
    ax.set_xlabel("Importance (MAE Δ)")
    ax.set_title(f"Top-{top_k} Most Important Sample Positions")
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{head}_feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    
    plt.close()


def save_importance_json(bitplane_imp, hf_imp, head, output_dir, metadata):
    """Save importance scores to JSON for later analysis."""
    output_dir = Path(output_dir)
    
    data = {
        "head": head,
        "method": metadata["method"],
        "n_samples": metadata["n_samples"],
        "bitplane_importance": bitplane_imp.tolist(),
        "hf_importance": hf_imp.tolist() if hf_imp is not None else None,
        "statistics": {
            "bitplane_mean": float(bitplane_imp.mean()),
            "bitplane_std": float(bitplane_imp.std()),
            "bitplane_max": float(bitplane_imp.max()),
            "bitplane_max_idx": int(bitplane_imp.argmax()),
            "bitplane_max_time_sec": float(bitplane_imp.argmax() / 360.0)
        }
    }
    
    if hf_imp is not None:
        hf_names = ["band_10_30_hz", "band_30_50_hz", "zcr", "tkeo"]
        data["statistics"]["hf_features"] = {
            name: float(val) for name, val in zip(hf_names, hf_imp)
        }
    
    output_path = output_dir / f"{head}_importance.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✅ Saved: {output_path}")


def analyze_head(head, models_dir, features_path, output_dir, method="ablation", limit=None):
    """Main analysis workflow for one head."""
    print(f"\n{'='*80}")
    print(f"🔬 ANALYZING {head.upper()} HEAD")
    print(f"{'='*80}")
    
    # Load
    model, X_val, y_val = load_model_and_data(models_dir, features_path, head, limit)
    
    # Baseline
    baseline_mae, baseline_pred = compute_baseline_mae(model, X_val, y_val)
    
    # Compute importance
    if method == "ablation":
        importance = ablation_importance(model, X_val, y_val, baseline_mae)
    elif method == "permutation":
        importance = permutation_importance(model, X_val, y_val, baseline_mae, n_repeats=5)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Group by bitplane
    bitplane_imp, hf_imp = group_by_bitplane(importance)
    
    print(f"\n📊 Importance Summary:")
    print(f"   Bitplane mean: {bitplane_imp.mean():.6f}")
    print(f"   Bitplane std:  {bitplane_imp.std():.6f}")
    print(f"   Max @ sample {bitplane_imp.argmax()} ({bitplane_imp.argmax()/360:.3f}s): {bitplane_imp.max():.6f}")
    
    if hf_imp is not None:
        print(f"\n   HF Features:")
        hf_names = ["Band 10-30Hz", "Band 30-50Hz", "ZCR", "TKEO"]
        for name, val in zip(hf_names, hf_imp):
            print(f"      {name:15s}: {val:.6f}")
    
    # Plot
    plot_importance(bitplane_imp, hf_imp, head, output_dir)
    
    # Save JSON
    metadata = {
        "method": method,
        "n_samples": len(X_val),
        "baseline_mae": float(baseline_mae)
    }
    save_importance_json(bitplane_imp, hf_imp, head, output_dir, metadata)
    
    return bitplane_imp, hf_imp


def plot_comparison(results, output_dir):
    """Plot side-by-side comparison of all heads."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Feature Importance Comparison Across Heads", fontsize=16, fontweight='bold')
    
    for idx, (head, (bitplane_imp, hf_imp)) in enumerate(results.items()):
        ax = axes[idx]
        
        time_axis = np.arange(len(bitplane_imp)) / 360.0
        ax.plot(time_axis, bitplane_imp, linewidth=1.5, label=head.upper())
        ax.fill_between(time_axis, 0, bitplane_imp, alpha=0.3)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Importance (MAE Δ)")
        ax.set_title(f"{head.upper()} Head")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    output_path = output_dir / "comparison_all_heads.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Feature Importance Analysis V2 (Ablation-based)")
    parser.add_argument("--models", required=True, help="Directory containing trained models")
    parser.add_argument("--features", required=True, help="Path to features dataset (H5)")
    parser.add_argument("--output", required=True, help="Output directory for plots/JSON")
    parser.add_argument("--heads", nargs="+", default=["bw", "emg", "pli"], 
                        help="Heads to analyze (default: bw emg pli)")
    parser.add_argument("--method", choices=["ablation", "permutation"], default="ablation",
                        help="Importance computation method (default: ablation)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit validation samples for speed (default: None = all)")
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 FEATURE IMPORTANCE ANALYSIS V2 - Ablation-Based")
    print("="*80)
    print(f"Models:   {args.models}")
    print(f"Features: {args.features}")
    print(f"Output:   {args.output}")
    print(f"Heads:    {args.heads}")
    print(f"Method:   {args.method}")
    print(f"Limit:    {args.limit if args.limit else 'None (use all)'}")
    
    # Analyze each head
    results = {}
    for head in args.heads:
        try:
            bitplane_imp, hf_imp = analyze_head(
                head, args.models, args.features, args.output, 
                method=args.method, limit=args.limit
            )
            results[head] = (bitplane_imp, hf_imp)
        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping {head}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error analyzing {head}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Comparison plot
    if len(results) > 1:
        plot_comparison(results, args.output)
    
    print("\n" + "="*80)
    print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("="*80)
    print(f"📁 Results saved to: {args.output}")
    print("\n🎓 Thesis usage:")
    print("   - Use ablation method (accurate but slow)")
    print("   - Limit to 2000 samples for speed (~10-15 min per head)")
    print("   - Focus on bitplane importance for interpretability")
    print("   - HF features validate EMG detector design")


if __name__ == "__main__":
    main()
