#!/usr/bin/env python3
"""
Enhanced Clause Weights Analysis V2
====================================

Analisi completa dei pesi delle clauses per evidenziare
la natura WHITE-BOX delle Tsetlin Machines.

Analisi:
1. **Distribuzione pesi**: Histogram + KDE
2. **Concentrazione**: Gini coefficient (sparità)
3. **Top-K clauses**: Quali clauses dominano?
4. **Cumulative weight**: 80/20 rule (Pareto)?
5. **Sparsità**: Quante clauses sono effettivamente attive?
6. **Weight magnitude**: Confronto positive vs negative

Per tesi:
- Dimostra interpretability via sparse representations
- Mostra che poche clauses spiegano maggior parte predizioni
- Evidenzia differenze BW vs EMG vs PLI

Usage:
    python explain_weights_simple_v2.py \
      --models models/tmu_v2_final \
      --output plots/weights_v2/ \
      --heads bw emg pli
"""

import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11


def compute_gini(weights):
    """
    Compute Gini coefficient for weight distribution.
    
    Gini = 0: perfect equality (all weights same)
    Gini = 1: perfect inequality (one weight has all mass)
    
    For interpretability: higher Gini = sparser = more interpretable
    """
    # Sort weights in ascending order
    sorted_weights = np.sort(np.abs(weights))
    n = len(sorted_weights)
    
    # Compute cumulative sum
    cumsum = np.cumsum(sorted_weights)
    
    # Gini coefficient formula
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_weights))) / (n * cumsum[-1]) - (n + 1) / n
    
    return gini


def load_model_weights(model_path):
    """Load TMU model and extract clause weights."""
    print(f"\n📦 Loading model: {model_path.name}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Extract weights from weight_bank
    if hasattr(model, 'weight_bank'):
        # TMURegressor has weight_bank
        weights = model.weight_bank.get_weights()
    elif hasattr(model, 'weights'):
        # Fallback for other structures
        weights = model.weights
    else:
        raise AttributeError("Model has no accessible weights")
    
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"   Mean weight: {weights.mean():.4f}")
    print(f"   Std weight: {weights.std():.4f}")
    
    return weights


def compute_weight_statistics(weights):
    """Compute comprehensive weight statistics."""
    stats = {}
    
    # Basic stats
    stats['n_clauses'] = len(weights)
    stats['mean'] = float(weights.mean())
    stats['std'] = float(weights.std())
    stats['min'] = float(weights.min())
    stats['max'] = float(weights.max())
    stats['median'] = float(np.median(weights))
    
    # Percentiles
    stats['p25'] = float(np.percentile(weights, 25))
    stats['p75'] = float(np.percentile(weights, 75))
    stats['p90'] = float(np.percentile(weights, 90))
    stats['p99'] = float(np.percentile(weights, 99))
    
    # Sparsity (weights close to zero)
    threshold = 0.01 * abs(weights).max()  # 1% of max weight
    stats['n_near_zero'] = int(np.sum(np.abs(weights) < threshold))
    stats['sparsity_ratio'] = float(stats['n_near_zero'] / len(weights))
    
    # Active clauses (|weight| > threshold)
    stats['n_active'] = int(len(weights) - stats['n_near_zero'])
    stats['active_ratio'] = float(stats['n_active'] / len(weights))
    
    # Positive vs negative
    stats['n_positive'] = int(np.sum(weights > 0))
    stats['n_negative'] = int(np.sum(weights < 0))
    stats['positive_ratio'] = float(stats['n_positive'] / len(weights))
    
    # Gini coefficient (concentration measure)
    # Gini = 0: perfectly uniform, Gini = 1: one clause has all weight
    abs_weights = np.abs(weights)
    stats['gini_coefficient'] = float(compute_gini(weights))
    
    # Effective rank (entropy-based)
    # High entropy → weights distributed, Low entropy → concentrated
    abs_weights_norm = abs_weights / abs_weights.sum() if abs_weights.sum() > 0 else abs_weights
    entropy = -np.sum(abs_weights_norm * np.log(abs_weights_norm + 1e-10))
    stats['entropy'] = float(entropy)
    stats['effective_rank'] = float(np.exp(entropy))  # Number of "effective" clauses
    
    # Top-K concentration (Pareto principle)
    sorted_abs = np.sort(np.abs(weights))[::-1]
    cumsum = np.cumsum(sorted_abs)
    total = cumsum[-1]
    
    stats['top10_weight_pct'] = float(100 * cumsum[9] / total) if len(cumsum) > 10 else 100.0
    stats['top20_weight_pct'] = float(100 * cumsum[19] / total) if len(cumsum) > 20 else 100.0
    stats['top50_weight_pct'] = float(100 * cumsum[49] / total) if len(cumsum) > 50 else 100.0
    stats['top100_weight_pct'] = float(100 * cumsum[99] / total) if len(cumsum) > 100 else 100.0
    
    # Find index where 80% weight is reached (80/20 rule)
    idx_80 = np.where(cumsum >= 0.8 * total)[0]
    stats['n_clauses_for_80pct'] = int(idx_80[0] + 1) if len(idx_80) > 0 else len(weights)
    stats['pareto_ratio_80'] = float(stats['n_clauses_for_80pct'] / len(weights))
    
    return stats


def plot_weight_analysis(weights, head, output_dir, stats):
    """
    Comprehensive weight visualization:
    1. Distribution (histogram + KDE)
    2. Top-K bar chart
    3. Cumulative weight curve (Pareto)
    4. Positive vs Negative comparison
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{head.upper()} Head - Clause Weights Analysis", fontsize=16, fontweight='bold')
    
    # 1. Distribution
    ax = axes[0, 0]
    
    # Histogram
    ax.hist(weights, bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True, label='Histogram')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(weights)
    x_range = np.linspace(weights.min(), weights.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero')
    ax.axvline(stats['mean'], color='green', linestyle='--', alpha=0.7, linewidth=1.5, label=f"Mean ({stats['mean']:.3f})")
    ax.axvline(stats['median'], color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f"Median ({stats['median']:.3f})")
    
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Density")
    ax.set_title("Weight Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Top-K clauses
    ax = axes[0, 1]
    
    top_k = 20
    abs_weights = np.abs(weights)
    top_indices = np.argsort(abs_weights)[-top_k:][::-1]
    top_values = weights[top_indices]
    
    colors = ['green' if v > 0 else 'red' for v in top_values]
    bars = ax.barh(range(top_k), np.abs(top_values), color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"Clause {idx}\n({weights[idx]:+.3f})" for idx in top_indices], fontsize=9)
    ax.set_xlabel("|Weight|")
    ax.set_title(f"Top-{top_k} Most Important Clauses")
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive (vote YES)'),
        Patch(facecolor='red', alpha=0.7, label='Negative (vote NO)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # 3. Cumulative weight (Pareto)
    ax = axes[1, 0]
    
    sorted_abs = np.sort(np.abs(weights))[::-1]
    cumsum = np.cumsum(sorted_abs)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    
    ax.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, linewidth=2, color='navy')
    ax.fill_between(range(1, len(cumsum_pct) + 1), 0, cumsum_pct, alpha=0.3, color='navy')
    
    # 80% line
    ax.axhline(80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
    ax.axvline(stats['n_clauses_for_80pct'], color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Annotate
    ax.text(stats['n_clauses_for_80pct'], 82, 
            f"{stats['n_clauses_for_80pct']} clauses\n({stats['pareto_ratio_80']*100:.1f}%)",
            fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel("Number of Clauses (ranked by |weight|)")
    ax.set_ylabel("Cumulative Weight (%)")
    ax.set_title("Pareto Chart: Cumulative Weight Contribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(weights))
    ax.set_ylim(0, 105)
    
    # 4. Positive vs Negative weights
    ax = axes[1, 1]
    
    pos_weights = weights[weights > 0]
    neg_weights = weights[weights < 0]
    
    data_to_plot = [pos_weights, np.abs(neg_weights)]
    labels = [f'Positive\n(n={len(pos_weights)})', f'Negative\n(n={len(neg_weights)})']
    colors_box = ['green', 'red']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel("|Weight|")
    ax.set_title("Positive vs Negative Weights Distribution")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
Gini Coefficient: {stats['gini_coefficient']:.3f}
Effective Rank: {stats['effective_rank']:.1f}/{stats['n_clauses']}
Sparsity: {stats['sparsity_ratio']*100:.1f}%
Active Clauses: {stats['n_active']} ({stats['active_ratio']*100:.1f}%)
    """.strip()
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{head}_weights_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    
    plt.close()


def save_statistics_json(stats, head, output_dir):
    """Save statistics to JSON."""
    output_dir = Path(output_dir)
    output_path = output_dir / f"{head}_weight_stats.json"
    
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✅ Saved: {output_path}")


def analyze_head(head, models_dir, output_dir):
    """Main analysis for one head."""
    print(f"\n{'='*80}")
    print(f"🔬 ANALYZING {head.upper()} HEAD")
    print(f"{'='*80}")
    
    # Load - try V7 naming first, fallback to V3/V4
    model_path = Path(models_dir) / f"tmu_{head}.pkl"
    if not model_path.exists():
        model_path = Path(models_dir) / f"rtm_intensity_{head}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    weights = load_model_weights(model_path)
    
    # Statistics
    stats = compute_weight_statistics(weights)
    
    print(f"\n📊 Weight Statistics:")
    print(f"   Total clauses:       {stats['n_clauses']}")
    print(f"   Active clauses:      {stats['n_active']} ({stats['active_ratio']*100:.1f}%)")
    print(f"   Sparsity:            {stats['sparsity_ratio']*100:.1f}%")
    print(f"   Gini coefficient:    {stats['gini_coefficient']:.3f} (0=uniform, 1=concentrated)")
    print(f"   Effective rank:      {stats['effective_rank']:.1f} / {stats['n_clauses']}")
    print(f"   Positive/Negative:   {stats['n_positive']}/{stats['n_negative']} ({stats['positive_ratio']*100:.1f}%/{100-stats['positive_ratio']*100:.1f}%)")
    print(f"\n   Pareto (80% weight): {stats['n_clauses_for_80pct']} clauses ({stats['pareto_ratio_80']*100:.1f}%)")
    print(f"   Top-10 weight:       {stats['top10_weight_pct']:.1f}%")
    print(f"   Top-20 weight:       {stats['top20_weight_pct']:.1f}%")
    print(f"   Top-50 weight:       {stats['top50_weight_pct']:.1f}%")
    
    # Plot
    plot_weight_analysis(weights, head, output_dir, stats)
    
    # Save JSON
    save_statistics_json(stats, head, output_dir)
    
    return stats


def plot_comparison(all_stats, output_dir):
    """Plot comparison across heads."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Clause Weights Comparison Across Heads", fontsize=16, fontweight='bold')
    
    heads = list(all_stats.keys())
    
    # 1. Sparsity comparison
    ax = axes[0]
    sparsity = [all_stats[h]['sparsity_ratio'] * 100 for h in heads]
    active = [all_stats[h]['active_ratio'] * 100 for h in heads]
    
    x = np.arange(len(heads))
    width = 0.35
    
    ax.bar(x - width/2, sparsity, width, label='Sparse (%)', color='lightcoral', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, active, width, label='Active (%)', color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels([h.upper() for h in heads])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Sparsity vs Active Clauses")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. Gini coefficient comparison
    ax = axes[1]
    gini_vals = [all_stats[h]['gini_coefficient'] for h in heads]
    
    colors = sns.color_palette("RdYlGn_r", len(heads))
    bars = ax.bar(heads, gini_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Weight Concentration (Gini)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, gini_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11)
    
    # 3. Pareto (80% rule)
    ax = axes[2]
    pareto = [all_stats[h]['pareto_ratio_80'] * 100 for h in heads]
    
    colors_pareto = sns.color_palette("Blues_r", len(heads))
    bars = ax.bar(heads, pareto, color=colors_pareto, alpha=0.8, edgecolor='black')
    ax.set_ylabel("% of Clauses")
    ax.set_title("Clauses Needed for 80% Weight (Pareto)")
    ax.axhline(20, color='red', linestyle='--', alpha=0.7, linewidth=2, label='80/20 Rule')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, pareto):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    output_path = output_dir / "comparison_all_heads.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Clause Weights Analysis V2")
    parser.add_argument("--models", required=True, help="Directory containing trained models")
    parser.add_argument("--output", required=True, help="Output directory for plots/JSON")
    parser.add_argument("--heads", nargs="+", default=["bw", "emg", "pli"],
                        help="Heads to analyze (default: bw emg pli)")
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 CLAUSE WEIGHTS ANALYSIS V2")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Output: {args.output}")
    print(f"Heads:  {args.heads}")
    
    # Analyze each head
    all_stats = {}
    for head in args.heads:
        try:
            stats = analyze_head(head, args.models, args.output)
            all_stats[head] = stats
        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping {head}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error analyzing {head}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Comparison
    if len(all_stats) > 1:
        plot_comparison(all_stats, args.output)
    
    print("\n" + "="*80)
    print("✅ WEIGHTS ANALYSIS COMPLETE")
    print("="*80)
    print(f"📁 Results saved to: {args.output}")
    print("\n🎓 Thesis usage:")
    print("   - Use Gini coefficient to show sparse representations")
    print("   - Pareto chart demonstrates interpretability (few clauses explain most)")
    print("   - Positive/Negative balance shows voting dynamics")
    print("   - Compare across heads to show domain-specific learning")


if __name__ == "__main__":
    main()
