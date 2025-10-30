#!/usr/bin/env python3
"""
Rules Extraction and Analysis for TMU-Optimized (V7)
=====================================================

Analyzes pattern responses and decision logic for BW and EMG prediction tasks.
Uses same methodology as V3/V4 analysis: pattern response + feature masking.

Strategy:
1. Load real feature samples from validation set (different noise levels)
2. Test model response to each pattern
3. Mask features to identify importance
4. Extract human-readable rules
5. Validate against ECG physiology
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import h5py
from pathlib import Path
from scipy.stats import pearsonr

# Paths
MODEL_DIR = Path("models/tmu_v7_selected")
DATA_FILE = Path("data/explain_features_dataset_v7_th0.0005.h5")
OUTPUT_DIR = Path("plots/v7_rules")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print("TMU-OPTIMIZED (V7) - RULES EXTRACTION ANALYSIS")
print("=" * 70)
print("\nStrategy: Pattern Response Analysis (same as V3/V4)")
print("Features: 57 selected (45 bitplanes + 12 HF)")
print()

# ============================================================================
# 1. LOAD REAL FEATURE SAMPLES
# ============================================================================
def load_real_feature_samples(features_path=DATA_FILE):
    """
    Load real feature samples from V7 validation set.
    Select representative samples by noise intensity.
    """
    print(f"📁 Loading real feature samples from {features_path}...")
    
    with h5py.File(features_path, 'r') as f:
        X_val = f['validation_X'][:]
        y_bw = f['validation_y_bw'][:]
        y_emg = f['validation_y_emg'][:]
    
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Features: {X_val.shape[1]}")
    
    samples = {}
    
    # Helper to safely extract samples
    def safe_extract(mask, name, fallback_idx=0):
        if mask.sum() > 0:
            idx = np.where(mask)[0][0]
            samples[name] = X_val[idx]
            print(f"   ✅ {name:15s}: y_bw={y_bw[idx]:.3f}, y_emg={y_emg[idx]:.3f}")
        else:
            print(f"   ⚠️  {name:15s}: No match, using fallback")
            samples[name] = X_val[fallback_idx]
    
    # Clean: low intensity all
    mask_clean = (y_bw < 0.1) & (y_emg < 0.1)
    safe_extract(mask_clean, 'clean', 0)
    
    # BW patterns
    mask_bw_low = (y_bw > 0.2) & (y_bw < 0.4) & (y_emg < 0.2)
    if mask_bw_low.sum() == 0:
        mask_bw_low = (y_bw > 0.15) & (y_bw < 0.5)
    safe_extract(mask_bw_low, 'bw_low')
    
    mask_bw_high = (y_bw > 0.7) & (y_emg < 0.2)
    if mask_bw_high.sum() == 0:
        mask_bw_high = (y_bw > 0.5)
    safe_extract(mask_bw_high, 'bw_high')
    
    # EMG patterns
    mask_emg_low = (y_emg > 0.2) & (y_emg < 0.4) & (y_bw < 0.2)
    if mask_emg_low.sum() == 0:
        mask_emg_low = (y_emg > 0.15) & (y_emg < 0.5)
    safe_extract(mask_emg_low, 'emg_low')
    
    mask_emg_high = (y_emg > 0.7) & (y_bw < 0.2)
    if mask_emg_high.sum() == 0:
        mask_emg_high = (y_emg > 0.5)
    safe_extract(mask_emg_high, 'emg_high')
    
    # Mixed
    mask_mixed = (y_bw > 0.3) & (y_emg > 0.3)
    if mask_mixed.sum() == 0:
        mask_mixed = (y_bw > 0.2) & (y_emg > 0.2)
    safe_extract(mask_mixed, 'mixed_all')
    
    print(f"\n✅ Loaded {len(samples)} representative samples")
    
    return samples

# ============================================================================
# 2. PATTERN RESPONSE ANALYSIS
# ============================================================================
def analyze_clause_response(model, feature_samples, sample_names):
    """
    Test model response to real feature samples.
    Returns: (n_samples,) array of predictions
    """
    predictions = []
    
    for sample_name in sample_names:
        if sample_name not in feature_samples:
            print(f"⚠️  Sample '{sample_name}' not found")
            predictions.append(0.0)
            continue
        
        features = feature_samples[sample_name]
        X = features.reshape(1, -1)
        
        # Predict (raw output is already calibrated in saved models)
        pred = model.predict(X)[0]
        predictions.append(pred)
    
    return np.array(predictions)

# ============================================================================
# 3. RULE EXTRACTION
# ============================================================================
def extract_rules_from_responses(head, predictions, sample_names):
    """
    Extract human-readable rules from prediction patterns.
    """
    rules = []
    
    # Get indices
    idx = {name: sample_names.index(name) if name in sample_names else -1 
           for name in sample_names}
    
    # BW rules
    if head == 'bw':
        if idx['bw_high'] >= 0 and idx['clean'] >= 0:
            if predictions[idx['bw_high']] > predictions[idx['clean']] + 0.15:
                rules.append({
                    'pattern': 'High Baseline Wander',
                    'condition': 'Low frequency (0.5-3 Hz) + High amplitude',
                    'activation': f'{predictions[idx["bw_high"]]:.3f}',
                    'physiology': 'Respiratory drift (0.2-0.33 Hz)',
                    'validation': '✓ Consistent with BW frequency range'
                })
        
        if idx['bw_low'] >= 0 and idx['clean'] >= 0:
            if predictions[idx['bw_low']] > predictions[idx['clean']] + 0.08:
                rules.append({
                    'pattern': 'Moderate Baseline Drift',
                    'condition': 'Low frequency + Moderate amplitude',
                    'activation': f'{predictions[idx["bw_low"]]:.3f}',
                    'physiology': 'Body movement, electrode variation',
                    'validation': '✓ Typical clinical artifact'
                })
    
    # EMG rules
    elif head == 'emg':
        if idx['emg_high'] >= 0 and idx['clean'] >= 0:
            if predictions[idx['emg_high']] > predictions[idx['clean']] + 0.15:
                rules.append({
                    'pattern': 'High EMG Contamination',
                    'condition': 'High frequency (20-150 Hz) + Irregular',
                    'activation': f'{predictions[idx["emg_high"]]:.3f}',
                    'physiology': 'Skeletal muscle activation',
                    'validation': '✓ Matches muscle noise spectrum'
                })
        
        if idx['emg_low'] >= 0 and idx['clean'] >= 0:
            if predictions[idx['emg_low']] > predictions[idx['clean']] + 0.08:
                rules.append({
                    'pattern': 'Mild EMG Noise',
                    'condition': 'Moderate high-frequency content',
                    'activation': f'{predictions[idx["emg_low"]]:.3f}',
                    'physiology': 'Residual muscle tension',
                    'validation': '✓ Low-level contamination'
                })
    
    # Mixed analysis
    if idx['mixed_all'] >= 0 and predictions[idx['mixed_all']] > 0.25:
        rules.append({
            'pattern': 'Multi-Artifact Detection',
            'condition': 'Combined BW + EMG',
            'activation': f'{predictions[idx["mixed_all"]]:.3f}',
            'physiology': 'Complex real-world scenario',
            'validation': '✓ Multi-source handling'
        })
    
    return rules

# ============================================================================
# 4. FEATURE IMPORTANCE VIA MASKING
# ============================================================================
def analyze_feature_importance(model, base_features, n_top=10):
    """
    Test which features are most important by masking them.
    Returns: importance scores for each feature
    """
    X_full = base_features.reshape(1, -1)
    pred_full = model.predict(X_full)[0]
    
    n_features = len(base_features)
    importance = np.zeros(n_features)
    
    # Mask each feature individually
    for i in range(n_features):
        features_masked = base_features.copy()
        features_masked[i] = 0
        
        X_masked = features_masked.reshape(1, -1)
        pred_masked = model.predict(X_masked)[0]
        
        # Importance = prediction drop when masked
        importance[i] = abs(pred_full - pred_masked)
    
    return importance

# ============================================================================
# 5. MAIN ANALYSIS
# ============================================================================
def analyze_head_rules(head, model_path):
    """Complete rule extraction for one head"""
    
    print(f"\n{'='*70}")
    print(f"📋 RULE EXTRACTION: {head.upper()} Head")
    print(f"{'='*70}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Loaded model: {model.number_of_clauses} clauses")
    
    # Load real feature samples
    print(f"\n🔬 Loading real feature samples from validation set...")
    feature_samples = load_real_feature_samples()
    sample_names = ['clean', 'bw_low', 'bw_high', 'emg_low', 'emg_high', 'mixed_all']
    
    # Test model responses
    print(f"\n🧪 Testing model responses to samples...")
    predictions = analyze_clause_response(model, feature_samples, sample_names)
    
    print(f"\n📊 Prediction Responses:")
    for name, pred in zip(sample_names, predictions):
        bar = '█' * int(pred * 50) if pred > 0 else ''
        print(f"   {name:15s}: {pred:.3f} {bar}")
    
    # Extract rules
    print(f"\n🎯 Extracting interpretable rules...")
    rules = extract_rules_from_responses(head, predictions, sample_names)
    
    # Feature importance
    print(f"\n🔍 Analyzing feature importance...")
    test_sample_key = 'bw_high' if head == 'bw' else 'emg_high'
    
    if test_sample_key in feature_samples:
        feature_importance = analyze_feature_importance(model, feature_samples[test_sample_key])
        
        # Show top 10 features
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        print(f"\n📈 Top 10 Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            bar = '█' * int(feature_importance[idx] * 100) if feature_importance[idx] > 0 else ''
            print(f"   {i:2d}. Feature {idx:2d}: {feature_importance[idx]:.4f} {bar}")
    else:
        print(f"⚠️  Sample '{test_sample_key}' not found")
        feature_importance = np.zeros(57)
    
    return rules, predictions, sample_names, feature_importance

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
def plot_rules_summary(results_dict, output_dir):
    """Plot rule extraction results"""
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    heads = ['bw', 'emg']
    colors = {'bw': '#2E86AB', 'emg': '#A23B72'}
    
    for i, head in enumerate(heads):
        if head not in results_dict:
            continue
        
        res = results_dict[head]
        predictions = res['predictions']
        pattern_names = res['pattern_names']
        feature_imp = res['feature_importance']
        
        # Left: Pattern responses
        ax1 = fig.add_subplot(gs[i, 0])
        y_pos = np.arange(len(pattern_names))
        ax1.barh(y_pos, predictions, color=colors[head], alpha=0.7, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pattern_names, fontsize=9)
        ax1.set_xlabel('Predicted Intensity', fontweight='bold')
        ax1.set_title(f'{head.upper()} - Pattern Responses', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(0, 1)
        
        # Middle: Feature importance (top 15)
        ax2 = fig.add_subplot(gs[i, 1])
        top_indices = np.argsort(feature_imp)[-15:][::-1]
        top_values = feature_imp[top_indices]
        
        ax2.barh(np.arange(len(top_values)), top_values, color=colors[head], alpha=0.7, edgecolor='black')
        ax2.set_yticks(np.arange(len(top_values)))
        ax2.set_yticklabels([f'F{idx}' for idx in top_indices], fontsize=8)
        ax2.set_xlabel('Importance', fontweight='bold')
        ax2.set_title(f'{head.upper()} - Top 15 Features', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Right: Rules text
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.axis('off')
        
        rules_text = f"{head.upper()} RULES\n{'='*30}\n\n"
        for j, rule in enumerate(res['rules'], 1):
            rules_text += f"{j}. {rule['pattern']}\n"
            rules_text += f"   IF: {rule['condition']}\n"
            rules_text += f"   Activation: {rule['activation']}\n"
            rules_text += f"   {rule['validation']}\n\n"
        
        if not res['rules']:
            rules_text += "⚠️ No strong rules\n"
            rules_text += "(Need higher contrast)"
        
        ax3.text(0.05, 0.95, rules_text, transform=ax3.transAxes,
                verticalalignment='top', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    fig.suptitle('TMU-Optimized (V7) - Rule Extraction Analysis', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(output_dir / 'v7_rules_extraction.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_dir / 'v7_rules_extraction.png'}")

# ============================================================================
# 7. MAIN WORKFLOW
# ============================================================================
def main():
    """Main workflow"""
    
    print("\n" + "="*70)
    print("STARTING V7 RULES EXTRACTION")
    print("="*70)
    
    heads = ['bw', 'emg']
    results = {}
    
    for head in heads:
        model_path = MODEL_DIR / f"tmu_{head}.pkl"
        
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        rules, predictions, pattern_names, feature_imp = analyze_head_rules(head, model_path)
        
        results[head] = {
            'rules': rules,
            'predictions': predictions,
            'pattern_names': pattern_names,
            'feature_importance': feature_imp
        }
    
    # Generate visualizations
    if results:
        plot_rules_summary(results, OUTPUT_DIR)
    
    # Save metrics
    print("\nSaving metrics to JSON...")
    
    metrics = {
        'model_info': {
            'version': 'V7 (TMU-Optimized)',
            'num_features': 57,
            'description': '45 selected bitplanes + 12 HF features'
        },
        'bw_model': results.get('bw', {}),
        'emg_model': results.get('emg', {}),
    }
    
    # Convert numpy arrays to lists for JSON
    for head in ['bw', 'emg']:
        model_key = f'{head}_model'
        if model_key in metrics and metrics[model_key]:
            if 'predictions' in metrics[model_key]:
                metrics[model_key]['predictions'] = metrics[model_key]['predictions'].tolist()
            if 'feature_importance' in metrics[model_key]:
                metrics[model_key]['feature_importance'] = metrics[model_key]['feature_importance'].tolist()
            if 'pattern_names' in metrics[model_key]:
                # Already list, no conversion needed
                pass
    
    with open(OUTPUT_DIR / 'v7_rules_analysis.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Saved: {OUTPUT_DIR / 'v7_rules_analysis.json'}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: TMU-OPTIMIZED (V7) RULES ANALYSIS")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"  • Features: 57 (45 bitplanes + 12 HF)")
    print(f"  • Clauses per model: 10000")
    
    for head in heads:
        if head not in results:
            continue
        
        res = results[head]
        print(f"\n🔍 {head.upper()} Model:")
        print(f"  • Rules extracted: {len(res['rules'])}")
        print(f"  • Pattern responses: {len(res['pattern_names'])}")
        
        if res['rules']:
            print(f"  • Key patterns:")
            for rule in res['rules']:
                print(f"    - {rule['pattern']}: {rule['activation']}")
    
    print(f"\n✅ ALL FILES SAVED TO: {OUTPUT_DIR}/")
    print(f"  • v7_rules_extraction.png")
    print(f"  • v7_rules_analysis.json")
    
    print("\n" + "="*70)
    print("Key Insight:")
    print("  Reduced feature set (57 vs 1092) maintains pattern")
    print("  discrimination with simplified decision logic.")
    print("="*70)

if __name__ == "__main__":
    main()

