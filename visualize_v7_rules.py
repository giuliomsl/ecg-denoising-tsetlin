"""
Visualizzazione Completa Rules Extraction V7
============================================
Genera plot dettagliati delle regole estratte dal modello V7.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configurazione plot
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

def load_rules_data():
    """Carica i dati dell'analisi delle regole V7."""
    with open('plots/v7_rules/v7_rules_analysis.json', 'r') as f:
        data = json.load(f)
    return data

def identify_feature_type(feature_idx, num_bitplanes=45):
    """
    Identifica il tipo di feature.
    0-44: Bitplanes (45 selezionati)
    45-56: HF features (12)
    """
    if feature_idx < num_bitplanes:
        return 'BP', f'BP_{feature_idx}'
    else:
        hf_idx = feature_idx - num_bitplanes
        hf_names = ['BW', 'PLI50', 'PLI60', 'EMG_low', 'EMG_mid', 'EMG_high', 
                    'HFNB1', 'HFNB2', 'HFNB3', 'HFNB4', 'HFNB5', 'HFNB6']
        return 'HF', hf_names[hf_idx] if hf_idx < len(hf_names) else f'HF_{hf_idx}'

def plot_v7_rules_comprehensive(data):
    """Genera visualizzazione completa delle regole V7."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Estrai importanze
    bw_importance = np.array(data['bw_model']['feature_importance'])
    emg_importance = np.array(data['emg_model']['feature_importance'])
    
    # 1. TOP 20 FEATURES - BW
    ax1 = fig.add_subplot(gs[0, 0])
    top20_bw_idx = np.argsort(bw_importance)[-20:][::-1]
    colors_bw = ['#e74c3c' if i >= 45 else '#3498db' for i in top20_bw_idx]
    
    ax1.barh(range(20), bw_importance[top20_bw_idx], color=colors_bw, alpha=0.7)
    ax1.set_yticks(range(20))
    feature_labels_bw = [identify_feature_type(i)[1] for i in top20_bw_idx]
    ax1.set_yticklabels(feature_labels_bw, fontsize=8)
    ax1.set_xlabel('Feature Importance', fontweight='bold')
    ax1.set_title('Top 20 Features - BW Model', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. TOP 20 FEATURES - EMG
    ax2 = fig.add_subplot(gs[0, 1])
    top20_emg_idx = np.argsort(emg_importance)[-20:][::-1]
    colors_emg = ['#e74c3c' if i >= 45 else '#3498db' for i in top20_emg_idx]
    
    ax2.barh(range(20), emg_importance[top20_emg_idx], color=colors_emg, alpha=0.7)
    ax2.set_yticks(range(20))
    feature_labels_emg = [identify_feature_type(i)[1] for i in top20_emg_idx]
    ax2.set_yticklabels(feature_labels_emg, fontsize=8)
    ax2.set_xlabel('Feature Importance', fontweight='bold')
    ax2.set_title('Top 20 Features - EMG Model', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. DISTRIBUZIONE BP vs HF
    ax3 = fig.add_subplot(gs[0, 2])
    
    bp_importance_bw = bw_importance[:45]
    hf_importance_bw = bw_importance[45:]
    bp_importance_emg = emg_importance[:45]
    hf_importance_emg = emg_importance[45:]
    
    positions = [1, 2, 4, 5]
    box_data = [bp_importance_bw, hf_importance_bw, bp_importance_emg, hf_importance_emg]
    bp = ax3.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=True, notch=True)
    
    colors_box = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['BP\n(BW)', 'HF\n(BW)', 'BP\n(EMG)', 'HF\n(EMG)'], fontsize=9)
    ax3.set_ylabel('Feature Importance', fontweight='bold')
    ax3.set_title('BP vs HF Importance Distribution', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axvline(3, color='gray', linestyle='--', alpha=0.5)
    
    # Statistics
    stats_text = f"BW: HF mean={np.mean(hf_importance_bw):.4f}, BP mean={np.mean(bp_importance_bw):.5f}\n"
    stats_text += f"EMG: HF mean={np.mean(hf_importance_emg):.4f}, BP mean={np.mean(bp_importance_emg):.5f}"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. CUMULATIVE IMPORTANCE - BW
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_bw = np.sort(bw_importance)[::-1]
    cumsum_bw = np.cumsum(sorted_bw) / np.sum(sorted_bw) * 100
    
    ax4.plot(range(1, len(cumsum_bw)+1), cumsum_bw, 'b-', linewidth=2, label='Cumulative')
    ax4.axhline(80, color='orange', linestyle='--', label='80%', linewidth=1.5)
    ax4.axhline(90, color='red', linestyle='--', label='90%', linewidth=1.5)
    
    # Find thresholds
    n80_bw = np.argmax(cumsum_bw >= 80) + 1
    n90_bw = np.argmax(cumsum_bw >= 90) + 1
    ax4.axvline(n80_bw, color='orange', linestyle=':', alpha=0.5)
    ax4.axvline(n90_bw, color='red', linestyle=':', alpha=0.5)
    
    ax4.text(n80_bw, 78, f'{n80_bw} feat', fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    ax4.text(n90_bw, 88, f'{n90_bw} feat', fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax4.set_xlabel('Number of Features', fontweight='bold')
    ax4.set_ylabel('Cumulative Importance (%)', fontweight='bold')
    ax4.set_title('Cumulative Feature Importance - BW', fontweight='bold', fontsize=12)
    ax4.legend(loc='lower right')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 57)
    
    # 5. CUMULATIVE IMPORTANCE - EMG
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_emg = np.sort(emg_importance)[::-1]
    cumsum_emg = np.cumsum(sorted_emg) / np.sum(sorted_emg) * 100
    
    ax5.plot(range(1, len(cumsum_emg)+1), cumsum_emg, 'g-', linewidth=2, label='Cumulative')
    ax5.axhline(80, color='orange', linestyle='--', label='80%', linewidth=1.5)
    ax5.axhline(90, color='red', linestyle='--', label='90%', linewidth=1.5)
    
    n80_emg = np.argmax(cumsum_emg >= 80) + 1
    n90_emg = np.argmax(cumsum_emg >= 90) + 1
    ax5.axvline(n80_emg, color='orange', linestyle=':', alpha=0.5)
    ax5.axvline(n90_emg, color='red', linestyle=':', alpha=0.5)
    
    ax5.text(n80_emg, 78, f'{n80_emg} feat', fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    ax5.text(n90_emg, 88, f'{n90_emg} feat', fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax5.set_xlabel('Number of Features', fontweight='bold')
    ax5.set_ylabel('Cumulative Importance (%)', fontweight='bold')
    ax5.set_title('Cumulative Feature Importance - EMG', fontweight='bold', fontsize=12)
    ax5.legend(loc='lower right')
    ax5.grid(alpha=0.3)
    ax5.set_xlim(0, 57)
    
    # 6. HEATMAP IMPORTANCE
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Crea matrice 57x2 (features x models)
    importance_matrix = np.column_stack([bw_importance, emg_importance])
    
    im = ax6.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['BW', 'EMG'], fontweight='bold')
    ax6.set_xlabel('Feature Index', fontweight='bold')
    ax6.set_title('Feature Importance Heatmap', fontweight='bold', fontsize=12)
    
    # Aggiungi separatore BP/HF
    ax6.axvline(44.5, color='white', linestyle='--', linewidth=2, label='BP/HF boundary')
    ax6.text(22, -0.5, 'Bitplanes (45)', ha='center', fontsize=9, color='blue', fontweight='bold')
    ax6.text(50, -0.5, 'HF (12)', ha='center', fontsize=9, color='red', fontweight='bold')
    
    plt.colorbar(im, ax=ax6, label='Importance')
    
    # 7. PATTERN RESPONSES - BW
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Estrai pattern names e predictions
    pattern_names = data['bw_model'].get('pattern_names', [])
    bw_predictions = data['bw_model'].get('predictions', [])
    
    if pattern_names and bw_predictions:
        # Mappa i nomi a etichette leggibili
        label_map = {
            'clean': 'Clean',
            'bw_low': 'BW Low',
            'bw_high': 'BW High',
            'emg_low': 'EMG Low',
            'emg_high': 'EMG High',
            'mixed_all': 'Mixed'
        }
        labels = [label_map.get(p, p) for p in pattern_names]
        colors_map = {
            'Clean': '#2ecc71',
            'BW Low': '#f39c12',
            'BW High': '#e67e22',
            'EMG Low': '#9b59b6',
            'EMG High': '#8e44ad',
            'Mixed': '#e74c3c'
        }
        colors = [colors_map.get(l, '#95a5a6') for l in labels]
        
        y_pos = np.arange(len(labels))
        bars = ax7.barh(y_pos, bw_predictions, color=colors, alpha=0.7)
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(labels, fontweight='bold', fontsize=9)
        ax7.set_xlabel('Predicted BW Intensity', fontweight='bold')
        ax7.set_title('Pattern Responses - BW Model', fontweight='bold', fontsize=12)
        ax7.grid(axis='x', alpha=0.3)
        ax7.invert_yaxis()
        
        # Valori sulle barre
        for i, val in enumerate(bw_predictions):
            ax7.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8)
    
    # 8. PATTERN RESPONSES - EMG
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Estrai pattern names e predictions per EMG
    emg_pattern_names = data['emg_model'].get('pattern_names', [])
    emg_predictions = data['emg_model'].get('predictions', [])
    
    if emg_pattern_names and emg_predictions:
        # Mappa i nomi a etichette leggibili
        label_map = {
            'clean': 'Clean',
            'bw_low': 'BW Low',
            'bw_high': 'BW High',
            'emg_low': 'EMG Low',
            'emg_high': 'EMG High',
            'mixed_all': 'Mixed'
        }
        labels = [label_map.get(p, p) for p in emg_pattern_names]
        colors_map = {
            'Clean': '#2ecc71',
            'BW Low': '#f39c12',
            'BW High': '#e67e22',
            'EMG Low': '#9b59b6',
            'EMG High': '#8e44ad',
            'Mixed': '#e74c3c'
        }
        colors = [colors_map.get(l, '#95a5a6') for l in labels]
        
        y_pos = np.arange(len(labels))
        bars = ax8.barh(y_pos, emg_predictions, color=colors, alpha=0.7)
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(labels, fontweight='bold', fontsize=9)
        ax8.set_xlabel('Predicted EMG Intensity', fontweight='bold')
        ax8.set_title('Pattern Responses - EMG Model', fontweight='bold', fontsize=12)
        ax8.grid(axis='x', alpha=0.3)
        ax8.invert_yaxis()
        
        # Valori sulle barre
        for i, val in enumerate(emg_predictions):
            ax8.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8)
    
    # 9. SUMMARY STATISTICS
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calcola statistiche
    top10_bw = top20_bw_idx[:10]
    top10_emg = top20_emg_idx[:10]
    
    hf_in_top10_bw = np.sum(top10_bw >= 45)
    hf_in_top10_emg = np.sum(top10_emg >= 45)
    
    bp_total_bw = np.sum(bp_importance_bw)
    hf_total_bw = np.sum(hf_importance_bw)
    bp_total_emg = np.sum(bp_importance_emg)
    hf_total_emg = np.sum(hf_importance_emg)
    
    summary_text = "📊 V7 RULES SUMMARY\n" + "="*40 + "\n\n"
    summary_text += "🎯 FEATURE SELECTION IMPACT:\n"
    summary_text += f"  • V7: 57 features (45 BP + 12 HF)\n"
    summary_text += f"  • V4: 1092 features (1080 BP + 12 HF)\n"
    summary_text += f"  • Reduction: 94.8%\n\n"
    
    summary_text += "🔝 TOP 10 FEATURES:\n"
    summary_text += f"  BW:  {hf_in_top10_bw}/10 HF features ({hf_in_top10_bw*10}%)\n"
    summary_text += f"  EMG: {hf_in_top10_emg}/10 HF features ({hf_in_top10_emg*10}%)\n\n"
    
    summary_text += "📈 IMPORTANCE TOTALS:\n"
    summary_text += f"  BW:  BP={bp_total_bw:.3f} | HF={hf_total_bw:.3f}\n"
    summary_text += f"       HF is {hf_total_bw/bp_total_bw:.1f}x BP\n"
    summary_text += f"  EMG: BP={bp_total_emg:.3f} | HF={hf_total_emg:.3f}\n"
    summary_text += f"       HF is {hf_total_emg/bp_total_emg:.1f}x BP\n\n"
    
    summary_text += "🎯 COMPACTNESS:\n"
    summary_text += f"  BW:  {n80_bw} features → 80% importance\n"
    summary_text += f"       {n90_bw} features → 90% importance\n"
    summary_text += f"  EMG: {n80_emg} features → 80% importance\n"
    summary_text += f"       {n90_emg} features → 90% importance\n\n"
    
    summary_text += "✅ FINDINGS:\n"
    summary_text += "  • HF features dominate (~60x BP)\n"
    summary_text += "  • 4-5 features capture 80% importance\n"
    summary_text += "  • Selected bitplanes contribute ~4%\n"
    summary_text += "  • V7 rules more interpretable than V4"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('TMU V7 - Comprehensive Rules Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    output_path = 'plots/v7_rules/v7_rules_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot salvato: {output_path}")
    
    plt.close()
    
    return {
        'n80_bw': n80_bw,
        'n90_bw': n90_bw,
        'n80_emg': n80_emg,
        'n90_emg': n90_emg,
        'hf_in_top10_bw': hf_in_top10_bw,
        'hf_in_top10_emg': hf_in_top10_emg
    }

def main():
    print("\n" + "="*60)
    print("🔍 VISUALIZZAZIONE RULES V7")
    print("="*60 + "\n")
    
    # Carica dati
    print("📂 Caricamento dati rules V7...")
    data = load_rules_data()
    
    # Genera plot comprehensive
    print("\n📊 Generazione plot comprehensive...")
    stats = plot_v7_rules_comprehensive(data)
    
    print("\n" + "="*60)
    print("✅ VISUALIZZAZIONE V7 RULES COMPLETATA!")
    print("="*60)
    print(f"\n📁 Output: plots/v7_rules/v7_rules_comprehensive.png")
    print(f"\n📊 Key Statistics:")
    print(f"  • BW:  {stats['n80_bw']} features → 80%, {stats['n90_bw']} features → 90%")
    print(f"  • EMG: {stats['n80_emg']} features → 80%, {stats['n90_emg']} features → 90%")
    print(f"  • Top 10 BW:  {stats['hf_in_top10_bw']}/10 HF features")
    print(f"  • Top 10 EMG: {stats['hf_in_top10_emg']}/10 HF features")
    print()

if __name__ == '__main__':
    main()
