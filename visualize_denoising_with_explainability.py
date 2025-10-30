#!/usr/bin/env python3
"""
Visualizzazione Completa Denoising + Explainability
====================================================

Genera plot publication-ready che mostra:
1. Segnale clean, noisy, denoised (confronto temporale)
2. Intensità stimate BW/EMG/PLI (explainability)
3. Analisi spettrale (FFT prima/dopo)
4. Metriche di qualità (SNR, correlazione)

Per tesi Section 5: dimostra che il sistema è interpretabile e efficace.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr

# Configurazione
INPUT_H5 = Path("data/explain_input_dataset.h5")
V7_PREDICTIONS = Path("results/v7_predictions/v7_test_predictions.npz")
OUTPUT_DIR = Path("plots/denoising_explainability_v7")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print("VISUALIZZAZIONE DENOISING + EXPLAINABILITY V7")
print("=" * 70)

# ============================================================================
# 1. CARICA DATI
# ============================================================================
print("\n[1/4] Caricamento dati...")

# Verifica file V7
if not V7_PREDICTIONS.exists():
    print(f"❌ File V7 non trovato: {V7_PREDICTIONS}")
    print("   Esegui prima l'inference V7")
    exit(1)

print(f"  📁 Input: {INPUT_H5}")
print(f"  📁 V7 Predictions: {V7_PREDICTIONS}")

# Carica dati
with h5py.File(INPUT_H5, 'r') as f:
    # Cerca test set
    if 'test_clean' in f:
        clean = f['test_clean'][:10]  # Prime 10 tracce
        noisy = f['test_noisy'][:10]
        split = 'test'
    elif 'val_clean' in f:
        clean = f['val_clean'][:10]
        noisy = f['val_noisy'][:10]
        split = 'val'
    else:
        clean = f['train_clean'][:10]
        noisy = f['train_noisy'][:10]
        split = 'train'
    
    fs = float(f.attrs.get('fs', 360.0))

print(f"  ✅ Caricati {len(clean)} segnali dal set '{split}'")
print(f"  ✅ Sampling frequency: {fs} Hz")
print(f"  ✅ Lunghezza segnale: {clean.shape[1]} samples")

# Carica predizioni V7
print(f"\n  📦 Caricamento predizioni V7...")
v7_data = np.load(V7_PREDICTIONS)
print(f"  📊 Keys disponibili: {list(v7_data.keys())}")

# Estrai intensità calibrate
y_pred_bw_cal = v7_data['y_pred_bw_cal'][:10]  # Prime 10
y_pred_emg_cal = v7_data['y_pred_emg_cal'][:10]

# Le predizioni V7 sono per-window, dobbiamo espanderle a serie temporali
# Assumiamo window=512, stride=256 (standard dal training)
window = 512
stride = 256

def predictions_to_timeseries(predictions, signal_length, window=512, stride=256):
    """Converte predizioni per-window in serie temporale con overlap-add"""
    n_windows = len(predictions)
    output = np.zeros(signal_length)
    counts = np.zeros(signal_length)
    
    for i, pred_val in enumerate(predictions):
        start = i * stride
        end = min(start + window, signal_length)
        output[start:end] += pred_val
        counts[start:end] += 1
    
    # Media dove c'è overlap
    counts[counts == 0] = 1  # Evita divisione per zero
    return output / counts

# Converti in serie temporali
print(f"  🔄 Conversione predizioni in serie temporali...")
n_samples = len(clean)
signal_length = clean.shape[1]

# Calcola quante finestre per segnale
n_windows_per_signal = (signal_length - window) // stride + 1
print(f"  📐 Windows per signal: {n_windows_per_signal}")

intensities = np.zeros((n_samples, 3, signal_length))

for idx in range(n_samples):
    # Estrai finestre per questo segnale
    start_win = idx * n_windows_per_signal
    end_win = start_win + n_windows_per_signal
    
    if end_win <= len(y_pred_bw_cal):
        bw_windows = y_pred_bw_cal[start_win:end_win]
        emg_windows = y_pred_emg_cal[start_win:end_win]
        
        # Converti in serie temporali
        intensities[idx, 0, :] = predictions_to_timeseries(bw_windows, signal_length, window, stride)
        intensities[idx, 1, :] = predictions_to_timeseries(emg_windows, signal_length, window, stride)
        intensities[idx, 2, :] = 0  # PLI non disponibile in V7

# Per denoising, non abbiamo implementazione in V7, quindi None
denoised = None

print(f"  ✅ Intensities shape: {intensities.shape}")
print(f"  ✅ BW range: [{intensities[:, 0, :].min():.3f}, {intensities[:, 0, :].max():.3f}]")
print(f"  ✅ EMG range: [{intensities[:, 1, :].min():.3f}, {intensities[:, 1, :].max():.3f}]")

# ============================================================================
# 2. CALCOLA METRICHE
# ============================================================================
print("\n[2/4] Calcolo metriche...")

def snr_db(clean_sig, noisy_sig):
    """Signal-to-Noise Ratio in dB"""
    noise = noisy_sig - clean_sig
    signal_power = np.mean(clean_sig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)

def rmse(sig1, sig2):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((sig1 - sig2) ** 2))

# Calcola metriche per ogni traccia
metrics = []
for i in range(len(clean)):
    snr_input = snr_db(clean[i], noisy[i])
    
    if denoised is not None:
        snr_output = snr_db(clean[i], denoised[i])
        snr_gain = snr_output - snr_input
        rmse_input = rmse(clean[i], noisy[i])
        rmse_output = rmse(clean[i], denoised[i])
        r_clean_deno, _ = pearsonr(clean[i], denoised[i])
    else:
        snr_output = snr_gain = rmse_output = r_clean_deno = None
    
    rmse_input = rmse(clean[i], noisy[i])
    r_clean_noisy, _ = pearsonr(clean[i], noisy[i])
    
    metrics.append({
        'snr_input': snr_input,
        'snr_output': snr_output,
        'snr_gain': snr_gain,
        'rmse_input': rmse_input,
        'rmse_output': rmse_output,
        'r_clean_noisy': r_clean_noisy,
        'r_clean_deno': r_clean_deno
    })

print(f"\n  📊 Metriche medie:")
print(f"    SNR input:  {np.mean([m['snr_input'] for m in metrics]):.2f} dB")
if denoised is not None:
    print(f"    SNR output: {np.mean([m['snr_output'] for m in metrics]):.2f} dB")
    print(f"    SNR gain:   {np.mean([m['snr_gain'] for m in metrics]):.2f} dB")
    print(f"    RMSE input:  {np.mean([m['rmse_input'] for m in metrics]):.4f}")
    print(f"    RMSE output: {np.mean([m['rmse_output'] for m in metrics]):.4f}")

# ============================================================================
# 3. SELEZIONA ESEMPI RAPPRESENTATIVI
# ============================================================================
print("\n[3/4] Selezione esempi rappresentativi...")

# Trova esempio con alto BW, alto EMG, alto PLI
if intensities is not None:
    bw_means = intensities[:, 0, :].mean(axis=1)
    emg_means = intensities[:, 1, :].mean(axis=1)
    pli_means = intensities[:, 2, :].mean(axis=1)
    
    # Trova esempi con caratteristiche distintive
    idx_high_bw = np.argmax(bw_means)
    idx_high_emg = np.argmax(emg_means)
    idx_high_pli = np.argmax(pli_means)
    idx_mixed = np.argmax(bw_means + emg_means + pli_means)
    
    examples = {
        'High BW': idx_high_bw,
        'High EMG': idx_high_emg,
        'High PLI': idx_high_pli,
        'Mixed Noise': idx_mixed
    }
    
    print(f"  ✅ Selezionati 4 esempi rappresentativi:")
    for name, idx in examples.items():
        print(f"    {name:15s}: trace {idx:2d} (BW={bw_means[idx]:.3f}, EMG={emg_means[idx]:.3f}, PLI={pli_means[idx]:.3f})")
else:
    # Fallback: usa primi 4
    examples = {f'Example {i+1}': i for i in range(min(4, len(clean)))}

# ============================================================================
# 4. GENERA PLOTS
# ============================================================================
print("\n[4/4] Generazione plots...")

def plot_denoising_example(idx, title, save_name):
    """Genera plot completo per un esempio"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    t = np.arange(len(clean[idx])) / fs
    
    # ========== LEFT COLUMN: SEGNALI TEMPORALI ==========
    
    # Plot 1: Clean vs Noisy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, clean[idx], 'g-', linewidth=1.5, label='Clean', alpha=0.8)
    ax1.plot(t, noisy[idx], 'r-', linewidth=1, label='Noisy', alpha=0.6)
    ax1.set_ylabel('Amplitude (mV)', fontsize=10, fontweight='bold')
    ax1.set_title('(A) Clean vs Noisy ECG', fontsize=11, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([t[0], t[-1]])
    
    # Aggiungi metriche
    snr_in = metrics[idx]['snr_input']
    r_in = metrics[idx]['r_clean_noisy']
    ax1.text(0.02, 0.95, f'SNR: {snr_in:.1f} dB\nr: {r_in:.3f}',
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Clean vs Denoised
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, clean[idx], 'g-', linewidth=1.5, label='Clean', alpha=0.8)
    if denoised is not None:
        ax2.plot(t, denoised[idx], 'b-', linewidth=1, label='Denoised', alpha=0.7)
        snr_out = metrics[idx]['snr_output']
        snr_gain = metrics[idx]['snr_gain']
        r_out = metrics[idx]['r_clean_deno']
        
        ax2.text(0.02, 0.95, f'SNR: {snr_out:.1f} dB (+{snr_gain:.1f} dB)\nr: {r_out:.3f}',
                 transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No denoised signal available',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=12, color='gray')
    
    ax2.set_ylabel('Amplitude (mV)', fontsize=10, fontweight='bold')
    ax2.set_title('(B) Clean vs Denoised ECG', fontsize=11, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([t[0], t[-1]])
    
    # Plot 3: Residuo
    ax3 = fig.add_subplot(gs[2, 0])
    noise_input = noisy[idx] - clean[idx]
    ax3.plot(t, noise_input, 'r-', linewidth=0.8, label='Input Noise', alpha=0.7)
    if denoised is not None:
        noise_output = denoised[idx] - clean[idx]
        ax3.plot(t, noise_output, 'b-', linewidth=0.8, label='Residual Noise', alpha=0.7)
    ax3.set_ylabel('Amplitude (mV)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax3.set_title('(C) Noise Comparison', fontsize=11, fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim([t[0], t[-1]])
    
    # Plot 4: Analisi Spettrale
    ax4 = fig.add_subplot(gs[3, 0])
    
    # FFT
    freqs_clean, psd_clean = signal.welch(clean[idx], fs, nperseg=min(256, len(clean[idx])//4))
    freqs_noisy, psd_noisy = signal.welch(noisy[idx], fs, nperseg=min(256, len(noisy[idx])//4))
    
    ax4.semilogy(freqs_clean, psd_clean, 'g-', linewidth=1.5, label='Clean', alpha=0.8)
    ax4.semilogy(freqs_noisy, psd_noisy, 'r-', linewidth=1, label='Noisy', alpha=0.6)
    
    if denoised is not None:
        freqs_deno, psd_deno = signal.welch(denoised[idx], fs, nperseg=min(256, len(denoised[idx])//4))
        ax4.semilogy(freqs_deno, psd_deno, 'b-', linewidth=1, label='Denoised', alpha=0.7)
    
    ax4.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('PSD (mV²/Hz)', fontsize=10, fontweight='bold')
    ax4.set_title('(D) Power Spectral Density', fontsize=11, fontweight='bold', loc='left')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(alpha=0.3, which='both')
    ax4.set_xlim([0, 100])
    
    # Evidenzia bande di rumore
    ax4.axvspan(0.5, 3, alpha=0.1, color='orange', label='BW band')
    ax4.axvspan(20, 100, alpha=0.1, color='purple', label='EMG band')
    ax4.axvspan(48, 52, alpha=0.2, color='red', label='PLI band')
    
    # ========== RIGHT COLUMN: EXPLAINABILITY ==========
    
    if intensities is not None:
        # Plot 5: Intensità BW
        ax5 = fig.add_subplot(gs[0, 1])
        ax5.plot(t, intensities[idx, 0, :], 'orange', linewidth=2, label='BW Intensity')
        ax5.fill_between(t, 0, intensities[idx, 0, :], alpha=0.3, color='orange')
        ax5.set_ylabel('Intensity', fontsize=10, fontweight='bold')
        ax5.set_title('(E) Baseline Wander Estimation (0.5-3 Hz)', 
                     fontsize=11, fontweight='bold', loc='left')
        ax5.set_ylim([0, 1])
        ax5.grid(alpha=0.3)
        ax5.set_xlim([t[0], t[-1]])
        
        mean_bw = intensities[idx, 0, :].mean()
        ax5.axhline(mean_bw, color='darkorange', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_bw:.3f}')
        ax5.legend(loc='upper right', fontsize=9)
        
        # Plot 6: Intensità EMG
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.plot(t, intensities[idx, 1, :], 'purple', linewidth=2, label='EMG Intensity')
        ax6.fill_between(t, 0, intensities[idx, 1, :], alpha=0.3, color='purple')
        ax6.set_ylabel('Intensity', fontsize=10, fontweight='bold')
        ax6.set_title('(F) EMG Artifact Estimation (20-150 Hz)', 
                     fontsize=11, fontweight='bold', loc='left')
        ax6.set_ylim([0, 1])
        ax6.grid(alpha=0.3)
        ax6.set_xlim([t[0], t[-1]])
        
        mean_emg = intensities[idx, 1, :].mean()
        ax6.axhline(mean_emg, color='darkviolet', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_emg:.3f}')
        ax6.legend(loc='upper right', fontsize=9)
        
        # Plot 7: Intensità PLI
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(t, intensities[idx, 2, :], 'red', linewidth=2, label='PLI Intensity')
        ax7.fill_between(t, 0, intensities[idx, 2, :], alpha=0.3, color='red')
        ax7.set_ylabel('Intensity', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax7.set_title('(G) Power-Line Interference Estimation (50/60 Hz)', 
                     fontsize=11, fontweight='bold', loc='left')
        ax7.set_ylim([0, 1])
        ax7.grid(alpha=0.3)
        ax7.set_xlim([t[0], t[-1]])
        
        mean_pli = intensities[idx, 2, :].mean()
        ax7.axhline(mean_pli, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_pli:.3f}')
        ax7.legend(loc='upper right', fontsize=9)
        
        # Plot 8: Confronto tutte le intensità
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.plot(t, intensities[idx, 0, :], 'orange', linewidth=1.5, label='BW', alpha=0.8)
        ax8.plot(t, intensities[idx, 1, :], 'purple', linewidth=1.5, label='EMG', alpha=0.8)
        ax8.plot(t, intensities[idx, 2, :], 'red', linewidth=1.5, label='PLI', alpha=0.8)
        ax8.set_ylabel('Intensity', fontsize=10, fontweight='bold')
        ax8.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax8.set_title('(H) Combined Noise Intensity Profile', 
                     fontsize=11, fontweight='bold', loc='left')
        ax8.set_ylim([0, 1])
        ax8.legend(loc='upper right', fontsize=9)
        ax8.grid(alpha=0.3)
        ax8.set_xlim([t[0], t[-1]])
    else:
        # Nessuna intensità disponibile
        ax5 = fig.add_subplot(gs[0:2, 1])
        ax5.text(0.5, 0.5, 'No intensity estimates available\n\nRun inference with TMU models to generate\nexplainability information',
                 transform=ax5.transAxes, ha='center', va='center',
                 fontsize=12, color='gray')
        ax5.axis('off')
    
    # Titolo principale
    fig.suptitle(f'{title} - TMU-Optimized (V7) ECG with Explainability', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Salva
    output_path = OUTPUT_DIR / f"{save_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Salvato: {output_path}")
    plt.close()

# Genera plot per ciascun esempio
for name, idx in examples.items():
    save_name = name.lower().replace(' ', '_')
    plot_denoising_example(idx, name, save_name)

# ============================================================================
# 5. GENERA SUMMARY FIGURE (4 esempi in una griglia)
# ============================================================================
print("\n[5/5] Generazione summary figure...")

if intensities is not None:
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for row_idx, (name, idx) in enumerate(examples.items()):
        t = np.arange(len(clean[idx])) / fs
        
        # Colonna 1: Segnali
        ax = axes[row_idx, 0]
        ax.plot(t, clean[idx], 'g-', linewidth=1, label='Clean', alpha=0.7)
        ax.plot(t, noisy[idx], 'r-', linewidth=0.8, label='Noisy', alpha=0.5)
        if denoised is not None:
            ax.plot(t, denoised[idx], 'b-', linewidth=0.8, label='Denoised', alpha=0.6)
        ax.set_ylabel('Amplitude', fontsize=9)
        if row_idx == 0:
            ax.set_title('ECG Signals', fontsize=10, fontweight='bold')
        if row_idx == 3:
            ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
        ax.text(-0.15, 0.5, name, transform=ax.transAxes, fontsize=10,
                fontweight='bold', rotation=90, va='center')
        
        # Colonna 2: BW
        ax = axes[row_idx, 1]
        ax.plot(t, intensities[idx, 0, :], 'orange', linewidth=1.5)
        ax.fill_between(t, 0, intensities[idx, 0, :], alpha=0.3, color='orange')
        ax.set_ylim([0, 1])
        if row_idx == 0:
            ax.set_title('BW Intensity', fontsize=10, fontweight='bold', color='orange')
        if row_idx == 3:
            ax.set_xlabel('Time (s)', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Colonna 3: EMG
        ax = axes[row_idx, 2]
        ax.plot(t, intensities[idx, 1, :], 'purple', linewidth=1.5)
        ax.fill_between(t, 0, intensities[idx, 1, :], alpha=0.3, color='purple')
        ax.set_ylim([0, 1])
        if row_idx == 0:
            ax.set_title('EMG Intensity', fontsize=10, fontweight='bold', color='purple')
        if row_idx == 3:
            ax.set_xlabel('Time (s)', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Colonna 4: PLI
        ax = axes[row_idx, 3]
        ax.plot(t, intensities[idx, 2, :], 'red', linewidth=1.5)
        ax.fill_between(t, 0, intensities[idx, 2, :], alpha=0.3, color='red')
        ax.set_ylim([0, 1])
        if row_idx == 0:
            ax.set_title('PLI Intensity', fontsize=10, fontweight='bold', color='red')
        if row_idx == 3:
            ax.set_xlabel('Time (s)', fontsize=9)
        ax.grid(alpha=0.3)
    
    fig.suptitle('TMU-Optimized (V7) - ECG with Explainability - Representative Examples', 
                 fontsize=16, fontweight='bold')
    
    summary_path = OUTPUT_DIR / "summary_grid.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Salvato summary: {summary_path}")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("✅ VISUALIZZAZIONE V7 COMPLETATA")
print("=" * 70)
print(f"\n📁 Output salvati in: {OUTPUT_DIR}/")
print(f"\n📊 File generati:")
for name in examples.keys():
    save_name = name.lower().replace(' ', '_')
    print(f"  • {save_name}.png")
print(f"  • summary_grid.png")

print(f"\n💡 Usa questi plot per:")
print(f"  - Section 5 tesi (interpretability V7)")
print(f"  - Confronto V4 vs V7")
print(f"  - Presentazione finale")

print(f"\n📝 Note V7:")
print(f"  - 57 features (45 bitplanes + 12 HF)")
print(f"  - 95% feature reduction vs V4")
print(f"  - Solo BW/EMG (PLI non incluso)")
print(f"  - 3-4x faster training")
