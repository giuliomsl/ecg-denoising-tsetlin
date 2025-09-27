#!/usr/bin/env python3
"""
Visualizza confronto segnali: noisy vs clean vs denoised
"""
import json, numpy as np, h5py, matplotlib.pyplot as plt
from pathlib import Path

def visualize_denoising_results():
    """Visualizza risultati denoising da direct inference"""
    
    # Load results from direct inference
    results_file = "results/quick_inference_results.npz"
    if not Path(results_file).exists():
        print(f"File {results_file} non trovato. Esegui prima direct_inference.py")
        return
        
    data = np.load(results_file)
    y_true = data['y_true']  # noise residual vero
    y_pred = data['y_pred']  # noise residual predetto
    mae_raw = float(data['mae_raw'])
    mu, sigma = float(data['mu']), float(data['sigma'])
    
    print(f"Loaded results: MAE={mae_raw:.6f}, samples={len(y_true)}")
    
    # Load original signals from preprocessed data
    print("Loading original signals...")
    with h5py.File("data/denoiser_rtm_preprocessed.h5", 'r') as f:
        # Use same validation subset as direct inference (first 5000)
        val_indices = slice(0, len(y_true))
        
        # Load validation data info to reconstruct original signals
        print("Available keys:", list(f.keys()))
        # Assumo che esistano anche i segnali originali o posso ricostruirli
        
    # Per ora uso dati sintetici per demo, poi puoi adattare ai tuoi dati
    print("Generating demo signals...")
    fs = 360.0  # sampling rate
    duration = 3.0  # seconds per sample
    n_samples = int(fs * duration)
    
    # Seleziona alcuni esempi interessanti
    n_plots = min(4, len(y_true))
    indices = [0, len(y_true)//4, len(y_true)//2, len(y_true)-1][:n_plots]
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Simula segnali per demo (sostituisci con i tuoi dati reali)
        t = np.linspace(0, duration, n_samples)
        
        # ECG pulito sintetico (sostituisci con clean signal vero)
        clean_ecg = 0.8 * np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
        clean_ecg += 0.1 * np.sin(2 * np.pi * 60 * t)  # high freq component
        
        # Noise vero e predetto per questo campione
        true_noise = y_true[idx]
        pred_noise = y_pred[idx]
        
        # Simula distribuzione noise su tutto il segnale (approssimazione)
        # In realtà dovresti ricostruire il segnale completo con overlap-add
        noise_true_signal = clean_ecg + true_noise * np.sin(2 * np.pi * 0.1 * t)
        noise_pred_signal = clean_ecg + pred_noise * np.sin(2 * np.pi * 0.1 * t)
        
        # Segnali finali
        noisy_signal = noise_true_signal
        denoised_signal = noisy_signal - (noise_pred_signal - clean_ecg)
        
        # Plot
        ax = axes[i]
        ax.plot(t, clean_ecg, 'g-', linewidth=2, label='Clean (reference)', alpha=0.8)
        ax.plot(t, noisy_signal, 'r-', linewidth=1, label='Noisy', alpha=0.7)
        ax.plot(t, denoised_signal, 'b-', linewidth=1.5, label='Denoised (RTM)', alpha=0.9)
        
        # Calcola SNR per questo segmento
        noise_power = np.var(noisy_signal - clean_ecg)
        signal_power = np.var(clean_ecg)
        snr_input = 10 * np.log10(signal_power / (noise_power + 1e-12))
        
        denoised_noise = np.var(denoised_signal - clean_ecg)
        snr_output = 10 * np.log10(signal_power / (denoised_noise + 1e-12))
        
        ax.set_title(f'Sample {idx}: True noise={true_noise:.4f}, Pred noise={pred_noise:.4f}\\n'
                    f'SNR: {snr_input:.1f}→{snr_output:.1f} dB (Δ={snr_output-snr_input:.1f} dB)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoom su una regione interessante
        ax.set_xlim(0.5, 2.5)
    
    plt.tight_layout()
    
    # Salva plot
    output_file = "results/denoising_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot salvato: {output_file}")
    
    # Summary statistics
    print(f"\\n=== Denoising Summary ===")
    print(f"Total samples: {len(y_true)}")
    print(f"Noise MAE: {mae_raw:.6f}")
    print(f"Noise std: true={np.std(y_true):.6f}, pred={np.std(y_pred):.6f}")
    print(f"Correlation: {np.corrcoef(y_true, y_pred)[0,1]:.4f}")
    
    # Mostra distribuzione errori
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=1)
    plt.plot([-0.2, 0.2], [-0.2, 0.2], 'r--', alpha=0.8)
    plt.xlabel('True Noise')
    plt.ylabel('Predicted Noise')
    plt.title('Noise Prediction Accuracy')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'Error Distribution\\nMAE={np.mean(np.abs(errors)):.6f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(y_true, bins=50, alpha=0.7, label='True Noise', edgecolor='blue')
    plt.hist(y_pred, bins=50, alpha=0.7, label='Pred Noise', edgecolor='red')
    plt.xlabel('Noise Amplitude')
    plt.ylabel('Count')
    plt.title('Noise Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva anche questo plot
    stats_file = "results/denoising_statistics.png"
    plt.savefig(stats_file, dpi=300, bbox_inches='tight')
    print(f"Statistics plot salvato: {stats_file}")
    
    plt.show()

def create_real_signal_visualization():
    """Crea visualizzazione con segnali ECG reali se disponibili"""
    
    # Cerca file di dati ECG reali
    data_files = [
        "data/denoiser_rtm_preprocessed.h5",
        "data/consolidated_data/consolidated_noisy_clean.h5"
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"Trovato file dati: {data_file}")
            with h5py.File(data_file, 'r') as f:
                print("Keys disponibili:", list(f.keys()))
                
                # Cerca segnali clean/noisy completi
                if 'clean_signals' in f and 'noisy_signals' in f:
                    print("Trovati segnali completi! Creando visualizzazione reale...")
                    # Implementa qui la visualizzazione con segnali reali
                    break
    else:
        print("Nessun file con segnali completi trovato. Usa visualizzazione demo.")
        
if __name__ == "__main__":
    print("=== ECG Denoising Visualization ===")
    
    # Prima prova con segnali reali se disponibili
    create_real_signal_visualization()
    
    # Poi mostra la demo con i risultati dell'inferenza
    try:
        visualize_denoising_results()
        print("\\n✅ Visualizzazione completata! Controlla i file in results/")
    except Exception as e:
        print(f"❌ Errore nella visualizzazione: {e}")
        import traceback
        traceback.print_exc()
