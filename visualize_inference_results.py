#!/usr/bin/env python3
"""
Visualizzazione semplice dei risultati dell'inferenza diretta
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_inference_results():
    """Visualizza i risultati salvati dall'inferenza diretta"""
    
    # Load results from direct inference
    results_file = "results/quick_inference_results.npz"
    if not Path(results_file).exists():
        print(f"File non trovato: {results_file}")
        print("Esegui prima direct_inference.py")
        return
    
    print("=== Visualizzazione Risultati Inferenza Diretta ===")
    
    data = np.load(results_file)
    
    # Extract all arrays
    clean_samples = data['clean_samples']
    noisy_samples = data['noisy_samples'] 
    denoised_samples = data['denoised_samples']
    mu = float(data['mu'])
    sigma = float(data['sigma'])
    
    print(f"Caricati {len(clean_samples)} campioni")
    print(f"Normalizzazione: mu={mu:.6f}, sigma={sigma:.6f}")
    
    # Calculate metrics
    mae_noisy = np.mean(np.abs(noisy_samples - clean_samples))
    mae_denoised = np.mean(np.abs(denoised_samples - clean_samples))
    improvement = (mae_noisy - mae_denoised) / mae_noisy * 100
    
    print(f"MAE noisy→clean: {mae_noisy:.6f}")
    print(f"MAE denoised→clean: {mae_denoised:.6f}")
    print(f"Miglioramento: {improvement:.1f}%")
    
    # SNR calculations
    def calculate_snr(signal, noise):
        signal_power = np.var(signal) 
        noise_power = np.var(noise)
        return 10 * np.log10(signal_power / (noise_power + 1e-12))
    
    input_noise = noisy_samples - clean_samples
    residual_noise = denoised_samples - clean_samples
    
    snr_input = calculate_snr(clean_samples, input_noise)
    snr_output = calculate_snr(clean_samples, residual_noise)
    snr_improvement = snr_output - snr_input
    
    print(f"SNR: {snr_input:.1f} → {snr_output:.1f} dB (Δ={snr_improvement:.1f} dB)")
    
    # Plot overview
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot 1: Sample comparison
    n_samples = min(500, len(clean_samples))
    indices = np.arange(n_samples)
    
    axes[0,0].scatter(indices, clean_samples[:n_samples], c='green', s=1, alpha=0.7, label='Clean')
    axes[0,0].scatter(indices, noisy_samples[:n_samples], c='red', s=1, alpha=0.5, label='Noisy')
    axes[0,0].scatter(indices, denoised_samples[:n_samples], c='blue', s=1, alpha=0.7, label='Denoised')
    axes[0,0].set_title('Sample Values Comparison')
    axes[0,0].set_xlabel('Sample Index')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    error_noisy = np.abs(noisy_samples - clean_samples)
    error_denoised = np.abs(denoised_samples - clean_samples)
    
    axes[0,1].hist(error_noisy, bins=50, alpha=0.7, color='red', label='Noisy Error', density=True)
    axes[0,1].hist(error_denoised, bins=50, alpha=0.7, color='blue', label='Denoised Error', density=True)
    axes[0,1].set_title('Error Distribution')
    axes[0,1].set_xlabel('Absolute Error')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Clean vs Noisy scatter
    axes[1,0].scatter(clean_samples[:n_samples], noisy_samples[:n_samples], 
                     c='red', s=1, alpha=0.5, label='Noisy')
    min_val, max_val = clean_samples.min(), clean_samples.max()
    axes[1,0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
    axes[1,0].set_title('Clean vs Noisy')
    axes[1,0].set_xlabel('Clean Amplitude')
    axes[1,0].set_ylabel('Noisy Amplitude')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Clean vs Denoised scatter
    axes[1,1].scatter(clean_samples[:n_samples], denoised_samples[:n_samples], 
                     c='blue', s=1, alpha=0.7, label='Denoised')
    axes[1,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
    axes[1,1].set_title('Clean vs Denoised')
    axes[1,1].set_xlabel('Clean Amplitude')
    axes[1,1].set_ylabel('Denoised Amplitude')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 5: Noise components
    axes[2,0].hist(input_noise, bins=50, alpha=0.7, color='red', label='Input Noise', density=True)
    axes[2,0].hist(residual_noise, bins=50, alpha=0.7, color='blue', label='Residual Noise', density=True)
    axes[2,0].set_title('Noise Components')
    axes[2,0].set_xlabel('Noise Amplitude')
    axes[2,0].set_ylabel('Density')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot 6: Performance metrics bar chart
    metrics = ['MAE\\nNoisy', 'MAE\\nDenoised', 'SNR\\nInput (dB)', 'SNR\\nOutput (dB)']
    values = [mae_noisy, mae_denoised, snr_input, snr_output]
    colors = ['red', 'blue', 'orange', 'green']
    
    bars = axes[2,1].bar(metrics, values, color=colors, alpha=0.7)
    axes[2,1].set_title('Performance Metrics')
    axes[2,1].set_ylabel('Value')
    axes[2,1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[2,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "results/direct_inference_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\nGrafico salvato: {output_file}")
    
    plt.show()
    
    # Create a detailed comparison plot for a subset
    create_signal_reconstruction_plot(clean_samples, noisy_samples, denoised_samples)

def create_signal_reconstruction_plot(clean, noisy, denoised):
    """Crea un plot che simula la ricostruzione di un segnale continuo"""
    
    # Take a subset and treat as time series
    start_idx = 1000
    length = 200
    
    clean_seg = clean[start_idx:start_idx+length]
    noisy_seg = noisy[start_idx:start_idx+length] 
    denoised_seg = denoised[start_idx:start_idx+length]
    
    # Create time axis (simulate 360 Hz sampling)
    t = np.arange(length) / 360.0
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: All signals
    axes[0].plot(t, clean_seg, 'g-', linewidth=2, label='Clean (reference)', alpha=0.9)
    axes[0].plot(t, noisy_seg, 'r-', linewidth=1, label='Noisy', alpha=0.7)
    axes[0].plot(t, denoised_seg, 'b-', linewidth=1.5, label='Denoised (RTM)', alpha=0.8)
    axes[0].set_title('ECG Signal Reconstruction Simulation')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Noise components
    input_noise = noisy_seg - clean_seg
    residual_noise = denoised_seg - clean_seg
    
    axes[1].plot(t, input_noise, 'r-', linewidth=1.5, label='Input Noise', alpha=0.8)
    axes[1].plot(t, residual_noise, 'b-', linewidth=1.5, label='Residual Noise', alpha=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title('Noise Components')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Noise Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = "results/signal_reconstruction_simulation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Simulazione ricostruzione salvata: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    try:
        visualize_inference_results()
        print("\\n✅ Visualizzazione completata!")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
