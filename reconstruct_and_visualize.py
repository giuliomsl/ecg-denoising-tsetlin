#!/usr/bin/env python3
"""
Ricostruzione completa segnali ECG denoisati con overlap-add
"""
import json, numpy as np, h5py, matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('src/denoiser')
from encoders import build_features, overlap_add

def reconstruct_full_signals():
    """Ricostruisce segnali completi usando overlap-add come nell'inferenza originale"""
    
    print("=== Full Signal Reconstruction ===")
    
    # Load quick training results for model parameters
    results_file = "results/quick_inference_results.npz"
    if not Path(results_file).exists():
        print(f"Esegui prima direct_inference.py per avere i parametri del modello")
        return
    
    data = np.load(results_file)
    mu, sigma = float(data['mu']), float(data['sigma'])
    print(f"Model normalization: mu={mu:.6f}, sigma={sigma:.6f}")
    
    # Simula training e inferenza per alcuni segnali demo
    print("Training quick model for full signal reconstruction...")
    
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
    
    # Load preprocessed data for training
    with h5py.File("data/denoiser_rtm_preprocessed.h5", 'r') as f:
        X_train = f["train_X"][:20000].astype(np.int32)  # Subset for speed
        y_train = f["train_y"][:20000].astype(np.float32)
        
    # Quick training with normalization
    mu_train = float(np.median(y_train))
    mad = float(np.median(np.abs(y_train - mu_train))) + 1e-9
    sigma_train = mad / 0.6745
    y_train_norm = np.clip((y_train - mu_train) / sigma_train, -5.0, 5.0)
    
    print("Training model...")
    model = RegressionTsetlinMachine(1500, 250, 2.75)  # Smaller for speed
    
    batch_size = 5000
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train_norm[i:i+batch_size]
        model.fit(X_batch, y_batch, epochs=2, incremental=True)
    
    print("Model trained!")
    
    # Now create synthetic ECG signals for demonstration
    print("Creating synthetic ECG signals...")
    
    fs = 360.0
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create multiple example ECGs
    n_examples = 3
    
    for example_idx in range(n_examples):
        print(f"\\nProcessing example {example_idx + 1}/{n_examples}")
        
        # Generate synthetic clean ECG
        clean_ecg = create_synthetic_ecg(t, example_idx)
        
        # Add realistic noise
        noisy_ecg = add_realistic_noise(clean_ecg, example_idx)
        
        # Denoise using RTM with overlap-add reconstruction
        denoised_ecg = denoise_signal_overlap_add(
            noisy_ecg, model, mu_train, sigma_train,
            window=128, stride=24  # Stride 24 come discusso
        )
        
        # Visualize results
        visualize_ecg_comparison(
            t, clean_ecg, noisy_ecg, denoised_ecg, 
            example_idx, fs
        )

def create_synthetic_ecg(t, variant=0):
    """Crea ECG sintetico realistico"""
    
    # Base ECG pattern
    heart_rate = 75 + variant * 10  # variazione frequenza cardiaca
    period = 60.0 / heart_rate
    
    ecg = np.zeros_like(t)
    
    # QRS complex + P/T waves
    for cycle_start in np.arange(0, t[-1], period):
        mask = (t >= cycle_start) & (t < cycle_start + period)
        if not np.any(mask):
            continue
            
        cycle_t = t[mask] - cycle_start
        cycle_phase = cycle_t / period * 2 * np.pi
        
        # P wave (0.1s duration)
        p_component = 0.2 * np.exp(-((cycle_phase - 0.5) / 0.3)**2)
        
        # QRS complex (0.08s duration) 
        qrs_component = 1.2 * np.exp(-((cycle_phase - np.pi) / 0.2)**2)
        qrs_component -= 0.3 * np.exp(-((cycle_phase - np.pi + 0.1) / 0.1)**2)  # Q wave
        
        # T wave (0.15s duration)
        t_component = 0.4 * np.exp(-((cycle_phase - 4.5) / 0.4)**2)
        
        ecg[mask] += p_component + qrs_component + t_component
    
    # Add baseline wander
    baseline = 0.1 * np.sin(2 * np.pi * 0.3 * t) * (variant + 1)
    
    # Add some individual variation
    if variant == 1:
        ecg *= 1.2  # Bigger amplitude
    elif variant == 2:
        ecg += 0.05 * np.sin(2 * np.pi * 15 * t)  # EMG-like noise
    
    return ecg + baseline

def add_realistic_noise(clean_ecg, variant=0):
    """Aggiunge rumore realistico all'ECG"""
    
    # Power line interference (50 Hz)
    pli_amplitude = 0.1 + variant * 0.05
    pli = pli_amplitude * np.sin(2 * np.pi * 50 * np.linspace(0, len(clean_ecg)/360, len(clean_ecg)))
    
    # Baseline wander (low frequency)
    bw_freq = 0.2 + variant * 0.1
    bw = 0.15 * np.sin(2 * np.pi * bw_freq * np.linspace(0, len(clean_ecg)/360, len(clean_ecg)))
    
    # Muscle artifact (EMG-like)
    if variant == 2:
        emg = 0.08 * np.random.randn(len(clean_ecg))
        emg = np.convolve(emg, np.ones(10)/10, mode='same')  # smooth
    else:
        emg = 0.03 * np.random.randn(len(clean_ecg))
    
    # Motion artifacts (sudden jumps)
    motion = np.zeros_like(clean_ecg)
    if variant == 1:
        jump_indices = np.random.choice(len(clean_ecg), 3, replace=False)
        for idx in jump_indices:
            motion[idx:idx+50] += 0.2 * np.exp(-np.arange(50)/20)
    
    return clean_ecg + pli + bw + emg + motion

def denoise_signal_overlap_add(noisy_signal, model, mu, sigma, window=128, stride=24):
    """Denoising con overlap-add reconstruction"""
    
    # Build features usando lo stesso encoder del training
    X = build_features_simple(noisy_signal, window, stride)
    
    print(f"  Built features: {X.shape}")
    
    # Predict noise residuals (normalized)
    noise_pred_norm = model.predict(X)
    
    # Denormalize
    noise_pred = noise_pred_norm * sigma + mu
    
    print(f"  Predicted {len(noise_pred)} noise samples")
    
    # Overlap-add reconstruction
    noise_signal_reconstructed = overlap_add(noise_pred, len(noisy_signal), window, stride)
    
    # Final denoised signal
    denoised = noisy_signal - noise_signal_reconstructed
    
    return denoised

def build_features_simple(signal, window, stride):
    """Simplified feature building (bitplanes encoder)"""
    
    # Z-score normalize and clip
    mu = np.mean(signal)
    std = np.std(signal) + 1e-8
    z = np.clip((signal - mu) / std, -3.0, 3.0)
    
    # Percentile scale to [0,1]
    p1, p99 = np.percentile(z, [1, 99])
    if p99 <= p1:
        p1, p99 = z.min(), z.max()
        if p99 <= p1:
            p99 = p1 + 1.0
    x01 = np.clip((z - p1) / (p99 - p1), 0.0, 1.0)
    
    # Quantize to 8 bits
    q = np.rint(x01 * 255).astype(np.int32)
    
    # Convert to bitplanes
    def int_to_bitplanes(q_arr, n_bits=8):
        out = np.empty((len(q_arr), n_bits), dtype=np.uint8)
        for b in range(n_bits):
            out[:, b] = ((q_arr >> b) & 1).astype(np.uint8)
        return out
    
    bp = int_to_bitplanes(q, 8)
    
    # Add derivative features
    dq = np.abs(np.diff(q, prepend=q[0]))
    bp_d = int_to_bitplanes(dq, 8)
    features = np.concatenate([bp, bp_d], axis=1)
    
    # Rolling windows
    if len(signal) < window:
        return np.empty((0, window * 16), dtype=np.uint8)
    
    n_windows = 1 + (len(signal) - window) // stride
    X = np.empty((n_windows, window * 16), dtype=np.uint8)
    
    for i in range(n_windows):
        start = i * stride
        end = start + window
        window_features = features[start:end]
        X[i] = window_features.flatten()
    
    return X

def visualize_ecg_comparison(t, clean, noisy, denoised, example_idx, fs):
    """Visualizza confronto completo dei segnali"""
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: All signals overlapped
    axes[0].plot(t, clean, 'g-', linewidth=2, label='Clean (reference)', alpha=0.9)
    axes[0].plot(t, noisy, 'r-', linewidth=1, label='Noisy', alpha=0.7)
    axes[0].plot(t, denoised, 'b-', linewidth=1.5, label='Denoised (RTM)', alpha=0.8)
    axes[0].set_title(f'ECG Denoising Comparison - Example {example_idx + 1}')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Noise components
    true_noise = noisy - clean
    residual_noise = denoised - clean
    axes[1].plot(t, true_noise, 'r-', label='Input Noise', alpha=0.8)
    axes[1].plot(t, residual_noise, 'b-', label='Residual Noise', alpha=0.8)
    axes[1].set_title('Noise Components')
    axes[1].set_ylabel('Noise Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Zoom su una regione interessante (1 secondo)
    zoom_start, zoom_end = 1.0, 2.5
    zoom_mask = (t >= zoom_start) & (t <= zoom_end)
    axes[2].plot(t[zoom_mask], clean[zoom_mask], 'g-', linewidth=3, label='Clean')
    axes[2].plot(t[zoom_mask], noisy[zoom_mask], 'r-', linewidth=1.5, label='Noisy', alpha=0.8)
    axes[2].plot(t[zoom_mask], denoised[zoom_mask], 'b--', linewidth=2, label='Denoised')
    axes[2].set_title(f'Detailed View ({zoom_start}s - {zoom_end}s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: SNR comparison
    def calculate_snr(signal, noise):
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        return 10 * np.log10(signal_power / (noise_power + 1e-12))
    
    snr_input = calculate_snr(clean, true_noise)
    snr_output = calculate_snr(clean, residual_noise)
    snr_improvement = snr_output - snr_input
    
    mae_noise = np.mean(np.abs(residual_noise))
    mae_original = np.mean(np.abs(true_noise))
    noise_reduction = (1 - mae_noise/mae_original) * 100
    
    # Bar plot delle metriche
    metrics = ['SNR Input\\n(dB)', 'SNR Output\\n(dB)', 'SNR Improvement\\n(dB)', 'Noise Reduction\\n(%)']
    values = [snr_input, snr_output, snr_improvement, noise_reduction]
    colors = ['red', 'blue', 'green', 'orange']
    
    bars = axes[3].bar(metrics, values, color=colors, alpha=0.7)
    axes[3].set_title('Denoising Performance Metrics')
    axes[3].set_ylabel('Value')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sui bar
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    for ax in axes[:-1]:
        ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Salva plot
    output_file = f"results/full_ecg_denoising_example_{example_idx + 1}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    # Stampa metriche
    print(f"  SNR: {snr_input:.1f} → {snr_output:.1f} dB (Δ={snr_improvement:.1f} dB)")
    print(f"  Noise reduction: {noise_reduction:.1f}%")
    print(f"  MAE: {mae_original:.6f} → {mae_noise:.6f}")
    
    plt.show()

if __name__ == "__main__":
    try:
        reconstruct_full_signals()
        print("\\n✅ Ricostruzione e visualizzazione completate!")
        print("Controlla i file salvati in results/")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
