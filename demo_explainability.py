#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo explainability system senza caricamento di stato
Retrain i modelli e fa inferenza diretta per evitare segfault
"""
import numpy as np
import h5py
import sys
sys.path.append('src')
from explain.features import build_features
from pyTsetlinMachine.tm import RegressionTsetlinMachine
import matplotlib.pyplot as plt
from pathlib import Path

def to_int32_bin(Xu8):
    return (Xu8 != 0).astype(np.int32, copy=False)

def overlap_add(center_vals, L, window, stride):
    out = np.zeros(L, dtype=np.float32)
    wsum = np.zeros(L, dtype=np.float32)
    half = window//2
    for i, y in enumerate(center_vals):
        c = i*stride + half
        if 0 <= c < L: 
            out[c] += float(y)
            wsum[c] += 1.0
    nz = wsum > 0
    out[nz] /= wsum[nz]
    if np.any(~nz): 
        out = np.interp(np.arange(L), np.where(nz)[0], out[nz])
    return out

def demo_explainability():
    """Demo completo del sistema explainability"""
    
    print("=== Demo Sistema Explainability ECG ===")
    
    # 1. Carica dataset features
    print("\\n1. Caricamento dataset...")
    with h5py.File("data/explain_features_dataset.h5", "r") as f:
        # Usa subset per velocità
        Xtr = to_int32_bin(f["train_X"][:20000])  # 20k esempi
        Xva = to_int32_bin(f["val_X"][:5000])     # 5k esempi
        
        # Target per i 3 tipi di rumore
        ytr_bw = f["train_y_bw"][:20000].astype(np.float32)
        ytr_emg = f["train_y_emg"][:20000].astype(np.float32) 
        ytr_pli = f["train_y_pli"][:20000].astype(np.float32)
        
        yva_bw = f["val_y_bw"][:5000].astype(np.float32)
        yva_emg = f["val_y_emg"][:5000].astype(np.float32)
        yva_pli = f["val_y_pli"][:5000].astype(np.float32)
    
    print(f"Features: {Xtr.shape}, Validation: {Xva.shape}")
    
    # 2. Training rapido dei 3 modelli specializzati
    print("\\n2. Training modelli specializzati...")
    models = {}
    targets = {
        "bw": (ytr_bw, yva_bw),
        "emg": (ytr_emg, yva_emg), 
        "pli": (ytr_pli, yva_pli)
    }
    
    for noise_type, (ytr, yva) in targets.items():
        print(f"\\n   Training {noise_type.upper()}...")
        model = RegressionTsetlinMachine(2000, 300, 2.8)  # Più piccolo per velocità
        
        # Training con batch
        batch_size = 5000
        for epoch in range(2):  # Solo 2 epoche per demo
            indices = np.random.permutation(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                batch_idx = indices[i:i+batch_size]
                model.fit(Xtr[batch_idx], ytr[batch_idx], epochs=1, incremental=True)
        
        # Valutazione
        pred_val = model.predict(Xva)
        mae = np.mean(np.abs(pred_val - yva))
        print(f"   {noise_type.upper()} - Val MAE: {mae:.4f}")
        
        models[noise_type] = model
    
    # 3. Inferenza su campioni di test
    print("\\n3. Inferenza explainability...")
    
    # Carica alcuni segnali di test
    with h5py.File("data/explain_input_dataset.h5", "r") as f:
        test_noisy = f["test_noisy"][:10].astype(np.float32)  # 10 campioni
        test_clean = f["test_clean"][:10].astype(np.float32)
    
    results = []
    
    for i, (noisy_sig, clean_sig) in enumerate(zip(test_noisy, test_clean)):
        print(f"   Processando campione {i+1}/10...")
        
        # Build features
        X = build_features(noisy_sig, window=512, stride=128,
                          encoder="bitplanes", bits=8, levels=64, include_deriv=True)
        Xbin = to_int32_bin(X)
        
        # Predizioni intensità
        pred_bw = models["bw"].predict(Xbin).astype(np.float32)
        pred_emg = models["emg"].predict(Xbin).astype(np.float32)
        pred_pli = models["pli"].predict(Xbin).astype(np.float32)
        
        # Ricostruzione serie temporali con overlap-add
        L = len(noisy_sig)
        intensity_bw = overlap_add(pred_bw, L, 512, 128)
        intensity_emg = overlap_add(pred_emg, L, 512, 128)
        intensity_pli = overlap_add(pred_pli, L, 512, 128)
        
        results.append({
            'noisy': noisy_sig,
            'clean': clean_sig,
            'intensity_bw': intensity_bw,
            'intensity_emg': intensity_emg, 
            'intensity_pli': intensity_pli
        })
    
    # 4. Analisi e visualizzazione
    print("\\n4. Analisi risultati...")
    
    # Statistiche aggregate
    all_bw = np.concatenate([r['intensity_bw'] for r in results])
    all_emg = np.concatenate([r['intensity_emg'] for r in results])
    all_pli = np.concatenate([r['intensity_pli'] for r in results])
    
    print(f"\\nStatistiche intensità predette:")
    print(f"BW  - Media: {np.mean(all_bw):.4f}, Std: {np.std(all_bw):.4f}, Range: [{np.min(all_bw):.4f}, {np.max(all_bw):.4f}]")
    print(f"EMG - Media: {np.mean(all_emg):.4f}, Std: {np.std(all_emg):.4f}, Range: [{np.min(all_emg):.4f}, {np.max(all_emg):.4f}]")
    print(f"PLI - Media: {np.mean(all_pli):.4f}, Std: {np.std(all_pli):.4f}, Range: [{np.min(all_pli):.4f}, {np.max(all_pli):.4f}]")
    
    # Visualizzazione di esempio
    visualize_explainability_results(results)
    
    return results

def visualize_explainability_results(results):
    """Visualizza risultati explainability"""
    
    # Prendi il primo campione per visualizzazione dettagliata
    sample = results[0]
    L = len(sample['noisy'])
    t = np.arange(L) / 360.0  # Tempo in secondi
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    
    # Plot 1: Segnali
    axes[0].plot(t, sample['clean'], 'g-', linewidth=2, label='Clean ECG', alpha=0.8)
    axes[0].plot(t, sample['noisy'], 'r-', linewidth=1, label='Noisy ECG', alpha=0.7)
    axes[0].set_title('ECG Signals')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rumore estratto
    noise = sample['noisy'] - sample['clean']
    axes[1].plot(t, noise, 'k-', linewidth=1, alpha=0.8)
    axes[1].set_title('Extracted Noise (Noisy - Clean)')
    axes[1].set_ylabel('Noise Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Intensità BW
    axes[2].plot(t, sample['intensity_bw'], 'b-', linewidth=2, label='BW Intensity')
    axes[2].set_title('Baseline Wander Intensity (0-0.7 Hz)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Intensità EMG
    axes[3].plot(t, sample['intensity_emg'], 'orange', linewidth=2, label='EMG Intensity')
    axes[3].set_title('Muscle Artifacts Intensity (20-120 Hz)')
    axes[3].set_ylabel('Intensity')
    axes[3].set_ylim([0, 1])
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Intensità PLI
    axes[4].plot(t, sample['intensity_pli'], 'purple', linewidth=2, label='PLI Intensity')
    axes[4].set_title('Power Line Interference Intensity (45-55 Hz)')
    axes[4].set_ylabel('Intensity')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylim([0, 1])
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva plot
    output_file = "results/explainability_demo.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\\n📊 Visualizzazione salvata: {output_file}")
    
    plt.show()
    
    # Plot distribuzione intensità
    plot_intensity_distributions(results)

def plot_intensity_distributions(results):
    """Plot distribuzione delle intensità per tutti i campioni"""
    
    # Raccogli tutte le intensità
    all_bw = np.concatenate([r['intensity_bw'] for r in results])
    all_emg = np.concatenate([r['intensity_emg'] for r in results])
    all_pli = np.concatenate([r['intensity_pli'] for r in results])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Istogrammi
    axes[0].hist(all_bw, bins=50, alpha=0.7, color='blue', density=True)
    axes[0].set_title('BW Intensity Distribution')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_emg, bins=50, alpha=0.7, color='orange', density=True)
    axes[1].set_title('EMG Intensity Distribution')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Density')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(all_pli, bins=50, alpha=0.7, color='purple', density=True)
    axes[2].set_title('PLI Intensity Distribution')
    axes[2].set_xlabel('Intensity')
    axes[2].set_ylabel('Density')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva plot
    output_file = "results/intensity_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Distribuzioni salvate: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    try:
        results = demo_explainability()
        print("\\n✅ Demo explainability completata con successo!")
        print("\\nIl sistema ha:")
        print("- Addestrato 3 modelli RTM per BW/EMG/PLI detection")
        print("- Predetto intensità di rumore per ogni tipo")
        print("- Ricostruito serie temporali continue con overlap-add")
        print("- Generato visualizzazioni dettagliate")
        
    except Exception as e:
        print(f"❌ Errore nella demo: {e}")
        import traceback
        traceback.print_exc()
