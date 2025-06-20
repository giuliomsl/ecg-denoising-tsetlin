# -*- coding: utf-8 -*-
"""
ECG Signal Visualizer - TEST FINALE DENOISING RTM
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from deploy_rtm_optimal import RTMECGDenoiser
from load_ecg import SAMPLE_DATA_DIR

def test_final_denoising():
    """Test finale del denoising RTM per verificare efficacia reale"""
    print("🎯 TEST FINALE DENOISING RTM")
    print("="*60)
    
    try:
        # 1. Inizializza RTM Denoiser ottimale
        print("📥 Inizializzazione RTM Denoiser ottimale...")
        denoiser = RTMECGDenoiser()
        
        # 2. Training del modello
        print("🚀 Training modello RTM ottimale...")
        success = denoiser.train(epochs=5)  # Training ridotto per test
        
        if not success:
            print("❌ Training fallito!")
            return False
        
        print(f"✅ Training completato - MSE: {denoiser.training_stats['final_mse']:.6f}")
        
        # 3. Carica dati di test
        print("\n📊 Caricamento dati di test...")
        clean_data = np.load(os.path.join(SAMPLE_DATA_DIR, 'clean_validation_original_center_samples.npy'))
        noisy_data = np.load(os.path.join(SAMPLE_DATA_DIR, 'noisy_validation_original_center_samples.npy'))
        X_val_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, 'X_validation_rtm_BINARIZED_q100.npy'))
        
        print(f"   Clean: {clean_data.shape}")
        print(f"   Noisy: {noisy_data.shape}")
        print(f"   Binarized: {X_val_binarized.shape}")
        
        # 4. Test denoising su campioni rappresentativi
        test_samples = 1000  # Test su 1000 campioni
        
        print(f"\n🧠 Test denoising su {test_samples} campioni...")
        noise_predictions = denoiser.denoise(X_val_binarized[:test_samples])
        
        # 5. Calcola segnali denoised
        clean_test = clean_data[:test_samples]
        noisy_test = noisy_data[:test_samples]
        denoised_test = noisy_test - noise_predictions
        
        # 6. Analisi prestazioni
        print("\n📈 ANALISI PRESTAZIONI DENOISING:")
        
        # Metriche globali
        rmse_original = np.sqrt(np.mean((clean_test - noisy_test)**2))
        rmse_denoised = np.sqrt(np.mean((clean_test - denoised_test)**2))
        rmse_improvement = ((rmse_original - rmse_denoised) / rmse_original) * 100
        
        # SNR
        signal_power = np.var(clean_test)
        noise_power_original = np.var(noisy_test - clean_test)
        noise_power_denoised = np.var(denoised_test - clean_test)
        
        snr_original = 10 * np.log10(signal_power / noise_power_original)
        snr_denoised = 10 * np.log10(signal_power / noise_power_denoised)
        snr_improvement = snr_denoised - snr_original
        
        # Correlazioni
        corr_original = np.corrcoef(clean_test, noisy_test)[0, 1]
        corr_denoised = np.corrcoef(clean_test, denoised_test)[0, 1]
        
        print(f"   RMSE Originale: {rmse_original:.6f}")
        print(f"   RMSE Denoised:  {rmse_denoised:.6f}")
        print(f"   RMSE Improvement: {rmse_improvement:+.2f}%")
        print(f"   ")
        print(f"   SNR Originale: {snr_original:.2f} dB")
        print(f"   SNR Denoised:  {snr_denoised:.2f} dB")
        print(f"   SNR Improvement: {snr_improvement:+.2f} dB")
        print(f"   ")
        print(f"   Correlazione Originale: {corr_original:.6f}")
        print(f"   Correlazione Denoised:  {corr_denoised:.6f}")
        
        # 7. Visualizzazione risultati
        print("\n📊 Generazione plot comparativo...")
        
        # Seleziona segmento rappresentativo per visualizzazione
        segment_start = 100
        segment_length = 360  # 1 secondo
        segment_end = segment_start + segment_length
        
        time_axis = np.arange(segment_length) / 360
        clean_segment = clean_test[segment_start:segment_end]
        noisy_segment = noisy_test[segment_start:segment_end]
        denoised_segment = denoised_test[segment_start:segment_end]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TEST FINALE DENOISING RTM - Risultati Reali', fontsize=16, fontweight='bold')
        
        # Plot principale
        axes[0, 0].plot(time_axis, clean_segment, 'g-', linewidth=2, label='Clean', alpha=0.9)
        axes[0, 0].plot(time_axis, noisy_segment, 'b-', linewidth=1, label='Noisy', alpha=0.7)
        axes[0, 0].plot(time_axis, denoised_segment, 'r-', linewidth=1.5, label='RTM Denoised', alpha=0.8)
        axes[0, 0].set_title('Confronto Segnali ECG')
        axes[0, 0].set_xlabel('Tempo (s)')
        axes[0, 0].set_ylabel('Ampiezza (mV)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot rumore
        noise_real = noisy_segment - clean_segment
        noise_estimated = denoised_segment - clean_segment
        axes[0, 1].plot(time_axis, noise_real, 'orange', linewidth=1.5, label='Rumore Reale', alpha=0.8)
        axes[0, 1].plot(time_axis, noise_estimated, 'red', linewidth=1, label='Rumore Residuo', alpha=0.6)
        axes[0, 1].set_title('Confronto Rumore')
        axes[0, 1].set_xlabel('Tempo (s)')
        axes[0, 1].set_ylabel('Ampiezza (mV)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter correlazione
        axes[1, 0].scatter(clean_test[::10], denoised_test[::10], alpha=0.5, s=10, color='red')
        axes[1, 0].plot([clean_test.min(), clean_test.max()], [clean_test.min(), clean_test.max()], 'k--', alpha=0.5)
        axes[1, 0].set_title(f'Clean vs Denoised (r={corr_denoised:.3f})')
        axes[1, 0].set_xlabel('Clean Signal')
        axes[1, 0].set_ylabel('Denoised Signal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metriche finali
        axes[1, 1].axis('off')
        metrics_text = f"""
RISULTATI FINALI DENOISING RTM:

RMSE:
  Originale: {rmse_original:.6f}
  Denoised:  {rmse_denoised:.6f}
  Miglioramento: {rmse_improvement:+.2f}%

SNR:
  Originale: {snr_original:.2f} dB
  Denoised:  {snr_denoised:.2f} dB
  Miglioramento: {snr_improvement:+.2f} dB

CORRELAZIONE:
  Clean-Noisy:    {corr_original:.6f}
  Clean-Denoised: {corr_denoised:.6f}

CAMPIONI TESTATI: {test_samples}
MODELLO: RTM Ottimale
        """
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Salva risultati
        os.makedirs('plots', exist_ok=True)
        save_path = 'plots/test_finale_denoising_rtm.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Plot salvato: {save_path}")
        
        # 8. Valutazione finale
        print(f"\n🎯 VALUTAZIONE FINALE:")
        
        if rmse_improvement > 10:
            print("🟢 DENOISING ECCELLENTE: Miglioramento significativo!")
            effectiveness = "ECCELLENTE"
        elif rmse_improvement > 5:
            print("🟡 DENOISING BUONO: Miglioramento moderato")
            effectiveness = "BUONO"
        elif rmse_improvement > 0:
            print("🟠 DENOISING LIMITATO: Miglioramento minimo")
            effectiveness = "LIMITATO"
        else:
            print("🔴 DENOISING INEFFICACE: Nessun miglioramento o peggioramento")
            effectiveness = "INEFFICACE"
        
        if snr_improvement > 2:
            print("🟢 SNR: Miglioramento significativo")
        elif snr_improvement > 0.5:
            print("🟡 SNR: Miglioramento moderato")
        else:
            print("🔴 SNR: Miglioramento insufficiente")
        
        # Raccomandazioni
        print(f"\n💡 RACCOMANDAZIONI:")
        if effectiveness in ["INEFFICACE", "LIMITATO"]:
            print("   • Considerare approcci alternativi (filtri digitali, reti neurali)")
            print("   • L'RTM potrebbe non essere ottimale per questo task")
            print("   • Valutare preprocessing diverso o feature engineering")
        else:
            print("   • RTM funziona correttamente per questo dataset")
            print("   • Possibile ottimizzazione fine dei parametri")
            print("   • Considera ensemble di modelli per miglioramenti ulteriori")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_final_denoising()
