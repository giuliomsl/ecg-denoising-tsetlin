#!/usr/bin/env python3
"""
Wavelet-RTM Hybrid Denoiser per ECG
===================================

SOLUZIONE INNOVATIVA 2: Combina trasformata wavelet con RTM.

STRATEGIA:
1. Decomposizione Wavelet del segnale ECG in coefficienti
2. RTM separate per ogni livello di dettaglio wavelet
3. Ricostruzione intelligente con weights adattivi
4. Post-processing con filtri ottimali

VANTAGGI:
- Lavora nel dominio wavelet dove il rumore è più separabile
- RTM focalizzate su bande di frequenza specifiche
- Preserva meglio le caratteristiche morfologiche ECG

Autore: Sistema RTM Innovativo  
Data: Giugno 2025
"""

import numpy as np
import os
import sys
import pywt
from pyTsetlinMachine.tm import RegressionTsetlinMachine
import pickle
from sklearn.metrics import mean_squared_error
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg')
from src.load_ecg import *
from src.ecg_utils import calculate_snr, bandpass_filter

class WaveletRTMDenoiser:
    """Denoiser ibrido Wavelet + RTM per ECG"""
    
    def __init__(self, wavelet='db8', levels=5):
        self.wavelet = wavelet
        self.levels = levels
        
        # RTM per coefficienti di approssimazione
        self.approx_rtm = RegressionTsetlinMachine(
            number_of_clauses=600,
            T=15000,
            s=3.5,
            boost_true_positive_feedback=0
        )
        
        # RTM separate per ogni livello di dettaglio
        self.detail_rtms = {}
        for level in range(1, levels + 1):
            self.detail_rtms[f'detail_{level}'] = RegressionTsetlinMachine(
                number_of_clauses=400 - level*50,  # Meno clausole per livelli alti
                T=10000 + level*2000,
                s=2.5 + level*0.3,
                boost_true_positive_feedback=0
            )
        
        # RTM per combinazione finale
        self.reconstruction_rtm = RegressionTsetlinMachine(
            number_of_clauses=300,
            T=8000,
            s=2.0,
            boost_true_positive_feedback=0
        )
        
        self.is_trained = False
        self.wavelet_stats = {}
    
    def wavelet_decompose(self, signals):
        """Decomposizione wavelet multi-livello"""
        decomposed_data = []
        
        for signal_data in signals:
            # Decomposizione wavelet
            coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.levels)
            
            # Separa approssimazione e dettagli
            approx = coeffs[0]
            details = coeffs[1:]
            
            decomposed_data.append({
                'approx': approx,
                'details': details,
                'original_length': len(signal_data)
            })
        
        return decomposed_data
    
    def prepare_wavelet_features(self, decomposed_data):
        """Prepara feature dai coefficienti wavelet per RTM"""
        print("=== PREPARAZIONE FEATURE WAVELET ===")
        
        # Estrae statistiche da coefficienti
        approx_features = []
        detail_features = {f'detail_{i+1}': [] for i in range(self.levels)}
        
        for data in decomposed_data:
            # Feature da coefficienti di approssimazione
            approx = data['approx']
            approx_feat = self._extract_coefficient_features(approx)
            approx_features.append(approx_feat)
            
            # Feature da coefficienti di dettaglio
            for i, detail in enumerate(data['details']):
                detail_key = f'detail_{i+1}'
                detail_feat = self._extract_coefficient_features(detail)
                detail_features[detail_key].append(detail_feat)
        
        return np.array(approx_features), {k: np.array(v) for k, v in detail_features.items()}
    
    def _extract_coefficient_features(self, coeffs):
        """Estrae feature da coefficienti wavelet"""
        # Padding/truncation a dimensione fissa
        target_size = 256
        if len(coeffs) > target_size:
            coeffs = coeffs[:target_size]
        else:
            coeffs = np.pad(coeffs, (0, target_size - len(coeffs)), 'constant')
        
        # Feature statistiche
        stats = [
            np.mean(coeffs),
            np.std(coeffs),
            np.max(coeffs),
            np.min(coeffs),
            np.median(coeffs),
            np.percentile(coeffs, 25),
            np.percentile(coeffs, 75),
            np.var(coeffs)
        ]
        
        # Combinazione: coefficienti + statistiche
        features = np.concatenate([coeffs, stats])
        
        # Binarizzazione per RTM
        thresholds = np.linspace(features.min(), features.max(), 10)
        binary_features = []
        for val in features:
            binary_features.extend([1 if val > th else 0 for th in thresholds])
        
        return np.array(binary_features)
    
    def train_wavelet_rtms(self, clean_signals, noisy_signals):
        """Training delle RTM sui coefficienti wavelet"""
        print("\\n=== TRAINING WAVELET-RTM SYSTEM ===")
        
        # Decomposizione wavelet
        print("Decomposizione wavelet segnali puliti...")
        clean_decomposed = self.wavelet_decompose(clean_signals)
        
        print("Decomposizione wavelet segnali rumorosi...")
        noisy_decomposed = self.wavelet_decompose(noisy_signals)
        
        # Prepara feature
        clean_approx_feat, clean_detail_feat = self.prepare_wavelet_features(clean_decomposed)
        noisy_approx_feat, noisy_detail_feat = self.prepare_wavelet_features(noisy_decomposed)
        
        # Training RTM approssimazione
        print("\\n🔧 Training RTM Approssimazione...")
        self.approx_rtm.fit(noisy_approx_feat, clean_approx_feat, epochs=15)
        
        # Test RTM approssimazione
        approx_pred = self.approx_rtm.predict(noisy_approx_feat)
        approx_mse = mean_squared_error(clean_approx_feat.flatten(), approx_pred.flatten())
        print(f"   Approssimazione MSE: {approx_mse:.6f}")
        
        # Training RTM dettagli
        for detail_key in self.detail_rtms.keys():
            print(f"\\n🔧 Training RTM {detail_key}...")
            
            clean_detail = clean_detail_feat[detail_key]
            noisy_detail = noisy_detail_feat[detail_key]
            
            self.detail_rtms[detail_key].fit(noisy_detail, clean_detail, epochs=12)
            
            # Test
            detail_pred = self.detail_rtms[detail_key].predict(noisy_detail)
            detail_mse = mean_squared_error(clean_detail.flatten(), detail_pred.flatten())
            print(f"   {detail_key} MSE: {detail_mse:.6f}")
        
        # Training RTM ricostruzione finale
        print("\\n🔧 Training RTM Ricostruzione...")
        self._train_reconstruction_rtm(clean_signals, noisy_signals, 
                                       clean_decomposed, noisy_decomposed)
        
        self.is_trained = True
        print("\\n🎉 TRAINING WAVELET-RTM COMPLETATO!")
    
    def _train_reconstruction_rtm(self, clean_signals, noisy_signals, 
                                  clean_decomposed, noisy_decomposed):
        """Training RTM per ricostruzione ottimale"""
        reconstruction_features = []
        reconstruction_targets = []
        
        for i in range(len(clean_signals)):
            # Feature: coefficienti predetti + statistiche originali
            clean_coeffs = clean_decomposed[i]
            noisy_coeffs = noisy_decomposed[i]
            
            # Statistiche multi-scala
            stats = []
            stats.extend([np.mean(clean_coeffs['approx']), np.std(clean_coeffs['approx'])])
            for detail in clean_coeffs['details']:
                stats.extend([np.mean(detail), np.std(detail)])
            
            # Energy ratios
            total_energy = sum([np.sum(d**2) for d in clean_coeffs['details']]) + np.sum(clean_coeffs['approx']**2)
            energy_ratios = []
            for detail in clean_coeffs['details']:
                energy_ratios.append(np.sum(detail**2) / (total_energy + 1e-8))
            
            # Combine features
            feature_vector = stats + energy_ratios + clean_coeffs['approx'][:50].tolist()
            reconstruction_features.append(feature_vector)
            
            # Target: differenza pulito - rumoroso
            target = clean_signals[i] - noisy_signals[i]
            reconstruction_targets.append(target)
        
        reconstruction_features = np.array(reconstruction_features)
        reconstruction_targets = np.array(reconstruction_targets)
        
        # Binarizzazione features
        binary_features = []
        for feat_vec in reconstruction_features:
            thresholds = np.linspace(feat_vec.min(), feat_vec.max(), 8)
            binary_feat = []
            for val in feat_vec:
                binary_feat.extend([1 if val > th else 0 for th in thresholds])
            binary_features.append(binary_feat)
        
        binary_features = np.array(binary_features)
        
        print(f"Training ricostruzione su {binary_features.shape} features")
        self.reconstruction_rtm.fit(binary_features, reconstruction_targets, epochs=10)
        
        # Test
        recon_pred = self.reconstruction_rtm.predict(binary_features)
        recon_mse = mean_squared_error(reconstruction_targets.flatten(), recon_pred.flatten())
        print(f"   Ricostruzione MSE: {recon_mse:.6f}")
    
    def denoise_signals(self, noisy_signals):
        """Applica denoising wavelet-RTM"""
        if not self.is_trained:
            raise ValueError("Modello non addestrato!")
        
        print("=== DENOISING CON WAVELET-RTM ===")
        
        denoised_signals = []
        
        for i, noisy_signal in enumerate(noisy_signals):
            # 1. Decomposizione wavelet
            coeffs = pywt.wavedec(noisy_signal, self.wavelet, level=self.levels)
            
            # 2. Predizione RTM per ogni componente
            # Approssimazione
            noisy_decomp = [{'approx': coeffs[0], 'details': coeffs[1:], 'original_length': len(noisy_signal)}]
            noisy_approx_feat, noisy_detail_feat = self.prepare_wavelet_features(noisy_decomp)
            
            clean_approx_pred = self.approx_rtm.predict(noisy_approx_feat)
            
            # Dettagli
            clean_details_pred = []
            for j, detail_key in enumerate(self.detail_rtms.keys()):
                detail_feat = noisy_detail_feat[detail_key]
                clean_detail_pred = self.detail_rtms[detail_key].predict(detail_feat)
                clean_details_pred.append(clean_detail_pred)
            
            # 3. Ricostruzione wavelet iniziale
            # Rebuild coefficients structure (simplified)
            clean_approx = clean_approx_pred[0][:len(coeffs[0])]
            clean_details = []
            for j, detail in enumerate(coeffs[1:]):
                if j < len(clean_details_pred):
                    clean_detail = clean_details_pred[j][0][:len(detail)]
                    clean_details.append(clean_detail)
                else:
                    clean_details.append(detail)  # Fallback
            
            # Ricostruzione
            try:
                reconstructed = pywt.waverec([clean_approx] + clean_details, self.wavelet)
                
                # Assicura stessa lunghezza
                if len(reconstructed) > len(noisy_signal):
                    reconstructed = reconstructed[:len(noisy_signal)]
                elif len(reconstructed) < len(noisy_signal):
                    reconstructed = np.pad(reconstructed, (0, len(noisy_signal) - len(reconstructed)), 'edge')
                
                denoised_signals.append(reconstructed)
                
            except Exception as e:
                print(f"Errore ricostruzione wavelet per segnale {i}: {e}")
                # Fallback: filtro passa-banda
                denoised = bandpass_filter(noisy_signal, lowcut=0.5, highcut=40)
                denoised_signals.append(denoised)
        
        return np.array(denoised_signals)

def load_ecg_data_for_training():
    """Carica dati ECG per training wavelet-RTM"""
    print("=== CARICAMENTO DATI ECG ===")
    
    try:
        # Usa i dati generati
        data_dir = "/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg/data/generated_signals"
        
        # Carica segnali puliti e rumorosi per alcuni record
        records = ['100', '101', '102', '103', '104']
        clean_signals = []
        noisy_signals = []
        
        for record in records:
            clean_file = os.path.join(data_dir, f"{record}_clean.npy")
            noisy_file = os.path.join(data_dir, f"{record}_noisy.npy")
            
            if os.path.exists(clean_file) and os.path.exists(noisy_file):
                clean = np.load(clean_file)
                noisy = np.load(noisy_file)
                
                clean_signals.append(clean)
                noisy_signals.append(noisy)
                print(f"Caricato record {record}: {len(clean)} samples")
        
        if len(clean_signals) == 0:
            raise FileNotFoundError("Nessun file di dati trovato")
            
        return np.array(clean_signals), np.array(noisy_signals)
        
    except Exception as e:
        print(f"Errore caricamento: {e}")
        print("Generazione dati sintetici di test...")
        
        # Dati sintetici per test
        fs = 360
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        
        clean_signals = []
        noisy_signals = []
        
        for i in range(5):
            # Segnale ECG sintetico
            ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
            ecg += 0.3 * np.sin(2 * np.pi * 0.8 * t)  # Variabilità
            
            # Rumore
            noise = 0.1 * np.random.randn(len(ecg))  # Gaussiano
            noise += 0.05 * np.sin(2 * np.pi * 50 * t)  # 50Hz PLI
            noise += 0.03 * signal.filtfilt([1, -0.95], [1], np.random.randn(len(ecg)))  # MA
            
            clean_signals.append(ecg)
            noisy_signals.append(ecg + noise)
        
        return np.array(clean_signals), np.array(noisy_signals)

def main():
    """Test principale del sistema Wavelet-RTM"""
    print("\\n" + "="*60)
    print("AVVIO WAVELET-RTM DENOISER")
    print("="*60)
    
    # Carica dati
    clean_signals, noisy_signals = load_ecg_data_for_training()
    print(f"\\nDati caricati: {len(clean_signals)} segnali di {len(clean_signals[0])} campioni")
    
    # Crea e addestra denoiser
    wrtm = WaveletRTMDenoiser(wavelet='db8', levels=4)
    
    # Training
    wrtm.train_wavelet_rtms(clean_signals, noisy_signals)
    
    # Test denoising
    print("\\n=== TEST DENOISING ===")
    test_signals = noisy_signals[:2]  # Test su primi 2 segnali
    denoised = wrtm.denoise_signals(test_signals)
    
    # Valutazione
    for i in range(len(test_signals)):
        original_snr = calculate_snr(clean_signals[i], test_signals[i])
        denoised_snr = calculate_snr(clean_signals[i], denoised[i])
        
        mse_before = mean_squared_error(clean_signals[i], test_signals[i])
        mse_after = mean_squared_error(clean_signals[i], denoised[i])
        
        print(f"\\nSegnale {i+1}:")
        print(f"  SNR: {original_snr:.2f} dB → {denoised_snr:.2f} dB (Δ{denoised_snr-original_snr:+.2f})")
        print(f"  MSE: {mse_before:.6f} → {mse_after:.6f} (Δ{mse_after-mse_before:+.6f})")
        print(f"  Miglioramento: {'✅' if denoised_snr > original_snr and mse_after < mse_before else '❌'}")
    
    # Salva modello
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'wavelet_rtm_denoiser.pkl')
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(wrtm, f)
    print(f"\\n💾 Modello salvato: {model_path}")

if __name__ == "__main__":
    main()
