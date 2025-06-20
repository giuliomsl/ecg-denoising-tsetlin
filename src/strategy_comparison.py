#!/usr/bin/env python3
"""
Confronto Strategico: Predizione Rumore vs Ricostruzione Diretta
===============================================================

Test sistematico delle due strategie principali per RTM ECG denoising:

STRATEGIA A: Predizione del Rumore (Attuale)
Input: Segnale Rumoroso → RTM → Rumore Predetto
Output: Segnale Rumoroso - Rumore Predetto

STRATEGIA B: Ricostruzione Diretta  
Input: Segnale Rumoroso → RTM → Segnale Pulito Diretto

Confronto sistematico per identificare l'approccio migliore.

Autore: Sistema di Analisi RTM
Data: Giugno 2025
"""

import numpy as np
import os
import sys
from pyTsetlinMachine.tm import RegressionTsetlinMachine
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg')
from src.load_ecg import *
from src.ecg_utils import calculate_snr

class StrategyComparator:
    """Confronta le due strategie principali per RTM denoising"""
    
    def __init__(self):
        # Configurazione RTM ottimizzata per entrambe le strategie
        self.base_config = {
            'number_of_clauses': 400,
            'T': 10000,
            's': 3.0,
            'boost_true_positive_feedback': 0
        }
        
        # RTM per predizione rumore
        self.noise_predictor = RegressionTsetlinMachine(**self.base_config)
        
        # RTM per ricostruzione diretta  
        self.direct_reconstructor = RegressionTsetlinMachine(**self.base_config)
        
        self.results = {}
    
    def load_data(self):
        """Carica dati per il confronto"""
        print("=== CARICAMENTO DATI PER CONFRONTO ===")
        
        try:
            # Dati binarizzati per RTM
            X_train = np.load(os.path.join(SAMPLE_DATA_DIR, 'X_train_rtm_BINARIZED_q100.npy'))
            X_val = np.load(os.path.join(SAMPLE_DATA_DIR, 'X_validation_rtm_BINARIZED_q100.npy'))
            
            # Target per strategia A: rumore aggregato
            y_noise_train = np.load(os.path.join(SAMPLE_DATA_DIR, 'y_train_rtm_aggregated_noise.npy'))
            y_noise_val = np.load(os.path.join(SAMPLE_DATA_DIR, 'y_validation_rtm_aggregated_noise.npy'))
            
            # Target per strategia B: segnali puliti originali
            clean_train = np.load(os.path.join(SAMPLE_DATA_DIR, 'clean_train_original_center_samples.npy'))
            clean_val = np.load(os.path.join(SAMPLE_DATA_DIR, 'clean_validation_original_center_samples.npy'))
            
            # Segnali rumorosi originali (per calcolo metriche)
            noisy_train = np.load(os.path.join(SAMPLE_DATA_DIR, 'noisy_train_original_center_samples.npy'))
            noisy_val = np.load(os.path.join(SAMPLE_DATA_DIR, 'noisy_validation_original_center_samples.npy'))
            
            print(f"Dati caricati:")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  Target rumore shape: {y_noise_train.shape}")
            print(f"  Target pulito shape: {clean_train.shape}")
            
            return {
                'X_train': X_train, 'X_val': X_val,
                'y_noise_train': y_noise_train, 'y_noise_val': y_noise_val,
                'clean_train': clean_train, 'clean_val': clean_val,
                'noisy_train': noisy_train, 'noisy_val': noisy_val
            }
            
        except Exception as e:
            print(f"ERRORE caricamento dati: {e}")
            return None
    
    def test_strategy_a_noise_prediction(self, data):
        """STRATEGIA A: Predizione del Rumore"""
        print("\\n" + "="*50)
        print("STRATEGIA A: PREDIZIONE DEL RUMORE")
        print("="*50)
        
        # Subset per training veloce
        n_samples = 5000
        X_train_subset = data['X_train'][:n_samples]
        y_noise_subset = data['y_noise_train'][:n_samples]
        
        print(f"Training RTM per predizione rumore su {n_samples} samples...")
        
        # Training
        self.noise_predictor.fit(X_train_subset, y_noise_subset, epochs=8)
        
        # Test su validation
        print("Test su validation set...")
        noise_predicted = self.noise_predictor.predict(data['X_val'])
        
        # Ricostruisco segnale pulito: rumoroso - rumore_predetto
        denoised_signals = data['noisy_val'] - noise_predicted
        
        # Metriche
        results_a = self.calculate_metrics(
            data['clean_val'], data['noisy_val'], denoised_signals,
            noise_predicted, data['y_noise_val'], "STRATEGIA A"
        )
        
        self.results['strategy_a'] = results_a
        return results_a
    
    def test_strategy_b_direct_reconstruction(self, data):
        """STRATEGIA B: Ricostruzione Diretta"""
        print("\\n" + "="*50)
        print("STRATEGIA B: RICOSTRUZIONE DIRETTA")
        print("="*50)
        
        # Subset per training veloce
        n_samples = 5000
        X_train_subset = data['X_train'][:n_samples]
        clean_subset = data['clean_train'][:n_samples]
        
        print(f"Training RTM per ricostruzione diretta su {n_samples} samples...")
        
        # Training
        self.direct_reconstructor.fit(X_train_subset, clean_subset, epochs=8)
        
        # Test su validation
        print("Test su validation set...")
        denoised_direct = self.direct_reconstructor.predict(data['X_val'])
        
        # Calcolo rumore "rimosso" per confronto
        noise_removed = data['noisy_val'] - denoised_direct
        
        # Metriche
        results_b = self.calculate_metrics(
            data['clean_val'], data['noisy_val'], denoised_direct,
            noise_removed, data['y_noise_val'], "STRATEGIA B"
        )
        
        self.results['strategy_b'] = results_b
        return results_b
    
    def calculate_metrics(self, clean_true, noisy_original, denoised_result, 
                         noise_component, noise_true, strategy_name):
        """Calcola metriche complete per una strategia"""
        
        # Subset per velocità
        n_test = min(1000, len(clean_true))
        clean_true = clean_true[:n_test]
        noisy_original = noisy_original[:n_test]
        denoised_result = denoised_result[:n_test]
        noise_component = noise_component[:n_test]
        noise_true = noise_true[:n_test]
        
        results = {}
        
        # 1. Metriche principali denoising
        mse_before = mean_squared_error(clean_true.flatten(), noisy_original.flatten())
        mse_after = mean_squared_error(clean_true.flatten(), denoised_result.flatten())
        
        # 2. SNR medio
        snr_before_list = []
        snr_after_list = []
        
        for i in range(n_test):
            snr_before = calculate_snr(clean_true[i], noisy_original[i])
            snr_after = calculate_snr(clean_true[i], denoised_result[i])
            
            if not np.isnan(snr_before) and not np.isinf(snr_before):
                snr_before_list.append(snr_before)
            if not np.isnan(snr_after) and not np.isinf(snr_after):
                snr_after_list.append(snr_after)
        
        snr_before_avg = np.mean(snr_before_list) if snr_before_list else 0
        snr_after_avg = np.mean(snr_after_list) if snr_after_list else 0
        
        # 3. Correlazione predizione vs target
        if strategy_name == "STRATEGIA A":
            # Correlazione rumore predetto vs rumore vero
            corr_target = np.corrcoef(noise_component.flatten(), noise_true.flatten())[0, 1]
            target_desc = "rumore"
        else:
            # Correlazione segnale ricostruito vs segnale pulito
            corr_target = np.corrcoef(denoised_result.flatten(), clean_true.flatten())[0, 1]
            target_desc = "segnale pulito"
        
        # 4. Percentuale campioni migliorati
        improved_samples = 0
        for i in range(n_test):
            mse_before_i = mean_squared_error(clean_true[i], noisy_original[i])
            mse_after_i = mean_squared_error(clean_true[i], denoised_result[i])
            if mse_after_i < mse_before_i:
                improved_samples += 1
        
        improvement_rate = (improved_samples / n_test) * 100
        
        results = {
            'mse_before': mse_before,
            'mse_after': mse_after,
            'mse_improvement': mse_before - mse_after,
            'mse_improvement_pct': ((mse_before - mse_after) / mse_before) * 100,
            'snr_before_avg': snr_before_avg,
            'snr_after_avg': snr_after_avg,
            'snr_improvement': snr_after_avg - snr_before_avg,
            'correlation_target': corr_target,
            'improvement_rate': improvement_rate,
            'target_description': target_desc
        }
        
        # Print risultati
        print(f"\\n📊 RISULTATI {strategy_name}:")
        print(f"   MSE: {mse_before:.6f} → {mse_after:.6f} (Δ{results['mse_improvement']:+.6f})")
        print(f"   MSE Miglioramento: {results['mse_improvement_pct']:+.2f}%")
        print(f"   SNR: {snr_before_avg:.2f} dB → {snr_after_avg:.2f} dB (Δ{results['snr_improvement']:+.2f})")
        print(f"   Correlazione {target_desc}: {corr_target:.4f}")
        print(f"   Campioni migliorati: {improvement_rate:.1f}%")
        print(f"   Verdetto: {'✅ MIGLIORAMENTO' if results['mse_improvement'] > 0 and results['snr_improvement'] > 0 else '❌ PEGGIORAMENTO'}")
        
        return results
    
    def compare_strategies(self):
        """Confronto finale tra le strategie"""
        print("\\n" + "="*60)
        print("CONFRONTO FINALE DELLE STRATEGIE")
        print("="*60)
        
        if 'strategy_a' not in self.results or 'strategy_b' not in self.results:
            print("ERRORE: Non tutte le strategie sono state testate")
            return
        
        a = self.results['strategy_a']
        b = self.results['strategy_b']
        
        print("\\n📈 CONFRONTO PRESTAZIONI:")
        print(f"\\n1. MSE MIGLIORAMENTO:")
        print(f"   Strategia A (Predizione Rumore):    {a['mse_improvement']:+.6f} ({a['mse_improvement_pct']:+.2f}%)")
        print(f"   Strategia B (Ricostruzione Diretta): {b['mse_improvement']:+.6f} ({b['mse_improvement_pct']:+.2f}%)")
        print(f"   🏆 VINCITORE MSE: {'Strategia A' if a['mse_improvement'] > b['mse_improvement'] else 'Strategia B'}")
        
        print(f"\\n2. SNR MIGLIORAMENTO:")
        print(f"   Strategia A: {a['snr_improvement']:+.2f} dB")
        print(f"   Strategia B: {b['snr_improvement']:+.2f} dB")
        print(f"   🏆 VINCITORE SNR: {'Strategia A' if a['snr_improvement'] > b['snr_improvement'] else 'Strategia B'}")
        
        print(f"\\n3. CORRELAZIONE TARGET:")
        print(f"   Strategia A ({a['target_description']}): {a['correlation_target']:.4f}")
        print(f"   Strategia B ({b['target_description']}): {b['correlation_target']:.4f}")
        print(f"   🏆 VINCITORE CORRELAZIONE: {'Strategia A' if a['correlation_target'] > b['correlation_target'] else 'Strategia B'}")
        
        print(f"\\n4. TASSO MIGLIORAMENTO:")
        print(f"   Strategia A: {a['improvement_rate']:.1f}% campioni migliorati")
        print(f"   Strategia B: {b['improvement_rate']:.1f}% campioni migliorati")
        print(f"   🏆 VINCITORE TASSO: {'Strategia A' if a['improvement_rate'] > b['improvement_rate'] else 'Strategia B'}")
        
        # Calcola score complessivo
        score_a = 0
        score_b = 0
        
        if a['mse_improvement'] > b['mse_improvement']: score_a += 1
        else: score_b += 1
        
        if a['snr_improvement'] > b['snr_improvement']: score_a += 1
        else: score_b += 1
        
        if a['correlation_target'] > b['correlation_target']: score_a += 1
        else: score_b += 1
        
        if a['improvement_rate'] > b['improvement_rate']: score_a += 1
        else: score_b += 1
        
        print(f"\\n🎯 RISULTATO FINALE:")
        print(f"   Strategia A (Predizione Rumore): {score_a}/4 punti")
        print(f"   Strategia B (Ricostruzione Diretta): {score_b}/4 punti")
        
        if score_a > score_b:
            winner = "STRATEGIA A - PREDIZIONE DEL RUMORE"
            recommendation = "Continuare con l'approccio attuale di predizione del rumore"
        elif score_b > score_a:
            winner = "STRATEGIA B - RICOSTRUZIONE DIRETTA"
            recommendation = "Cambiare a ricostruzione diretta del segnale pulito"
        else:
            winner = "PAREGGIO"
            recommendation = "Testare entrambe le strategie con più dati/epoche"
        
        print(f"\\n🏆 VINCITORE ASSOLUTO: {winner}")
        print(f"📋 RACCOMANDAZIONE: {recommendation}")
        
        # Salva risultati
        results_path = os.path.join(MODEL_OUTPUT_DIR, 'strategy_comparison.pkl')
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"\\n💾 Risultati salvati in: {results_path}")
        
        return winner, recommendation

def main():
    """Test principale del confronto strategico"""
    print("\\n" + "="*60)
    print("CONFRONTO STRATEGICO RTM ECG DENOISING")
    print("="*60)
    
    comparator = StrategyComparator()
    
    # Carica dati
    data = comparator.load_data()
    if data is None:
        print("ERRORE: Impossibile caricare i dati")
        return
    
    # Test Strategia A
    results_a = comparator.test_strategy_a_noise_prediction(data)
    
    # Test Strategia B  
    results_b = comparator.test_strategy_b_direct_reconstruction(data)
    
    # Confronto finale
    winner, recommendation = comparator.compare_strategies()
    
    print(f"\\n🎉 ANALISI COMPLETATA!")
    print(f"Prossimo step: {recommendation}")

if __name__ == "__main__":
    main()
