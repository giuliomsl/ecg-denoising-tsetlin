#!/usr/bin/env python3
"""
Test di diverse configurazioni RTM per trovare la migliore per il denoising ECG.
Questo script testa sistematicamente diverse combinazioni di parametri.
"""

import numpy as np
import yaml
import sys
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Aggiungi la directory src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ecg_utils import calculate_snr
from load_ecg import load_ecg_data

try:
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
except ImportError:
    print("PyTsetlinMachine non installato. Installare con: pip install pyTsetlinMachine")
    sys.exit(1)

class RTMConfigurationTester:
    def __init__(self):
        self.test_results = []
        
    def thermometer_encode(self, data, num_bits=8):
        """Codifica termometrica per RTM"""
        data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        thresholds = np.linspace(0, 1, num_bits)
        encoded = np.zeros((len(data), num_bits), dtype=int)
        
        for i, value in enumerate(data_normalized):
            encoded[i] = (value >= thresholds).astype(int)
        
        return encoded
    
    def create_training_data(self, clean_signal, noisy_signal, window_size=32):
        """Crea dataset di training con finestre scorrevoli"""
        X = []
        y = []
        
        for i in range(len(noisy_signal) - window_size + 1):
            noisy_window = noisy_signal[i:i + window_size]
            clean_target = clean_signal[i + window_size // 2]
            
            encoded_window = self.thermometer_encode(noisy_window)
            encoded_target = self.thermometer_encode([clean_target])
            
            X.append(encoded_window.flatten())
            y.append(encoded_target.flatten())
        
        return np.array(X), np.array(y)
    
    def test_configuration(self, config, clean_signal, noisy_signal, test_name=""):
        """Testa una specifica configurazione RTM"""
        
        print(f"\n--- Testing: {test_name} ---")
        print(f"Config: {config}")
        
        # Split train/test
        split_idx = int(0.8 * len(clean_signal))
        clean_train, clean_test = clean_signal[:split_idx], clean_signal[split_idx:]
        noisy_train, noisy_test = noisy_signal[:split_idx], noisy_signal[split_idx:]
        
        # Crea dataset
        X_train, y_train = self.create_training_data(clean_train, noisy_train)
        X_test, y_test = self.create_training_data(clean_test, noisy_test)
        
        # Inizializza RTM
        try:
            rtm = RegressionTsetlinMachine(
                number_of_clauses=config['number_of_clauses'],
                T=config['T'],
                s=config['s'],
                number_of_regressor_clauses=config.get('number_of_regressor_clauses', 1),
                weighted_clauses=config.get('weighted_clauses', True)
            )
            
            # Training
            rtm.fit(X_train, y_train, epochs=config.get('epochs', 100))
            
            # Predizioni
            y_pred_test = rtm.predict(X_test)
            
            # Metriche
            mse = mean_squared_error(y_test, y_pred_test)
            correlation = np.corrcoef(y_test.flatten(), y_pred_test.flatten())[0, 1]
            
            # Ricostruzione segnale per SNR
            reconstructed = self.reconstruct_signal(rtm, noisy_test)
            if reconstructed is not None:
                snr_improvement = calculate_snr(clean_test[:len(reconstructed)], reconstructed) - \
                                calculate_snr(clean_test[:len(reconstructed)], noisy_test[:len(reconstructed)])
            else:
                snr_improvement = 0
            
            result = {
                'config': config,
                'test_name': test_name,
                'mse': mse,
                'correlation': correlation,
                'snr_improvement': snr_improvement,
                'success': True
            }
            
            print(f"MSE: {mse:.6f}")
            print(f"Correlation: {correlation:.4f}")
            print(f"SNR Improvement: {snr_improvement:.2f} dB")
            
        except Exception as e:
            print(f"Errore nella configurazione: {e}")
            result = {
                'config': config,
                'test_name': test_name,
                'mse': float('inf'),
                'correlation': 0,
                'snr_improvement': 0,
                'success': False,
                'error': str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def reconstruct_signal(self, rtm, noisy_signal, window_size=32):
        """Ricostruisce il segnale usando RTM"""
        try:
            reconstructed = []
            
            for i in range(len(noisy_signal) - window_size + 1):
                noisy_window = noisy_signal[i:i + window_size]
                encoded_window = self.thermometer_encode(noisy_window)
                
                pred_encoded = rtm.predict(encoded_window.flatten().reshape(1, -1))
                decoded_value = np.sum(pred_encoded) / len(pred_encoded)
                reconstructed.append(decoded_value)
            
            return np.array(reconstructed)
        except:
            return None
    
    def run_comprehensive_test(self, clean_signal, noisy_signal):
        """Esegue test completo di diverse configurazioni"""
        
        configurations = [
            # Configurazioni base
            {
                'name': 'Base Small',
                'number_of_clauses': 100,
                'T': 15,
                's': 3.0,
                'epochs': 100
            },
            {
                'name': 'Base Medium',
                'number_of_clauses': 500,
                'T': 20,
                's': 3.9,
                'epochs': 150
            },
            {
                'name': 'Base Large',
                'number_of_clauses': 1000,
                'T': 25,
                's': 4.5,
                'epochs': 200
            },
            
            # Variazioni su T (threshold)
            {
                'name': 'Low Threshold',
                'number_of_clauses': 500,
                'T': 10,
                's': 3.9,
                'epochs': 150
            },
            {
                'name': 'High Threshold',
                'number_of_clauses': 500,
                'T': 50,
                's': 3.9,
                'epochs': 150
            },
            
            # Variazioni su s (specificity)
            {
                'name': 'Low Specificity',
                'number_of_clauses': 500,
                'T': 20,
                's': 2.0,
                'epochs': 150
            },
            {
                'name': 'High Specificity',
                'number_of_clauses': 500,
                'T': 20,
                's': 6.0,
                'epochs': 150
            },
            
            # Configurazioni avanzate con parametri aggiuntivi
            {
                'name': 'Weighted Clauses',
                'number_of_clauses': 500,
                'T': 20,
                's': 3.9,
                'epochs': 150,
                'weighted_clauses': True,
                'number_of_regressor_clauses': 2
            },
            {
                'name': 'Multiple Regressors',
                'number_of_clauses': 800,
                'T': 25,
                's': 4.0,
                'epochs': 200,
                'weighted_clauses': True,
                'number_of_regressor_clauses': 5
            },
            
            # Configurazioni per dati ad alta dimensionalità
            {
                'name': 'High Dimension Optimized',
                'number_of_clauses': 2000,
                'T': 30,
                's': 5.0,
                'epochs': 300,
                'weighted_clauses': True,
                'number_of_regressor_clauses': 3
            }
        ]
        
        print("=== RTM CONFIGURATION TESTING ===")
        print(f"Testing {len(configurations)} different configurations...")
        
        for config in configurations:
            self.test_configuration(config, clean_signal, noisy_signal, config['name'])
        
        # Analizza risultati
        self.analyze_results()
    
    def analyze_results(self):
        """Analizza e visualizza i risultati dei test"""
        
        print("\n" + "="*80)
        print("RISULTATI COMPLETI DEI TEST")
        print("="*80)
        
        # Ordina per MSE (migliore = MSE più basso)
        successful_results = [r for r in self.test_results if r['success']]
        failed_results = [r for r in self.test_results if not r['success']]
        
        if successful_results:
            successful_results.sort(key=lambda x: x['mse'])
            
            print("\nTOP 5 CONFIGURAZIONI (per MSE):")
            for i, result in enumerate(successful_results[:5]):
                print(f"{i+1}. {result['test_name']}")
                print(f"   MSE: {result['mse']:.6f}")
                print(f"   Correlation: {result['correlation']:.4f}")
                print(f"   SNR Improvement: {result['snr_improvement']:.2f} dB")
                print(f"   Config: {result['config']}")
                print()
            
            # Ordina per correlazione
            successful_results.sort(key=lambda x: x['correlation'], reverse=True)
            
            print("\nTOP 5 CONFIGURAZIONI (per Correlazione):")
            for i, result in enumerate(successful_results[:5]):
                print(f"{i+1}. {result['test_name']}")
                print(f"   Correlation: {result['correlation']:.4f}")
                print(f"   MSE: {result['mse']:.6f}")
                print(f"   SNR Improvement: {result['snr_improvement']:.2f} dB")
                print()
            
            # Migliore configurazione complessiva
            best_overall = min(successful_results, 
                             key=lambda x: x['mse'] - x['correlation'] + abs(x['snr_improvement']))
            
            print("\nMIGLIORE CONFIGURAZIONE COMPLESSIVA:")
            print(f"Nome: {best_overall['test_name']}")
            print(f"MSE: {best_overall['mse']:.6f}")
            print(f"Correlation: {best_overall['correlation']:.4f}")
            print(f"SNR Improvement: {best_overall['snr_improvement']:.2f} dB")
            print(f"Config: {best_overall['config']}")
            
        if failed_results:
            print(f"\nCONFIGURAZIONI FALLITE: {len(failed_results)}")
            for result in failed_results:
                print(f"- {result['test_name']}: {result.get('error', 'Unknown error')}")
        
        # Salva risultati migliori
        self.save_best_configuration()
        
        # Plot comparativo
        self.plot_comparison()
    
    def save_best_configuration(self):
        """Salva la migliore configurazione in un file YAML"""
        if not any(r['success'] for r in self.test_results):
            print("Nessuna configurazione di successo da salvare.")
            return
        
        successful_results = [r for r in self.test_results if r['success']]
        best_config = min(successful_results, key=lambda x: x['mse'])
        
        config_to_save = {
            'rtm_optimal': best_config['config'],
            'performance': {
                'mse': float(best_config['mse']),
                'correlation': float(best_config['correlation']),
                'snr_improvement': float(best_config['snr_improvement'])
            },
            'test_name': best_config['test_name']
        }
        
        with open('../config/rtm_optimal_config.yaml', 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)
        
        print(f"\nMigliore configurazione salvata in: ../config/rtm_optimal_config.yaml")
    
    def plot_comparison(self):
        """Crea grafici di confronto delle configurazioni"""
        successful_results = [r for r in self.test_results if r['success']]
        
        if not successful_results:
            print("Nessun risultato di successo da plottare.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        names = [r['test_name'] for r in successful_results]
        mse_values = [r['mse'] for r in successful_results]
        corr_values = [r['correlation'] for r in successful_results]
        snr_values = [r['snr_improvement'] for r in successful_results]
        
        # MSE comparison
        axes[0, 0].barh(names, mse_values)
        axes[0, 0].set_title('MSE per Configurazione')
        axes[0, 0].set_xlabel('MSE')
        
        # Correlation comparison
        axes[0, 1].barh(names, corr_values)
        axes[0, 1].set_title('Correlazione per Configurazione')
        axes[0, 1].set_xlabel('Correlazione')
        
        # SNR improvement comparison
        axes[1, 0].barh(names, snr_values)
        axes[1, 0].set_title('Miglioramento SNR per Configurazione')
        axes[1, 0].set_xlabel('SNR Improvement (dB)')
        
        # Scatter plot MSE vs Correlation
        axes[1, 1].scatter(mse_values, corr_values)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (mse_values[i], corr_values[i]), 
                              fontsize=8, rotation=45)
        axes[1, 1].set_xlabel('MSE')
        axes[1, 1].set_ylabel('Correlazione')
        axes[1, 1].set_title('MSE vs Correlazione')
        
        plt.tight_layout()
        plt.savefig('../plots/rtm_configuration_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Grafici di confronto salvati in: ../plots/rtm_configuration_comparison.png")

def main():
    """Funzione principale per testare le configurazioni RTM"""
    
    # Carica dati di test
    try:
        clean_signal = np.load("../data/generated_signals/100_clean.npy")[:1500]
        noisy_signal = np.load("../data/generated_signals/100_noisy.npy")[:1500]
    except:
        print("Impossibile caricare i dati. Assicurati che esistano i file dei segnali generati.")
        return
    
    print(f"Caricati {len(clean_signal)} campioni per il test")
    
    # Inizializza tester
    tester = RTMConfigurationTester()
    
    # Esegui test completo
    tester.run_comprehensive_test(clean_signal, noisy_signal)

if __name__ == "__main__":
    main()
