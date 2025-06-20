#!/usr/bin/env python3
"""
S6 RTM Training - Training Corretto con Input Binari e Target Continui
====================================================================

Addestra RegressionTsetlinMachine per predizione del rumore ECG usando:
- INPUT: Dati binari da s5_rtm_preprocessing.py (X_*_BINARIZED_q*.npy)
- TARGET: Rumore continuo (y_*_aggregated_noise.npy)

Pipeline Corretta:
1. Carica dati preprocessati (input binari + target continui)
2. Addestra RTM con configurazione ottimizzata
3. Valuta performance e salva modello
4. Gestisce serialization issues di RTM

Autore: Pipeline RTM Corretta
Data: Giugno 2025
"""

import numpy as np
import os
import sys
import time
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importa utility ECG
sys.path.append('/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg')

try:
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
    print("✅ PyTsetlinMachine importata correttamente")
except ImportError:
    print("❌ PyTsetlinMachine non trovata. Installare con: pip install pyTsetlinMachine")
    sys.exit(1)

# Configurazione paths
BASE_PATH = '/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg'
DATA_PATH = os.path.join(BASE_PATH, 'data', 'samplewise')
MODEL_PATH = os.path.join(BASE_PATH, 'models', 'rtm_denoiser')

# Configurazione RTM ottimizzata per iniziare con parametri moderati
RTM_CONFIG = {
    'number_of_clauses': 100,      # Iniziamo con un numero ridotto
    'T': 5000,                     # Temperatura moderata
    's': 2.0,                      # Sensibilità ridotta
    'boost_true_positive_feedback': 0,
    'number_of_state_bits': 8
}

def ensure_directory(path):
    """Crea directory se non esiste"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✅ Creata directory: {path}")

def load_preprocessed_data(num_quantiles=20):
    """
    Carica i dati preprocessati da s5_rtm_preprocessing.py
    
    Returns:
        Dict con dati train/val/test o None se errore
    """
    print("📂 Caricamento dati preprocessati...")
    
    suffix = f"_BINARIZED_q{num_quantiles}"
    
    # File da caricare con target continui
    files_map = {
        'X_train': f"X_train_rtm{suffix}.npy",
        'X_val': f"X_validation_rtm{suffix}.npy", 
        'X_test': f"X_test_rtm{suffix}.npy",
        'y_train': "y_train_rtm_aggregated_noise.npy",
        'y_val': "y_validation_rtm_aggregated_noise.npy",
        'y_test': "y_test_rtm_aggregated_noise.npy"
    }
    
    data = {}
    
    for key, filename in files_map.items():
        filepath = os.path.join(DATA_PATH, filename)
        
        if os.path.exists(filepath):
            try:
                data[key] = np.load(filepath)
                print(f"   ✅ {key}: {data[key].shape} - {filepath}")
            except Exception as e:
                print(f"   ❌ Errore caricamento {key}: {e}")
                return None
        else:
            print(f"   ❌ File non trovato: {filename}")
            print(f"      Eseguire prima: python src/s5_rtm_preprocessing.py")
            return None
    
    # Validazione dati
    print("\\n🔍 Validazione dati:")
    print(f"   🔸 X_train shape: {data['X_train'].shape}, dtype: {data['X_train'].dtype}")
    print(f"   🔸 y_train shape: {data['y_train'].shape}, dtype: {data['y_train'].dtype}")
    
    # Verifica che X sia binario
    unique_X = np.unique(data['X_train'])
    print(f"   🔸 Valori unici X: {unique_X}")
    if not (len(unique_X) <= 2 and all(val in [0, 1] for val in unique_X)):
        print(f"   ⚠️  ATTENZIONE: X non è perfettamente binario!")
    else:
        print(f"   ✅ X è correttamente binario")
    
    # Statistiche target continui
    print(f"   🔸 Target y range: [{data['y_train'].min():.6f}, {data['y_train'].max():.6f}]")
    print(f"   🔸 Target y mean±std: {data['y_train'].mean():.6f} ± {data['y_train'].std():.6f}")
    
    # Verifica compatibilità dimensioni
    if data['X_train'].shape[0] != data['y_train'].shape[0]:
        print(f"   ❌ Errore: Numero campioni diverso X:{data['X_train'].shape[0]} vs y:{data['y_train'].shape[0]}")
        return None
    
    print("✅ Dati validati correttamente")
    return data

class RTMTrainer:
    """Classe per training RTM con gestione robusta"""
    
    def __init__(self, config=None):
        self.config = config or RTM_CONFIG
        self.rtm = None
        self.training_history = []
        self.is_trained = False
        
    def create_rtm(self):
        """Crea istanza RTM con configurazione specificata"""
        print("🔧 Creazione istanza RTM...")
        print(f"   📊 Configurazione: {self.config}")
        
        try:
            self.rtm = RegressionTsetlinMachine(
                number_of_clauses=self.config['number_of_clauses'],
                T=self.config['T'],
                s=self.config['s'],
                boost_true_positive_feedback=self.config['boost_true_positive_feedback'],
                number_of_state_bits=self.config.get('number_of_state_bits', 8)
            )
            print("✅ RTM creata con successo")
            return True
        except Exception as e:
            print(f"❌ Errore creazione RTM: {e}")
            return False
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=5):
        """
        Training del modello RTM con validazione
        
        Args:
            X_train, y_train: Dati di training
            X_val, y_val: Dati di validazione  
            epochs: Numero di epoche
        """
        print(f"\\n🚀 INIZIO TRAINING RTM")
        print("=" * 50)
        
        if self.rtm is None and not self.create_rtm():
            return False
        
        print(f"📊 Dataset info:")
        print(f"   Train: X{X_train.shape} → y{y_train.shape}")
        print(f"   Val:   X{X_val.shape} → y{y_val.shape}")
        
        best_val_mse = float('inf')
        patience = 2
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            print(f"\\n📍 Epoca {epoch+1}/{epochs}")
            
            try:
                # Shuffle dati per questa epoca
                X_epoch, y_epoch = shuffle(X_train, y_train, random_state=42+epoch)
                
                # Training RTM (1 epoca)
                print("   🔄 Training RTM...")
                self.rtm.fit(X_epoch, y_epoch, epochs=1)
                
                # Validazione
                print("   🔄 Validazione...")
                
                # Train metrics (su subset per velocità)
                train_subset = min(1000, len(X_train))
                train_idx = np.random.choice(len(X_train), train_subset, replace=False)
                train_pred = self.rtm.predict(X_train[train_idx])
                train_mse = mean_squared_error(y_train[train_idx].flatten(), train_pred.flatten())
                train_corr = np.corrcoef(y_train[train_idx].flatten(), train_pred.flatten())[0,1]
                if np.isnan(train_corr):
                    train_corr = 0.0
                
                # Validation metrics
                val_pred = self.rtm.predict(X_val)
                val_mse = mean_squared_error(y_val.flatten(), val_pred.flatten())
                val_corr = np.corrcoef(y_val.flatten(), val_pred.flatten())[0,1]
                if np.isnan(val_corr):
                    val_corr = 0.0
                
                epoch_time = time.time() - start_time
                
                # Log risultati
                print(f"   📊 Train MSE: {train_mse:.6f}, Corr: {train_corr:.4f}")
                print(f"   📊 Val   MSE: {val_mse:.6f}, Corr: {val_corr:.4f}")
                print(f"   ⏱️  Tempo: {epoch_time:.1f}s")
                
                # Salva history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'train_corr': train_corr,
                    'val_corr': val_corr,
                    'time': epoch_time
                })
                
                # Early stopping check
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    patience_counter = 0
                    print(f"   🎯 Nuovo miglior val_mse: {val_mse:.6f}")
                else:
                    patience_counter += 1
                    print(f"   ⏳ Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"   🛑 Early stopping")
                        break
                        
            except Exception as e:
                print(f"   ❌ Errore epoca {epoch+1}: {e}")
                return False
        
        self.is_trained = True
        print(f"\\n✅ TRAINING COMPLETATO!")
        print(f"🎯 Miglior val MSE: {best_val_mse:.6f}")
        return True
    
    def evaluate(self, X_test, y_test):
        """Valutazione finale del modello"""
        if not self.is_trained:
            print("❌ Modello non addestrato")
            return None
        
        print("\\n📊 VALUTAZIONE FINALE")
        print("=" * 30)
        
        try:
            # Predizione
            y_pred = self.rtm.predict(X_test)
            
            # Metriche
            mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            correlation = np.corrcoef(y_test.flatten(), y_pred.flatten())[0,1]
            if np.isnan(correlation):
                correlation = 0.0
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation,
                'n_test_samples': len(X_test)
            }
            
            print(f"🎯 RISULTATI FINALI:")
            print(f"   MSE:          {mse:.6f}")
            print(f"   RMSE:         {rmse:.6f}")
            print(f"   MAE:          {mae:.6f}")
            print(f"   Correlazione: {correlation:.4f}")
            print(f"   Campioni:     {len(X_test)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Errore valutazione: {e}")
            return None
    
    def save_results(self, results, num_quantiles):
        """Salva i risultati (evitando problemi pickle con RTM)"""
        ensure_directory(MODEL_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepara dati da salvare (SENZA oggetto RTM)
        save_data = {
            'config': self.config,
            'training_history': self.training_history,
            'final_results': results,
            'num_quantiles': num_quantiles,
            'timestamp': timestamp,
            'note': 'RTM object not saved due to serialization issues'
        }
        
        # Salva risultati
        results_file = os.path.join(MODEL_PATH, f"rtm_results_q{num_quantiles}_{timestamp}.pkl")
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"💾 Risultati salvati: {results_file}")
        except Exception as e:
            print(f"⚠️  Errore salvataggio risultati: {e}")
        
        # Plot training history
        if self.training_history:
            try:
                plt.figure(figsize=(12, 4))
                
                # MSE plot
                plt.subplot(1, 2, 1)
                epochs = [h['epoch'] for h in self.training_history]
                train_mse = [h['train_mse'] for h in self.training_history]
                val_mse = [h['val_mse'] for h in self.training_history]
                
                plt.plot(epochs, train_mse, 'b-o', label='Train MSE', markersize=4)
                plt.plot(epochs, val_mse, 'r-o', label='Validation MSE', markersize=4)
                plt.xlabel('Epoca')
                plt.ylabel('MSE')
                plt.title('Training History - MSE')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Correlation plot
                plt.subplot(1, 2, 2)
                train_corr = [h['train_corr'] for h in self.training_history]
                val_corr = [h['val_corr'] for h in self.training_history]
                
                plt.plot(epochs, train_corr, 'b-o', label='Train Correlation', markersize=4)
                plt.plot(epochs, val_corr, 'r-o', label='Validation Correlation', markersize=4)
                plt.xlabel('Epoca')
                plt.ylabel('Correlazione')
                plt.title('Training History - Correlation')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = os.path.join(MODEL_PATH, f"training_plot_q{num_quantiles}_{timestamp}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📈 Plot salvato: {plot_file}")
                
            except Exception as e:
                print(f"⚠️  Errore creazione plot: {e}")

def main():
    """Funzione principale"""
    print("🚀 S6 RTM Training - Pipeline Corretta")
    print("=" * 60)
    
    # Parametri
    NUM_QUANTILES = 20  # Deve corrispondere a s5_rtm_preprocessing.py
    EPOCHS = 3          # Poche epoche per test iniziale
    
    # Crea directory output
    ensure_directory(MODEL_PATH)
    
    # Carica dati
    data = load_preprocessed_data(NUM_QUANTILES)
    if data is None:
        print("❌ Impossibile caricare dati. Uscita.")
        return

    # Carica il massimo valore assoluto del target di rumore (per normalizzazione)
    y_train_max_abs_noise_path = os.path.join(DATA_PATH, "y_train_max_abs_noise.npy")
    if not os.path.exists(y_train_max_abs_noise_path):
        print(f"❌ File y_train_max_abs_noise.npy non trovato in {DATA_PATH}. Esegui prima s5_rtm_preprocessing.py.")
        return
    y_train_max_abs_noise = np.load(y_train_max_abs_noise_path)
    print(f"✅ y_train_max_abs_noise caricato: {y_train_max_abs_noise:.6f}")

    # Normalizza i target
    data['y_train'] = data['y_train'] / y_train_max_abs_noise
    data['y_val'] = data['y_val'] / y_train_max_abs_noise
    data['y_test'] = data['y_test'] / y_train_max_abs_noise
    print(f"✅ Target normalizzati in [-1, 1] (circa)")

    # Crea trainer
    trainer = RTMTrainer(RTM_CONFIG)
    
    # Training
    success = trainer.train_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=EPOCHS
    )
    
    if not success:
        print("❌ Training fallito")
        return
    
    # Valutazione
    results = trainer.evaluate(data['X_test'], data['y_test'])
    
    if results:
        # Salva tutto
        trainer.save_results(results, NUM_QUANTILES)
        
        # Riassunto finale
        print(f"\\n" + "="*50)
        print(f"🎯 RIASSUNTO FINALE")
        print(f"="*50)
        print(f"📊 Configurazione RTM: {RTM_CONFIG}")
        print(f"📊 Quantili usati: {NUM_QUANTILES}")
        print(f"📊 Epoche training: {len(trainer.training_history)}")
        print(f"📊 MSE finale: {results['mse']:.6f}")
        print(f"📊 Correlazione: {results['correlation']:.4f}")
        
        # Valutazione performance
        if results['correlation'] > 0.5:
            print("🎉 ECCELLENTE! Correlazione > 0.5")
        elif results['correlation'] > 0.3:
            print("🎊 BUONA! Correlazione > 0.3")
        elif results['correlation'] > 0.1:
            print("🤔 MODERATA. Correlazione > 0.1")
        else:
            print("😞 BASSA. Correlazione < 0.1 - Serve ottimizzazione")
            
        print("\\n✨ PIPELINE COMPLETATA CON SUCCESSO!")
    else:
        print("❌ Valutazione fallita")

if __name__ == "__main__":
    main()
