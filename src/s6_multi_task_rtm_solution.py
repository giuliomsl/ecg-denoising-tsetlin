#!/usr/bin/env python3
"""
Multi-Task RTM Gerarchica per ECG Denoising
===========================================

SOLUZIONE INNOVATIVA ai risultati scadenti RTM tradizionale.

STRATEGIA:
1. RTM Classificatrice: Identifica tipo di rumore dominante (BW, MA, PLI)  
2. RTM Specializzate: Una per ogni tipo di rumore
3. RTM Ensemble Finale: Combina predizioni con pesi adattivi

VANTAGGI:
- Task specifici invece di regressione generica
- Learning più focalizzato per ogni tipo di rumore
- Ensemble intelligente basato su classificazione

Autore: Sistema RTM Innovativo
Data: Giugno 2025
"""

import numpy as np
import os
import sys
from pyTsetlinMachine.tm import MultiClassTsetlinMachine, RegressionTsetlinMachine
import pickle
from sklearn.metrics import mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg')
from src.load_ecg import *
from src.ecg_utils import calculate_snr

class MultiTaskRTMDenoiser:
    """RTM Multi-Task per ECG Denoising con approccio gerarchico"""
    
    def __init__(self):
        # RTM Classificatrice per tipo di rumore
        self.noise_classifier = MultiClassTsetlinMachine(
            number_of_clauses=800,
            T=15000,
            s=4.0,
            boost_true_positive_feedback=1
        )
        
        # RTM Specializzate per ogni tipo di rumore
        self.denoisers = {
            'bw': RegressionTsetlinMachine(
                number_of_clauses=600,
                T=12000,
                s=3.5,
                boost_true_positive_feedback=0
            ),
            'ma': RegressionTsetlinMachine(
                number_of_clauses=500,
                T=10000,
                s=3.0,
                boost_true_positive_feedback=0
            ),
            'pli': RegressionTsetlinMachine(
                number_of_clauses=400,
                T=8000,
                s=2.5,
                boost_true_positive_feedback=0
            )
        }
        
        # RTM Ensemble finale
        self.ensemble_combiner = RegressionTsetlinMachine(
            number_of_clauses=300,
            T=5000,
            s=2.0,
            boost_true_positive_feedback=0
        )
        
        self.is_trained = False
        
    def prepare_classification_data(self):
        """Prepara dati per classificazione tipo di rumore"""
        print("=== PREPARAZIONE DATI CLASSIFICAZIONE ===")
        
        # Carica dati raw per analisi rumore
        try:
            # Genera labels per tipo di rumore dominante basato sui file separati
            bw_noise = np.load(os.path.join(SAMPLE_DATA_DIR, 'y_train_rtm_bw_noise.npy'))
            ma_noise = np.load(os.path.join(SAMPLE_DATA_DIR, 'y_train_rtm_ma_noise.npy'))
            pli_noise = np.load(os.path.join(SAMPLE_DATA_DIR, 'y_train_rtm_pli_noise.npy'))
            
            # Determina tipo di rumore dominante per ogni sample
            noise_types = []
            
            # Calcola magnitudini per ogni tipo di rumore
            if bw_noise.ndim > 1:
                bw_mag = np.abs(bw_noise).mean(axis=1)
                ma_mag = np.abs(ma_noise).mean(axis=1)
                pli_mag = np.abs(pli_noise).mean(axis=1)
            else:
                bw_mag = np.array([np.abs(bw_noise).mean()])
                ma_mag = np.array([np.abs(ma_noise).mean()])
                pli_mag = np.array([np.abs(pli_noise).mean()])
            
            noise_magnitudes = np.stack([bw_mag, ma_mag, pli_mag], axis=1)
            
            dominant_noise = np.argmax(noise_magnitudes, axis=1)
            noise_labels = ['bw', 'ma', 'pli']
            
            return dominant_noise, noise_labels, {
                'bw': bw_noise,
                'ma': ma_noise, 
                'pli': pli_noise
            }
            
        except Exception as e:
            print(f"Errore caricamento dati: {e}")
            return None, None, None
    
    def train_classifier(self, X_train, noise_classes):
        """Addestra la RTM classificatrice"""
        print("=== TRAINING RTM CLASSIFICATRICE ===")
        
        print(f"Training su {len(X_train)} samples, {len(np.unique(noise_classes))} classi")
        
        self.noise_classifier.fit(X_train, noise_classes, epochs=15)
        
        # Valutazione
        train_pred = self.noise_classifier.predict(X_train)
        accuracy = np.mean(train_pred == noise_classes)
        print(f"Accuratezza classificazione training: {accuracy:.3f}")
        
        return accuracy
    
    def train_specialized_denoisers(self, X_train, noise_data, noise_classes):
        """Addestra RTM specializzate per ogni tipo di rumore"""
        print("=== TRAINING RTM SPECIALIZZATE ===")
        
        noise_labels = ['bw', 'ma', 'pli']
        
        for i, noise_type in enumerate(noise_labels):
            print(f"Training denoiser per rumore: {noise_type.upper()}")
            
            # Seleziona solo i samples di questo tipo di rumore
            mask = (noise_classes == i)
            X_subset = X_train[mask]
            y_subset = noise_data[noise_type][mask]
            
            if len(X_subset) > 0:
                print(f"  Training su {len(X_subset)} samples specifici")
                self.denoisers[noise_type].fit(X_subset, y_subset, epochs=12)
                
                # Test predizione
                pred = self.denoisers[noise_type].predict(X_subset)
                mse = mean_squared_error(y_subset, pred)
                corr = np.corrcoef(y_subset.flatten(), pred.flatten())[0,1]
                print(f"  MSE: {mse:.6f}, Correlazione: {corr:.4f}")
            else:
                print(f"  Nessun sample per tipo {noise_type}")
    
    def train_ensemble_combiner(self, X_train, noise_classes, noise_data):
        """Addestra RTM ensemble che combina le predizioni specializzate"""
        print("=== TRAINING RTM ENSEMBLE COMBINER ===")
        
        # Genera predizioni da tutte le RTM specializzate
        noise_labels = ['bw', 'ma', 'pli']
        ensemble_features = []
        ensemble_targets = []
        
        for i in range(len(X_train)):
            # Predizioni dalle RTM specializzate
            specialized_preds = []
            for noise_type in noise_labels:
                pred = self.denoisers[noise_type].predict(X_train[i:i+1])
                specialized_preds.extend(pred.flatten())
            
            # Predizione classificatore (probabilità soft)
            class_pred = self.noise_classifier.predict(X_train[i:i+1])[0]
            class_confidence = [1.0 if j == class_pred else 0.0 for j in range(3)]
            
            # Feature per ensemble: predizioni + confidenze + features originali ridotte
            feature_vector = specialized_preds + class_confidence + X_train[i][:100].tolist()
            ensemble_features.append(feature_vector)
            
            # Target: rumore aggregato reale
            real_noise = noise_data['bw'][i] + noise_data['ma'][i] + noise_data['pli'][i] 
            ensemble_targets.append(real_noise)
        
        ensemble_features = np.array(ensemble_features)
        ensemble_targets = np.array(ensemble_targets)
        
        print(f"Training ensemble su feature shape: {ensemble_features.shape}")
        self.ensemble_combiner.fit(ensemble_features, ensemble_targets, epochs=10)
        
        # Test ensemble
        ensemble_pred = self.ensemble_combiner.predict(ensemble_features)
        mse = mean_squared_error(ensemble_targets.flatten(), ensemble_pred.flatten())
        corr = np.corrcoef(ensemble_targets.flatten(), ensemble_pred.flatten())[0,1]
        print(f"Ensemble MSE: {mse:.6f}, Correlazione: {corr:.4f}")
    
    def full_training_pipeline(self):
        """Pipeline completa di training multi-task"""
        print("\\n" + "="*60)
        print("TRAINING MULTI-TASK RTM DENOISER")
        print("="*60)
        
        # 1. Carica dati
        X_train = np.load(os.path.join(SAMPLE_DATA_DIR, 'X_train_rtm_BINARIZED_q100.npy'))
        
        # 2. Prepara classificazione rumore
        noise_classes, noise_labels, noise_data = self.prepare_classification_data()
        if noise_classes is None:
            print("ERRORE: Impossibile preparare dati classificazione")
            return False
        
        # 3. Training classificatore
        classifier_acc = self.train_classifier(X_train, noise_classes)
        
        # 4. Training denoisers specializzati
        self.train_specialized_denoisers(X_train, noise_data, noise_classes)
        
        # 5. Training ensemble combiner
        self.train_ensemble_combiner(X_train, noise_classes, noise_data)
        
        self.is_trained = True
        print("\\n🎉 TRAINING MULTI-TASK COMPLETATO!")
        return True
    
    def predict_denoised(self, X_input):
        """Predice segnale denoised usando pipeline multi-task"""
        if not self.is_trained:
            raise ValueError("Modello non ancora addestrato!")
        
        # 1. Classifica tipo di rumore
        noise_type_pred = self.noise_classifier.predict(X_input)
        
        # 2. Predizioni da RTM specializzate
        noise_labels = ['bw', 'ma', 'pli']
        specialized_preds = {}
        for noise_type in noise_labels:
            specialized_preds[noise_type] = self.denoisers[noise_type].predict(X_input)
        
        # 3. Combina con RTM ensemble
        ensemble_features = []
        for i in range(len(X_input)):
            # Feature per ensemble
            preds_flat = []
            for noise_type in noise_labels:
                preds_flat.extend(specialized_preds[noise_type][i].flatten())
            
            class_conf = [1.0 if j == noise_type_pred[i] else 0.0 for j in range(3)]
            feature_vector = preds_flat + class_conf + X_input[i][:100].tolist()
            ensemble_features.append(feature_vector)
        
        ensemble_features = np.array(ensemble_features)
        final_prediction = self.ensemble_combiner.predict(ensemble_features)
        
        return final_prediction, noise_type_pred, specialized_preds

def main():
    """Test del sistema multi-task RTM"""
    print("AVVIO TEST MULTI-TASK RTM DENOISER")
    
    # Crea e addestra il sistema
    mtrtm = MultiTaskRTMDenoiser()
    
    success = mtrtm.full_training_pipeline()
    if not success:
        print("ERRORE: Training fallito")
        return
    
    # Test su validation set
    print("\\n=== TEST SU VALIDATION SET ===")
    try:
        X_val = np.load(os.path.join(SAMPLE_DATA_DIR, 'X_validation_rtm_BINARIZED_q100.npy'))
        y_val_true = np.load(os.path.join(SAMPLE_DATA_DIR, 'y_validation_rtm_aggregated_noise.npy'))
        
        # Predizione
        denoised_pred, noise_types, specialized_preds = mtrtm.predict_denoised(X_val[:100])  # Test su subset
        
        # Valutazione
        mse = mean_squared_error(y_val_true[:100].flatten(), denoised_pred.flatten())
        rmse = np.sqrt(mse)
        corr = np.corrcoef(y_val_true[:100].flatten(), denoised_pred.flatten())[0,1]
        
        print(f"\\n🎯 RISULTATI MULTI-TASK RTM:")
        print(f"   MSE: {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Correlazione: {corr:.4f}")
        print(f"   Tipi rumore predetti: {np.unique(noise_types, return_counts=True)}")
        
        # Salva modello
        model_path = os.path.join(MODEL_OUTPUT_DIR, 'multi_task_rtm_denoiser.pkl')
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(mtrtm, f)
        print(f"\\n💾 Modello salvato in: {model_path}")
        
    except Exception as e:
        print(f"Errore nel test: {e}")

if __name__ == "__main__":
    main()
