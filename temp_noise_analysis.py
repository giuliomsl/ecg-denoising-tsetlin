import numpy as np
import os

# Definisci il percorso alla directory dei dati campionati
# Assumendo che lo script sia eseguito dalla root del progetto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # o specifica il percorso assoluto
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "samplewise")

# Nome del file target di validazione
y_val_filename = "y_validation_rtm_aggregated_noise.npy"
y_val_filepath = os.path.join(SAMPLE_DATA_DIR, y_val_filename)

try:
    # Carica i dati target di validazione
    y_val_target_noise = np.load(y_val_filepath)
    
    if y_val_target_noise.size == 0:
        print("ERRORE: Il file dei target di validazione è vuoto.")
    else:
        # 1. Calcola la varianza del rumore target
        variance_y_val = np.var(y_val_target_noise)
        print(f"Varianza di y_validation_rtm_aggregated_noise: {variance_y_val:.6f}")
        
        # 2. Calcola l'MSE di una baseline che predice sempre zero
        mse_baseline_zero_prediction = np.mean(y_val_target_noise**2)
        print(f"MSE baseline (predizione sempre zero rumore): {mse_baseline_zero_prediction:.6f}")
        
        # Ricorda l'MSE del modello RTM (dal training precedente)
        mse_rtm_model = 0.302302 # Valore dal log di training precedente
        print(f"MSE del modello RTM (Set 1, epoca 2): {mse_rtm_model:.6f}")
        
        if mse_rtm_model < mse_baseline_zero_prediction:
            improvement_over_baseline = mse_baseline_zero_prediction - mse_rtm_model
            percentage_improvement = (improvement_over_baseline / mse_baseline_zero_prediction) * 100
            print(f"Il modello RTM è migliore della baseline.")
            print(f"  Miglioramento assoluto MSE rispetto alla baseline: {improvement_over_baseline:.6f}")
            print(f"  Miglioramento percentuale MSE rispetto alla baseline: {percentage_improvement:.2f}%")
        else:
            print("ATTENZIONE: Il modello RTM NON è migliore della baseline che predice sempre zero.")

except FileNotFoundError:
    print(f"ERRORE: File non trovato: {y_val_filepath}")
    print("Assicurati che 's5_rtm_preprocessing.py' sia stato eseguito correttamente.")
except Exception as e:
    print(f"Si è verificato un errore: {e}")
