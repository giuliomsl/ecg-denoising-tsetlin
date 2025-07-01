
import optuna
import numpy as np
import os

def analyze_study():
    """
    Carica lo studio Optuna, stampa il miglior trial, calcola l'MSE denormalizzato
    e lo confronta con la baseline.
    """
    storage_path = "sqlite:///optuna_rtm_study.db"
    study_name = "rtm_denoising_search"

    # Controlla se il file del database esiste
    db_file = storage_path.replace("sqlite:///", "")
    if not os.path.exists(db_file):
        print(f"Errore: Il database di Optuna '{db_file}' non è stato trovato.")
        print("Assicurati che la ricerca Optuna sia stata completata e che il file si trovi nella directory principale.")
        return

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
    except KeyError:
        print(f"Errore: Lo studio '{study_name}' non è stato trovato in '{db_file}'.")
        print("Controlla il nome dello studio in `optuna_rtm_search.py`.")
        return

    best_trial = study.best_trial

    print("--- Risultati della Ricerca Iperparametrica Optuna ---")
    print(f"Miglior Trial (Numero {best_trial.number}):")
    print(f"  Value (MSE Normalizzato): {best_trial.value:.6f}")
    print("  Parametri Ottimali:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Calcolo dell'MSE denormalizzato
    try:
        max_abs_noise_path = 'data/samplewise/y_train_max_abs_noise.npy'
        if not os.path.exists(max_abs_noise_path):
            print(f"\nErrore: File di normalizzazione non trovato in '{max_abs_noise_path}'.")
            print("Esegui prima lo script di preprocessing.")
            return
            
        max_abs_noise = np.load(max_abs_noise_path)
        # La normalizzazione era y / max_abs_noise, quindi l'errore è MSE * (max_abs_noise**2)
        denormalized_mse = best_trial.value * (max_abs_noise**2)
        
        print(f"\nFattore di denormalizzazione (max_abs_noise^2): {max_abs_noise**2:.6f}")
        print(f"MSE Denormalizzato del Miglior Trial: {denormalized_mse:.6f}")

        # Confronto con la baseline
        baseline_mse = 0.036910
        print(f"Baseline MSE Denormalizzato (predire 'zero'): {baseline_mse:.6f}")

        if denormalized_mse < baseline_mse:
            improvement = ((baseline_mse - denormalized_mse) / baseline_mse) * 100
            print(f"\nRISULTATO: CONGRATULAZIONI! La baseline è stata superata.")
            print(f"Miglioramento rispetto alla baseline: {improvement:.2f}%")
        else:
            difference = ((denormalized_mse - baseline_mse) / baseline_mse) * 100
            print(f"\nRISULTATO: La baseline non è stata ancora superata.")
            print(f"L'MSE è superiore alla baseline del {difference:.2f}%.")

    except FileNotFoundError:
        print(f"\nErrore: Impossibile trovare '{max_abs_noise_path}'.")
        print("Non è possibile calcolare l'MSE denormalizzato.")
    print("----------------------------------------------------")


if __name__ == "__main__":
    analyze_study()
