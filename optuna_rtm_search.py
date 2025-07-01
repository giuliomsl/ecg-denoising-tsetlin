import optuna
import numpy as np
import os
import yaml
from src.s5_rtm_train import train_rtm_denoiser, NUM_QUANTILES_USED_IN_PREPROCESSING, SAMPLE_DATA_DIR

def objective(trial):
    # Spazio di ricerca iperparametri affinato
    number_of_clauses = trial.suggest_int('number_of_clauses', 8000, 16000, step=500)
    T = trial.suggest_int('T', 2000, 7000, step=250)
    s = trial.suggest_float('s', 2.0, 4.0, step=0.1)
    boost_true_positive_feedback = trial.suggest_categorical('boost_true_positive_feedback', [1, 2])
    num_quantiles = trial.suggest_categorical('num_quantiles_for_data', [15, 20, 25, 30])
    rtm_config = {
        'number_of_clauses': number_of_clauses,
        'T': T,
        's': s,
        'boost_true_positive_feedback': boost_true_positive_feedback,
        'number_of_state_bits': 8
    }
    # Carica y_train_max_abs_noise
    y_train_max_abs_noise_path = os.path.join(SAMPLE_DATA_DIR, "y_train_max_abs_noise.npy")
    y_train_max_abs_noise = np.load(y_train_max_abs_noise_path)
    # Training con pruning e più epoche
    _, _, val_losses = train_rtm_denoiser(
        noise_type_to_predict="aggregated",
        rtm_config=rtm_config,
        num_epochs=30,
        num_quantiles_for_data=num_quantiles,
        early_stopping_patience=7,
        model_save_name_prefix="optuna_trial",
        y_train_max_abs_noise=y_train_max_abs_noise,
        trial=trial
    )
    best_val_loss = np.nanmin(val_losses) if val_losses else float('inf')
    return best_val_loss

if __name__ == "__main__":
    storage_name = "sqlite:///optuna_rtm_study.db"
    study_name = "rtm_denoising_search"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # Permette di riprendere lo studio se interrotto
        direction="minimize", 
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=50)

    print("--- Analisi Studio Optuna Completata ---")
    print("Miglior trial:")
    print(f"  Value (MSE Normalizzato): {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Salvataggio dei parametri migliori in un file YAML
    with open("optuna_rtm_best_params.yaml", "w") as f:
        yaml.dump(study.best_params, f)
    print("\nFile 'optuna_rtm_best_params.yaml' salvato.")

    # Generazione e salvataggio dei grafici di analisi di Optuna
    print("Generazione grafici di analisi...")
    try:
        if optuna.visualization.is_available():
            # Grafico della cronologia dell'ottimizzazione
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_image("optuna_history.png")
            print("- Grafico 'optuna_history.png' salvato.")

            # Grafico dell'importanza degli iperparametri
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image("optuna_param_importances.png")
            print("- Grafico 'optuna_param_importances.png' salvato.")

            # Grafico delle slice per vedere l'impatto di ogni iperparametro
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.write_image("optuna_slice.png")
            print("- Grafico 'optuna_slice.png' salvato.")
        else:
            print("ATTENZIONE: La visualizzazione di Optuna non è disponibile. Installa plotly.")
    except Exception as e:
        print(f"ERRORE durante la generazione dei grafici: {e}")
        print("Potrebbe essere necessario installare 'plotly' e 'kaleido': pip install plotly kaleido")
