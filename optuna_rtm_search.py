import optuna
import numpy as np
import os
import yaml
from src.s5_rtm_train import train_rtm_denoiser, NUM_QUANTILES_USED_IN_PREPROCESSING, SAMPLE_DATA_DIR

def objective(trial):
    # Spazio di ricerca iperparametri
    number_of_clauses = trial.suggest_int('number_of_clauses', 5000, 15000)
    T = trial.suggest_int('T', 500, 5000)
    s = trial.suggest_float('s', 1.5, 5.0)
    num_quantiles = trial.suggest_categorical('num_quantiles_for_data', [15, 20, 25, 30, 35, 40])
    boost_true_positive_feedback = trial.suggest_categorical('boost_true_positive_feedback', [1, 2])
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
    # Training (usa early stopping breve per rapidità)
    _, _, val_losses = train_rtm_denoiser(
        noise_type_to_predict="aggregated",
        rtm_config=rtm_config,
        num_epochs=20,
        num_quantiles_for_data=num_quantiles,
        early_stopping_patience=5,
        model_save_name_prefix="optuna_trial",
        y_train_max_abs_noise=y_train_max_abs_noise
    )
    # Restituisci la migliore validation loss
    best_val_loss = np.nanmin(val_losses) if val_losses else float('inf')
    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    print(study.best_params)
    # Salva i risultati
    with open("optuna_rtm_best_params.yaml", "w") as f:
        yaml.dump(study.best_params, f)
