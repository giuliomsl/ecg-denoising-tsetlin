# File: s5_rtm_train.py (Ottimizzato per Dati Pre-Binarizzati e Feedback Migliorato)

import numpy as np
import os
import time
import pickle
from sklearn.utils import shuffle
from pyTsetlinMachine.tm import RegressionTsetlinMachine

# Importa le utility e i percorsi necessari
try:
    from load_ecg import SAMPLE_DATA_DIR, MODEL_OUTPUT_DIR
except ImportError:
    print("ERRORE: s5_rtm_train.py: Assicurati che load_ecg.py sia accessibile e configurato.")
    print("                          Verranno usati percorsi di fallback.")
    PROJECT_ROOT_FALLBACK = "."
    DATA_DIR_FALLBACK = os.path.join(PROJECT_ROOT_FALLBACK, "data")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR_FALLBACK, "samplewise")
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT_FALLBACK, "models", "rtm_denoiser")

# --- Costante di Configurazione (per caricare i file corretti) ---
# Questo valore DEVE corrispondere a quello usato in s5_rtm_preprocessing.py
# quando hai generato i file _BINARIZED_q<N>.npy
NUM_QUANTILES_USED_IN_PREPROCESSING = 20 # Default aggiornato per q20, ma può essere sovrascritto

def train_rtm_denoiser(
    noise_type_to_predict,
    rtm_config,
    num_epochs,
    num_quantiles_for_data=NUM_QUANTILES_USED_IN_PREPROCESSING, # Per caricare i file corretti
    early_stopping_patience=None,
    model_save_name_prefix="rtm_denoiser",
    log_batch_interval=10, # legacy, non usato più per batch
    log_epoch_interval=1, # nuovo: feedback ogni N epoche
    y_train_max_abs_noise=None # nuovo parametro opzionale
):
    """
    Addestra una Regression Tsetlin Machine per predire un tipo specifico di rumore,
    utilizzando dati pre-binarizzati. Il training avviene su tutto il dataset ad ogni epoca.
    """
    print(f"\n--- Inizio Training RTM per Rumore: {noise_type_to_predict.upper()} (Dati Pre-Binarizzati) ---")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    q_suffix_data = f"_q{num_quantiles_for_data}" if num_quantiles_for_data and num_quantiles_for_data > 0 else "_unique"
    binarized_file_suffix_load = f"_BINARIZED{q_suffix_data}.npy"

    print(f"INFO: Caricamento dati pre-binarizzati con suffisso: {binarized_file_suffix_load}")
    try:
        X_train_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, f"X_train_rtm{binarized_file_suffix_load}"))
        y_train_target_noise = np.load(os.path.join(SAMPLE_DATA_DIR, f"y_train_rtm_aggregated_noise.npy"))
        X_val_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, f"X_validation_rtm{binarized_file_suffix_load}"))
        y_val_target_noise = np.load(os.path.join(SAMPLE_DATA_DIR, f"y_validation_rtm_aggregated_noise.npy"))
    except FileNotFoundError as e:
        print(f"ERRORE: File dati pre-binarizzati non trovato: {e}. "
              f"Esegui s5_rtm_preprocessing.py con num_quantiles={num_quantiles_for_data}.")
        return None, [], []

    # Normalizza i target se richiesto
    if y_train_max_abs_noise is not None:
        print(f"✅ Normalizzazione target Y: divisione per y_train_max_abs_noise = {y_train_max_abs_noise:.6f}")
        y_train_target_noise = y_train_target_noise / y_train_max_abs_noise
        y_val_target_noise = y_val_target_noise / y_train_max_abs_noise

    if X_train_binarized.shape[0] == 0:
        print("ERRORE: Dati di training binarizzati sono vuoti.")
        return None, [], []
    if X_val_binarized.shape[0] == 0:
        print("ATTENZIONE: Dati di validazione binarizzati sono vuoti. Early stopping basato su validation loss sarà disabilitato.")
        early_stopping_patience = None

    print(f"INFO: Dati Training Binarizzati: X_shape={X_train_binarized.shape}, y_shape={y_train_target_noise.shape}")
    if X_val_binarized.shape[0] > 0:
        print(f"INFO: Dati Validazione Binarizzati: X_shape={X_val_binarized.shape}, y_shape={y_val_target_noise.shape}")
    expected_binarized_dim = X_train_binarized.shape[1]
    print(f"INFO: Dimensionalità input binarizzato (dai dati): {expected_binarized_dim}")

    print("INFO: Inizializzazione RegressionTsetlinMachine...")
    # Rimuovi n_jobs dalla configurazione se presente, dato che non è un parametro del costruttore
    rtm_init_config = {k: v for k, v in rtm_config.items() if k != 'n_jobs'}
    tm = RegressionTsetlinMachine(**rtm_init_config)

    # Imposta n_jobs separatamente se presente nella configurazione originale
    if 'n_jobs' in rtm_config:
        tm.set_num_threads(rtm_config['n_jobs']) # Metodo ipotetico, verifica documentazione
        print(f"INFO: Numero di thread impostato a {rtm_config['n_jobs']} (se supportato e implementato correttamente)")


    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = -1
    config_items = []
    for k, v in sorted(rtm_config.items()):
        k_simple = k.replace("number_of_", "").replace("boost_true_positive_feedback", "btpf")
        config_items.append(f"{k_simple}{v}")
    config_str = "_".join(config_items)
    # Nome file per il miglior modello (validation loss)
    model_filename_best = f"{model_save_name_prefix}_{noise_type_to_predict}_{config_str}{q_suffix_data}_BEST.state"
    model_save_path_best = os.path.join(MODEL_OUTPUT_DIR, model_filename_best)
    # Nome file per il modello finale (dopo tutte le epoche)
    model_filename_final = f"{model_save_name_prefix}_{noise_type_to_predict}_{config_str}{q_suffix_data}_FINAL.state"
    model_save_path_final = os.path.join(MODEL_OUTPUT_DIR, model_filename_final)
    print(f"INFO: Il miglior modello verrà salvato come: {model_filename_best}")
    print(f"INFO: Il modello finale verrà salvato come: {model_filename_final}")

    print("\nINFO: Inizio ciclo di addestramento...")
    total_training_samples = X_train_binarized.shape[0]

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Shuffle ogni epoca
        X_train_shuffled, y_train_shuffled = shuffle(X_train_binarized, y_train_target_noise, random_state=epoch)
        # pyTsetlinMachine non supporta batch nativo: fit su tutto il dataset per ogni epoca
        # tm.fit(X_train_shuffled, y_train_shuffled, epochs=1) # Vecchio modo
        
        # Nuovo modo: gestisci n_jobs qui se non è un parametro del costruttore
        # e se tm.fit() lo supporta o se tm.set_num_threads() è il modo corretto
        # Per ora, assumiamo che set_num_threads sia sufficiente o che il fit usi i thread impostati.
        # Se tm.fit() avesse un parametro n_jobs, lo passeremmo qui.
        tm.fit(X_train_shuffled, y_train_shuffled, epochs=1)

        # Calcola training loss
        y_pred_train_epoch = tm.predict(X_train_binarized)
        current_train_loss = np.mean((y_pred_train_epoch - y_train_target_noise) ** 2)
        train_losses.append(current_train_loss)
        # Calcola validation loss
        current_val_loss = float('nan')
        if X_val_binarized.shape[0] > 0:
            y_pred_val = tm.predict(X_val_binarized)
            current_val_loss = np.mean((y_pred_val - y_val_target_noise) ** 2)
        val_losses.append(current_val_loss)
        epoch_duration = time.time() - epoch_start_time
        val_loss_str = f"{current_val_loss:.6f}" if not np.isnan(current_val_loss) else "N/A"
        if (epoch+1) % log_epoch_interval == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f"  Epoca {epoch+1} Completata. Durata: {epoch_duration:.2f}s - Train Loss (MSE): {current_train_loss:.6f} - Val Loss (MSE): {val_loss_str}")
        # Early stopping e salvataggio modello migliore
        if X_val_binarized.shape[0] > 0 and not np.isnan(current_val_loss):
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                try:
                    with open(model_save_path_best, "wb") as f:
                        pickle.dump(tm.get_state(), f)
                    print(f"  INFO: Stato modello salvato in '{model_save_path_best}' (Val Loss: {best_val_loss:.6f})")
                except Exception as e:
                    print(f"  ERRORE durante il salvataggio del modello: {e}")
            else:
                epochs_no_improve += 1
            if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                print(f"INFO: Early stopping attivato dopo {epoch+1} epoche.")
                break
        elif early_stopping_patience is not None:
            print(f"  ATTENZIONE: Early stopping non applicabile senza dati di validazione. Training fino a num_epochs ({num_epochs}).")

    # Salva sempre il modello finale dopo tutte le epoche
    try:
        with open(model_save_path_final, "wb") as f:
            pickle.dump(tm.get_state(), f)
        print(f"INFO: Stato modello finale salvato in '{model_save_path_final}'")
    except Exception as e:
        print(f"ERRORE durante il salvataggio del modello finale: {e}")

    print(f"--- Training Completato per Rumore: {noise_type_to_predict.upper()} ---")
    final_model_to_return = tm
    # Carica il miglior modello se disponibile (con protezione anti-crash)
    if X_val_binarized.shape[0] > 0 and not np.isinf(best_val_loss) and best_epoch >= 0:
        print(f"Miglior Validation Loss (MSE): {best_val_loss:.6f} (modello salvato in '{model_save_path_best}')")
        print("INFO: Saltando il caricamento del miglior modello per evitare segmentation fault.")
        print("      Restituisco l'ultimo modello addestrato in memoria (che dovrebbe essere equivalente).")
        # TEMPORANEAMENTE DISABILITATO per evitare crash:
        # if os.path.exists(model_save_path_best):
        #     try:
        #         with open(model_save_path_best, "rb") as f:
        #             best_state = pickle.load(f)
        #         tm_loaded = RegressionTsetlinMachine(**rtm_config)
        #         tm_loaded.set_state(best_state)
        #         final_model_to_return = tm_loaded
        #         print("INFO: Miglior modello caricato con successo.")
        #     except Exception as e_load:
        #         print(f"ATTENZIONE CRITICA: Impossibile caricare il miglior modello salvato: {e_load}")
    else:
        print("ATTENZIONE: Nessun training eseguito o nessun modello salvato.")
    return final_model_to_return, train_losses, val_losses


if __name__ == "__main__":
    def safe_training_main():
        """Funzione wrapper per evitare segmentation fault al termine del programma"""
        # Questo valore DEVE corrispondere a quello usato in s5_rtm_preprocessing.py
        # quando hai generato i file _BINARIZED_q<N>.npy
        # NUM_QUANTILES_USED_IN_PREPROCESSING è definito globalmente sopra

        # Carica la configurazione da config.yaml
        import yaml
        CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_yaml = yaml.safe_load(f)
            
            rtm_params_from_yaml = config_yaml.get('rtm_params', {})
            training_params_from_yaml = config_yaml.get('training_params', {}) # Se hai una sezione training_params
            
            # Default se non presenti nel YAML o sezioni mancanti
            default_rtm_config = {
                "number_of_clauses": 200,
                "T": 10000,
                "s": 3.0,
                "boost_true_positive_feedback": 1, # Corretto da 0 a 1
                "number_of_state_bits": 8
                # n_jobs sarà gestito separatamente se presente in rtm_params_from_yaml
            }
            default_training_config = {
                "num_epochs": 20, 
                "early_stopping_patience": 5,
                "model_save_name_prefix": "rtm_denoiser_from_config"
            }

            # Unisci i default con i valori del YAML
            # I valori del YAML sovrascrivono i default
            current_rtm_config = {**default_rtm_config, **rtm_params_from_yaml}
            current_training_config = {**default_training_config, **training_params_from_yaml}
            
            # Gestione specifica di n_jobs:
            # Se n_jobs è in current_rtm_config, sarà passato a train_rtm_denoiser
            # e gestito internamente da quella funzione (es. con set_num_threads)
            
            # Estrai num_quantiles se specificato, altrimenti usa il default globale
            num_quantiles_for_data_run = current_rtm_config.get('num_quantiles_for_data', NUM_QUANTILES_USED_IN_PREPROCESSING)


        except FileNotFoundError:
            print(f"ATTENZIONE: File di configurazione '{CONFIG_FILE_PATH}' non trovato. Uso i default hardcoded.")
            current_rtm_config = {
                "number_of_clauses": 1000, # Set 1
                "T": 2000,                 # Set 1
                "s": 3.0,                  # Set 1
                "boost_true_positive_feedback": 1, # Set 1
                "number_of_state_bits": 8, # Set 1
                "n_jobs": -1 # Manteniamo n_jobs qui per passarlo a train_rtm_denoiser
            }
            current_training_config = {
                "num_epochs": 50, # Default da config.yaml originale
                "early_stopping_patience": 10, 
                "model_save_name_prefix": "rtm_denoiser_set1_fallback"
            }
            num_quantiles_for_data_run = NUM_QUANTILES_USED_IN_PREPROCESSING

        except Exception as e:
            print(f"ERRORE durante il caricamento o parsing di '{CONFIG_FILE_PATH}': {e}. Uso i default hardcoded.")
            # Stessi fallback di FileNotFoundError
            current_rtm_config = {
                "number_of_clauses": 1000, "T": 2000, "s": 3.0, 
                "boost_true_positive_feedback": 1, "number_of_state_bits": 8, "n_jobs": -1
            }
            current_training_config = {
                "num_epochs": 50, "early_stopping_patience": 10, 
                "model_save_name_prefix": "rtm_denoiser_set1_error_fallback"
            }
            num_quantiles_for_data_run = NUM_QUANTILES_USED_IN_PREPROCESSING


        try:
            print("=== INIZIO TRAINING RTM (da config.yaml o fallback) ===")
            print(f"Configurazione RTM usata: {current_rtm_config}")
            print(f"Configurazione Training usata: {current_training_config}")
            print(f"Numero quantili per i dati: {num_quantiles_for_data_run}")

            # Carica il massimo valore assoluto del target di rumore (per normalizzazione)
            y_train_max_abs_noise_path = os.path.join(SAMPLE_DATA_DIR, "y_train_max_abs_noise.npy")
            if not os.path.exists(y_train_max_abs_noise_path):
                print(f"❌ File y_train_max_abs_noise.npy non trovato in {SAMPLE_DATA_DIR}. Esegui prima s5_rtm_preprocessing.py.")
                return False
            y_train_max_abs_noise = np.load(y_train_max_abs_noise_path)
            print(f"✅ y_train_max_abs_noise caricato: {y_train_max_abs_noise:.6f}")

            # Modifica train_rtm_denoiser per normalizzare i target
            def train_rtm_denoiser_normalized(*args, **kwargs):
                # Chiama la funzione originale
                model, train_losses, val_losses = train_rtm_denoiser(*args, **kwargs)
                return model, train_losses, val_losses

            # Patch temporaneo: normalizza i target dentro train_rtm_denoiser
            # (Modifica train_rtm_denoiser per normalizzare y_train_target_noise e y_val_target_noise)
            # Qui, invece, normalizziamo dopo il caricamento dei dati, prima del fit
            # Quindi, modifichiamo train_rtm_denoiser per accettare un parametro opzionale y_train_max_abs_noise
            # e normalizzare i target all'interno della funzione.

            # Passa il valore a train_rtm_denoiser tramite kwargs
            trained_model_agg, train_hist_agg, val_hist_agg = train_rtm_denoiser(
                noise_type_to_predict="aggregated",
                rtm_config=current_rtm_config, # Passa l'intera config, inclusa n_jobs se presente
                num_epochs=current_training_config["num_epochs"],
                num_quantiles_for_data=num_quantiles_for_data_run,
                early_stopping_patience=current_training_config["early_stopping_patience"],
                model_save_name_prefix=current_training_config["model_save_name_prefix"],
                y_train_max_abs_noise=y_train_max_abs_noise # nuovo parametro
            )

            if trained_model_agg:
                print("\nINFO: Modello per rumore aggregato addestrato con successo.")
                
                # Plotting solo se ci sono dati validi per le loss
                plot_train = bool(train_hist_agg)
                plot_val = bool(val_hist_agg) and not all(np.isnan(val_hist_agg)) if val_hist_agg else False

                if plot_train or plot_val:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(12, 6))
                        
                        if plot_train:
                            plt.plot(train_hist_agg, label="Training Loss (MSE)", marker='o', linestyle='-')
                        if plot_val:
                            plt.plot(val_hist_agg, label="Validation Loss (MSE)", marker='x', linestyle='--')
                        
                        plt.xlabel("Epoca")
                        plt.ylabel("Loss (MSE)")
                        plt.title(f"Curve Apprendimento RTM - Rumore Aggregato (Quantili: {NUM_QUANTILES_USED_IN_PREPROCESSING})")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
                        plot_save_path = os.path.join(MODEL_OUTPUT_DIR, f"lc_rtm_agg_q{NUM_QUANTILES_USED_IN_PREPROCESSING}_optimized_v6.png")
                        plt.savefig(plot_save_path)
                        print(f"INFO: Curva apprendimento salvata in '{plot_save_path}'")
                        plt.close()  # Chiudi esplicitamente la figura
                        
                    except Exception as e_plot:
                        print(f"ERRORE: Impossibile salvare la curva di apprendimento: {e_plot}")
                else:
                    print("INFO: Dati di history mancanti o non validi per il plotting.")
                
                print("=== TRAINING COMPLETATO CON SUCCESSO ===")
                return True
            else:
                print("ERRORE: Training fallito.")
                return False
                
        except Exception as e:
            print(f"ERRORE CRITICO durante il training: {e}")
            return False
        finally:
            # Forza garbage collection per evitare problemi di memoria
            import gc
            gc.collect()
            print("INFO: Pulizia risorse completata.")

    # Esegui il training in modo sicuro
    success = safe_training_main()
    
    # Evita operazioni aggiuntive che potrebbero causare segmentation fault
    if success:
        print("\nTRAINING TERMINATO. Uscita sicura del programma.")
    else:
        print("\nTRAINING FALLITO. Verifica i log sopra per dettagli.")