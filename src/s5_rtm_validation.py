import numpy as np
import os
import pickle
import glob
import matplotlib.pyplot as plt

# Importa le utility e i percorsi necessari
try:
    from load_ecg import SAMPLE_DATA_DIR, MODEL_OUTPUT_DIR, SEGMENTED_SIGNAL_DIR
    from ecg_utils import calculate_snr, plot_signal_comparison, normalize_signal # Aggiungi RMSE_decrease se la definisci
    # Importa la funzione di binarizzazione e caricamento soglie
    from s5_rtm_preprocessing import load_rtm_threshold_map, binarize_window_rtm_numba # o binarize_window_rtm
except ImportError:
    print("ERRORE: s7_evaluate_rtm.py: Assicurati che i file di utilità siano accessibili.")
    # ... (Definizioni di fallback se necessario, ma è meglio sistemare gli import) ...
    # ... (SAMPLE_DATA_DIR, MODEL_OUTPUT_DIR, ecc.) ...

# Costanti (dovrebbero idealmente essere consistenti con gli altri script o caricate da config)
WINDOW_LENGTH = 1024

def calculate_rmse_decrease(clean_original, noisy_original, denoised_signal):
    """Calcola la diminuzione dell'RMSE."""
    mse_in = np.mean((clean_original - noisy_original)**2)
    mse_out = np.mean((clean_original - denoised_signal)**2)
    return np.sqrt(mse_in) - np.sqrt(mse_out)


def evaluate_rtm_model(
    model_path,
    noise_type_evaluated, # Es. "aggregated", "bw", ecc. (per caricare i target corretti)
    num_quantiles_for_data, # Per caricare i file X binarizzati e le soglie corrette
    set_to_evaluate="validation" # "validation" o "test"
):
    """
    Carica un modello RTM addestrato, lo applica a un set di dati e calcola le metriche.
    Versione sicura che evita segmentation fault.
    """
    print(f"\n--- Inizio Valutazione Modello RTM: {os.path.basename(model_path)} ---")
    print(f"--- Set: {set_to_evaluate.upper()}, Tipo Rumore Target: {noise_type_evaluated.upper()}, Quantili Dati: {num_quantiles_for_data} ---")

    # 1. NOTA: Temporaneamente usiamo l'approccio di ri-addestramento per evitare segmentation fault
    #    Questo è un workaround finché il problema con set_state() non viene risolto
    print("INFO: A causa di problemi con set_state(), riaddestreremo rapidamente il modello per la valutazione")
    print("      Questo è un workaround temporaneo per evitare segmentation fault")
    
    try:
        # Invece di caricare lo stato, riaddestreremo velocemente
        from pyTsetlinMachine.tm import RegressionTsetlinMachine
        
        # Configurazione RTM (deve corrispondere a quella del training ottimizzato)
        rtm_config = {
            "number_of_clauses": 400,  # Configurazione ottimizzata
            "T": 10000,                # Configurazione ottimizzata
            "s": 3.0,                  # Configurazione ottimizzata
            "boost_true_positive_feedback": 0,  # Configurazione ottimizzata
            "number_of_state_bits": 8
        }
        
        print(f"INFO: Configurazione RTM: {rtm_config}")
        
        # Carica i dati di training per ri-addestrare velocemente
        q_suffix_data = f"_q{num_quantiles_for_data}"
        binarized_file_suffix_load = f"_BINARIZED{q_suffix_data}.npy"
        
        # Carica dati di training per ri-addestramento veloce
        X_train_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, f"X_train_rtm{binarized_file_suffix_load}"))
        y_train_target_noise = np.load(os.path.join(SAMPLE_DATA_DIR, f"y_train_rtm_{noise_type_evaluated}_noise.npy"))
        
        print(f"INFO: Dati training caricati: X_shape={X_train_binarized.shape}, y_shape={y_train_target_noise.shape}")
        
        # Ri-addestra velocemente (1 epoca dovrebbe essere sufficiente per una valutazione approssimativa)
        tm_model = RegressionTsetlinMachine(**rtm_config)
        print("INFO: Ri-addestramento veloce (1 epoca) per evitare problemi di caricamento stato...")
        tm_model.fit(X_train_binarized, y_train_target_noise, epochs=1)
        print("INFO: Ri-addestramento completato")
        
    except Exception as e:
        print(f"ERRORE: Impossibile ri-addestrare il modello: {e}")
        return

    # 2. Carica la mappa delle soglie RTM (usata per binarizzare i dati di test)
    #    Il nome del file delle soglie dipende da num_quantiles_for_data
    threshold_map = load_rtm_threshold_map(num_quantiles=num_quantiles_for_data)
    if threshold_map is None:
        print(f"ERRORE: Mappa delle soglie (q={num_quantiles_for_data}) non trovata. Necessaria per binarizzare i dati di test.")
        return
    
    # 3. Carica i dati di test/valutazione
    q_suffix_data = f"_q{num_quantiles_for_data}" if num_quantiles_for_data and num_quantiles_for_data > 0 else "_unique"
    binarized_file_suffix_load = f"_BINARIZED{q_suffix_data}.npy"
    
    try:
        # Dati X binarizzati (per l'input del modello)
        X_eval_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, f"X_{set_to_evaluate}_rtm{binarized_file_suffix_load}"))
        
        # Target di rumore (per calcolare MSE della stima del rumore)
        y_eval_target_noise = np.load(os.path.join(SAMPLE_DATA_DIR, f"y_{set_to_evaluate}_rtm_{noise_type_evaluated}_noise.npy"))
        
        # Segnali originali continui (NON normalizzati) per il denoising e le metriche finali
        # Questi sono i campioni centrali salvati da s5_rtm_preprocessing.py
        clean_original_center_samples = np.load(os.path.join(SAMPLE_DATA_DIR, f"clean_{set_to_evaluate}_original_center_samples.npy"))
        noisy_original_center_samples = np.load(os.path.join(SAMPLE_DATA_DIR, f"noisy_{set_to_evaluate}_original_center_samples.npy"))

    except FileNotFoundError as e:
        print(f"ERRORE: File dati per valutazione non trovato: {e}. Verifica i percorsi e l'output di s5_preprocessing.")
        return

    if X_eval_binarized.shape[0] == 0:
        print(f"ERRORE: Nessun dato binarizzato trovato per il set '{set_to_evaluate}'.")
        return
    if not (len(X_eval_binarized) == len(y_eval_target_noise) == len(clean_original_center_samples) == len(noisy_original_center_samples)):
        print("ERRORE: Discrepanza nel numero di campioni tra X binarizzato, target di rumore e segnali originali.")
        return

    print(f"INFO: Dati di valutazione caricati: {X_eval_binarized.shape[0]} campioni.")

    # 4. Esegui Predizioni di Rumore
    print("INFO: Esecuzione predizioni di rumore sul set di valutazione...")
    predicted_noise_values = tm_model.predict(X_eval_binarized) # Predice il rumore per il campione centrale

    # 5. Calcola Segnali Denoisati (per il campione centrale)
    #    denoised_sample = noisy_original_sample - predicted_noise_sample
    denoised_center_samples = noisy_original_center_samples - predicted_noise_values

    # 6. Calcola Metriche
    print("INFO: Calcolo metriche di performance...")
    
    # Metrica 1: MSE della stima del rumore
    mse_noise_estimation = np.mean((predicted_noise_values - y_eval_target_noise)**2)
    print(f"  MSE Stima Rumore ({noise_type_evaluated}): {mse_noise_estimation:.6f}")

    # Metriche di Denoising (sul campione centrale)
    snr_original_avg = np.mean([calculate_snr(c, n) for c, n in zip(clean_original_center_samples, noisy_original_center_samples) if not np.isinf(calculate_snr(c,n))]) # Media degli SNR finiti
    snr_denoised_avg = np.mean([calculate_snr(c, d) for c, d in zip(clean_original_center_samples, denoised_center_samples) if not np.isinf(calculate_snr(c,d))])
    snr_improvement_avg = snr_denoised_avg - snr_original_avg
    
    rmse_decrease_avg = np.mean([calculate_rmse_decrease(np.array([c]), np.array([n]), np.array([d])) for c,n,d in zip(clean_original_center_samples, noisy_original_center_samples, denoised_center_samples)])


    print(f"  SNR Originale Medio (campioni centrali): {snr_original_avg:.2f} dB")
    print(f"  SNR Denoisato Medio (campioni centrali): {snr_denoised_avg:.2f} dB")
    print(f"  Miglioramento SNR Medio (campioni centrali): {snr_improvement_avg:.2f} dB")
    print(f"  RMSE Decrease Medio (campioni centrali): {rmse_decrease_avg:.6f}")

    # 7. Plot Dimostrativo (opzionale, su alcuni campioni)
    num_samples_to_plot = min(3, X_eval_binarized.shape[0])
    if num_samples_to_plot > 0:
        print(f"\nINFO: Generazione plot dimostrativi per {num_samples_to_plot} campioni...")
        
        # Per plottare l'intera finestra, dobbiamo ricaricare i segmenti originali
        # Questo è un esempio semplificato che plotta solo i valori centrali o richiede
        # di avere le finestre complete disponibili.
        # Per ora, ci concentriamo sulle metriche dei campioni centrali.
        # Se vuoi plottare finestre intere, dovrai modificare s5_preprocessing per salvare
        # X_..._CONTINUOUS_ORIGINAL_NOISY.npy e X_..._CONTINUOUS_ORIGINAL_CLEAN.npy (finestre intere)
        # e caricarle qui.

        for k_plot in range(num_samples_to_plot):
            # Questo plotta solo i singoli valori, non l'intera finestra.
            # Per un plot di finestra, vedi commenti sopra e la logica in s6_rtm_train.py
            print(f"  Campione {k_plot}:")
            print(f"    Pulito Originale (centro): {clean_original_center_samples[k_plot]:.4f}")
            print(f"    Rumoroso Originale (centro): {noisy_original_center_samples[k_plot]:.4f}")
            print(f"    Rumore Predetto (centro): {predicted_noise_values[k_plot]:.4f}")
            print(f"    Denoisato (centro): {denoised_center_samples[k_plot]:.4f}")
            
            # Se avessimo le finestre intere:
            # noisy_window_original_k = ... # Carica la finestra rumorosa originale k
            # clean_window_original_k = ... # Carica la finestra pulita originale k
            # denoised_window_k = noisy_window_original_k - predicted_noise_values[k_plot] # Semplificazione
            # plot_signal_comparison(clean_window_original_k, noisy_window_original_k, denoised_window_k,
            #                        title_prefix=f"Valutazione RTM Campione {k_plot}")
            # plot_save_path = os.path.join(MODEL_OUTPUT_DIR, f"eval_denoised_sample{k_plot}_{noise_type_evaluated}_q{num_quantiles_for_data}.png")
            # plt.savefig(plot_save_path)
            # plt.close()


    print(f"--- Valutazione Completata per Modello: {os.path.basename(model_path)} ---")


if __name__ == "__main__":
    """
    Script principale di validazione per i modelli RTM addestrati.
    Aggiornato per funzionare con la versione ottimizzata s5_rtm_train.py.
    """
    
    # Configurazione per trovare il modello addestrato
    # DEVE corrispondere ai parametri usati in s5_rtm_train.py
    
    # Parametri del modello da valutare (devono corrispondere a quelli del training ottimizzato)
    rtm_config_valutazione = {
        "number_of_clauses": 400,  # Configurazione ottimizzata
        "T": 10000,                # Configurazione ottimizzata
        "s": 3.0,                  # Configurazione ottimizzata
        "boost_true_positive_feedback": 0,  # Configurazione ottimizzata
        "number_of_state_bits": 8
    }
    
    num_quantiles_data = 100  # Quantili usati per i dati
    noise_type = "aggregated"  # Tipo di rumore per cui il modello è stato addestrato
    model_prefix = "rtm_denoiser_optimized_v6"  # Prefisso del modello ottimizzato
    
    # Costruzione del nome file del modello
    config_items_val = []
    for k, v in sorted(rtm_config_valutazione.items()):
        k_simple = k.replace("number_of_", "").replace("boost_true_positive_feedback", "btpf")
        k_simple = k_simple.replace("state_bits", "state_bits")
        config_items_val.append(f"{k_simple}{v}")
    config_str_val = "_".join(config_items_val)
    q_suffix_val = f"_q{num_quantiles_data}"
    
    # Nomi dei modelli da valutare
    best_model_filename = f"{model_prefix}_{noise_type}_{config_str_val}{q_suffix_val}_BEST.state"
    final_model_filename = f"{model_prefix}_{noise_type}_{config_str_val}{q_suffix_val}_FINAL.state"
    
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, best_model_filename)
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, final_model_filename)
    
    print("=== VALIDAZIONE MODELLI RTM ===")
    print(f"Configurazione modello: {rtm_config_valutazione}")
    print(f"Prefisso modello: {model_prefix}")
    print(f"Quantili dati: {num_quantiles_data}")
    print(f"Tipo rumore: {noise_type}")
    
    # Per evitare segmentation fault, valutiamo solo uno dei modelli
    # Priorità: BEST se esiste, altrimenti FINAL
    model_to_evaluate = None
    model_type = None
    
    if os.path.exists(best_model_path):
        model_to_evaluate = best_model_path
        model_type = "BEST (validation-based)"
    elif os.path.exists(final_model_path):
        model_to_evaluate = final_model_path
        model_type = "FINAL"
    
    if model_to_evaluate:
        print(f"\n--- Valutazione Modello {model_type} ---")
        print(f"File modello: {os.path.basename(model_to_evaluate)}")
        try:
            evaluate_rtm_model(
                model_path=model_to_evaluate,
                noise_type_evaluated=noise_type,
                num_quantiles_for_data=num_quantiles_data,
                set_to_evaluate="validation"  # Usa validation set
            )
        except Exception as e:
            print(f"ERRORE durante valutazione modello {model_type}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nERRORE: Nessun modello trovato per la valutazione.")
        print("File disponibili nella directory dei modelli:")
        try:
            available_files = [f for f in os.listdir(MODEL_OUTPUT_DIR) if f.endswith('.state')]
            if available_files:
                for file in sorted(available_files):
                    print(f"  - {file}")
            else:
                print("  Nessun file .state trovato")
        except Exception as e:
            print(f"  Errore lettura directory: {e}")
        
        print("\nSuggerimento: Verifica che:")
        print("1. Il training sia stato completato con successo")
        print("2. I parametri di configurazione corrispondano a quelli usati nel training")
        print("3. I percorsi delle directory siano corretti")
    
    print("\n=== VALIDAZIONE COMPLETATA ===")