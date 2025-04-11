# -*- coding: utf-8 -*-
import numpy as np
import pickle
# Assicurati che il nome del file e della classe siano corretti
from multilabel_tm_class import MultiLabelTsetlinMachine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, hamming_loss
from ecg_dataset import ECGSamplewiseDataset
from ecg_utils import calculate_snr, plot_signal_comparison
from binarization import load_binarization_info
import os
import time
import sys
import errno # Per gestione errori directory

# --- Rilevamento Ambiente e Definizione Percorsi ---

# Verifica se siamo in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("INFO: Rilevato ambiente Google Colab.")
    from google.colab import drive
    # Monta Google Drive se non già montato
    if not os.path.exists('/content/drive/MyDrive'):
         print("INFO: Montaggio Google Drive...")
         drive.mount('/content/drive')
         # Attendi un attimo per assicurarti che il mount sia completo
         import time
         time.sleep(5)
    else:
         print("INFO: Google Drive già montato.")

    # Definisci i percorsi base per Colab (ASSUMENDO la tua struttura su Drive)
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/" # Modifica se necessario
    REPO_NAME = "TUO_REPO_NAME" # Il nome della cartella clonata da GitHub
    PROJECT_ROOT_COLAB = f"/content/{REPO_NAME}/" # Percorso del progetto clonato

    # Percorsi Dati su Drive
    DATA_DIR_COLAB = os.path.join(GDRIVE_BASE, "data")
    ECG_DIR = os.path.join(DATA_DIR_COLAB, "mit-bih/")
    NOISE_DIR = os.path.join(DATA_DIR_COLAB, "noise_stress_test/")
    NOISY_ECG_DIR = os.path.join(DATA_DIR_COLAB, "noisy_ecg/")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR_COLAB, "samplewise/")

    # Percorsi Output Modelli su Drive
    MODEL_OUTPUT_DIR = os.path.join(GDRIVE_BASE, "models/multi_tm_denoiser/")

    # Percorso per info binarizzazione su Drive
    BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

    # Assicurati che le directory di output esistano su Drive
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    # Potrebbe essere necessario creare anche NOISY_ECG_DIR e SAMPLE_DATA_DIR se
    # esegui anche i preprocessing su Colab la prima volta.
    os.makedirs(NOISY_ECG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

else:
    print("INFO: Rilevato ambiente Locale.")
    # Definisci i percorsi relativi per l'ambiente locale
    # Assumiamo che lo script sia eseguito dalla root del progetto
    # o che i percorsi relativi funzionino dalla posizione dello script.
    # Se esegui da src/, potresti dover usare '../data' etc.
    PROJECT_ROOT_LOCAL = "." # O specifica il percorso assoluto/relativo corretto
    DATA_DIR_LOCAL = os.path.join(PROJECT_ROOT_LOCAL, "data")

    ECG_DIR = os.path.join(DATA_DIR_LOCAL, "mit-bih/")
    NOISE_DIR = os.path.join(DATA_DIR_LOCAL, "noise_stress_test/")
    NOISY_ECG_DIR = os.path.join(DATA_DIR_LOCAL, "noisy_ecg/")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR_LOCAL, "samplewise/")
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT_LOCAL, "models/multi_tm_denoiser/")
    BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

    # Assicurati che le directory di output esistano localmente
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOISY_ECG_DIR, exist_ok=True) # Se generate_noisy_ecg è separato
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True) # Se preprocessing è separato

# --- Configurazione ---
# Parametri Binarizzazione (devono corrispondere a preprocessing)
BINARIZATION_METHOD = "combined"
BINARIZATION_PARAMS = {"quant_n_bins": 4}
NUM_OUTPUT_BITS = 7 # (2 mean + 1 grad + 4 quant con n_bins=4) - Verifica coerenza!
CONTEXT_K = 10

# Parametri Tsetlin Machine (da passare al manager)
# Assicurati siano compatibili con la classe TM sottostante (MultiClassTsetlinMachine)
TM_PARAMS = {
    "number_of_clauses": 1000,
    "T": 800,
    "s": 5.0,
    "number_of_state_bits": 8,
    "boost_true_positive_feedback": 1, # Comune per binario
    "indexed": True, # Parametro specifico di MultiClassTsetlinMachine
    # Aggiungi altri parametri kwargs se necessario (es. weighted_clauses, s_range, etc.)
}
EPOCHS_PER_LABEL = 10 # Epoche per *ogni* TM binaria interna
USE_VALIDATION_FOR_BEST_STATE = True # Salva il miglior stato basato su val_acc

# Percorsi
SAMPLE_DATA_DIR = "data/samplewise/"
MODEL_OUTPUT_DIR = "models/multi_tm_denoiser/"
MODEL_FILENAME = "noise_tm_manager.pkl" # Nome file per salvare/caricare il manager
BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)

# Flag per controllare riesecuzione training
FORCE_RETRAIN = False # Metti a True per forzare nuovo addestramento

# --- Funzione di Decodifica Inversa (invariata) ---
def inverse_binarize_combined(y_pred_bin, binarization_info):
    # ... (La tua funzione di decodifica come definita prima) ...
    if binarization_info is None or "bin_edges" not in binarization_info:
        print("⚠️ Errore: Informazioni di binarizzazione (bin_edges) non disponibili per la decodifica.")
        return np.zeros(y_pred_bin.shape[0])
    if y_pred_bin.shape[1] != NUM_OUTPUT_BITS:
         print(f"⚠️ Attenzione: y_pred_bin ha {y_pred_bin.shape[1]} colonne, ma NUM_OUTPUT_BITS è {NUM_OUTPUT_BITS}.")
    bin_edges = binarization_info["bin_edges"]
    n_bins_quant = binarization_info.get("n_bins_quant", 4)
    num_samples = y_pred_bin.shape[0]
    y_pred_numeric = np.zeros(num_samples)
    quant_bits_start_idx = 3
    quant_bits_end_idx = min(quant_bits_start_idx + n_bins_quant, y_pred_bin.shape[1])
    quant_bits = y_pred_bin[:, quant_bits_start_idx : quant_bits_end_idx]
    if quant_bits.shape[1] != n_bins_quant:
         print(f"⚠️ Attenzione: Estratti {quant_bits.shape[1]} bit per quantizzazione, ma n_bins_quant è {n_bins_quant}.")
    for i in range(num_samples):
        quantized_level = np.sum(quant_bits[i, :])
        quantized_level = min(int(quantized_level), n_bins_quant -1)
        if quantized_level + 1 < len(bin_edges):
            lower_edge = bin_edges[quantized_level]
            upper_edge = bin_edges[quantized_level + 1]
            y_pred_numeric[i] = (lower_edge + upper_edge) / 2.0
        else:
            y_pred_numeric[i] = bin_edges[-1]
            # if i == 0: print(f"⚠️ Attenzione: Livello quantizzato {quantized_level} fuori range per bin_edges (len={len(bin_edges)}). Usato ultimo edge.")
    return y_pred_numeric

# --- Main Script Logic ---
if __name__ == "__main__":

    # --- 1. Caricamento Dati ---
    print("--- Fase 1: Caricamento Dati ---")
    # Dati Binarizzati
    print("🔹 Loading sample-centric data (binarized)...")
    try:
        dataset = ECGSamplewiseDataset(SAMPLE_DATA_DIR)
        X_samples_bin, y_samples_bin = dataset.get_data()
        dataset.summary()
    except FileNotFoundError:
        print(f"❌ Errore: File campionati non trovati in '{SAMPLE_DATA_DIR}'.")
        print("   Esegui prima 'preprocessing_multilabel.py'.")
        exit()
    except Exception as e:
        print(f"❌ Errore caricamento dati campionati: {e}")
        exit()

    # Verifica Coerenza Bit
    if y_samples_bin.shape[1] != NUM_OUTPUT_BITS:
        print(f"❌ Errore Critico: NUM_OUTPUT_BITS ({NUM_OUTPUT_BITS}) non corrisponde alle colonne di y_samples_bin ({y_samples_bin.shape[1]})!")
        exit()

    # Dati Numerici Puliti Originali
    print("🔹 Loading original numeric data for evaluation...")
    try:
        y_clean_numeric_samples = np.load(os.path.join(SAMPLE_DATA_DIR, "y_clean_numeric_samples.npy"))
    except FileNotFoundError:
        print("❌ Errore: File 'y_clean_numeric_samples.npy' non trovato.")
        exit()

    # Verifica Coerenza Dimensioni Campioni
    if X_samples_bin.shape[0] != y_clean_numeric_samples.shape[0]:
         print("❌ Errore: Incoerenza tra numero campioni binarizzati e numerici originali.")
         exit()

    # Informazioni Binarizzazione
    print("🔹 Loading binarization info...")
    binarization_info = load_binarization_info(BIN_INFO_PATH)
    if binarization_info is None:
        print("❌ Errore: Impossibile procedere senza informazioni di binarizzazione.")
        exit()
    # Verifica n_bins (opzionale ma utile)
    loaded_n_bins = binarization_info.get("n_bins_quant")
    if loaded_n_bins is None: print("⚠️ 'n_bins_quant' non trovato in binarization_info.pkl.")
    elif loaded_n_bins != BINARIZATION_PARAMS["quant_n_bins"]: print(f"⚠️ n_bins info ({loaded_n_bins}) != n_bins params ({BINARIZATION_PARAMS['quant_n_bins']})")

    # --- 2. Split Train/Validation ---
    print("\n--- Fase 2: Split Dati Train/Validation ---")
    indices = np.arange(X_samples_bin.shape[0])
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    X_train_bin, X_val_bin = X_samples_bin[train_indices], X_samples_bin[val_indices]
    y_train_bin, y_val_bin = y_samples_bin[train_indices], y_samples_bin[val_indices]
    # Dati numerici corrispondenti per la valutazione
    y_train_clean_num = y_clean_numeric_samples[train_indices]
    y_val_clean_num = y_clean_numeric_samples[val_indices]

    print(f"Train shapes: X_bin={X_train_bin.shape}, y_bin={y_train_bin.shape}")
    print(f"Val shapes:   X_bin={X_val_bin.shape}, y_bin={y_val_bin.shape}")

    # --- INIZIO BLOCCO DI DEBUG DISTRIBUZIONE CLASSI ---
    print("\n--- Verifica Distribuzione Classi Label 6 (Indice 6) ---")
    label_index_to_check = 6 # Indice dell'ultima etichetta
    if y_train_bin.shape[1] > label_index_to_check: # Assicurati che l'indice sia valido
        unique_train, counts_train = np.unique(y_train_bin[:, label_index_to_check], return_counts=True)
        print(f"  Training Label {label_index_to_check}: Valori={unique_train}, Conteggi={counts_train}")
        if len(unique_train) < 2:
            print(f"  ⚠️ ATTENZIONE: Manca una classe (0 o 1) nel training set per Label {label_index_to_check}!")

        if y_val_bin.shape[1] > label_index_to_check:
            unique_val, counts_val = np.unique(y_val_bin[:, label_index_to_check], return_counts=True)
            print(f"  Validation Label {label_index_to_check}: Valori={unique_val}, Conteggi={counts_val}")
            if len(unique_val) < 2:
                print(f"  ⚠️ ATTENZIONE: Manca una classe (0 o 1) nel validation set per Label {label_index_to_check}!")
        else:
            print(f"  Errore: Indice {label_index_to_check} non valido per y_val_bin (shape: {y_val_bin.shape})")
    else:
        print(f"  Errore: Indice {label_index_to_check} non valido per y_train_bin (shape: {y_train_bin.shape})")

    # --- 3. Training o Caricamento Modello ---
    print("\n--- Fase 3: Training o Caricamento Modello ---")
    manager = None
    # Controlla se esiste un modello salvato e se non si forza il retrain
    if os.path.exists(MODEL_SAVE_PATH) and not FORCE_RETRAIN:
        print(f"🔹 Trovato modello esistente in '{MODEL_SAVE_PATH}'. Caricamento...")
        try:
            # Usa il metodo load della classe
            manager = MultiLabelTsetlinMachine.load(MODEL_SAVE_PATH)
            # Verifica rapida se i parametri caricati corrispondono a quelli attuali
            if manager.n_labels != NUM_OUTPUT_BITS or manager.tm_params != TM_PARAMS:
                 print("⚠️ Attenzione: Parametri del modello caricato differiscono da quelli correnti.")
                 print("   Parametri caricati:", manager.tm_params)
                 print("   Parametri correnti:", TM_PARAMS)
                 if input("   Forzare retrain con parametri correnti? (s/N): ").lower() == 's':
                      manager = None # Forza retrain
                 else:
                      print("   Utilizzo parametri del modello caricato.")
        except Exception as e:
            print(f"❌ Errore caricamento modello: {e}. Si procederà con il training.")
            manager = None # Assicura che si faccia retrain

    # Se il manager non è stato caricato o si forza il retrain
    if manager is None:
        print("🔹 Inizializzazione e Training nuovo modello...")
        # Inizializza il manager
        manager = MultiLabelTsetlinMachine(
            n_labels=NUM_OUTPUT_BITS,
            **TM_PARAMS # Passa i parametri TM definiti sopra
        )
        # Addestra il manager
        manager.fit(
            X_train_bin,
            y_train_bin,
            epochs=EPOCHS_PER_LABEL,
            X_val=X_val_bin,
            Y_val=y_val_bin,
            use_best_state=USE_VALIDATION_FOR_BEST_STATE,
            verbose=True
        )
        # Salva il modello addestrato
        manager.save(MODEL_SAVE_PATH)
    else:
        print("✅ Modello caricato.")

    # --- 4. Valutazione Denoising sul Set di Validazione ---
    print("\n--- Fase 4: Valutazione Denoising (Validation Set) ---")

    # 1. Predizione Multi-Label Binarizzata
    #    Ora chiamiamo il metodo predict del manager *una sola volta*
    y_pred_bin_val = manager.predict(X_val_bin)

    # Valutazione Accuratezza Multi-Label (opzionale)
    print("\n  Valutazione Accuratezza Binaria (per label):")
    avg_accuracy = 0
    for i in range(NUM_OUTPUT_BITS):
         acc = accuracy_score(y_val_bin[:, i], y_pred_bin_val[:, i])
         print(f"    Accuratezza Bit {i}: {acc:.4f}")
         avg_accuracy += acc
    print(f"    Accuratezza Media per Bit: {avg_accuracy / NUM_OUTPUT_BITS:.4f}")
    # Metrica multi-label: Hamming Loss (frazione di etichette predette errate)
    h_loss = hamming_loss(y_val_bin, y_pred_bin_val)
    print(f"    Hamming Loss: {h_loss:.4f}")


    # 2. Decodifica Inversa (Inverse Binarization)
    print("\n  Decoding binary predictions to numeric...")
    start_decode_time = time.time()
    y_pred_numeric_val = inverse_binarize_combined(y_pred_bin_val, binarization_info)
    end_decode_time = time.time()
    print(f"  Decoding time: {end_decode_time - start_decode_time:.2f}s")

    # 3. Calcolo Metriche Numeriche Denoising
    print("\n  Calculating denoising metrics...")
    if len(y_pred_numeric_val) == len(y_val_clean_num):
        try:
            mse_denoised = mean_squared_error(y_val_clean_num, y_pred_numeric_val)
            rmse_denoised = np.sqrt(mse_denoised)
        except Exception as e:
            print(f"  Errore calcolo MSE/RMSE: {e}")
            rmse_denoised = np.nan

        try:
            snr_output = calculate_snr(y_val_clean_num, y_pred_numeric_val)
        except Exception as e:
            print(f"  Errore calcolo SNR: {e}")
            snr_output = np.nan

        print(f"\n  --- Denoising Results (Validation Set) ---")
        print(f"    RMSE (Denoised vs Clean): {rmse_denoised:.6f}")
        print(f"    SNR Output (Denoised vs Clean): {snr_output:.2f} dB")

        # Calcolo Baseline (Rumoroso vs Pulito)
        try:
            y_val_noisy_num = np.load(os.path.join(SAMPLE_DATA_DIR, "y_noisy_numeric_samples.npy"))[val_indices]
            mse_noisy = mean_squared_error(y_val_clean_num, y_val_noisy_num)
            rmse_noisy = np.sqrt(mse_noisy)
            snr_input = calculate_snr(y_val_clean_num, y_val_noisy_num)
            print(f"\n    --- Baseline (Noisy vs Clean) ---")
            print(f"    RMSE (Noisy vs Clean):    {rmse_noisy:.6f}")
            print(f"    SNR Input (Noisy vs Clean): {snr_input:.2f} dB")
            print(f"\n    --- Improvement ---")
            if not np.isnan(rmse_denoised) and not np.isnan(rmse_noisy):
                 print(f"    RMSE Decrease: {rmse_noisy - rmse_denoised:.6f} {'✅' if rmse_noisy > rmse_denoised else '❌'}")
            else: print(f"    RMSE Decrease: N/A")
            if not np.isnan(snr_output) and not np.isnan(snr_input):
                 print(f"    SNR Improvement: {snr_output - snr_input:.2f} dB {'✅' if snr_output > snr_input else '❌'}")
            else: print(f"    SNR Improvement: N/A")
        except FileNotFoundError:
            print("\n    (Skipping baseline comparison: 'y_noisy_numeric_samples.npy' not found)")
        except Exception as e:
             print(f"\n    (Skipping baseline comparison due to error: {e})")

    else:
        print("❌ Errore: Lunghezza predizioni numeriche non corrisponde ai target numerici.")

    # 4. Visualizzazione Esempio (Opzionale)
    print("\n  Visualizing example reconstruction...")
    try:
        plot_idx = 50 # Indice nel set di validazione
        clean_sample_to_plot = y_val_clean_num[plot_idx]
        denoised_sample_to_plot = y_pred_numeric_val[plot_idx]

        print(f"    Visualizing sample index {plot_idx} from validation set:")
        print(f"      Original Clean Value: {clean_sample_to_plot:.4f}")
        try:
             y_val_noisy_num = np.load(os.path.join(SAMPLE_DATA_DIR, "y_noisy_numeric_samples.npy"))[val_indices]
             noisy_sample_to_plot = y_val_noisy_num[plot_idx]
             print(f"      Original Noisy Value: {noisy_sample_to_plot:.4f}")
        except Exception: print("      Original Noisy Value: (non disponibile)")
        print(f"      Denoised Value (TM):  {denoised_sample_to_plot:.4f}")

        # Plot di un segmento più lungo (richiede assemblaggio)
        # Esempio: Assembla i primi N campioni del validation set
        N_plot_samples = 1000 # Numero di campioni da plottare
        if len(y_val_clean_num) >= N_plot_samples:
             plot_signal_comparison(
                 clean=y_val_clean_num[:N_plot_samples],
                 noisy=y_val_noisy_num[:N_plot_samples] if 'y_val_noisy_num' in locals() else None, # Passa noisy solo se caricato
                 denoised=y_pred_numeric_val[:N_plot_samples],
                 title_prefix=f"ECG Denoising TM (Val Set Samples 0-{N_plot_samples-1})"
             )

    except IndexError: print("    (Skipping visualization: plot_idx out of bounds)")
    except Exception as e: print(f"    (Skipping visualization due to error: {e})")

    print("\n🏁 Script execution finished.")