# -*- coding: utf-8 -*-
import numpy as np
import pickle
from multilabel_tm_class import MultiLabelTsetlinMachine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from ecg_dataset import ECGSamplewiseDataset
from ecg_utils import calculate_snr, plot_signal_comparison # Assumiamo esista plot_signal_comparison
from binarization import load_binarization_info # Per caricare info decodifica
import os
import time # Per misurare tempi

# --- Config ---
BINARIZATION_METHOD = "combined" # Assicurati sia lo stesso usato nel preprocessing
BINARIZATION_PARAMS = {"quant_n_bins": 4} # Parametri usati in binarize()
# CORREZIONE: Imposta a 7 per corrispondere ai dati caricati e a binarize_combined
NUM_OUTPUT_BITS = 7 # (2 mean + 1 grad + 4 quant con n_bins=4)
CONTEXT_K = 10 # Deve corrispondere a preprocessing_multilabel.py

TM_PARAMS = {"number_of_clauses": 1000, "T": 800, "s": 5.0, "number_of_state_bits": 8} # Esempio - DA OTTIMIZZARE!
EPOCHS = 10 # Riduci per test rapidi, aumenta per training reale
MODEL_OUTPUT_DIR = "models/multi_tm_denoiser/"
BIN_INFO_PATH = os.path.join("data/samplewise/", "binarization_info.pkl") # Dove salvare/caricare info decodifica
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# --- Funzione di Decodifica Inversa ---
def inverse_binarize_combined(y_pred_bin, binarization_info):
    """
    Riconverte l'output binario predetto in un valore numerico approssimativo.
    ATTENZIONE: Questa è un'implementazione SEMPLIFICATA che decodifica
              SOLO la parte quantizzata (thermometer). Le altre parti (mean, grad)
              sono ignorate per semplicità iniziale. Richiede miglioramenti.

    Args:
        y_pred_bin (np.ndarray): Array (n_samples, NUM_OUTPUT_BITS) di bit predetti.
        binarization_info (dict): Dizionario contenente info salvate da binarize(),
                                   in particolare 'bin_edges' e 'n_bins_quant'.

    Returns:
        np.ndarray: Array (n_samples,) di valori numerici ricostruiti.
    """
    if binarization_info is None or "bin_edges" not in binarization_info:
        print("⚠️ Errore: Informazioni di binarizzazione (bin_edges) non disponibili per la decodifica.")
        return np.zeros(y_pred_bin.shape[0])

    # Verifica coerenza NUM_OUTPUT_BITS con y_pred_bin
    if y_pred_bin.shape[1] != NUM_OUTPUT_BITS:
         print(f"⚠️ Attenzione: y_pred_bin ha {y_pred_bin.shape[1]} colonne, ma NUM_OUTPUT_BITS è {NUM_OUTPUT_BITS}.")
         # Potrebbe essere necessario aggiustare gli indici sotto

    bin_edges = binarization_info["bin_edges"]
    n_bins_quant = binarization_info.get("n_bins_quant", 4) # Usa default se manca
    num_samples = y_pred_bin.shape[0]
    y_pred_numeric = np.zeros(num_samples)

    # Estrai i bit corrispondenti alla quantizzazione thermometer
    # Assumendo: 2 bit mean + 1 bit grad = 3 bit iniziali
    quant_bits_start_idx = 3
    # Assicurati che l'indice finale non superi il numero di colonne
    quant_bits_end_idx = min(quant_bits_start_idx + n_bins_quant, y_pred_bin.shape[1])
    quant_bits = y_pred_bin[:, quant_bits_start_idx : quant_bits_end_idx]

    # Verifica se abbiamo estratto il numero corretto di bit quantizzati
    if quant_bits.shape[1] != n_bins_quant:
         print(f"⚠️ Attenzione: Estratti {quant_bits.shape[1]} bit per quantizzazione, ma n_bins_quant è {n_bins_quant}.")
         # La decodifica potrebbe essere errata. Controlla la struttura dei bit.

    for i in range(num_samples):
        # Decodifica Thermometer: conta quanti '1' ci sono
        # La nostra implementazione in binarization.py ora fa:
        # Livello 0 -> [0,0,0,0], Livello 1 -> [1,0,0,0], Livello 2 -> [1,1,0,0], etc.
        # Quindi la somma degli '1' dà direttamente il livello quantizzato.
        quantized_level = np.sum(quant_bits[i, :])

        # Mappa il livello quantizzato al valore numerico (es. centro del bin)
        # Assicurati che il livello non ecceda i limiti degli indici di bin_edges
        # bin_edges ha n_bins + 1 elementi. Livello k usa bin_edges[k] e bin_edges[k+1]
        quantized_level = min(int(quantized_level), n_bins_quant -1) # Assicura sia intero e nel range valido

        # Calcola il punto medio del bin corrispondente
        # Gestisci potenziali errori se bin_edges non ha la lunghezza attesa
        if quantized_level + 1 < len(bin_edges):
            lower_edge = bin_edges[quantized_level]
            upper_edge = bin_edges[quantized_level + 1]
            y_pred_numeric[i] = (lower_edge + upper_edge) / 2.0
        else:
            # Fallback: usa l'ultimo bin edge o un valore di default
            y_pred_numeric[i] = bin_edges[-1]
            if i == 0: # Stampa avviso solo una volta
                 print(f"⚠️ Attenzione: Livello quantizzato {quantized_level} fuori range per bin_edges (len={len(bin_edges)}). Usato ultimo edge.")


        # --- Placeholder per decodifica Mean/Gradient (Bit 0-2) ---
        # ... (come prima) ...

    return y_pred_numeric

# --- Load Sample-Centric Data ---
print("🔹 Loading sample-centric data (binarized)...")
dataset = ECGSamplewiseDataset("data/samplewise/")
X_samples_bin, y_samples_bin = dataset.get_data()
dataset.summary()

# Verifica coerenza tra NUM_OUTPUT_BITS e dati caricati
if y_samples_bin.shape[1] != NUM_OUTPUT_BITS:
    print(f"❌ Errore Critico: NUM_OUTPUT_BITS ({NUM_OUTPUT_BITS}) non corrisponde alle colonne di y_samples_bin ({y_samples_bin.shape[1]})!")
    print("   Verifica la configurazione (NUM_OUTPUT_BITS) e l'output di binarization.py.")
    exit()

# --- Load Original Numeric Data (per valutazione) ---
# ... (come prima, ma assicurati che il file esista) ...
print("🔹 Loading original numeric data for evaluation...")
try:
    y_clean_numeric_samples = np.load("data/samplewise/y_clean_numeric_samples.npy")
except FileNotFoundError:
    print("❌ Errore: File 'data/samplewise/y_clean_numeric_samples.npy' non trovato.")
    print("   Assicurati che 'preprocessing_multilabel.py' lo abbia creato correttamente.")
    exit()

# Verifica coerenza dimensioni
if X_samples_bin.shape[0] != y_clean_numeric_samples.shape[0]:
     print("❌ Errore: Incoerenza tra numero campioni binarizzati e numerici originali.")
     exit()

# --- Load Binarization Info (per decodifica) ---
# ... (come prima) ...
print("🔹 Loading binarization info...")
binarization_info = load_binarization_info(BIN_INFO_PATH)
if binarization_info is None:
    print("❌ Errore: Impossibile procedere senza informazioni di binarizzazione.")
    exit()
# Verifica che n_bins corrisponda
loaded_n_bins = binarization_info.get("n_bins_quant")
if loaded_n_bins is None:
     print("⚠️ Attenzione: 'n_bins_quant' non trovato in binarization_info.pkl.")
elif loaded_n_bins != BINARIZATION_PARAMS["quant_n_bins"]:
    print(f"⚠️ Attenzione: n_bins nelle info ({loaded_n_bins}) "
          f"diverso da quello nei parametri ({BINARIZATION_PARAMS['quant_n_bins']})")


# --- Train/Validation Split ---
# ... (come prima) ...
indices = np.arange(X_samples_bin.shape[0])
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True) # Shuffle è ok qui

X_train_bin, X_val_bin = X_samples_bin[train_indices], X_samples_bin[val_indices]
y_train_bin, y_val_bin = y_samples_bin[train_indices], y_samples_bin[val_indices]
y_train_clean_num = y_clean_numeric_samples[train_indices]
y_val_clean_num = y_clean_numeric_samples[val_indices]
# y_val_noisy_num = X_noisy_numeric_samples[val_indices] # Se disponibile per SNR input

print(f"Train shapes: X_bin={X_train_bin.shape}, y_bin={y_train_bin.shape}, y_num={y_train_clean_num.shape}")
print(f"Val shapes:   X_bin={X_val_bin.shape}, y_bin={y_val_bin.shape}, y_num={y_val_clean_num.shape}")


# --- Train one TM per output bit ---
trained_tm_states = []
training_times = []
print("\n⚙️ Starting Training...")

for i in range(NUM_OUTPUT_BITS): # Ora itererà da 0 a 6
    print(f"\n--- Training TM for Output Bit {i} ---")
    start_time_bit = time.time()
    # Estrai il target binario per questo bit
    y_train_bit_i = y_train_bin[:, i]
    y_val_bit_i = y_val_bin[:, i]

    # Inizializza TM (MultiClass funziona anche per binario)
    # RIMOSSO n_jobs=-1
    tm = MultiLabelTsetlinMachine(
        number_of_clauses=TM_PARAMS["number_of_clauses"],
        T=TM_PARAMS["T"],
        s=TM_PARAMS["s"],
        number_of_state_bits=TM_PARAMS["number_of_state_bits"],
        boost_true_positive_feedback=1 # Default
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        tm.fit(X_train_bin, y_train_bit_i, epochs=1) # Addestra per 1 epoca alla volta
        # Calcola accuratezza (gestisci possibile divisione per zero se val set è piccolo)
        try:
            train_pred = tm.predict(X_train_bin)
            train_acc = accuracy_score(y_train_bit_i, train_pred)
        except Exception as e:
            print(f"  Errore calcolo train accuracy: {e}")
            train_acc = 0.0
        try:
            val_pred = tm.predict(X_val_bin)
            val_acc = accuracy_score(y_val_bit_i, val_pred)
        except Exception as e:
            print(f"  Errore calcolo val accuracy: {e}")
            val_acc = 0.0

        print(f"  Epoch {epoch+1}/{EPOCHS} - Bit {i}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Salva il modello migliore basato sulla validazione
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                best_state = tm.get_state()
                print(f"    * New best validation accuracy for bit {i}: {best_val_acc:.4f}")
            except Exception as e:
                 print(f"  Errore ottenimento stato TM: {e}")
                 best_state = None # Non salvare stato se c'è errore

    end_time_bit = time.time()
    elapsed_bit = end_time_bit - start_time_bit
    training_times.append(elapsed_bit)
    print(f"  Bit {i} training time: {elapsed_bit:.2f}s")

    # Salva lo stato migliore (o l'ultimo se non c'è stato miglioramento o errore)
    state_to_save = None
    if best_state is not None:
        state_to_save = best_state
    else:
        try:
            # Prova a ottenere l'ultimo stato come fallback
            state_to_save = tm.get_state()
            print(f"  ⚠️ No best state found based on validation for bit {i}, saving last state.")
        except Exception as e:
            print(f"  ❌ Errore critico: Impossibile ottenere lo stato finale della TM per bit {i}: {e}")
            # Aggiungi uno stato placeholder o gestisci l'errore
            trained_tm_states.append(None) # Segnala che questo modello non è valido
            continue # Salta al prossimo bit

    trained_tm_states.append(state_to_save)
    model_bit_path = os.path.join(MODEL_OUTPUT_DIR, f"tm_bit_{i}_state.pkl")
    try:
        with open(model_bit_path, "wb") as f:
            pickle.dump(state_to_save, f)
        print(f"  💾 Saved state for bit {i} to {model_bit_path}")
    except Exception as e:
        print(f"  ❌ Errore salvataggio stato per bit {i}: {e}")


print(f"\n✅ Training complete for {len(trained_tm_states)}/{NUM_OUTPUT_BITS} bits. Total time: {sum(training_times):.2f}s")

# --- Valutazione Denoising sul Set di Validazione ---
print("\n📊 Evaluating Denoising Performance on Validation Set...")

# 1. Ricostruzione Binarizzata
print("  Predicting bits on validation set...")
# Verifica che abbiamo uno stato valido per ogni bit prima di procedere
if any(state is None for state in trained_tm_states):
     print("❌ Errore: Manca lo stato di almeno una TM addestrata. Impossibile valutare.")
     exit()

y_pred_bin_val = np.zeros_like(y_val_bin, dtype=np.uint8)
prediction_times = []

for i in range(NUM_OUTPUT_BITS):
    start_pred_bit = time.time()
    # Inizializza una nuova TM per predire
    tm_eval = MultiLabelTsetlinMachine(
        number_of_clauses=TM_PARAMS["number_of_clauses"],
        T=TM_PARAMS["T"],
        s=TM_PARAMS["s"],
        number_of_state_bits=TM_PARAMS["number_of_state_bits"]
    )
    try:
        tm_eval.set_state(trained_tm_states[i]) # Usa lo stato salvato
        y_pred_bin_val[:, i] = tm_eval.predict(X_val_bin)
    except Exception as e:
         print(f"❌ Errore durante la predizione per bit {i}: {e}")
         # Imposta predizioni a un valore di default (es. 0) o gestisci
         y_pred_bin_val[:, i] = 0 # Esempio: imposta a 0 in caso di errore

    end_pred_bit = time.time()
    prediction_times.append(end_pred_bit - start_pred_bit)

print(f"  Bit prediction time: {sum(prediction_times):.2f}s")

# 2. Decodifica Inversa (Inverse Binarization)
# ... (come prima) ...
print("  Decoding binary predictions to numeric...")
start_decode_time = time.time()
y_pred_numeric_val = inverse_binarize_combined(y_pred_bin_val, binarization_info)
end_decode_time = time.time()
print(f"  Decoding time: {end_decode_time - start_decode_time:.2f}s")


# 3. Calcolo Metriche Numeriche
# ... (come prima, ma aggiungi gestione errori per SNR) ...
print("  Calculating denoising metrics...")
if len(y_pred_numeric_val) == len(y_val_clean_num):
    try:
        mse_denoised = mean_squared_error(y_val_clean_num, y_pred_numeric_val)
        rmse_denoised = np.sqrt(mse_denoised)
    except Exception as e:
        print(f"  Errore calcolo MSE/RMSE: {e}")
        rmse_denoised = np.nan # Not a Number

    try:
        snr_output = calculate_snr(y_val_clean_num, y_pred_numeric_val)
    except Exception as e:
        print(f"  Errore calcolo SNR: {e}")
        snr_output = np.nan # Not a Number

    print(f"\n  --- Denoising Results (Validation Set) ---")
    print(f"    RMSE (Denoised vs Clean): {rmse_denoised:.6f}")
    print(f"    SNR Output (Denoised vs Clean): {snr_output:.2f} dB")

    # Blocco opzionale per baseline (come prima)
    try:
        # Assicurati che il file esista e sia stato creato da preprocessing_multilabel.py
        y_val_noisy_num = np.load("data/samplewise/y_noisy_numeric_samples.npy")[val_indices]
        mse_noisy = mean_squared_error(y_val_clean_num, y_val_noisy_num)
        rmse_noisy = np.sqrt(mse_noisy)
        snr_input = calculate_snr(y_val_clean_num, y_val_noisy_num)
        print(f"\n    --- Baseline (Noisy vs Clean) ---")
        print(f"    RMSE (Noisy vs Clean):    {rmse_noisy:.6f}")
        print(f"    SNR Input (Noisy vs Clean): {snr_input:.2f} dB")
        print(f"\n    --- Improvement ---")
        # Verifica che le metriche siano numeri validi prima del confronto
        if not np.isnan(rmse_denoised) and not np.isnan(rmse_noisy):
             print(f"    RMSE Decrease: {rmse_noisy - rmse_denoised:.6f} {'✅' if rmse_noisy > rmse_denoised else '❌'}")
        else:
             print(f"    RMSE Decrease: N/A (calcolo fallito)")
        if not np.isnan(snr_output) and not np.isnan(snr_input):
             print(f"    SNR Improvement: {snr_output - snr_input:.2f} dB {'✅' if snr_output > snr_input else '❌'}")
        else:
             print(f"    SNR Improvement: N/A (calcolo fallito)")
    except FileNotFoundError:
        print("\n    (Skipping baseline comparison: 'y_noisy_numeric_samples.npy' not found)")
    except Exception as e:
         print(f"\n    (Skipping baseline comparison due to error: {e})")

else:
    print("❌ Errore: Lunghezza predizioni numeriche non corrisponde ai target numerici.")


# 4. Visualizzazione Esempio (Opzionale)
# ... (come prima) ...
print("\n  Visualizing example reconstruction...")
try:
    plot_idx = 50 # Indice del campione nel set di validazione da visualizzare
    clean_sample_to_plot = y_val_clean_num[plot_idx]
    denoised_sample_to_plot = y_pred_numeric_val[plot_idx]

    print(f"    Visualizing sample index {plot_idx} from validation set:")
    print(f"      Original Clean Value: {clean_sample_to_plot:.4f}")
    # Carica e stampa il valore rumoroso se disponibile
    try:
         y_val_noisy_num = np.load("data/samplewise/y_noisy_numeric_samples.npy")[val_indices]
         noisy_sample_to_plot = y_val_noisy_num[plot_idx]
         print(f"      Original Noisy Value: {noisy_sample_to_plot:.4f}")
    except FileNotFoundError:
         print("      Original Noisy Value: (file non trovato)")
    except IndexError:
         print("      Original Noisy Value: (indice fuori range nel file noisy)")

    print(f"      Denoised Value (TM):  {denoised_sample_to_plot:.4f}")

    # Codice per plottare segmenti (più complesso, richiede assemblaggio)
    # ...

except IndexError:
    print("    (Skipping visualization: plot_idx out of bounds for validation set)")
except Exception as e:
    print(f"    (Skipping visualization due to error: {e})")


print("\n🏁 Script execution finished.")