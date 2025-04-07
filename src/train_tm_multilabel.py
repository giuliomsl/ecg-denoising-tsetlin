# -*- coding: utf-8 -*-
import numpy as np
import pickle
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
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
NUM_OUTPUT_BITS = 6 # Basato su binarize_combined con quant_n_bins=4 (2 mean + 1 grad + 3 quant)
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
        # Restituisce zero o solleva un errore
        return np.zeros(y_pred_bin.shape[0])

    bin_edges = binarization_info["bin_edges"]
    n_bins_quant = binarization_info["n_bins_quant"]
    num_samples = y_pred_bin.shape[0]
    y_pred_numeric = np.zeros(num_samples)

    # Estrai i bit corrispondenti alla quantizzazione thermometer
    # Assumendo: 2 bit mean + 1 bit grad = 3 bit iniziali
    quant_bits_start_idx = 3
    quant_bits = y_pred_bin[:, quant_bits_start_idx : quant_bits_start_idx + n_bins_quant]

    for i in range(num_samples):
        # Decodifica Thermometer: conta quanti '1' ci sono (o trova l'ultimo '1')
        # Esempio n_bins=4 (3 bit): [0,0,0]->0, [1,0,0]->1, [1,1,0]->2, [1,1,1]->3
        quantized_level = np.sum(quant_bits[i, :]) # Somma gli 1 per ottenere il livello

        # Mappa il livello quantizzato al valore numerico (es. centro del bin)
        # Assicurati che il livello non ecceda i limiti dei bin_edges
        quantized_level = min(quantized_level, n_bins_quant - 1) # n_bins_quant livelli (0 a n_bins_quant-1)

        # Calcola il punto medio del bin corrispondente
        lower_edge = bin_edges[quantized_level]
        upper_edge = bin_edges[quantized_level + 1]
        y_pred_numeric[i] = (lower_edge + upper_edge) / 2.0

        # --- Placeholder per decodifica Mean/Gradient (Bit 0-2) ---
        # La decodifica di questi bit è complessa e richiederebbe:
        # 1. Mappe precalcolate (durante preprocessing) che associano pattern
        #    binari (es. [0,1] per bit 0-1) al valore numerico medio originale
        #    che li ha generati.
        # 2. Una strategia per *combinare* il valore decodificato dalla parte
        #    quantizzata con le informazioni qualitative/medie da mean/gradient.
        # Per ora, ignoriamo questi bit nella ricostruzione numerica.
        # mean_bits = y_pred_bin[i, 0:2]
        # grad_bit = y_pred_bin[i, 2]
        # numeric_value_from_mean = lookup_mean_value(mean_bits) # Funzione fittizia
        # numeric_value_from_grad = lookup_grad_value(grad_bit) # Funzione fittizia
        # y_pred_numeric[i] = combine_decoded_values(quant_value, mean_value, grad_value) # Funzione fittizia
        # ----------------------------------------------------------

    return y_pred_numeric

# --- Load Sample-Centric Data ---
print("🔹 Loading sample-centric data (binarized)...")
dataset = ECGSamplewiseDataset("data/samplewise/")
X_samples_bin, y_samples_bin = dataset.get_data()
dataset.summary()

# --- Load Original Numeric Data (per valutazione) ---
# Assumiamo che i dati originali siano salvati o possano essere ricaricati
# Questo è un ESEMPIO, adatta il percorso e la logica al tuo salvataggio
print("🔹 Loading original numeric data for evaluation...")
try:
    # Potresti aver salvato X originale e y originale separatamente
    # o doverli ricaricare e segmentare qui.
    # Per semplicità, assumiamo esistano file .npy con i dati numerici
    # corrispondenti ai campioni binarizzati.
    # NOTA: Questo richiede che preprocessing_multilabel.py salvi anche
    #       i target numerici originali (y_clean_numeric_samples.npy)
    y_clean_numeric_samples = np.load("data/samplewise/y_clean_numeric_samples.npy") # File fittizio! Devi crearlo.
    # Potrebbe servire anche il segnale rumoroso originale per SNR input
    # X_noisy_numeric_samples = np.load("data/samplewise/X_noisy_numeric_samples.npy") # File fittizio!
except FileNotFoundError:
    print("❌ Errore: File dati numerici originali non trovati. Impossibile valutare il denoising.")
    print("   Assicurati che 'preprocessing_multilabel.py' salvi 'y_clean_numeric_samples.npy'.")
    exit()

# Verifica coerenza dimensioni
if X_samples_bin.shape[0] != y_clean_numeric_samples.shape[0]:
     print("❌ Errore: Incoerenza tra numero campioni binarizzati e numerici originali.")
     exit()

# --- Load Binarization Info (per decodifica) ---
print("🔹 Loading binarization info...")
binarization_info = load_binarization_info(BIN_INFO_PATH)
if binarization_info is None:
    print("❌ Errore: Impossibile procedere senza informazioni di binarizzazione.")
    exit()
# Verifica che n_bins corrisponda
if binarization_info.get("n_bins_quant") != BINARIZATION_PARAMS["quant_n_bins"]:
    print(f"⚠️ Attenzione: n_bins nelle info ({binarization_info.get('n_bins_quant')}) "
          f"diverso da quello nei parametri ({BINARIZATION_PARAMS['quant_n_bins']})")


# --- Train/Validation Split ---
# Split mantenendo corrispondenza tra binarizzato e numerico
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

for i in range(NUM_OUTPUT_BITS):
    print(f"\n--- Training TM for Output Bit {i} ---")
    start_time_bit = time.time()
    # Estrai il target binario per questo bit
    y_train_bit_i = y_train_bin[:, i]
    y_val_bit_i = y_val_bin[:, i]

    # Inizializza TM (MultiClass funziona anche per binario)
    # Usiamo parametri da TM_PARAMS
    tm = MultiClassTsetlinMachine(
        number_of_clauses=TM_PARAMS["number_of_clauses"],
        T=TM_PARAMS["T"],
        s=TM_PARAMS["s"],
        number_of_state_bits=TM_PARAMS["number_of_state_bits"],
        boost_true_positive_feedback=1, # Default
        n_jobs=-1 # Usa tutti i core
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        tm.fit(X_train_bin, y_train_bit_i, epochs=1) # Addestra per 1 epoca alla volta
        train_acc = accuracy_score(y_train_bit_i, tm.predict(X_train_bin))
        val_acc = accuracy_score(y_val_bit_i, tm.predict(X_val_bin))
        print(f"  Epoch {epoch+1}/{EPOCHS} - Bit {i}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Salva il modello migliore basato sulla validazione
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = tm.get_state()
            print(f"    * New best validation accuracy for bit {i}: {best_val_acc:.4f}")

    end_time_bit = time.time()
    elapsed_bit = end_time_bit - start_time_bit
    training_times.append(elapsed_bit)
    print(f"  Bit {i} training time: {elapsed_bit:.2f}s")

    # Salva lo stato migliore (o l'ultimo se non c'è stato miglioramento)
    state_to_save = best_state if best_state is not None else tm.get_state()
    trained_tm_states.append(state_to_save)
    model_bit_path = os.path.join(MODEL_OUTPUT_DIR, f"tm_bit_{i}_state.pkl")
    with open(model_bit_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  💾 Saved best state for bit {i} to {model_bit_path}")


print(f"\n✅ Training complete for all bits. Total time: {sum(training_times):.2f}s")

# --- Valutazione Denoising sul Set di Validazione ---
print("\n📊 Evaluating Denoising Performance on Validation Set...")

# 1. Ricostruzione Binarizzata
print("  Predicting bits on validation set...")
y_pred_bin_val = np.zeros_like(y_val_bin, dtype=np.uint8)
prediction_times = []

for i in range(NUM_OUTPUT_BITS):
    start_pred_bit = time.time()
    # Carica lo stato salvato e inizializza una nuova TM per predire
    tm_eval = MultiClassTsetlinMachine(
        number_of_clauses=TM_PARAMS["number_of_clauses"],
        T=TM_PARAMS["T"],
        s=TM_PARAMS["s"],
        number_of_state_bits=TM_PARAMS["number_of_state_bits"]
        # Non servono n_jobs o boost per la sola predizione
    )
    tm_eval.set_state(trained_tm_states[i])
    y_pred_bin_val[:, i] = tm_eval.predict(X_val_bin)
    end_pred_bit = time.time()
    prediction_times.append(end_pred_bit - start_pred_bit)

print(f"  Bit prediction time: {sum(prediction_times):.2f}s")

# 2. Decodifica Inversa (Inverse Binarization)
print("  Decoding binary predictions to numeric...")
start_decode_time = time.time()
y_pred_numeric_val = inverse_binarize_combined(y_pred_bin_val, binarization_info)
end_decode_time = time.time()
print(f"  Decoding time: {end_decode_time - start_decode_time:.2f}s")

# 3. Calcolo Metriche Numeriche
print("  Calculating denoising metrics...")
# Assicurati che y_val_clean_num sia disponibile e abbia la stessa lunghezza
if len(y_pred_numeric_val) == len(y_val_clean_num):
    mse_denoised = mean_squared_error(y_val_clean_num, y_pred_numeric_val)
    rmse_denoised = np.sqrt(mse_denoised)
    snr_output = calculate_snr(y_val_clean_num, y_pred_numeric_val)

    print(f"\n  --- Denoising Results (Validation Set) ---")
    print(f"    RMSE (Denoised vs Clean): {rmse_denoised:.6f}")
    print(f"    SNR Output (Denoised vs Clean): {snr_output:.2f} dB")

    # Opzionale: Calcola metriche baseline (Rumoroso vs Pulito) se hai y_val_noisy_num
    # try:
    #     y_val_noisy_num = np.load("data/samplewise/X_noisy_numeric_samples.npy")[val_indices] # Esempio caricamento
    #     mse_noisy = mean_squared_error(y_val_clean_num, y_val_noisy_num)
    #     rmse_noisy = np.sqrt(mse_noisy)
    #     snr_input = calculate_snr(y_val_clean_num, y_val_noisy_num)
    #     print(f"\n    --- Baseline (Noisy vs Clean) ---")
    #     print(f"    RMSE (Noisy vs Clean):    {rmse_noisy:.6f}")
    #     print(f"    SNR Input (Noisy vs Clean): {snr_input:.2f} dB")
    #     print(f"\n    --- Improvement ---")
    #     print(f"    RMSE Decrease: {rmse_noisy - rmse_denoised:.6f} {'✅' if rmse_noisy > rmse_denoised else '❌'}")
    #     print(f"    SNR Improvement: {snr_output - snr_input:.2f} dB {'✅' if snr_output > snr_input else '❌'}")
    # except FileNotFoundError:
    #     print("\n    (Skipping baseline comparison: noisy numeric data not found)")
    # except NameError:
    #      print("\n    (Skipping baseline comparison: noisy numeric data not loaded)")

else:
    print("❌ Errore: Lunghezza predizioni numeriche non corrisponde ai target numerici.")

# 4. Visualizzazione Esempio (Opzionale)
print("\n  Visualizing example reconstruction...")
try:
    plot_idx = 50 # Indice del campione nel set di validazione da visualizzare
    # Trova il campione rumoroso originale corrispondente (se disponibile)
    # noisy_sample_to_plot = y_val_noisy_num[plot_idx]
    clean_sample_to_plot = y_val_clean_num[plot_idx]
    denoised_sample_to_plot = y_pred_numeric_val[plot_idx]

    print(f"    Visualizing sample index {plot_idx} from validation set:")
    print(f"      Original Clean Value: {clean_sample_to_plot:.4f}")
    # print(f"      Original Noisy Value: {noisy_sample_to_plot:.4f}") # Se disponibile
    print(f"      Denoised Value (TM):  {denoised_sample_to_plot:.4f}")

    # Potresti anche voler plottare un segmento più lungo ricostruito
    # Questo richiederebbe di assemblare y_pred_numeric_val in segmenti
    # e plottarli contro i segmenti originali puliti e rumorosi.
    # plot_signal_comparison(segmento_pulito, segmento_rumoroso, segmento_denoisato)

except IndexError:
    print("    (Skipping visualization: plot_idx out of bounds)")
# except NameError:
#     print("    (Skipping visualization: necessary numeric data not available)")


print("\n🏁 Script execution finished.")