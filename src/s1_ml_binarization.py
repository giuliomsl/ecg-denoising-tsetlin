# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
import pickle
import os

# --- Suggerimenti Generali per Sperimentazione ---
# 1.  **Normalizzazione Pre-Binarizzazione:**
#     Considera di normalizzare il segnale *prima* di applicare metodi di quantizzazione
#     (es., `binarize_quantized`, `binarize_quantized_adaptive`).
#     - Scalare tra 0 e 1: `scaler = MinMaxScaler(); signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()`
#     - Standardizzare (media 0, std 1): `scaler = StandardScaler(); signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()`
#     Questo può influenzare l'efficacia delle strategie 'uniform' e 'quantile' in KBinsDiscretizer.
#     Applica la normalizzazione prima di chiamare `binarize()` in `preprocessing_multilabel.py`.

# 2.  **Metodi Singoli vs Combinati:**
#     Prova a usare metodi di binarizzazione singoli (es., `method="quantized"`)
#     invece di `"combined"` per vedere quale cattura meglio le informazioni rilevanti.
#     Modifica il parametro `method` nella chiamata a `binarize()` in `preprocessing_multilabel.py`.

# 3.  **Parametri dei Metodi:**
#     - `n_bins` in `binarize_quantized*`: Valore cruciale. Prova 3, 4, 5, 8, 10...
#       Un numero maggiore di bin cattura più dettagli ma aumenta la complessità (più bit).
#     - `window_size` in `binarize_mean_filter` e `binarize_gradient_window`:
#       Influenza la scala temporale. Finestre più grandi catturano trend più lenti,
#       finestre più piccole catturano variazioni più rapide. Prova valori diversi (es., 5, 10, 20, 50, 100).

def binarize_median(signal):
    """Binarizza in base alla mediana del segnale."""
    threshold = np.median(signal)
    binary = (signal > threshold).astype(np.uint8) # Usiamo uint8 per efficienza
    diff = np.diff(signal, prepend=signal[0])
    binary_diff = (diff > 0).astype(np.uint8)
    return np.column_stack((binary, binary_diff))

def binarize_mean_filter(signal, window_size=50):
    """Binarizza in base a una soglia media mobile."""
    # window_size: Iperparametro da ottimizzare (es. 10, 20, 50, 100)
    smooth = uniform_filter1d(signal, size=window_size)
    binary = (signal > smooth).astype(np.uint8)
    diff = np.diff(signal, prepend=signal[0])
    binary_diff = (diff > 0).astype(np.uint8)
    return np.column_stack((binary, binary_diff))

def binarize_quantized(signal, n_bins=4, strategy='uniform'):
    """Quantizza il segnale in più livelli, poi codifica binariamente (thermometer)."""
    # n_bins: Iperparametro cruciale (es. 3, 4, 5, 8)
    # strategy: 'uniform' o 'quantile'
    signal = signal.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None) # subsample=None per usare tutti i dati
    quantized = est.fit_transform(signal).astype(int).flatten()

    # Codifica ogni livello come vettore binario (thermometer encoding)
    thermometer = np.array([[1 if i < val else 0 for i in range(n_bins)] for val in quantized], dtype=np.uint8)
    # Salva i bin edges per la decodifica inversa (IMPORTANTE!)
    # Nota: Questo dovrebbe essere fatto nel preprocessing e salvato
    # qui è solo un esempio di come ottenerli
    bin_edges = est.bin_edges_[0]
    return thermometer, bin_edges # Restituisce anche i bordi per la decodifica

def binarize_quantized_adaptive(signal, n_bins): # Rimosso default=4
    """Quantizzazione adattiva (per quantili) con thermometer encoding standard."""
    if not isinstance(n_bins, int) or n_bins <= 1:
         raise ValueError("n_bins deve essere un intero maggiore di 1")
    signal = signal.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
    quantized = est.fit_transform(signal).astype(int).flatten()
    bin_edges = est.bin_edges_[0]

    thermometer = np.zeros((len(quantized), n_bins), dtype=np.uint8)
    for i in range(len(quantized)):
        level = quantized[i]
        for j in range(n_bins):
            if level >= j:
                thermometer[i, j] = 1
    return thermometer, bin_edges

def binarize_gradient_window(signal, window_size=10):
    """Binarizza usando la tendenza (trend) locale su finestre mobili."""
    # window_size: Iperparametro da ottimizzare (es. 5, 10, 20)
    if len(signal) <= window_size:
        # Gestisce segnali corti restituendo un trend costante (es. 0)
        return np.zeros((len(signal), 1), dtype=np.uint8)
    trend = np.sign(signal[window_size:] - signal[:-window_size])
    # Sostituisci 0 (nessun cambiamento) con -1 (decrescente) o +1 (crescente)
    # Qui scegliamo di mappare 0 a 0 (bit per decrescente/stabile)
    trend[trend == 0] = -1 # Considera 0 come decrescente/stabile ai fini del bit
    trend = np.pad(trend, (window_size // 2, window_size - window_size // 2), mode='edge') # Padding più bilanciato
    binary = (trend > 0).astype(np.uint8) # 1 se crescente, 0 se decrescente/stabile
    return binary.reshape(-1, 1)

def binarize_combined(signal, quant_n_bins, mean_filt_window=50, grad_window=10): # quant_n_bins ora è richiesto
    """
    Combinare diversi metodi di binarizzazione.
    quant_n_bins è ora un argomento richiesto.
    """
    mean_bin = binarize_mean_filter(signal, window_size=mean_filt_window) # 2 colonne
    grad_bin = binarize_gradient_window(signal, window_size=grad_window)  # 1 colonna
    quant_bin, bin_edges = binarize_quantized_adaptive(signal, n_bins=quant_n_bins) # Usa l'argomento

    # Il numero totale di bit ora è 2 + 1 + quant_n_bins
    n_total_bits = 2 + 1 + quant_n_bins
    print(f"DEBUG (binarize_combined): n_bins={quant_n_bins} -> quant_bin shape={quant_bin.shape} -> total bits={n_total_bits}")

    combined_binary = np.concatenate((mean_bin, grad_bin, quant_bin), axis=1)
    # Salva n_bins usato nelle info ausiliarie
    aux_info = {"bin_edges": bin_edges, "n_bins_quant": quant_n_bins}

    return combined_binary, aux_info


# Interfaccia unica
def binarize(signal, method="combined", **kwargs):
    """Interfaccia unica per binarizzazione."""
    if method == "median":
        return binarize_median(signal), {}
    elif method == "mean":
        # Passa window_size se fornito
        ws = kwargs.get('mean_filt_window', 50)
        return binarize_mean_filter(signal, window_size=ws), {}
    elif method == "quantized":
        nb = kwargs.get('quant_n_bins', 4)
        strat = kwargs.get('quant_strategy', 'uniform')
        # binarize_quantized restituisce anche bin_edges
        bin_data, edges = binarize_quantized(signal, n_bins=nb, strategy=strat)
        return bin_data, {"bin_edges": edges, "n_bins_quant": nb}
    elif method == "quantized_adaptive":
        nb = kwargs.get('quant_n_bins', 4)
        bin_data, edges = binarize_quantized_adaptive(signal, n_bins=nb)
        return bin_data, {"bin_edges": edges, "n_bins_quant": nb}
    elif method == "gradient":
        ws = kwargs.get('grad_window', 10)
        return binarize_gradient_window(signal, window_size=ws), {}
    elif method == "combined":
        # Estrai quant_n_bins da kwargs, altrimenti errore perché è richiesto
        if 'quant_n_bins' not in kwargs:
             raise ValueError("Argomento 'quant_n_bins' è richiesto per method='combined'")
        return binarize_combined(signal,
                                 quant_n_bins=kwargs['quant_n_bins'], # Passa il valore
                                 mean_filt_window=kwargs.get('mean_filt_window', 50),
                                 grad_window=kwargs.get('grad_window', 10))
    else:
        raise ValueError(f"Metodo di binarizzazione sconosciuto: {method}")

# --- Funzione per salvare/caricare info ausiliarie ---
def save_binarization_info(info, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(info, f)

def load_binarization_info(filepath):
     try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
     except FileNotFoundError:
         print(f"⚠️ File informazioni binarizzazione non trovato: {filepath}")
         return None