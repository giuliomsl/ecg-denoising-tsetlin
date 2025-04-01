import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import KBinsDiscretizer

def binarize_median(signal):
    """Binarizza in base alla mediana del segnale."""
    threshold = np.median(signal)
    binary = (signal > threshold).astype(int)
    diff = np.diff(signal, prepend=signal[0])
    binary_diff = (diff > 0).astype(int)
    return np.column_stack((binary, binary_diff))

def binarize_mean_filter(signal, window_size=50):
    """Binarizza in base a una soglia media mobile."""
    smooth = uniform_filter1d(signal, size=window_size)
    binary = (signal > smooth).astype(int)
    diff = np.diff(signal, prepend=signal[0])
    binary_diff = (diff > 0).astype(int)
    return np.column_stack((binary, binary_diff))

def binarize_quantized(signal, n_bins=4):
    """Quantizza il segnale in più livelli, poi codifica binariamente."""
    signal = signal.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    quantized = est.fit_transform(signal).astype(int).flatten()

    # Codifica ogni livello come vettore binario (thermometer encoding)
    thermometer = np.array([[1 if i <= val else 0 for i in range(n_bins)] for val in quantized])
    return thermometer  # shape: (len(signal), n_bins)

def binarize_quantized_adaptive(signal, n_bins=4):
    """Quantizzazione adattiva (per quantili) con thermometer encoding."""
    signal = signal.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    quantized = est.fit_transform(signal).astype(int).flatten()
    
    thermometer = np.array([[1 if i <= val else 0 for i in range(n_bins)] for val in quantized])
    return thermometer

def binarize_gradient_window(signal, window_size=10):
    """Binarizza usando la tendenza (trend) locale su finestre mobili."""
    trend = np.sign(signal[window_size:] - signal[:-window_size])
    trend = np.pad(trend, (window_size, 0), mode='edge')
    binary = (trend > 0).astype(int)
    return binary.reshape(-1, 1)

def binarize_combined(signal):
    mean_bin = binarize_mean_filter(signal)       # 2 colonne
    grad_bin = binarize_gradient_window(signal)   # 1 colonna
    quant_bin = binarize_quantized_adaptive(signal, n_bins=3)  # 3 colonne
    return np.concatenate((mean_bin, grad_bin, quant_bin), axis=1)



def binarize(signal, method="median"):
    """Interfaccia unica per binarizzazione."""
    if method == "median":
        return binarize_median(signal)
    elif method == "mean":
        return binarize_mean_filter(signal)
    elif method == "quantized":
        return binarize_quantized(signal)
    elif method == "gradient":
        return binarize_gradient_window(signal)
    elif method == "combined":
        return binarize_combined(signal)
    else:
        raise ValueError(f"Metodo di binarizzazione sconosciuto: {method}")
