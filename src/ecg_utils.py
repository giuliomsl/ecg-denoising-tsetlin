import numpy as np
import matplotlib.pyplot as plt

def calculate_snr(clean_signal, noisy_signal):
    """
    Calcola il Signal-to-Noise Ratio (SNR) in decibel tra il segnale pulito e quello rumoroso.

    Args:
        clean_signal (np.ndarray): Segnale di riferimento pulito.
        noisy_signal (np.ndarray): Segnale contaminato o denoised.

    Returns:
        float: Valore di SNR in dB.
    """
    noise = clean_signal - noisy_signal
    power_signal = np.mean(clean_signal ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0:
        return float('inf')
    return 10 * np.log10(power_signal / power_noise)

def plot_signal_comparison(clean, noisy, denoised=None, title_prefix="ECG"):
    """
    Plotta il confronto tra segnali: pulito, rumoroso, e opzionalmente denoised.

    Si aspetta segnali numerici quindi va usato dopo la ricostruzione!!!!!

    Args:
        clean (np.ndarray): Segnale pulito.
        noisy (np.ndarray): Segnale rumoroso.
        denoised (np.ndarray, optional): Segnale denoised.
        title_prefix (str): Titolo base del grafico.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(clean, label="Pulito", color='green')
    plt.plot(noisy, label="Rumoroso", color='red', alpha=0.5)
    if denoised is not None:
        plt.plot(denoised, label="Denoised", color='blue', alpha=0.7)
    plt.xlabel("Campioni")
    plt.ylabel("Ampiezza")
    plt.title(f"{title_prefix} - Confronto Segnali")
    plt.legend()
    plt.tight_layout()
    plt.show()

def normalize_signal(signal):
    """
    Normalizza un segnale ECG tra -1 e 1.

    Args:
        signal (np.ndarray): Segnale da normalizzare.

    Returns:
        np.ndarray: Segnale normalizzato.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return 2 * (signal - min_val) / (max_val - min_val) - 1
