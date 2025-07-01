import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=360, order=4):
    """
    Applies a bandpass filter to the ECG signal.
    
    Args:
        data (np.ndarray): Signal to be filtered
        lowcut (float): Low cut-off frequency in Hz
        highcut (float): High cut-off frequency in Hz  
        fs (float): Sampling frequency in Hz
        order (int): Filter order
    
    Returns:
        np.ndarray: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in decibels between the clean and noisy signals.

    Args:
        clean_signal (np.ndarray): Reference clean signal.
        noisy_signal (np.ndarray): Contaminated or denoised signal.

    Returns:
        float: SNR value in dB.
    """
    noise = clean_signal - noisy_signal
    power_signal = np.mean(clean_signal ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0:
        return float('inf')
    return 10 * np.log10(power_signal / power_noise)

def plot_signal_comparison(clean, noisy, denoised=None, title_prefix="ECG"):
    """
    Plots the comparison between signals: clean, noisy, and optionally denoised.

    Expects numerical signals so use it after reconstruction!!!!!

    Args:
        clean (np.ndarray): Clean signal.
        noisy (np.ndarray): Noisy signal.
        denoised (np.ndarray, optional): Denoised signal.
        title_prefix (str): Base title of the graph.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(clean, label="Clean", color='green')
    plt.plot(noisy, label="Noisy", color='red', alpha=0.5)
    if denoised is not None:
        plt.plot(denoised, label="Denoised", color='blue', alpha=0.7)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(f"{title_prefix} - Signal Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

def normalize_signal(signal):
    """
    Normalizes an ECG signal between -1 and 1.

    Args:
        signal (np.ndarray): Signal to normalize.

    Returns:
        np.ndarray: Normalized signal.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return 2 * (signal - min_val) / (max_val - min_val) - 1


def thermometer_encode(x, thresholds):
    """Thermometer encoding: x.shape = (n,), thresholds = sorted list."""
    return np.array([[int(val <= t) for t in thresholds] for val in x])

def difference_encode(x, thresholds=[-0.1, 0, 0.1]):
    """Encode differences x(t) - x(t-1) on a ternary basis with thresholds."""
    dx = np.diff(x, prepend=x[0])
    return np.array([[int(d <= t) for t in thresholds] for d in dx])

def moving_avg_dev(x, window_size=15, thresholds=[-0.1, 0.1]):
    """Deviation from the binarized moving average value."""
    pad = window_size // 2
    padded = np.pad(x, (pad, pad), mode='edge')
    ma = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    delta = x - ma
    return np.array([[int(d <= thresholds[0]), int(d >= thresholds[1])] for d in delta])

import os
import re

def find_best_model_path(model_dir, model_prefix, noise_type, q_suffix):
    """
    Finds the path to the best-performing model based on a naming convention.

    The function searches for files ending with "_BEST.state" that match the
    specified prefix, noise type, and quantile suffix.

    Args:
        model_dir (str): The directory where the models are stored.
        model_prefix (str): The prefix of the model file name (e.g., 'rtm_denoiser').
        noise_type (str): The type of noise the model was trained for (e.g., 'aggregated').
        q_suffix (str): The quantile suffix used during training (e.g., '_q20').

    Returns:
        str: The full path to the best model file found.
        None: If no matching model file is found.
    """
    # Regex to find the model file, allowing for any configuration details in the middle
    # Example: rtm_denoiser_aggregated_clauses200..._q20_BEST.state
    pattern = re.compile(f"^{re.escape(model_prefix)}_{re.escape(noise_type)}.*{re.escape(q_suffix)}_BEST\.state$")
    
    try:
        for filename in os.listdir(model_dir):
            if pattern.match(filename):
                print(f"Found best model: {filename}")
                return os.path.join(model_dir, filename)
    except FileNotFoundError:
        print(f"Error: Model directory not found at '{model_dir}'")
        return None
        
    print(f"Warning: No model matching the pattern found in '{model_dir}'")
    return None
