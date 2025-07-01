#!/usr/bin/env python3
"""
Generate Noisy ECG Segments
===========================

This script automates the creation of a dataset for ECG denoising. It performs the following steps:

1.  **Loads Clean ECG Signals**: Reads full ECG recordings from the MIT-BIH Arrhythmia Database.
2.  **Generates Realistic Noise**: Creates a composite noise signal by combining:
    -   Baseline Wander (BW) from the NSTDB.
    -   Muscle Artifact (MA) from the NSTDB.
    -   Powerline Interference (PLI), synthetically generated with optional harmonics.
3.  **Creates Noisy ECGs**: Adds the composite noise to the clean ECG signals.
4.  **Segments Signals**: Splits the clean signals, noisy signals, and individual noise components into fixed-length windows (segments).
5.  **Saves Segments**: Saves each segment as a separate .npy file for use in training and evaluation.

This process ensures that for each segment, we have the clean version, the noisy version, and the constituent noise components, which is useful for various denoising model architectures.

Author: Your Name
Date: June 2025
"""

import os
import numpy as np
import wfdb
import random
from tqdm import tqdm

# --- Environment Flag ---
# Set this to True if running on Google Colab to handle Drive mounting.
RUNNING_ON_COLAB = False
# -------------------------

# --- General Configurations ---
SEGMENT_LENGTH = 1024  # The length of each output window/segment.
OVERLAP_LENGTH = 0     # Overlap between consecutive segments.
# -----------------------------

# --- Path Definitions ---
if RUNNING_ON_COLAB:
    print("INFO: RUNNING_ON_COLAB flag is set to True.")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("INFO: Mounting Google Drive...")
            drive.mount('/content/drive', force_remount=True)
            import time
            time.sleep(5) # Allow time for mounting
        else:
            print("INFO: Google Drive is already mounted.")
    except ImportError:
        print("ERROR: Could not import google.colab. Exiting.")
        exit()
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/"
    DATA_DIR = os.path.join(GDRIVE_BASE, "data")
else:
    print("INFO: RUNNING_ON_COLAB flag is set to False (Local Environment).")
    # Assuming the script is run from the project root `denoising_ecg`
    PROJECT_ROOT = "."
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
SEGMENTED_SIGNAL_DIR = os.path.join(DATA_DIR, "segmented_signals") # Output directory for segments

print(f"INFO: Using ECG_DIR: {ECG_DIR}")
print(f"INFO: Using NOISE_DIR: {NOISE_DIR}")
print(f"INFO: Generated Segment Output Directory: {SEGMENTED_SIGNAL_DIR}")
os.makedirs(SEGMENTED_SIGNAL_DIR, exist_ok=True)

# --- Noise Generation Configurations ---
RANDOMIZE_NOISE_LEVELS = True   # If True, noise component weights are randomized.
LEVEL_MIN_WEIGHT = 0.1          # Min weight for a noise component.
LEVEL_MAX_WEIGHT = 0.6          # Max weight for a noise component.
RANDOMIZE_PLI_AMPLITUDE = True  # If True, PLI amplitude is randomized.
PLI_AMP_MIN = 0.05              # Min amplitude for PLI.
PLI_AMP_MAX = 0.25              # Max amplitude for PLI.
ADD_PLI_HARMONICS_PROB = 0.15   # Probability of adding 3rd and 5th harmonics to PLI.
RECORDS_TO_PROCESS = []         # List of specific record names to process. Leave empty to process all.
ENABLE_PLOTTING_DEBUG = False   # If True, plots generated signals for debugging (requires matplotlib).

# --- Helper Functions ---

def load_ecg_full(record_name):
    """
    Loads the full ECG signal for a given record name.

    Args:
        record_name (str): The base name of the record (e.g., '100').

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The full ECG signal (channel 0).
            - int: The sampling frequency (fs).
        Returns (None, None) on failure.
    """
    record_path = os.path.join(ECG_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        ecg_signal_full = record.p_signal[:, 0]
        fs = record.fs
        if len(ecg_signal_full) == 0:
            raise ValueError("ECG signal is empty.")
        return ecg_signal_full, fs
    except Exception as e:
        print(f"❌ Error loading record {record_name}: {e}")
        return None, None

def load_noise_realistically(noise_name, target_length):
    """
    Loads a noise signal from the noise directory and adjusts its length.

    If the noise signal is shorter than target_length, it's tiled.
    If longer, a random segment is extracted.

    Args:
        noise_name (str): The name of the noise file (e.g., 'bw', 'ma').
        target_length (int): The desired length of the output noise signal.

    Returns:
        np.ndarray: The noise signal of the specified target_length.
    """
    noise_path = os.path.join(NOISE_DIR, noise_name)
    try:
        noise_record = wfdb.rdrecord(noise_path)
        full_noise_signal = noise_record.p_signal[:, 0]
    except Exception as e:
        print(f"❌ Error loading noise {noise_name}: {e}")
        return np.zeros(target_length)
    
    current_len = len(full_noise_signal)
    if current_len == 0:
        return np.zeros(target_length)
    
    if current_len < target_length:
        # Tile the noise signal to match the target length
        return np.tile(full_noise_signal, int(np.ceil(target_length / current_len)))[:target_length]
    elif current_len > target_length:
        # Extract a random segment from the noise signal
        start_index = random.randint(0, current_len - target_length)
        return full_noise_signal[start_index : start_index + target_length]
    
    return full_noise_signal

def generate_pli(length, fs=360, base_freq=50):
    """
    Generates a synthetic Powerline Interference (PLI) signal.

    Features randomized amplitude, frequency deviation, and optional harmonics.

    Args:
        length (int): The desired length of the PLI signal.
        fs (int): The sampling frequency.
        base_freq (int): The base frequency of the PLI (e.g., 50Hz or 60Hz).

    Returns:
        np.ndarray: The generated PLI signal.
    """
    t = np.arange(length) / fs
    amplitude = random.uniform(PLI_AMP_MIN, PLI_AMP_MAX) if RANDOMIZE_PLI_AMPLITUDE else 0.1
    
    # Introduce slight frequency deviation for realism
    deviation_factor = 1 + random.uniform(0, 0.08)
    effective_freq = base_freq * deviation_factor
    
    pli_signal = amplitude * np.sin(2 * np.pi * effective_freq * t)
    
    # Add 3rd and 5th harmonics with a certain probability
    if random.random() < ADD_PLI_HARMONICS_PROB:
        pli_signal += amplitude * random.uniform(0.1, 0.4) * np.sin(2 * np.pi * (3 * effective_freq) * t)
    if random.random() < ADD_PLI_HARMONICS_PROB:
        pli_signal += amplitude * random.uniform(0.05, 0.3) * np.sin(2 * np.pi * (5 * effective_freq) * t)
        
    return pli_signal

def generate_noise_weights(n_weights=3, min_w=0.1, max_w=0.6, total_sum=1.0):
    """
    Generates random weights for noise components that sum to a target value.

    This is a non-trivial problem. This implementation uses an iterative approach
    to generate weights that are within the [min_w, max_w] range and sum close to total_sum.

    Args:
        n_weights (int): The number of weights to generate.
        min_w (float): The minimum value for any weight.
        max_w (float): The maximum value for any weight.
        total_sum (float): The target sum of the weights.

    Returns:
        np.ndarray: An array of generated weights.
    """
    # Start with a Dirichlet distribution, which ensures the sum is total_sum.
    weights = np.random.dirichlet(np.ones(n_weights)) * total_sum
    
    # Iteratively adjust weights to meet min/max constraints
    for _ in range(10):
        if np.all(weights >= min_w) and np.all(weights <= max_w):
            break # Constraints satisfied
        
        weights = np.clip(weights, min_w, max_w)
        current_sum = np.sum(weights)
        
        if current_sum > 0:
            weights = weights * (total_sum / current_sum) # Re-normalize
        else:
            weights = np.ones(n_weights) * (total_sum / n_weights) # Fallback
            
    # Final clip and re-normalization to be safe
    weights = np.clip(weights, min_w, max_w)
    if not np.isclose(np.sum(weights), total_sum):
        weights = weights * (total_sum / np.sum(weights))
        
    return weights

def generate_noisy_ecg_components(ecg_signal, bw_template_signal, ma_template_signal, pli_template_signal):
    """
    Combines noise templates and adds them to a clean ECG signal.

    Args:
        ecg_signal (np.ndarray): The clean ECG signal.
        bw_template_signal (np.ndarray): The baseline wander noise template.
        ma_template_signal (np.ndarray): The muscle artifact noise template.
        pli_template_signal (np.ndarray): The powerline interference noise template.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The final noisy ECG signal.
            - np.ndarray: The weighted baseline wander component.
            - np.ndarray: The weighted muscle artifact component.
            - np.ndarray: The weighted powerline interference component.
    """
    if RANDOMIZE_NOISE_LEVELS:
        bw_weight, ma_weight, pli_weight = generate_noise_weights(
            min_w=LEVEL_MIN_WEIGHT, max_w=LEVEL_MAX_WEIGHT
        )
    else:
        # Fixed weights if not randomizing
        bw_weight, ma_weight, pli_weight = (0.33, 0.33, 0.34)

    # Apply weights to the noise templates
    bw_component = bw_template_signal * bw_weight
    ma_component = ma_template_signal * ma_weight
    pli_component = pli_template_signal * pli_weight
    
    # Create the final noisy signal
    noisy_ecg = ecg_signal + bw_component + ma_component + pli_component
    
    return noisy_ecg, bw_component, ma_component, pli_component

def segment_signal(signal, segment_len, overlap_len):
    """
    Splits a signal into overlapping or non-overlapping segments.

    Args:
        signal (np.ndarray): The input signal to segment.
        segment_len (int): The length of each segment.
        overlap_len (int): The number of samples to overlap between segments.

    Returns:
        np.ndarray: A 2D array where each row is a segment.
    """
    segments = []
    step = segment_len - overlap_len
    for i in range(0, len(signal) - segment_len + 1, step):
        segments.append(signal[i : i + segment_len])
    return np.array(segments) if segments else np.empty((0, segment_len))

def save_segmented_signals(output_dir, base_record_name, segment_idx, signals_dict_for_segment):
    """
    Saves a dictionary of signal segments to .npy files.

    Args:
        output_dir (str): The directory to save the files in.
        base_record_name (str): The name of the original ECG record.
        segment_idx (int): The index of the current segment.
        signals_dict_for_segment (dict): A dictionary where keys are signal types
                                         (e.g., 'clean', 'noisy') and values are the
                                         signal data arrays.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        for signal_type, signal_data in signals_dict_for_segment.items():
            # Consistent file naming convention
            file_name = f"{base_record_name}_{segment_idx:03d}_{signal_type}.npy"
            file_path = os.path.join(output_dir, file_name)
            np.save(file_path, signal_data)
        return True
    except Exception as e:
        print(f"❌ Error saving segments for {base_record_name}_{segment_idx}: {e}")
        return False

def process_record_and_save_segments(ecg_record_name, output_dir, seg_len, overlap_len):
    """
    Main processing pipeline for a single ECG record.

    Loads ECG, generates noise, creates segments, and saves them.

    Args:
        ecg_record_name (str): The name of the ECG record to process.
        output_dir (str): The directory to save the segmented files.
        seg_len (int): The length of each segment.
        overlap_len (int): The overlap between segments.

    Returns:
        int: The number of segments successfully generated and saved.
    """
    ecg_signal_full, fs = load_ecg_full(ecg_record_name)
    if ecg_signal_full is None:
        return 0

    full_length = len(ecg_signal_full)
    
    # Generate noise templates with the same length as the full ECG signal
    bw_noise_template = load_noise_realistically("bw", full_length)
    ma_noise_template = load_noise_realistically("ma", full_length)
    pli_noise_template = generate_pli(full_length, fs)
    
    # Create the full noisy signal and get individual noise components
    noisy_ecg_full, bw_comp_full, ma_comp_full, pli_comp_full = generate_noisy_ecg_components(
        ecg_signal_full, bw_noise_template, ma_noise_template, pli_noise_template
    )
    
    # Dictionary of all signals to be segmented
    all_signals_to_segment = {
        "clean": ecg_signal_full,
        "noisy": noisy_ecg_full,
        "noise_bw": bw_comp_full,
        "noise_ma": ma_comp_full,
        "noise_pli": pli_comp_full
    }
    
    segmented_data = {}
    num_segments = -1

    # Segment all signals
    for sig_type, sig_full in all_signals_to_segment.items():
        segments = segment_signal(sig_full, seg_len, overlap_len)
        if num_segments == -1:
            num_segments = len(segments)
        elif len(segments) != num_segments:
            print(f"⚠️ Segment count mismatch for {ecg_record_name} ({sig_type}). Skipping record.")
            return 0
        segmented_data[sig_type] = segments

    if num_segments == 0:
        return 0

    segments_processed_count = 0
    for i in range(num_segments):
        # Create a dictionary for the i-th segment of all signal types
        signals_for_this_segment = {st: segmented_data[st][i] for st in segmented_data}
        if save_segmented_signals(output_dir, ecg_record_name, i, signals_for_this_segment):
            segments_processed_count += 1
            
    # Optional plotting for debugging the first segment
    if ENABLE_PLOTTING_DEBUG and num_segments > 0:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 1, 1)
            plt.title(f"Record: {ecg_record_name} - First Segment")
            plt.plot(segmented_data["clean"][0], label="Clean ECG")
            plt.plot(segmented_data["noisy"][0], label="Noisy ECG", alpha=0.7)
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.title("Noise Components")
            plt.plot(segmented_data["noise_bw"][0], label="BW Noise")
            plt.plot(segmented_data["noise_ma"][0], label="MA Noise")
            plt.plot(segmented_data["noise_pli"][0], label="PLI Noise")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.title("Total Added Noise")
            total_noise = segmented_data["noisy"][0] - segmented_data["clean"][0]
            plt.plot(total_noise, label="Total Noise")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("WARNING: matplotlib is not installed. Cannot plot debug graphs.")

    return segments_processed_count

if __name__ == "__main__":
    print(f"--- Generating Noisy Segmented ECG Dataset (Segment Length: {SEGMENT_LENGTH}) ---")
    try:
        all_ecg_files = sorted([f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")])
    except FileNotFoundError:
        print(f"❌ ERROR: ECG directory '{ECG_DIR}' not found. Please check the path.")
        exit()

    if not all_ecg_files:
        print(f"❌ ERROR: No .dat files found in '{ECG_DIR}'.")
        exit()

    records_to_run = all_ecg_files if not RECORDS_TO_PROCESS else [r for r in RECORDS_TO_PROCESS if r in all_ecg_files]
    
    if not records_to_run:
        print("INFO: No records selected for processing. Exiting.")
        exit()
        
    print(f"▶️ Processing {len(records_to_run)} ECG records...")
    total_segments_generated = 0
    
    # Use tqdm for a progress bar
    for ecg_name in tqdm(records_to_run, desc="Processing Records"):
        total_segments_generated += process_record_and_save_segments(
            ecg_name, SEGMENTED_SIGNAL_DIR, SEGMENT_LENGTH, OVERLAP_LENGTH
        )
        
    print(f"\n✅ Generation complete. Total segments created: {total_segments_generated}")
    print(f"   Files saved in: '{SEGMENTED_SIGNAL_DIR}'")