"""
File: load_ecg.py
Description: 
    This script provides a centralized system for managing file paths and loading 
    data for the ECG denoising project. It ensures that the project runs 
    seamlessly in different environments by dynamically configuring paths. It also 
    includes utility functions to load ECG records and noise signals, and it 
    handles the creation of necessary output directories.
"""

import os
import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
import errno
import sys

# --- Project Directory Setup ---
# This setup assumes the script is in a 'src' folder, and navigates up to the project root.
# This makes the script runnable from any location within the project structure.
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback for interactive environments (like Jupyter) where __file__ is not defined.
    PROJECT_ROOT = os.path.abspath('.')
    print(f"Warning: __file__ not defined. Assuming project root is the current directory: {PROJECT_ROOT}")

# --- Core Data and Model Directories ---
# Define key directories relative to the project root for consistency.
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Derived Project Paths ---
# These paths are constructed from the base directories and are used project-wide.

# Source data directories
MIT_BIH_DIR = os.path.join(DATA_DIR, "mit-bih/")              # MIT-BIH Arrhythmia Database
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")  # MIT-BIH Noise Stress Test Database

# Directories for generated/processed data
GENERATED_SIGNAL_DIR = os.path.join(DATA_DIR, 'generated_signals') # For signals with synthetic noise
SEGMENTED_SIGNAL_DIR = os.path.join(DATA_DIR, "segmented_signals")  # Output of generate_noisy_ecg.py
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samplewise")             # Output of s5_rtm_preprocessing.py
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "rtm_denoiser")          # Output of training scripts

# --- Global Constants ---
SAMPLE_RATE = 360  # Sampling rate for the MIT-BIH dataset in Hz.

# --- Directory Creation ---
# Ensures that all necessary output directories exist.
def create_directories():
    """
    Creates all necessary output directories if they do not already exist.
    """
    dirs_to_create = [
        DATA_DIR,
        MODEL_DIR,
        GENERATED_SIGNAL_DIR,
        SEGMENTED_SIGNAL_DIR,
        SAMPLE_DATA_DIR,
        MODEL_OUTPUT_DIR
    ]
    for d in dirs_to_create:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            # A simple check to avoid crashing if the directory already exists.
            if e.errno != errno.EEXIST:
                print(f"ERROR: Could not create directory {d}. Reason: {e}")
                sys.exit(1) # Exit if a critical directory cannot be created.

# --- Data Loading Utilities ---

def load_mit_bih_records(data_dir=MIT_BIH_DIR):
    """
    Loads all unique MIT-BIH record names from the specified directory.

    It scans the directory for files with the '.dat' extension and extracts
    the base name of each record.

    Args:
        data_dir (str): The path to the directory containing MIT-BIH files.

    Returns:
        list: A sorted list of unique record names (e.g., ['100', '101']).
    """
    try:
        record_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
        record_names = sorted(list(set([f.split('.')[0] for f in record_files])))
        print(f"Found {len(record_names)} unique records in '{data_dir}'.")
        return record_names
    except FileNotFoundError:
        print(f"ERROR: Data directory not found at '{data_dir}'. Please check the path.")
        return []

def get_train_test_split(test_size=0.3, validation_size=0.2, random_state=42, data_dir=MIT_BIH_DIR):
    """
    Splits the available MIT-BIH records into training, validation, and test sets.

    Args:
        test_size (float): The proportion of the dataset to allocate to the test set.
        validation_size (float): The proportion of the *training* set to allocate to the validation set.
        random_state (int): The seed for the random number generator for reproducibility.
        data_dir (str): The directory to scan for records.

    Returns:
        tuple: A tuple containing three lists of record names: (train_ids, val_ids, test_ids).
    """
    all_records = load_mit_bih_records(data_dir)
    if not all_records:
        return [], [], [] # Return empty lists if no records were found

    # Split into initial training set and test set
    train_val_ids, test_ids = train_test_split(
        all_records, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Split the initial training set into final training and validation sets
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=validation_size,
        random_state=random_state
    )
    
    print(f"Data split: {len(train_ids)} training, {len(val_ids)} validation, {len(test_ids)} test records.")
    return train_ids, val_ids, test_ids

def load_ecg_signal(record_name, data_dir=MIT_BIH_DIR, channel=0):
    """
    Loads a single-channel ECG signal from a specified MIT-BIH record.

    Args:
        record_name (str): The name of the record to load (e.g., '100').
        data_dir (str): The directory where the record files are located.
        channel (int): The channel index to extract from the record.

    Returns:
        np.array: A 1D NumPy array containing the ECG signal data.
                  Returns None if the record cannot be loaded.
    """
    record_path = os.path.join(data_dir, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, channel]
        return signal
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None

# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- ECG Data Loading and Path Utility ---")
    
    # Create directories first
    create_directories()
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"MIT-BIH Data Directory: {MIT_BIH_DIR}")
    
    # Demonstrate loading record names
    record_names = load_mit_bih_records()
    if record_names:
        print(f"First 5 records: {record_names[:5]}")

    # Demonstrate train-test split
    train, val, test = get_train_test_split()
    if train:
        print(f"Sample Train IDs: {train[:3]}")
        print(f"Sample Validation IDs: {val[:3]}")
        print(f"Sample Test IDs: {test[:3]}")

    # Demonstrate loading a single signal
    if record_names:
        sample_signal = load_ecg_signal(record_names[0])
        if sample_signal is not None:
            print(f"Successfully loaded signal '{record_names[0]}' with length: {len(sample_signal)}")