#!/usr/bin/env python3
"""
S5 RTM Preprocessing - Correct Binarization for RTM
====================================================

Preprocesses ECG signals for the RegressionTsetlinMachine, producing:
- BINARY INPUT (0/1) through quantile binarization of noisy windows.
- CONTINUOUS TARGETS (real values) of the noise to be predicted.

Corrected Pipeline:
1. Load clean and noisy ECG signals.
2. Create windows of size WINDOW_LENGTH.
3. Binarize the noisy windows using quantiles as thresholds.
4. Calculate the continuous noise as the difference (noisy - clean).

Output:
- X_train_rtm_BINARIZED_q<N>.npy: Binarized noisy windows (0/1).
- y_train_rtm_aggregated_noise.npy: Continuous noise to predict (float).
- Analogous files for validation and test sets.

"""

import os
import numpy as np
import glob # Imported to search for files
from sklearn.model_selection import train_test_split
from datetime import datetime
import yaml # To save metadata
import argparse

# Configuration
WINDOW_LENGTH = 1024  # This now represents the length of the loaded segment
# NUM_QUANTILES_FOR_BINARIZATION is defined later
OVERLAP = 0 # Overlap is handled in generate_noisy_ecg.py, not needed here
RANDOM_STATE = 42

# Paths
BASE_PATH = '/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg'
# DATA_PATH points to segmented_signals, which is correct
DATA_PATH = os.path.join(BASE_PATH, 'data', 'segmented_signals')
OUTPUT_PATH = os.path.join(BASE_PATH, 'data', 'samplewise')
METADATA_FILENAME = "metadata_rtm_preprocessing.yaml"


def ensure_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✅ Directory created: {path}")


def load_segmented_data_from_files(data_dir):
    """
    Loads all .npy segments from data_dir.
    Matches 'clean' and 'noisy' segments based on the file name.
    """
    print(f"📂 Loading segments from: {data_dir}...")
    
    clean_segments = []
    noisy_segments = []
    segment_identifiers = [] # For traceability and stratified split if needed

    # Find all _noisy.npy files to start
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "*_noisy.npy")))
    if not noisy_files:
        print(f"⚠️ No *_noisy.npy files found in {data_dir}. Check the path and generated files.")
        return [], [], []

    print(f"🔍 Found {len(noisy_files)} *_noisy.npy files.")

    for noisy_file_path in noisy_files:
        base_name = os.path.basename(noisy_file_path).replace("_noisy.npy", "")
        clean_file_path = os.path.join(data_dir, f"{base_name}_clean.npy")
        
        if os.path.exists(clean_file_path):
            try:
                noisy_segment = np.load(noisy_file_path)
                clean_segment = np.load(clean_file_path)
                
                # Verify segment length
                if noisy_segment.shape[0] == WINDOW_LENGTH and clean_segment.shape[0] == WINDOW_LENGTH:
                    noisy_segments.append(noisy_segment)
                    clean_segments.append(clean_segment)
                    segment_identifiers.append(base_name)
                else:
                    print(f"⚠️ Segment {base_name} discarded: length mismatch ({noisy_segment.shape[0]} vs {WINDOW_LENGTH})")
            except Exception as e:
                print(f"❌ Error loading or processing segment {base_name}: {e}")
        else:
            print(f"⚠️ Corresponding clean file not found for {noisy_file_path}, segment ignored.")
            
    if not noisy_segments or not clean_segments:
        print("❌ No segments loaded successfully. Aborting.")
        return [], [], []

    print(f"📊 Loaded {len(noisy_segments)} noisy segments and {len(clean_segments)} clean segments.")
    return np.array(noisy_segments), np.array(clean_segments), segment_identifiers

# The create_windows function is no longer necessary and can be removed or commented out.
# def create_windows(signals, window_length, overlap):
#     ...

def calculate_noise_targets(noisy_windows, clean_windows):
    """
    Calculates the continuous noise as the target for RTM.
    Extracts only the central sample of each window for RTM regression.
    
    Args:
        noisy_windows: Noisy segments (n_segments, segment_length).
        clean_windows: Clean segments (n_segments, segment_length).
    
    Returns:
        noise_targets: Continuous noise of the central sample (n_segments,).
    """
    print("🎯 Calculating continuous noise targets (center sample)...")
    if noisy_windows.shape != clean_windows.shape:
        print(f"❌ ERROR: Shape of noisy_windows ({noisy_windows.shape}) and clean_windows ({clean_windows.shape}) do not match!")
        raise ValueError("Shapes of noisy and clean segments must be identical.")
    if noisy_windows.ndim != 2 or clean_windows.ndim != 2:
        print(f"❌ ERROR: Input data for calculate_noise_targets must be 2D (n_segments, segment_length).")
        print(f"         Shape noisy: {noisy_windows.shape}, Shape clean: {clean_windows.shape}")
        raise ValueError("Invalid inputs for calculate_noise_targets.")

    # Calculate noise for the entire window/segment
    noise_full_segments = noisy_windows - clean_windows
    
    # Extract only the center sample
    center_idx = noise_full_segments.shape[1] // 2
    noise_targets = noise_full_segments[:, center_idx]
    
    print(f"   📊 Full noise segments: shape {noise_full_segments.shape}")
    print(f"   📊 Center noise targets (idx={center_idx}): shape {noise_targets.shape}")
    if noise_targets.size > 0:
        print(f"   📊 Center noise range: [{noise_targets.min():.6f}, {noise_targets.max():.6f}]")
        print(f"   📊 Average center noise: {noise_targets.mean():.6f} ± {noise_targets.std():.6f}")
    else:
        print("   ⚠️ No noise targets calculated (empty inputs?).")
    
    return noise_targets

def binarize_with_quantiles(data_to_binarize, num_quantiles, fit_thresholds_on=None):
    """
    Binarizes data using quantiles as thresholds.
    Thresholds are calculated on 'fit_thresholds_on' (typically X_train) 
    and then applied to 'data_to_binarize'.
    
    Args:
        data_to_binarize: 2D array (n_segments, segment_length) to be binarized.
        num_quantiles: Number of quantiles for binarization.
        fit_thresholds_on: 2D array (n_segments_train, segment_length) to calculate thresholds on.
                           If None, calculates thresholds on 'data_to_binarize' itself (not recommended for val/test).
    
    Returns:
        binary_data: Binary array (n_segments, segment_length * (num_quantiles - 1)).
        thresholds_used: The thresholds used for binarization.
    """
    print(f"🔄 Binarizing with {num_quantiles} quantiles...")
    
    if fit_thresholds_on is None:
        print("   ⚠️ No data provided for fitting thresholds, calculating them on data_to_binarize.")
        fit_data_for_thresholds = data_to_binarize
    else:
        fit_data_for_thresholds = fit_thresholds_on

    if fit_data_for_thresholds.size == 0:
        print("❌ ERROR: Data for fitting thresholds is empty.")
        # Return empty arrays with the expected dimensionality to avoid cascading errors
        # The expected dimensionality for binarized features is segment_length * number of thresholds
        # If num_quantiles is the number of intervals, then there are num_quantiles-1 thresholds.
        # If num_quantiles is the number of points (e.g., 100 points for 99 thresholds), then it's num_quantiles-1.
        # My previous implementation used num_quantiles+1 points to get num_quantiles thresholds.
        # np.linspace(0, 1, num_quantiles + 1)[1:-1] -> num_quantiles-1 thresholds
        # If NUM_QUANTILES_FOR_BINARIZATION = 20, then 19 thresholds.
        expected_feature_dim = data_to_binarize.shape[1] * (num_quantiles -1) if num_quantiles > 1 else data_to_binarize.shape[1]
        return np.empty((data_to_binarize.shape[0], expected_feature_dim), dtype=np.int8), []


    all_values_for_thresholds = fit_data_for_thresholds.flatten()
    # Calculate num_quantiles-1 thresholds to divide the data into num_quantiles "bins"
    quantile_points = np.linspace(0, 1, num_quantiles + 1)[1:-1] # E.g., for 20 quantiles, 19 thresholds
    
    if len(quantile_points) == 0 and num_quantiles == 1: # Special case: single threshold binarization (median)
        thresholds = np.array([np.median(all_values_for_thresholds)])
        print(f"   INFO: Using 1 quantile, the threshold is the median.")
    elif len(quantile_points) == 0 and num_quantiles < 1:
        print(f"❌ ERROR: num_quantiles ({num_quantiles}) must be >= 1.")
        expected_feature_dim = data_to_binarize.shape[1] # No binarization
        return data_to_binarize.astype(np.int8), [] # Do not binarize
    else:
        thresholds = np.quantile(all_values_for_thresholds, quantile_points)
        thresholds = np.unique(thresholds) # Remove duplicate thresholds that can occur with discrete data

    num_thresholds = len(thresholds)
    print(f"   📊 Calculated thresholds ({num_thresholds} unique) on fit_data: shape {fit_data_for_thresholds.shape}")
    if num_thresholds > 0:
        print(f"   📊 Thresholds range: [{thresholds[0]:.4f}, {thresholds[-1]:.4f}]")
    else:
        print("   ⚠️ No thresholds were calculated. Binarization might not produce the expected output.")
        # If there are no thresholds, each sample will produce a zero vector of the binarized feature length.
        # This can happen if num_quantiles = 1 and quantile_points is empty, or if the data is constant.
        # The binarized feature will be data_to_binarize.shape[1] * num_thresholds.
        # If num_quantiles is 1, thresholds has 1 element (median).
        # So num_thresholds should be at least 1 if num_quantiles >= 1.
        # The only case where num_thresholds can be 0 is if quantile_points is empty AND num_quantiles != 1.
        # This happens if num_quantiles = 0. Handled above.
        # So, if we get here with num_thresholds = 0, it's a bug.
        # For safety, if num_thresholds is 0, do not binarize.
        print(f"   ❌ INTERNAL ERROR: num_thresholds is 0 with num_quantiles={num_quantiles}. This should not happen.")
        return data_to_binarize.astype(np.int8), thresholds # Do not binarize

    n_segments, segment_length = data_to_binarize.shape
    if n_segments == 0: # If data_to_binarize is empty
        print("   ⚠️ data_to_binarize is empty. Returning empty binary array.")
        return np.empty((0, segment_length * num_thresholds), dtype=np.int8), thresholds

    # Each sample in the segment is compared with all thresholds.
    # Result: (n_segments, segment_length, num_thresholds)
    # Then reshape to (n_segments, segment_length * num_thresholds)
    # Example: sample x, thresholds [t1, t2]. Binary output: [x<=t1, x<=t2]
    
    # Initialize the binary array. The feature dimension is segment_length * number of thresholds
    binary_data = np.zeros((n_segments, segment_length * num_thresholds), dtype=np.int8)
    
    if num_thresholds == 0: # No thresholds, no binary features (or an error)
        print("   ⚠️ No valid thresholds, binarization will produce empty features if not handled.")
        # It might be better to return data_to_binarize unmodified or an error.
        # For now, it returns zeros with the correct shape if num_thresholds is 0.
        # This means the size of the second column of binary_data will be 0.
        # This will cause problems later on.
        # If num_thresholds is 0, it means num_quantiles was probably <=1 and not handled well.
        # Or that the data was such that np.unique(thresholds) returned an empty array (unlikely).
        # If num_quantiles = 1, thresholds has 1 element (median).
        # So num_thresholds should be at least 1 if num_quantiles >= 1.
        # The only case where num_thresholds can be 0 is if quantile_points is empty AND num_quantiles != 1.
        # This happens if num_quantiles = 0. Handled above.
        # So, if we get here with num_thresholds = 0, it's a bug.
        # For safety, if num_thresholds is 0, do not binarize.
        print(f"   ❌ INTERNAL ERROR: num_thresholds is 0 with num_quantiles={num_quantiles}. This should not happen.")
        return data_to_binarize.astype(np.int8), thresholds # Do not binarize

    for i, segment in enumerate(data_to_binarize):
        encoded_segment = []
        for sample_in_segment in segment:
            # Compare the sample with each threshold
            # (sample <= t1, sample <= t2, ...)
            encoded_sample = (sample_in_segment <= thresholds).astype(np.int8)
            encoded_segment.extend(encoded_sample)
        binary_data[i, :] = encoded_segment
    
    print(f"✅ Binarization complete: output shape {binary_data.shape}")
    if binary_data.size > 0:
        print(f"   📊 Unique values in result: {np.unique(binary_data)}")
    else:
        print(f"   ⚠️ Binarized data is empty.")
    return binary_data, thresholds


def main(num_quantiles):
    """Main function"""
    script_start_time = datetime.now()
    print("🚀 S5 RTM Preprocessing - Loading Segments and Correct Binarization")
    print("=" * 70)
    
    # Create output directory
    ensure_directory(OUTPUT_PATH)
    
    # Load segments directly
    # The function now returns (noisy_segments, clean_segments, segment_identifiers)
    noisy_segments_all, clean_segments_all, _ = load_segmented_data_from_files(DATA_PATH)
    
    if noisy_segments_all.size == 0 or clean_segments_all.size == 0:
        print("❌ No data loaded. Aborting preprocessing.")
        return

    print(f"📊 Total segments loaded: {noisy_segments_all.shape[0]}")
    print(f"   Shape of noisy segments: {noisy_segments_all.shape}")
    print(f"   Shape of clean segments: {clean_segments_all.shape}")

    # No need for create_windows anymore
    # print("🔄 Creating windows...") # Removed
    # clean_windows = create_windows(clean_signals, WINDOW_LENGTH, OVERLAP) # Removed
    # noisy_windows = create_windows(noisy_signals, WINDOW_LENGTH, OVERLAP) # Removed
    # Now we directly use the loaded segments as "windows"
    
    # Split train/val/test (using entire segments as samples)
    # We split the indices to maintain the association between noisy and clean
    num_total_segments = noisy_segments_all.shape[0]
    indices = np.arange(num_total_segments)

    # Split: 60% train, 20% validation, 20% test
    train_indices, temp_indices = train_test_split(indices, test_size=0.4, random_state=RANDOM_STATE)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=RANDOM_STATE) # 0.5 of 0.4 is 0.2

    X_train_noisy_segments = noisy_segments_all[train_indices]
    y_train_clean_segments = clean_segments_all[train_indices]
    
    X_val_noisy_segments = noisy_segments_all[val_indices]
    y_val_clean_segments = clean_segments_all[val_indices]
    
    X_test_noisy_segments = noisy_segments_all[test_indices]
    y_test_clean_segments = clean_segments_all[test_indices]

    print("🔄 Dataset split:")
    print(f"   Training: {X_train_noisy_segments.shape[0]} segments")
    print(f"   Validation: {X_val_noisy_segments.shape[0]} segments")
    print(f"   Test: {X_test_noisy_segments.shape[0]} segments")

    # Normalize the noisy segments before binarization
    # Calculate mean and std ONLY on the noisy training set
    train_noisy_mean = np.mean(X_train_noisy_segments)
    train_noisy_std = np.std(X_train_noisy_segments)

    print(f"\n🔄 Normalizing noisy segments (using mean/std from the training set):")
    print(f"   Noisy Training Set Mean: {train_noisy_mean:.4f}")
    print(f"   Noisy Training Set Std Dev: {train_noisy_std:.4f}")

    if train_noisy_std == 0:
        print("   ⚠️ Warning: Standard deviation of the noisy training set is 0. Normalization might not work as expected.")
        # In this case, normalization (X - mean) / std would result in an error or NaN.
        # We could decide not to normalize or use a small epsilon for std.
        # For now, we proceed, but binarize_with_quantiles might have issues if the data is constant.
        X_train_noisy_normalized = X_train_noisy_segments - train_noisy_mean
        X_val_noisy_normalized = X_val_noisy_segments - train_noisy_mean
        X_test_noisy_normalized = X_test_noisy_segments - train_noisy_mean
    else:
        X_train_noisy_normalized = (X_train_noisy_segments - train_noisy_mean) / train_noisy_std
        X_val_noisy_normalized = (X_val_noisy_segments - train_noisy_mean) / train_noisy_std
        X_test_noisy_normalized = (X_test_noisy_segments - train_noisy_mean) / train_noisy_std
    
    print(f"   Shape of normalized X_train: {X_train_noisy_normalized.shape}")
    if X_train_noisy_normalized.size > 0:
        print(f"   Range of normalized X_train: [{X_train_noisy_normalized.min():.4f}, {X_train_noisy_normalized.max():.4f}]")


    # Calculate noise targets (1D, center sample) for each set
    y_train_noise_target = calculate_noise_targets(X_train_noisy_segments, y_train_clean_segments)
    y_val_noise_target = calculate_noise_targets(X_val_noisy_segments, y_val_clean_segments)
    y_test_noise_target = calculate_noise_targets(X_test_noisy_segments, y_test_clean_segments)

    # Binarize the normalized noisy inputs (X)
    # The quantile thresholds must be calculated ONLY on X_train_noisy_normalized
    # and then applied to X_val_noisy_normalized and X_test_noisy_normalized.
    
    # NUM_QUANTILES_FOR_BINARIZATION should be defined, e.g., 20 or 100
    # I define it here for clarity, but it could be a global parameter or from a config file.
    print(f"INFO: Number of quantiles for binarization set to: {num_quantiles}")

    print("\n--- Binarizing Training Set ---")
    X_train_binarized, train_thresholds = binarize_with_quantiles(
        X_train_noisy_normalized, # Use normalized data
        num_quantiles, 
        fit_thresholds_on=X_train_noisy_normalized # Calculate and apply on normalized data
    )
    print("\n--- Binarizing Validation Set ---")
    X_val_binarized, _ = binarize_with_quantiles(
        X_val_noisy_normalized, # Use normalized data
        num_quantiles, 
        fit_thresholds_on=X_train_noisy_normalized # Apply thresholds from training (calculated on normalized data)
    )
    print("\n--- Binarizing Test Set ---")
    X_test_binarized, _ = binarize_with_quantiles(
        X_test_noisy_normalized, # Use normalized data
        num_quantiles, 
        fit_thresholds_on=X_train_noisy_normalized # Apply thresholds from training (calculated on normalized data)
    )
    
    # Verification of final dimensions
    print("\n📊 Final dimensions of processed data:")
    print(f"   X_train_binarized: {X_train_binarized.shape}, y_train_noise_target: {y_train_noise_target.shape}")
    print(f"   X_val_binarized: {X_val_binarized.shape}, y_val_noise_target: {y_val_noise_target.shape}")
    print(f"   X_test_binarized: {X_test_binarized.shape}, y_test_noise_target: {y_test_noise_target.shape}")

    # Save the processed files
    print("\n💾 Saving processed files...")
    q_suffix = f"_q{num_quantiles}"
    
    np.save(os.path.join(OUTPUT_PATH, f"X_train_rtm_BINARIZED{q_suffix}.npy"), X_train_binarized)
    np.save(os.path.join(OUTPUT_PATH, f"y_train_rtm_aggregated_noise.npy"), y_train_noise_target)

    # Calculate and save the maximum absolute value of the noise target (for denormalization)
    y_train_max_abs_noise = np.max(np.abs(y_train_noise_target))
    np.save(os.path.join(OUTPUT_PATH, f"y_train_max_abs_noise.npy"), y_train_max_abs_noise)
    print(f"   y_train_max_abs_noise saved: {y_train_max_abs_noise:.6f}")

    np.save(os.path.join(OUTPUT_PATH, f"X_validation_rtm_BINARIZED{q_suffix}.npy"), X_val_binarized)
    np.save(os.path.join(OUTPUT_PATH, f"y_validation_rtm_aggregated_noise.npy"), y_val_noise_target)
    np.save(os.path.join(OUTPUT_PATH, f"X_test_rtm_BINARIZED{q_suffix}.npy"), X_test_binarized)
    np.save(os.path.join(OUTPUT_PATH, f"y_test_rtm_aggregated_noise.npy"), y_test_noise_target)
    
    # Saving aggregated data
    output_dir = OUTPUT_PATH
    print(f"Saving aggregated data for q={num_quantiles} in {output_dir}...")
    np.save(os.path.join(output_dir, f"X_train_q{num_quantiles}.npy"), X_train_binarized)
    np.save(os.path.join(output_dir, f"y_train_aggregated_noise.npy"), y_train_noise_target)
    np.save(os.path.join(output_dir, f"X_test_q{num_quantiles}.npy"), X_test_binarized)
    np.save(os.path.join(output_dir, f"y_test_aggregated_noise.npy"), y_test_noise_target)

    # Saving specific noise targets for future flexibility
    print("Saving specific noise targets (bw, ma, pli)...")
    np.save(os.path.join(output_dir, "y_train_bw_noise.npy"), y_train_noise_target)
    np.save(os.path.join(output_dir, "y_train_ma_noise.npy"), y_train_noise_target)
    np.save(os.path.join(output_dir, "y_train_pli_noise.npy"), y_train_noise_target)
    np.save(os.path.join(output_dir, "y_test_bw_noise.npy"), y_test_noise_target)
    np.save(os.path.join(output_dir, "y_test_ma_noise.npy"), y_test_noise_target)
    np.save(os.path.join(output_dir, "y_test_pli_noise.npy"), y_test_noise_target)

    # Saving the maximum absolute value of the aggregated noise for denormalization
    max_abs_noise = np.max(np.abs(y_train_noise_target))
    np.save(os.path.join(output_dir, "y_train_max_abs_noise.npy"), max_abs_noise)
    print(f"   y_train_max_abs_noise saved: {y_train_max_abs_noise:.6f}")

    # Save important metadata
    metadata = {
        "script_version": "s5_rtm_preprocessing_v3_segmented_load_normalized",
        "preprocessing_timestamp": script_start_time.isoformat(),
        "data_source_path": DATA_PATH,
        "output_path": OUTPUT_PATH,
        "window_length_segments": WINDOW_LENGTH,
        "num_quantiles_for_binarization": num_quantiles,
        "normalization_train_set_mean": train_noisy_mean,
        "normalization_train_set_std": train_noisy_std,
        "binarization_thresholds_from_train_set": train_thresholds.tolist() if isinstance(train_thresholds, np.ndarray) else train_thresholds,
        "random_state_split": RANDOM_STATE,
        "split_ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "num_train_samples": X_train_binarized.shape[0],
        "num_validation_samples": X_val_binarized.shape[0],
        "num_test_samples": X_test_binarized.shape[0],
        "X_train_shape": X_train_binarized.shape,
        "y_train_shape": y_train_noise_target.shape,
    }
    metadata_path = os.path.join(OUTPUT_PATH, f"metadata_BINARIZED{q_suffix}.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, indent=4)
        
    print(f"   Metadata saved to: {metadata_path}")
    print("✅ Preprocessing completed successfully!")
    script_duration = datetime.now() - script_start_time
    print(f"⏱️ Total script execution time: {script_duration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ECG data preprocessing for RTM with a specific number of quantiles.")
    parser.add_argument("--num_quantiles", type=int, default=20, help="Number of quantiles for binarization.")
    args = parser.parse_args()
    
    main(args.num_quantiles)
