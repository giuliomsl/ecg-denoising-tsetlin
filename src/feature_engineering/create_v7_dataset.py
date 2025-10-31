#!/usr/bin/env python3
"""
Create V7 Dataset: Selected Bitplanes + HF Features
====================================================

Extract only the important bitplanes (threshold 0.0005 = 45 bitplanes) + 12 HF.
Total: 57 features (95% reduction from 1092).
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create V7 dataset with selected features")
    parser.add_argument("--strategy", type=str, default="th0.0005", 
                        help="Selection strategy: top10, top20, top50, top100, th0.0001, th0.0005, th0.0010")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("V7: SELECTED FEATURES DATASET")
    print("="*80)
    print(f"\nStrategy: {args.strategy}")
    
    # Load selected bitplane indices
    if args.strategy.startswith("top"):
        k = int(args.strategy[3:])
        selected_file = f"data/v7_selected_bitplanes_top{k}.npy"
        strategy_name = f"Top {k}"
        threshold_value = None
    elif args.strategy.startswith("th"):
        threshold = args.strategy[2:]
        selected_file = f"data/v7_selected_bitplanes_th{threshold}.npy"
        strategy_name = f"Threshold {threshold}"
        threshold_value = float(threshold)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    print(f"Loading: {selected_file}")
    selected_bitplanes = np.load(selected_file)
    print(f"Selected bitplanes: {len(selected_bitplanes)}")
    
    # HF features are always at the end (indices 1080-1091)
    hf_indices = np.arange(1080, 1092)
    
    # Combined indices: selected bitplanes + all HF
    all_indices = np.concatenate([selected_bitplanes, hf_indices])
    all_indices = np.sort(all_indices)
    
    n_features = len(all_indices)
    print(f"Total features: {n_features} ({len(selected_bitplanes)} bitplanes + 12 HF)")
    print(f"Reduction: 1092 → {n_features} ({(1092-n_features)/1092*100:.1f}%)")
    
    # Process dataset
    input_path = "data/explain_features_dataset_v2.h5"
    output_path = f"data/explain_features_dataset_v7_{args.strategy}.h5"
    
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    
    with h5py.File(input_path, "r") as f_in, \
         h5py.File(output_path, "w") as f_out:
        
        # Process each split
        for split in ["train", "validation", "test"]:
            X_key = f"{split}_X"
            
            print(f"\n{split.upper()}:")
            
            # Load full features
            X_full = f_in[X_key][:]
            y_bw = f_in[f"{split}_y_bw"][:]
            y_emg = f_in[f"{split}_y_emg"][:]
            
            # Select features
            X_selected = X_full[:, all_indices]
            
            print(f"  Samples: {X_selected.shape[0]}")
            print(f"  Features: {X_full.shape[1]} → {X_selected.shape[1]}")
            
            # Save
            f_out.create_dataset(f"{split}_X", data=X_selected, compression="gzip", compression_opts=4)
            f_out.create_dataset(f"{split}_y_bw", data=y_bw, compression="gzip", compression_opts=4)
            f_out.create_dataset(f"{split}_y_emg", data=y_emg, compression="gzip", compression_opts=4)
            
            # PLI if exists
            if f"{split}_y_pli" in f_in:
                y_pli = f_in[f"{split}_y_pli"][:]
                f_out.create_dataset(f"{split}_y_pli", data=y_pli, compression="gzip", compression_opts=4)
        
        # Copy and update metadata
        for attr_name, attr_value in f_in.attrs.items():
            f_out.attrs[attr_name] = attr_value
        
        f_out.attrs["version"] = f"V7_{args.strategy}"
        f_out.attrs["n_features"] = n_features
        f_out.attrs["n_bitplanes_selected"] = len(selected_bitplanes)
        f_out.attrs["n_hf_features"] = 12
        f_out.attrs["selection_strategy"] = strategy_name
        if threshold_value is not None:
            f_out.attrs["selection_threshold"] = threshold_value
        f_out.attrs["selected_bitplane_indices"] = selected_bitplanes
        f_out.attrs["selected_all_indices"] = all_indices  # Complete feature indices
    
    print("\n" + "="*80)
    print(f"✅ V7 DATASET CREATED: {args.strategy}")
    print("="*80)
    print(f"\nSaved: {output_path}")
    print(f"Features: {n_features} ({len(selected_bitplanes)} bitplanes + 12 HF)")
    print(f"Reduction: {(1092-n_features)/1092*100:.1f}%")
    print("\nNext: python quick_test_v7.py --strategy " + args.strategy)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
