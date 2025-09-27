#!/usr/bin/env python3
"""
Simple inference script per debugging segfault
"""
import json, numpy as np, h5py
from pathlib import Path
from pyTsetlinMachine.tm import RegressionTsetlinMachine

def load_model_safe(stem_path):
    """Carica modello con gestione errori"""
    stem = Path(stem_path)
    meta_file = stem.with_suffix(".meta.json")
    state_file = stem.with_suffix(".state.npz")
    
    print(f"Loading meta from {meta_file}")
    meta = json.loads(meta_file.read_text())
    
    clauses = meta["clauses"]
    T = meta["T"] 
    s = meta["s"]
    print(f"Model params: clauses={clauses}, T={T}, s={s}")
    
    print("Creating model...")
    model = RegressionTsetlinMachine(clauses, T, s)
    
    print(f"Loading state from {state_file}")
    data = dict(np.load(str(state_file)))
    print(f"State keys: {list(data.keys())}")
    
    # Handle sequence format
    if meta.get("state_format") == "sequence":
        state_len = meta.get("state_len", len(data))
        state_list = [data[f"s{i}"] for i in range(state_len)]
        print(f"Converting to sequence format, len={len(state_list)}")
        print("Setting state...")
        model.set_state(state_list)
    else:
        print("Setting state as mapping...")
        model.set_state(data)
    
    print("Model loaded successfully!")
    return model, meta

def simple_inference(model_stem, data_h5, split="val", max_samples=1000):
    """Inferenza semplificata"""
    print(f"=== Simple Inference on {split} ===")
    
    # Load model
    try:
        model, meta = load_model_safe(model_stem)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data sample
    print(f"Loading data from {data_h5}")
    with h5py.File(data_h5, 'r') as f:
        X_key = f"{split}_X"
        y_key = f"{split}_y"
        
        if X_key not in f:
            print(f"Split {split} not found, available: {list(f.keys())}")
            return
            
        X = f[X_key][:max_samples].astype(np.int32)
        y = f[y_key][:max_samples].astype(np.float32)
        
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    
    # Load yscaler
    yscaler_file = Path(model_stem).with_suffix(".yscaler.json")
    yscaler = json.loads(yscaler_file.read_text())
    mu, sigma = yscaler["mu"], yscaler["sigma"]
    print(f"Y scaler: mu={mu:.6f}, sigma={sigma:.6f}")
    
    # Predict
    print("Predicting...")
    try:
        y_pred_norm = model.predict(X)
        print(f"Prediction shape: {y_pred_norm.shape}")
        
        # Denormalize
        y_pred_raw = y_pred_norm * sigma + mu
        
        # Compute metrics
        mae_raw = np.mean(np.abs(y - y_pred_raw))
        rmse_raw = np.sqrt(np.mean((y - y_pred_raw)**2))
        
        print(f"Results on {len(y)} samples:")
        print(f"  MAE (raw): {mae_raw:.6f}")
        print(f"  RMSE (raw): {rmse_raw:.6f}")
        print(f"  Best training MAE: {meta['best_val_mae_raw']:.6f}")
        
        return {
            "mae_raw": mae_raw,
            "rmse_raw": rmse_raw,
            "y_true": y,
            "y_pred": y_pred_raw
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # Test small model first
    print("Testing small model...")
    result = simple_inference(
        "models/rtm_specialists_final/rtm_denoiser_smoke2",
        "data/denoiser_rtm_preprocessed.h5",
        split="val",
        max_samples=500
    )
    
    if result:
        print("✅ Small model inference successful!")
        print("\nTesting optimized model...")
        result2 = simple_inference(
            "models/rtm_specialists_final/rtm_denoiser_opt",
            "data/denoiser_rtm_preprocessed.h5", 
            split="val",
            max_samples=1000
        )
        if result2:
            print("✅ Optimized model inference successful!")
