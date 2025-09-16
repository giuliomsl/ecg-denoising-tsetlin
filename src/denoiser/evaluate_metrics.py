#!/usr/bin/env python3
from __future__ import annotations
import argparse, numpy as np, h5py

def snr_db(sig, noise):
    p_sig = np.mean(sig**2) + 1e-12
    p_noise = np.mean(noise**2) + 1e-12
    return 10.0 * np.log10(p_sig / p_noise)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-h5", required=True)
    args = ap.parse_args()
    with h5py.File(str(args.input_h5), "r") as f:
        noisy = f["test_noisy"][:]
        den = f["test_denoised"][:]
        clean = f["test_clean"][:] if "test_clean" in f else None
    if clean is None:
        print("No clean reference found.")
        return
    n = noisy.shape[0]
    snr_in, snr_out, mae, rmse = [], [], [], []
    for i in range(n):
        noise_in = noisy[i] - clean[i]
        noise_out = den[i] - clean[i]
        snr_in.append(snr_db(clean[i], noise_in))
        snr_out.append(snr_db(clean[i], noise_out))
        d = den[i] - clean[i]
        mae.append(float(np.mean(np.abs(d))))
        rmse.append(float(np.sqrt(np.mean(d**2))))
    snr_in, snr_out = np.array(snr_in), np.array(snr_out)
    mae, rmse = np.array(mae), np.array(rmse)
    print(f"SNR_in  mean={snr_in.mean():.3f} dB  median={np.median(snr_in):.3f} dB")
    print(f"SNR_out mean={snr_out.mean():.3f} dB  median={np.median(snr_out):.3f} dB")
    print(f"SNR_impr mean={(snr_out-snr_in).mean():.3f} dB  median={np.median(snr_out-snr_in):.3f} dB")
    print(f"MAE mean={mae.mean():.6f}  median={np.median(mae):.6f}")
    print(f"RMSE mean={rmse.mean():.6f}  median={np.median(rmse):.6f}")

if __name__ == "__main__":
    main()
