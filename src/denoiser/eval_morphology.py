#!/usr/bin/env python3
from __future__ import annotations
import argparse, numpy as np, h5py

def pearson_corr(a, b):
    a = (a - a.mean())/(a.std()+1e-9)
    b = (b - b.mean())/(b.std()+1e-9)
    return float(np.mean(a*b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-h5", required=True)
    ap.add_argument("--fs", type=float, default=360.0)
    args = ap.parse_args()
    with h5py.File(str(args.input_h5), "r") as f:
        clean = f["test_clean"][:]
        den = f["test_denoised"][:]
    N = clean.shape[0]
    cors = [pearson_corr(clean[i], den[i]) for i in range(N)]
    print(f"[Corr] mean={np.mean(cors):.4f}  median={np.median(cors):.4f}")
    try:
        import neurokit2 as nk
        diffs = []
        for i in range(N):
            try:
                _, info_c = nk.ecg_peaks(clean[i], sampling_rate=args.fs)
                _, info_d = nk.ecg_peaks(den[i], sampling_rate=args.fs)
                r_c = np.asarray(info_c["ECG_R_Peaks"], dtype=int)
                r_d = np.asarray(info_d["ECG_R_Peaks"], dtype=int)
                if len(r_c) == 0 or len(r_d) == 0: continue
                used = set()
                for rc in r_c:
                    idx = int(np.argmin(np.abs(r_d - rc)))
                    if idx in used: continue
                    used.add(idx)
                    diffs.append((r_d[idx]-rc)*1000.0/args.fs)
            except Exception:
                continue
        if diffs:
            diffs = np.array(diffs)
            print(f"[R-peak Δt ms] mean={diffs.mean():.2f}  median={np.median(diffs):.2f}")
        else:
            print("[R-peak] no peaks detected/alignment failed.")
    except Exception:
        print("[neurokit2] not installed. Skipping R-peak analysis.")

if __name__ == "__main__":
    main()
