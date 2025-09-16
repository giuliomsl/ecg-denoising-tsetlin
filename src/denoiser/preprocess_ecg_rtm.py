#!/usr/bin/env python3
from __future__ import annotations
import argparse, numpy as np, h5py
from pathlib import Path
try:
    from encoders import build_features, build_targets
    from prefilters import notch_filter, highpass_butter, wavelet_denoise
except ImportError:
    from src.denoiser.encoders import build_features, build_targets
    from src.denoiser.prefilters import notch_filter, highpass_butter, wavelet_denoise

def maybe_prefilter(noisy, fs, args):
    y = noisy.astype(np.float32)
    if args.notch:
        y = notch_filter(y, fs=fs, f0=args.notch_freq, Q=args.notch_Q)
    if args.hp:
        y = highpass_butter(y, fs=fs, cutoff=args.hp_cutoff, order=args.hp_order)
    if args.wavelet is not None:
        y = wavelet_denoise(y, wavelet=args.wavelet, level=args.wavelet_level, mode=args.wavelet_mode)
    return y

def build_dataset(noisy_list, clean_list, window, stride, encoder, bits, levels, include_deriv, fs, args):
    X_all, y_all = [], []
    for noisy, clean in zip(noisy_list, clean_list):
        nz = maybe_prefilter(noisy, fs, args)
        cz = maybe_prefilter(clean, fs, args)
        X, _ = build_features(nz, window, stride, encoder=encoder, bits=bits, levels=levels, include_deriv=include_deriv)
        y = build_targets(nz, cz, window, stride)
        n = min(len(X), len(y))
        if n <= 0:
            continue
        X_all.append(X[:n]); y_all.append(y[:n])
    X_all = np.vstack(X_all).astype(np.uint8) if X_all else np.empty((0, window), dtype=np.uint8)
    y_all = np.concatenate(y_all).astype(np.float32) if y_all else np.empty((0,), dtype=np.float32)
    return X_all, y_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--encoder", choices=["bitplanes","thermometer","onehot2d"], default="bitplanes")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--no-deriv", action="store_true")
    ap.add_argument("--fs", type=float, default=360.0)
    ap.add_argument("--notch", action="store_true")
    ap.add_argument("--notch-freq", type=float, default=50.0)
    ap.add_argument("--notch-Q", type=float, default=30.0)
    ap.add_argument("--hp", action="store_true")
    ap.add_argument("--hp-cutoff", type=float, default=0.5)
    ap.add_argument("--hp-order", type=int, default=2)
    ap.add_argument("--wavelet", type=str, default=None)
    ap.add_argument("--wavelet-level", type=int, default=None)
    ap.add_argument("--wavelet-mode", type=str, default="soft", choices=["soft","hard"])
    args = ap.parse_args()

    include_deriv = not args.no_deriv
    with h5py.File(str(Path(args.input_h5)), "r") as f:
        # usa fs dal file se presente
        if "fs" in f.attrs:
            args.fs = float(f.attrs["fs"])
        train_noisy = f["train_noisy"][:]; train_clean = f["train_clean"][:]
        # gestisci 'val' o 'validation'
        if "val_noisy" in f and "val_clean" in f:
            val_noisy = f["val_noisy"][:]; val_clean = f["val_clean"][:]
        elif "validation_noisy" in f and "validation_clean" in f:
            val_noisy = f["validation_noisy"][:]; val_clean = f["validation_clean"][:]
        else:
            val_noisy = None; val_clean = None
        test_noisy = f["test_noisy"][:] if "test_noisy" in f else None
        test_clean = f["test_clean"][:] if "test_clean" in f else None

    Xtr, ytr = build_dataset(train_noisy, train_clean, args.window, args.stride, args.encoder, args.bits, args.levels, include_deriv, args.fs, args)
    if val_noisy is not None and val_clean is not None:
        Xva, yva = build_dataset(val_noisy, val_clean, args.window, args.stride, args.encoder, args.bits, args.levels, include_deriv, args.fs, args)
    else:
        n = Xtr.shape[0]; n_val = max(1, n//10)
        Xva, yva = Xtr[-n_val:], ytr[-n_val:]; Xtr, ytr = Xtr[:-n_val], ytr[:-n_val]
    if test_noisy is not None and test_clean is not None:
        Xte, yte = build_dataset(test_noisy, test_clean, args.window, args.stride, args.encoder, args.bits, args.levels, include_deriv, args.fs, args)
    else:
        Xte, yte = (np.empty((0, Xtr.shape[1]), dtype=np.uint8) if Xtr.size else np.empty((0, args.window), dtype=np.uint8),
                    np.empty((0,), dtype=np.float32))

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(outp), "w") as h:
        h.create_dataset("train_X", data=Xtr, compression="gzip", compression_opts=4)
        h.create_dataset("train_y", data=ytr, compression="gzip", compression_opts=4)
        h.create_dataset("val_X", data=Xva, compression="gzip", compression_opts=4)
        h.create_dataset("val_y", data=yva, compression="gzip", compression_opts=4)
        h.create_dataset("test_X", data=Xte, compression="gzip", compression_opts=4)
        h.create_dataset("test_y", data=yte, compression="gzip", compression_opts=4)
        h.attrs.update({"window": args.window, "stride": args.stride, "encoder": args.encoder,
                        "bits": args.bits, "levels": args.levels, "include_deriv": int(include_deriv),
                        "fs": float(args.fs), "prefilter_notch": int(args.notch),
                        "notch_freq": float(args.notch_freq), "notch_Q": float(args.notch_Q),
                        "prefilter_hp": int(args.hp), "hp_cutoff": float(args.hp_cutoff),
                        "hp_order": int(args.hp_order), "wavelet": args.wavelet or "",
                        "wavelet_level": -1 if args.wavelet_level is None else int(args.wavelet_level),
                        "wavelet_mode": args.wavelet_mode})
    print(f"[OK] {args.encoder} -> {outp}  train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

if __name__ == "__main__":
    main()
