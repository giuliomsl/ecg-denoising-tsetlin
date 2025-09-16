#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_and_reconstruct.py — Inference with state-based RTM restore (portable).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, h5py

def notch_filter(x, fs=360.0, f0=50.0, Q=30.0):
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(w0=f0/(fs/2.0), Q=Q)
    return filtfilt(b, a, x).astype(np.float32)

def highpass_butter(x, fs=360.0, cutoff=0.5, order=2):
    from scipy.signal import butter, filtfilt
    wn = cutoff/(fs/2.0)
    b, a = butter(order, wn, btype='highpass')
    return filtfilt(b, a, x).astype(np.float32)

def zscore_clip(x, clip_sigma=3.0):
    mu = np.mean(x); sd = np.std(x) + 1e-8
    z = (x - mu) / sd
    return np.clip(z, -clip_sigma, clip_sigma)

def percentile_scale(x, lo=1.0, hi=99.0):
    p_lo, p_hi = np.percentile(x, [lo, hi])
    if p_hi <= p_lo:
        p_lo, p_hi = float(np.min(x)), float(np.max(x))
        if p_hi <= p_lo: p_hi = p_lo + 1.0
    y = (x - p_lo) / (p_hi - p_lo); y = np.clip(y, 0.0, 1.0)
    return y

def quantize_levels(x01, n_levels):
    q = np.floor(x01 * n_levels).astype(np.int32)
    q[q == n_levels] = n_levels - 1
    return q

def int_to_bitplanes(q, n_bits):
    out = np.empty(q.shape + (n_bits,), dtype=np.uint8)
    for b in range(n_bits):
        out[..., b] = ((q >> b) & 1).astype(np.uint8)
    return out

def thermometer_encode(q, n_levels):
    L = q.shape[0]
    t = np.zeros((L, n_levels), dtype=np.uint8)
    idx = np.arange(n_levels, dtype=np.int32)[None, :]
    t[:, :] = (q[:, None] >= idx).astype(np.uint8)
    return t

def onehot_encode(q, n_levels):
    L = q.shape[0]
    oh = np.zeros((L, n_levels), dtype=np.uint8)
    oh[np.arange(L), q] = 1
    return oh

def rolling_windows(arr, window, stride):
    from numpy.lib.stride_tricks import as_strided
    L = arr.shape[0]
    if L < window: return np.empty((0, window) + arr.shape[1:], dtype=arr.dtype)
    n = 1 + (L - window)//stride
    new_shape = (n, window) + arr.shape[1:]
    new_strides = (arr.strides[0]*stride,) + arr.strides
    return as_strided(arr, shape=new_shape, strides=new_strides)

def build_features(sig, window, stride, encoder="bitplanes", bits=8, levels=64, include_deriv=True):
    x01 = percentile_scale(zscore_clip(sig.astype(np.float32)))
    if encoder == "bitplanes":
        n_bits = int(bits)
        q = np.rint(x01 * ((1 << n_bits) - 1)).astype(np.int32)
        q = np.clip(q, 0, (1 << n_bits) - 1)
        bp = int_to_bitplanes(q, n_bits)
        feats = bp
        if include_deriv:
            dq = np.abs(np.diff(q, prepend=q[:1]))
            bp_d = int_to_bitplanes(dq, n_bits)
            feats = np.concatenate([bp, bp_d], axis=1)
        wins = rolling_windows(feats, window, stride)
        return wins.reshape(wins.shape[0], wins.shape[1]*wins.shape[2]).astype(np.uint8)
    elif encoder == "thermometer":
        L = int(levels)
        q = quantize_levels(x01, L)
        th = thermometer_encode(q, L)
        feats = th
        if include_deriv:
            dq = np.abs(np.diff(q, prepend=q[:1])); dq = np.minimum(dq, L-1)
            th_d = thermometer_encode(dq, L)
            feats = np.concatenate([th, th_d], axis=1)
        wins = rolling_windows(feats, window, stride)
        return wins.reshape(wins.shape[0], wins.shape[1]*wins.shape[2]).astype(np.uint8)
    elif encoder == "onehot2d":
        L = int(levels)
        q = quantize_levels(x01, L)
        oh = onehot_encode(q, L)
        wins = rolling_windows(oh, window, stride)
        return wins.reshape(wins.shape[0], window*L).astype(np.uint8)
    else:
        raise ValueError("Encoder sconosciuto")

def overlap_add(center_preds, L, window, stride):
    out = np.zeros(L, dtype=np.float32); wsum = np.zeros(L, dtype=np.float32)
    half = window//2
    for i, y in enumerate(center_preds):
        c = i*stride + half
        if 0 <= c < L: out[c] += float(y); wsum[c] += 1.0
    nz = wsum > 0; out[nz] /= wsum[nz]
    if np.any(~nz): out = np.interp(np.arange(L), np.where(nz)[0], out[nz])
    return out

def apply_prefilters(x, fs, cfg):
    y = x.astype(np.float32)
    if cfg.get("notch", False):
        y = notch_filter(y, fs=fs, f0=float(cfg.get("f0", 50.0)), Q=float(cfg.get("Q", 30.0)))
    if cfg.get("hp", False):
        y = highpass_butter(y, fs=fs, cutoff=float(cfg.get("cutoff", 0.5)), order=int(cfg.get("order", 2)))
    return y

def load_model(stem: Path, clauses=None, T=None, s=None):
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
    state_npz = stem.with_suffix(".state.npz")
    meta_json = stem.with_suffix(".meta.json")
    if state_npz.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text())
        if meta.get("format") != "state":
            raise RuntimeError("Meta non indica 'state'.")
        c = int(meta.get("clauses") if clauses is None else clauses)
        TT = int(meta.get("T") if T is None else T)
        ss = float(meta.get("s") if s is None else s)
        model = RegressionTsetlinMachine(c, TT, ss)
        if not hasattr(model, "set_state"):
            raise RuntimeError("pyTsetlinMachine non ha set_state(). Aggiorna.")
        data = dict(np.load(str(state_npz)))
        model.set_state(data)
        return model
    rtm_file = stem.with_suffix(".rtm")
    if rtm_file.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text())
        c = int(meta.get("clauses") if clauses is None else clauses)
        TT = int(meta.get("T") if T is None else T)
        ss = float(meta.get("s") if s is None else s)
        model = RegressionTsetlinMachine(c, TT, ss)
        if not hasattr(model, "load"):
            raise RuntimeError("pyTsetlinMachine non ha load(). Usa .state.npz")
        model.load(str(rtm_file))
        return model
    jb_file = stem.with_suffix(".joblib")
    if jb_file.exists():
        import joblib
        obj = joblib.load(str(jb_file))
        model = obj.get("model", None) if isinstance(obj, dict) else obj
        if model is None:
            raise RuntimeError("joblib non valido (manca 'model').")
        return model
    raise FileNotFoundError(f"Modello non trovato per stem: {stem}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["auto","test","val","train"], default="auto")
    ap.add_argument("--encoder", choices=["bitplanes","thermometer","onehot2d"], default="bitplanes")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--no-deriv", action="store_true")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--fs", type=float, default=360.0)
    ap.add_argument("--notch", action="store_true")
    ap.add_argument("--notch-freq", type=float, default=50.0)
    ap.add_argument("--notch-Q", type=float, default=30.0)
    ap.add_argument("--hp", action="store_true")
    ap.add_argument("--hp-cutoff", type=float, default=0.5)
    ap.add_argument("--hp-order", type=int, default=2)
    ap.add_argument("--clauses", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--s", type=float, default=None)
    args = ap.parse_args()

    stem = Path(args.model)
    if stem.suffix in (".state", ".npz", ".rtm", ".joblib"):
        stem = stem.with_suffix("")

    model = load_model(stem, clauses=args.clauses, T=args.T, s=args.s)
    mu, sigma = 0.0, 1.0
    yscaler = stem.with_suffix(".yscaler.json")
    if yscaler.exists():
        sc = json.loads(yscaler.read_text())
        mu = float(sc.get("mu", sc.get("median", 0.0)))
        sigma = float(sc.get("sigma", 1.0)) or 1.0

    enc = {"encoder": args.encoder, "bits": args.bits, "levels": args.levels,
           "window": args.window, "stride": args.stride, "include_deriv": (not args.no-deriv if False else not args.no_deriv),
           "prefilter": {"notch": args.notch, "f0": args.notch_freq, "Q": args.notch_Q,
                         "hp": args.hp, "cutoff": args.hp_cutoff, "order": args.hp_order},
           "fs": args.fs}

    # Try load pre.json to override
    preconf = stem.with_suffix(".pre.json")
    if preconf.exists():
        pc = json.loads(preconf.read_text())
        for k in ("encoder","bits","levels","window","stride"):
            if k in pc: enc[k] = pc[k]
        if "prefilter" in pc and isinstance(pc["prefilter"], dict):
            enc["prefilter"].update(pc["prefilter"])
        if "fs" in pc: enc["fs"] = float(pc["fs"])

    include_deriv = bool(enc.get("include_deriv", True))
    encoder = str(enc["encoder"]).lower()
    bits = int(enc.get("bits", 8))
    levels = int(enc.get("levels", 64))
    window = int(enc.get("window", 128))
    stride = int(enc.get("stride", 32))
    fs = float(enc.get("fs", args.fs))
    pre_cfg = enc.get("prefilter", {})

    with h5py.File(str(Path(args.input_h5)), "r") as f:
        split = args.split
        if split == "auto":
            split = "test" if "test_noisy" in f else ("val" if "val_noisy" in f else "train")
        noisy = f[f"{split}_noisy"][:]
        clean = f.get(f"{split}_clean")
        clean = None if clean is None else clean[:]

    den_list, mae_list = [], []
    snr_noisy_list, snr_den_list = [], []
    corr_noisy_list, corr_den_list = [], []
    for i in range(noisy.shape[0]):
        x = noisy[i].astype(np.float32)
        x_pref = apply_prefilters(x, fs=fs, cfg=pre_cfg) if pre_cfg else x
        X = build_features(x_pref, window=window, stride=stride, encoder=encoder, bits=bits, levels=levels, include_deriv=include_deriv)
        if X.shape[0] == 0:
            den = x
        else:
            pred_norm = model.predict(X).astype(np.float32)
            pred = pred_norm * sigma + mu
            noise_hat = overlap_add(pred, L=len(x_pref), window=window, stride=stride)
            den = x_pref - noise_hat
        den_list.append(den.astype(np.float32))
        if clean is not None:
            # Prefilter clean for coherent comparison
            clean_pref = apply_prefilters(clean[i].astype(np.float32), fs=fs, cfg=pre_cfg) if pre_cfg else clean[i].astype(np.float32)
            mae_list.append(float(np.mean(np.abs(den - clean_pref))))
            # SNR: 10*log10(||clean||^2 / ||clean - signal||^2)
            eps = 1e-12
            num = float(np.sum(clean_pref**2)) + eps
            den_noisy = float(np.sum((clean_pref - x_pref)**2)) + eps
            den_denoised = float(np.sum((clean_pref - den)**2)) + eps
            snr_noisy_list.append(10.0 * np.log10(num / den_noisy))
            snr_den_list.append(10.0 * np.log10(num / den_denoised))
            # Pearson correlation
            def _corr(a,b):
                a = a - np.mean(a); b = b - np.mean(b)
                na = np.linalg.norm(a) + eps; nb = np.linalg.norm(b) + eps
                return float(np.dot(a,b) / (na*nb))
            corr_noisy_list.append(_corr(x_pref, clean_pref))
            corr_den_list.append(_corr(den, clean_pref))

    den = np.stack(den_list, axis=0)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(outp), "w") as h:
        h.create_dataset("test_denoised", data=den, compression="gzip", compression_opts=4)
        if clean is not None:
            h.create_dataset("test_clean", data=clean, compression="gzip", compression_opts=4)
        h.attrs["encoder"] = encoder; h.attrs["bits"] = bits; h.attrs["levels"] = levels
        h.attrs["window"] = window; h.attrs["stride"] = stride; h.attrs["fs"] = fs
        h.attrs["prefilter"] = json.dumps(pre_cfg); h.attrs["mu"] = mu; h.attrs["sigma"] = sigma; h.attrs["split"] = split
        if mae_list:
            h.attrs["test_mae_mean"] = float(np.mean(mae_list)); h.attrs["test_mae_median"] = float(np.median(mae_list))
            if snr_noisy_list and snr_den_list:
                h.attrs["snr_noisy_mean"] = float(np.mean(snr_noisy_list))
                h.attrs["snr_denoised_mean"] = float(np.mean(snr_den_list))
                h.attrs["delta_snr_mean"] = float(np.mean(np.array(snr_den_list) - np.array(snr_noisy_list)))
            if corr_noisy_list and corr_den_list:
                h.attrs["corr_noisy_mean"] = float(np.mean(corr_noisy_list))
                h.attrs["corr_denoised_mean"] = float(np.mean(corr_den_list))
    print(f"[OK] wrote {outp}  den={den.shape}  split={split}")
    if mae_list:
        print(f"[TEST] MAE mean={np.mean(mae_list):.6f} median={np.median(mae_list):.6f}")
        if snr_noisy_list and snr_den_list:
            import numpy as _np
            dsnr = _np.mean(_np.array(snr_den_list) - _np.array(snr_noisy_list))
            print(f"[TEST] SNR noisy={_np.mean(snr_noisy_list):.3f} dB  denoised={_np.mean(snr_den_list):.3f} dB  ΔSNR={dsnr:.3f} dB")
        if corr_noisy_list and corr_den_list:
            print(f"[TEST] corr noisy={np.mean(corr_noisy_list):.4f}  denoised={np.mean(corr_den_list):.4f}")

if __name__ == "__main__":
    main()
