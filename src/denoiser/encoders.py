#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

def zscore_clip(x: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    mu = np.mean(x); sd = np.std(x) + 1e-8
    z = (x - mu) / sd
    return np.clip(z, -clip_sigma, clip_sigma)

def percentile_scale(x: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    p_lo, p_hi = np.percentile(x, [lo, hi])
    if p_hi <= p_lo:
        p_lo, p_hi = float(np.min(x)), float(np.max(x))
        if p_hi <= p_lo:
            p_hi = p_lo + 1.0
    y = (x - p_lo) / (p_hi - p_lo)
    y = np.clip(y, 0.0, 1.0)
    return y, float(p_lo), float(p_hi)

def quantize_levels(x01: np.ndarray, n_levels: int) -> np.ndarray:
    n_levels = int(n_levels)
    q = np.floor(x01 * n_levels).astype(np.int32)
    q[q == n_levels] = n_levels - 1
    return q

def int_to_bitplanes(q: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.empty(q.shape + (n_bits,), dtype=np.uint8)
    for b in range(n_bits):
        out[..., b] = ((q >> b) & 1).astype(np.uint8)
    return out

def thermometer_encode(q: np.ndarray, n_levels: int) -> np.ndarray:
    L = q.shape[0]
    t = np.zeros((L, n_levels), dtype=np.uint8)
    idx = np.arange(n_levels, dtype=np.int32)[None, :]
    t[:, :] = (q[:, None] >= idx).astype(np.uint8)
    return t

def onehot_encode(q: np.ndarray, n_levels: int) -> np.ndarray:
    L = q.shape[0]
    oh = np.zeros((L, n_levels), dtype=np.uint8)
    oh[np.arange(L), q] = 1
    return oh

def rolling_windows(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided
    L = arr.shape[0]
    if L < window:
        return np.empty((0, window) + arr.shape[1:], dtype=arr.dtype)
    n = 1 + (L - window) // stride
    new_shape = (n, window) + arr.shape[1:]
    new_strides = (arr.strides[0]*stride, ) + arr.strides
    return as_strided(arr, shape=new_shape, strides=new_strides)

def build_features(sig, window, stride, encoder="bitplanes", bits=8, levels=64, include_deriv=True):
    encoder = encoder.lower()
    z = zscore_clip(sig.astype(np.float32), 3.0)
    x01, lo, hi = percentile_scale(z, 1.0, 99.0)

    if encoder == "bitplanes":
        n_bits = int(bits)
        q = np.rint(x01 * ((1 << n_bits) - 1)).astype(np.int32)
        q = np.clip(q, 0, (1 << n_bits) - 1)
        bp = int_to_bitplanes(q, n_bits)  # (L, bits)
        feats = bp
        if include_deriv:
            dq = np.abs(np.diff(q, prepend=q[:1]))
            bp_d = int_to_bitplanes(dq, n_bits)
            feats = np.concatenate([bp, bp_d], axis=1)
        wins = rolling_windows(feats, window, stride)
        X = wins.reshape(wins.shape[0], wins.shape[1]*wins.shape[2]).astype(np.uint8)
        meta = {"scale_lo": lo, "scale_hi": hi, "encoder": encoder, "bits": n_bits, "window": window, "stride": stride}

    elif encoder == "thermometer":
        L = int(levels)
        q = quantize_levels(x01, L)
        th = thermometer_encode(q, L)
        feats = th
        if include_deriv:
            dq = np.abs(np.diff(q, prepend=q[:1]))
            dq = np.minimum(dq, L-1)
            th_d = thermometer_encode(dq, L)
            feats = np.concatenate([th, th_d], axis=1)
        wins = rolling_windows(feats, window, stride)
        X = wins.reshape(wins.shape[0], wins.shape[1]*wins.shape[2]).astype(np.uint8)
        meta = {"scale_lo": lo, "scale_hi": hi, "encoder": encoder, "levels": L, "window": window, "stride": stride}

    elif encoder == "onehot2d":
        L = int(levels)
        q = quantize_levels(x01, L)
        oh = onehot_encode(q, L)
        wins = rolling_windows(oh, window, stride)
        X = wins.reshape(wins.shape[0], window*L).astype(np.uint8)
        meta = {"scale_lo": lo, "scale_hi": hi, "encoder": encoder, "levels": L, "window": window, "stride": stride}
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    return X, meta

def build_targets(noisy: np.ndarray, clean: np.ndarray, window: int, stride: int) -> np.ndarray:
    noise = noisy.astype(np.float32) - clean.astype(np.float32)
    half = window // 2
    if len(noisy) < window:
        return np.empty((0,), dtype=np.float32)
    n = 1 + (len(noisy) - window) // stride
    y = np.empty((n,), dtype=np.float32)
    for i in range(n):
        c = i*stride + half
        y[i] = noise[c]
    return y

def overlap_add(center_preds: np.ndarray, L: int, window: int, stride: int) -> np.ndarray:
    out = np.zeros(L, dtype=np.float32); wsum = np.zeros(L, dtype=np.float32); half = window // 2
    for i, y in enumerate(center_preds):
        c = i*stride + half
        if 0 <= c < L:
            out[c] += float(y); wsum[c] += 1.0
    nz = wsum > 0; out[nz] /= wsum[nz]
    if np.any(~nz):
        out = np.interp(np.arange(L), np.where(nz)[0], out[nz])
    return out
