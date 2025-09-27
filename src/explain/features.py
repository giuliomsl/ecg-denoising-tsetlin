#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature encoder for TM: bit-planes / thermometer / onehot2d + optional derivative channel.
Outputs uint8 features; cast to int32 (0/1) before feeding TM.
"""
from __future__ import annotations
import numpy as np

def zscore_clip(x, clip_sigma=3.0):
    mu = np.mean(x); sd = np.std(x) + 1e-8
    z = (x - mu) / sd
    return np.clip(z, -clip_sigma, clip_sigma)

def percentile_scale(x, lo=1.0, hi=99.0):
    p_lo, p_hi = np.percentile(x, [lo, hi])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        p_lo, p_hi = float(np.min(x)), float(np.max(x))
        if p_hi <= p_lo: p_hi = p_lo + 1.0
    y = (x - p_lo) / (p_hi - p_lo)
    y = np.clip(y, 0.0, 1.0)
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
    if L < window:
        return np.empty((0, window) + arr.shape[1:], dtype=arr.dtype)
    n = 1 + (L - window)//stride
    new_shape = (n, window) + arr.shape[1:]
    new_strides = (arr.strides[0]*stride,) + arr.strides
    return as_strided(arr, shape=new_shape, strides=new_strides)

def build_features(sig, window=128, stride=32, encoder="bitplanes", bits=8, levels=64, include_deriv=True):
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
        raise ValueError("encoder sconosciuto")
