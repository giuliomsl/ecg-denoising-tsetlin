#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter
try:
    import pywt
except Exception:
    pywt = None

def notch_filter(x, fs=360.0, f0=50.0, Q=30.0):
    b, a = iirnotch(w0=f0/(fs/2.0), Q=Q)
    return filtfilt(b, a, x).astype(np.float32)

def highpass_butter(x, fs=360.0, cutoff=0.5, order=2):
    wn = cutoff/(fs/2.0)
    b, a = butter(order, wn, btype='highpass')
    return filtfilt(b, a, x).astype(np.float32)

def wavelet_denoise(x, wavelet="bior2.6", level=None, mode="soft"):
    if pywt is None:
        raise RuntimeError("pywt not installed. pip install PyWavelets")
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745 + 1e-9
    thr = sigma * (2*np.log(len(x)))**0.5
    def shrink(c):
        if mode == "soft":
            return np.sign(c)*np.maximum(np.abs(c)-thr, 0.0)
        elif mode == "hard":
            return c * (np.abs(c) >= thr)
        return c
    den = [coeffs[0]] + [shrink(c) for c in coeffs[1:]]
    y = pywt.waverec(den, wavelet=wavelet)
    return y.astype(np.float32)
