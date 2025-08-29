# ============================================
# src/classifier/data_generator_classifier.py
# ============================================
"""
ECG Noise Classifier - Data Generator
-----------------------------------------
Genera dataset HDF5 "raw" per la classificazione del rumore ECG:
  classi: {0:CLEAN, 1:BW, 2:MA, 3:PLI, 4:MIXED (>=2 rumori)}

Obiettivi:
- Split per record (niente leakage)
- Bilanciamento per classe e per record
- Finestre distanziate (min_spacing) per ridurre duplicati
- Modelli di rumore più realistici:
    * BW: 1/f-like (random walk) + lowpass (0.05–0.5 Hz)
    * MA: burst con envelope smooth, posizioni casuali
    * PLI: 50/60 Hz, armoniche, AM lieve, occasional phase jumps
- SNR tramite RMS (20*log10(rms(sig)/rms(noise))) con scaling stabile
- Metadati sample-level: y_multi (presenza BW/MA/PLI), snr_db per ciascun tipo,
  record id e indice di start nel segnale

Output HDF5 (default ./data/bin/classifier_raw.h5):
  group train/val/test:
    - sig       : (N, L) float32
    - y         : (N,)   uint32
    - y_multi   : (N, 3) int32       [BW, MA, PLI] in {0,1}
    - snr_db    : (N, 3) float32     SNR assegnato (NaN se non usato)
    - rid       : (N,)   |Sxx        record id
    - start_idx : (N,)   int64       indice di inizio finestra in samples
  attrs:
    - fs, L, classes_map, noise_types, snr_dom_grid, snr_mixed_range, etc.

Compatibilità:
- Preprocess: src/classifier/preprocess_ctm_classifier.py (usa solo 'sig' e 'y')
"""

import os
import math
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import h5py

# wfdb per MIT-BIH / NSTDB
try:
    import wfdb
    _WFDB_OK = True
except Exception:
    _WFDB_OK = False

# ---------------- Const ----------------
TARGET_FS = 360.0
CLASS_MAP = {0: "CLEAN", 1: "BW", 2: "MA", 3: "PLI", 4: "MIXED"}
NOISE_TYPES = ["BW","MA","PLI"]

# -------------- Utils base --------------

def set_repro(seed: int):
    np.random.seed(seed)

def list_records(mit_dir: str) -> List[str]:
    # MIT-BIH: di solito coppie .dat/.hea -> qui filtriamo da .dat
    recs = []
    for f in os.listdir(mit_dir):
        if f.endswith(".dat"):
            recs.append(os.path.splitext(f)[0])
    return sorted(list(set(recs)))

def resample_nearest(x: np.ndarray, fs_orig: float, fs_tgt: float) -> np.ndarray:
    if abs(fs_orig - fs_tgt) < 1e-9:
        return x.astype(np.float32, copy=False)
    ratio = fs_tgt / fs_orig
    idx = (np.arange(int(round(len(x)*ratio))) / ratio).astype(int)
    idx = np.clip(idx, 0, len(x)-1)
    return x[idx].astype(np.float32)

def read_signal(rec_path: str, channel: int = 0) -> Tuple[np.ndarray, float]:
    r = wfdb.rdrecord(rec_path)
    sig = r.p_signal[:, channel].astype(np.float32)
    fs = float(getattr(r, "fs", TARGET_FS))
    if fs != TARGET_FS:
        sig = resample_nearest(sig, fs, TARGET_FS)
        fs = TARGET_FS
    return sig, fs

def zscore(win: np.ndarray) -> np.ndarray:
    m = float(win.mean()); s = float(win.std()) + 1e-8
    return ((win - m) / s).astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32))))) + 1e-12

def scale_for_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scala 'noise' per avere 20*log10(rms(clean)/rms(noise_scaled)) ~= snr_db."""
    pn = rms(noise)
    if pn <= 0.0:
        return np.zeros_like(clean, dtype=np.float32)
    target = rms(clean) / (10.0**(snr_db/20.0))
    alpha = target / pn
    return (alpha * noise).astype(np.float32)

def slice_random(x: np.ndarray, L: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    if len(x) <= L:
        if len(x) < L:
            reps = int(np.ceil(L/len(x)))
            x = np.tile(x, reps)
        return x[:L].astype(np.float32), 0
    i0 = int(rng.integers(0, len(x)-L+1))
    return x[i0:i0+L].astype(np.float32), i0

def spaced_starts(Lsig: int, L: int, min_spacing: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Seleziona 'n' start index con distanza >= min_spacing, greedy + tentativi."""
    if Lsig <= L:
        return np.array([0], dtype=np.int64)[:n]
    starts = []
    attempt = 0
    while len(starts) < n and attempt < n*20:
        s0 = int(rng.integers(0, Lsig - L))
        if all(abs(s0 - s) >= min_spacing for s in starts):
            starts.append(s0)
        attempt += 1
    if len(starts) < n:
        # fallback: stride fisso
        step = max(min_spacing, 1)
        starts = list(range(0, min(Lsig-L, step*n), step))[:n]
    return np.array(starts, dtype=np.int64)

# ----------- Modelli di rumore PRO -----------

def noise_bw(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Baseline Wander realistico: random-walk (1/f-ish) + lowpass a ~0.3 Hz."""
    white = rng.normal(0.0, 1.0, size=length).astype(np.float32)
    # random walk
    walk = np.cumsum(white).astype(np.float32)
    walk = walk - walk.mean()
    # lowpass grossolano con finestra lunga
    k = max(1, int(fs * 2.5))  # ~2.5 s
    kernel = np.ones((k,), dtype=np.float32) / k
    lw = np.convolve(walk, kernel, mode="same")
    lw = lw - lw.mean()
    return lw.astype(np.float32)

def noise_ma(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Motion Artifact: burst sparsi con envelope smooth e pause."""
    base = rng.normal(0.0, 1.0, size=length).astype(np.float32)
    env = np.zeros(length, dtype=np.float32)
    # numero burst ~ 2-5 su finestra
    n_burst = int(rng.integers(2, 6))
    for _ in range(n_burst):
        center = int(rng.integers(0, length))
        w = int(rng.integers(int(0.1*fs), int(0.6*fs)))  # 0.1–0.6 s
        left = max(0, center - w//2); right = min(length, center + w//2)
        # finestra gaussiana locale
        x = np.linspace(-1, 1, right-left, dtype=np.float32)
        bump = np.exp(-4.0*(x**2)).astype(np.float32)
        env[left:right] += bump
    # smooth generale dell'envelope
    k = max(1, int(fs*0.15))
    env = np.convolve(env, np.ones((k,), dtype=np.float32)/k, mode="same")
    env = env / (env.max() + 1e-8)
    return (base * env).astype(np.float32)

def noise_pli(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """PLI con 50/60 Hz, armoniche leggere, AM lieve e salti di fase rari."""
    f_main = 50.0 if rng.uniform() < 0.8 else 60.0
    t = np.arange(length, dtype=np.float32) / float(fs)
    # fase con drift lento
    phase = rng.uniform(0, 2*np.pi)
    drift = 2*np.pi * rng.normal(0.0, 0.02, size=length).astype(np.float32)
    phase_t = phase + np.cumsum(drift)
    s = np.sin(2*np.pi*f_main*t + phase_t)
    # 2a armonica leggera
    if rng.uniform() < 0.5:
        s += 0.25*np.sin(2*np.pi*(2*f_main)*t + phase_t*1.1)
    # AM lieve
    if rng.uniform() < 0.5:
        am = 1.0 + 0.1*np.sin(2*np.pi*0.4*t + rng.uniform(0, 2*np.pi))
        s = s * am
    # occasional phase jump
    if rng.uniform() < 0.15:
        j = int(rng.integers(int(0.2*length), int(0.8*length)))
        s[j:] = -s[j:]
    # normalize
    s = s.astype(np.float32)
    s = s - s.mean()
    s = s / (s.std() + 1e-8)
    return s

def get_noise(kind: str, L: int, fs: float, rng: np.random.Generator,
              bw_track: np.ndarray = None, ma_track: np.ndarray = None) -> np.ndarray:
    if kind == "BW":
        if bw_track is not None and len(bw_track) >= L:
            seg, _ = slice_random(bw_track, L, rng)
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            return seg.astype(np.float32)
        return noise_bw(L, fs, rng)
    if kind == "MA":
        if ma_track is not None and len(ma_track) >= L:
            seg, _ = slice_random(ma_track, L, rng)
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            return seg.astype(np.float32)
        return noise_ma(L, fs, rng)
    if kind == "PLI":
        return noise_pli(L, fs, rng)
    raise ValueError(kind)

def read_nstdb_tracks(nstdb_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Legge 'bw' e 'ma' da NSTDB e risincronizza a TARGET_FS."""
    def _read_one(name):
        rr = wfdb.rdrecord(os.path.join(nstdb_dir, name))
        x = rr.p_signal[:,0].astype(np.float32)
        fs = float(getattr(rr, "fs", TARGET_FS))
        if fs != TARGET_FS:
            x = resample_nearest(x, fs, TARGET_FS)
        # standardizza per sicurezza
        return (x - x.mean()) / (x.std() + 1e-8)
    try:
        bw = _read_one("bw")
        ma = _read_one("ma")
        return bw.astype(np.float32), ma.astype(np.float32)
    except Exception:
        # fallback: nessun file → None → generator sintetico
        return None, None

# ----------- Core: build split ------------

def build_split(
    split_name: str,
    rec_ids: List[str],
    mit_dir: str,
    out_group,
    L: int,
    per_record_per_class: int,
    snr_dom_grid: List[float],
    snr_mixed_range: Tuple[float, float],
    p_three_in_mixed: float,
    contam_secondary: bool,
    contam_margin_db: float,
    min_spacing: int,
    seed: int,
    bw_track: np.ndarray,
    ma_track: np.ndarray,
):
    rng = np.random.default_rng(seed + hash(split_name) % 1000003)
    classes = [0,1,2,3,4]  # CLEAN,BW,MA,PLI,MIXED
    fs = TARGET_FS

    # prealloc
    N = len(rec_ids) * per_record_per_class * len(classes)
    ds_sig = out_group.create_dataset('sig', shape=(N, L), maxshape=(N, L),
                                      dtype=np.float32, chunks=True)
    ds_y   = out_group.create_dataset('y', shape=(N,), dtype=np.uint32, chunks=True)
    ds_ym  = out_group.create_dataset('y_multi', shape=(N,3), dtype=np.int32, chunks=True)
    ds_snr = out_group.create_dataset('snr_db',  shape=(N,3), dtype=np.float32, chunks=True)
    dt_rid = h5py.string_dtype(encoding='utf-8')
    ds_rid = out_group.create_dataset('rid', shape=(N,), dtype=dt_rid, chunks=True)
    ds_pos = out_group.create_dataset('start_idx', shape=(N,), dtype=np.int64, chunks=True)

    wptr = 0
    # ciclo per record
    for rid in rec_ids:
        sig, _ = read_signal(os.path.join(mit_dir, rid))
        if len(sig) < L:
            continue
        # start non troppo vicini tra loro
        starts = spaced_starts(len(sig), L, min_spacing, per_record_per_class, rng)

        for cls in classes:
            for s0 in starts:
                clean = zscore(sig[s0:s0+L])
                present = np.zeros(3, dtype=np.int32)
                snrvec  = np.full(3, np.nan, dtype=np.float32)

                if cls == 0:
                    noisy = clean.copy()
                elif cls in (1,2,3):
                    kind = NOISE_TYPES[cls-1]
                    dom_snr = float(rng.choice(snr_dom_grid))
                    n = get_noise(kind, L, fs, rng, bw_track, ma_track)
                    noisy = clean + scale_for_snr(clean, n, dom_snr)
                    idx = cls - 1
                    present[idx] = 1; snrvec[idx] = dom_snr
                    # contaminazione secondaria (opzionale) a +margin dB → molto debole
                    if contam_secondary:
                        for j, k in enumerate(NOISE_TYPES):
                            if j == idx:
                                continue
                            if rng.uniform() < 0.15:  # 15% chance
                                sec = get_noise(k, L, fs, rng, bw_track, ma_track)
                                noisy += scale_for_snr(clean, sec, dom_snr + contam_margin_db)
                                present[j] = 1; snrvec[j] = dom_snr + contam_margin_db
                else:
                    # MIXED: 2 o 3 componenti con SNR simili + jitter
                    kinds = NOISE_TYPES.copy(); rng.shuffle(kinds)
                    selected = kinds if (rng.uniform() < p_three_in_mixed) else kinds[:2]
                    base = float(rng.uniform(*snr_mixed_range))
                    noisy = clean.copy()
                    for k in selected:
                        jitter = rng.uniform(-1.5, 1.5)
                        sdb = max(0.0, base + jitter)
                        n = get_noise(k, L, fs, rng, bw_track, ma_track)
                        noisy += scale_for_snr(clean, n, sdb)
                        idx = NOISE_TYPES.index(k)
                        present[idx] = 1; snrvec[idx] = sdb

                # write
                ds_sig[wptr, :] = noisy
                ds_y[wptr]      = np.uint32(cls)
                ds_ym[wptr, :]  = present
                ds_snr[wptr, :] = snrvec
                ds_rid[wptr]    = rid
                ds_pos[wptr]    = int(s0)
                wptr += 1

    out_group.attrs['fs'] = float(fs)
    out_group.attrs['n_samples'] = int(wptr)

# --------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mit_dir', type=str, default='./data/mit-bih')
    ap.add_argument('--nstdb_dir', type=str, default='./data/noise_stress_test')
    ap.add_argument('--out', type=str, default='./data/bin/classifier_raw.h5')
    ap.add_argument('--L', type=int, default=1024)
    ap.add_argument('--per_record_per_class', type=int, default=20)
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--snr_dom_grid', type=float, nargs='+', default=[0.0, 3.0, 6.0, 9.0, 12.0])
    ap.add_argument('--snr_mix_min', type=float, default=8.0)
    ap.add_argument('--snr_mix_max', type=float, default=14.0)
    ap.add_argument('--mixed_prob_three', type=float, default=0.5)
    ap.add_argument('--contam_secondary', action='store_true', help='Consenti contaminazioni deboli nei single-noise')
    ap.add_argument('--contam_margin_db', type=float, default=12.0)
    ap.add_argument('--overlap', type=float, default=0.5, help='Per la distanza minima tra finestre: min_spacing = L*(1-overlap)')
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    if not _WFDB_OK:
        raise RuntimeError("wfdb non disponibile. Esegui: pip install wfdb")

    set_repro(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    recs = list_records(args.mit_dir)
    if not recs:
        raise RuntimeError(f"Nessun record MIT-BIH trovato in {args.mit_dir}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(recs)
    n_tot = len(recs)
    n_tr  = int(round(n_tot * args.train_ratio))
    n_va  = int(round(n_tot * args.val_ratio))
    n_te  = max(1, n_tot - n_tr - n_va)
    splits = {
        'train': recs[:n_tr],
        'val'  : recs[n_tr:n_tr+n_va],
        'test' : recs[n_tr+n_va:]
    }

    bw_track, ma_track = read_nstdb_tracks(args.nstdb_dir)
    min_spacing = max(1, int(args.L * (1.0 - args.overlap)))

    meta = dict(
        classes_map=CLASS_MAP,
        noise_types=NOISE_TYPES,
        fs=TARGET_FS,
        L=int(args.L),
        snr_dom_grid=[float(x) for x in args.snr_dom_grid],
        snr_mixed_range=(float(args.snr_mix_min), float(args.snr_mix_max)),
        mixed_prob_three=float(args.mixed_prob_three),
        contam_secondary=bool(args.contam_secondary),
        contam_margin_db=float(args.contam_margin_db),
        min_spacing=int(min_spacing),
        per_record_per_class=int(args.per_record_per_class),
        seed=int(args.seed),
    )

    with h5py.File(args.out, 'w') as h5:
        h5.attrs['meta_json'] = json.dumps(meta)
        for name in ('train','val','test'):
            print(f"[BUILD] split={name} | #records={len(splits[name])}")
            grp = h5.create_group(name)
            build_split(
                name, splits[name], args.mit_dir, grp,
                L=args.L,
                per_record_per_class=args.per_record_per_class,
                snr_dom_grid=[float(x) for x in args.snr_dom_grid],
                snr_mixed_range=(args.snr_mix_min, args.snr_mix_max),
                p_three_in_mixed=args.mixed_prob_three,
                contam_secondary=bool(args.contam_secondary),
                contam_margin_db=float(args.contam_margin_db),
                min_spacing=min_spacing,
                seed=args.seed,
                bw_track=bw_track, ma_track=ma_track,
            )
        print("Wrote:", args.out)

if __name__ == "__main__":
    main()
