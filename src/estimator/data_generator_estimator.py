# ============================================
# File: data_generator_estimator.py
# Pipeline: MIT-BIH (clean) + NSTDB (BW/MA) + PLI sintetico  ->  HDF5 "estimation"
# ============================================
"""
Generatore dati per Noise Estimation (per tipo: BW, MA, PLI) su finestre ECG.

Output HDF5 (default: ./data/bin/estimator_raw.h5), gruppi: train/val/test
  - sig         : (N, L) float32              -> finestra noisy (z-score)
  - y_present   : (N, 3) uint8  in {0,1}      -> presenza [BW, MA, PLI]
  - y_snr_db    : (N, 3) float32              -> SNR dB per tipo (absent_val se non presente)
  - y_snr_bin   : (N, 3) uint8                -> bin SNR per tipo (0 = absent)
  - attrs sul gruppo:
        fs, L, snr_absent_value, snr_bin_edges (float[]), bin_def_json (str)

Feature “full gas”:
  - Split per record (no leakage)
  - Copertura bilanciata: 1-via, 2-vie, 3-vie (+ quota di clean)
  - SNR ben controllato: dominio & “mixed” con jitter
  - NSTDB: BW/MA veri; PLI sintetico (50 Hz con jitter + armonica opzionale)
  - Resampling robusto via np.interp (no dipendenze extra)
  - Chunking + gzip per I/O
  - Metadati completi per il preprocessing/estimation

Richiede:
  - wfdb (pip install wfdb)
  - MIT-BIH in --mit_dir (cartelle .dat/.hea)
  - NSTDB in --nstdb_dir (records 'bw', 'ma')

Uso:
  python src/estimator/data_generator_estimator.py \
    --mit_dir ./data/mit-bih --nstdb_dir ./data/noise_stress_test \
    --out ./data/bin/estimator_raw.h5 --L 1024 --fs 360 \
    --n_train 16000 --n_val 4000 --n_test 4000 \
    --p_clean 0.05 --p_single 0.55 --p_double 0.30 --p_triple 0.10 \
    --seed 123
"""

from __future__ import annotations
import os
import json
import math
import argparse
import random
from typing import List, Tuple, Dict

import numpy as np
import h5py

try:
    import wfdb
    _WFDB_OK = True
except Exception:
    _WFDB_OK = False


# ------------------ Costanti & util ------------------

NOISE_NAMES = ["BW", "MA", "PLI"]  # ordine fisso
IDX = {n:i for i,n in enumerate(NOISE_NAMES)}

DEFAULT_SNR_BINS = [0.0, 3.0, 6.0, 9.0, 12.0, 18.0]  # bin edges (dB); bin 0 è "absent"

def set_repro(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def list_mit_records(mit_dir: str) -> List[str]:
    recs = []
    for f in os.listdir(mit_dir):
        if f.endswith(".hea"):
            recs.append(os.path.splitext(f)[0])
        elif f.endswith(".dat"):
            recs.append(os.path.splitext(f)[0])
    return sorted(list(set(recs)))

def rd_sig(rec_path_base: str, channel: int = 0) -> Tuple[np.ndarray, float]:
    """Legge un record WFDB e ritorna (signal float32, fs)."""
    if not _WFDB_OK:
        raise RuntimeError("wfdb non disponibile. `pip install wfdb`.")
    rec = wfdb.rdrecord(rec_path_base)
    fs = float(rec.fs)
    sig = rec.p_signal[:, channel].astype(np.float32)
    return sig, fs

def resample_linear(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resampling lineare con np.interp (stabile, senza scipy)."""
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * fs_out / fs_in))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(xq, xp, x.astype(np.float64, copy=False)).astype(np.float32)
    return y

def zscore_win(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        sd = 1e-8
    return ((x - mu) / sd).astype(np.float32)

def slice_random(x: np.ndarray, L: int) -> np.ndarray:
    if x.shape[0] <= L:
        if x.shape[0] < L:
            return np.pad(x, (0, L - x.shape[0])).astype(np.float32)
        return x.astype(np.float32, copy=False)
    i0 = np.random.randint(0, x.shape[0] - L + 1)
    return x[i0:i0+L].astype(np.float32)

def gen_pli(L: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    # PLI 50Hz con jitter ±0.5 Hz e 2a armonica opzionale
    f0 = 50.0 + rng.uniform(-0.5, 0.5)
    t = np.arange(L, dtype=np.float32) / float(fs)
    s = np.sin(2*np.pi*f0*t + rng.uniform(0, 2*np.pi)).astype(np.float32)
    if rng.uniform() < 0.35:
        s += 0.25 * np.sin(2*np.pi*2*f0*t + rng.uniform(0, 2*np.pi)).astype(np.float32)
    return s

def power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float32) ** 2)) + 1e-12

def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scala il noise per ottenere l'SNR desiderato (dB) rispetto a 'clean'."""
    if noise.shape[0] < clean.shape[0]:
        reps = int(np.ceil(clean.shape[0] / noise.shape[0]))
        noise = np.tile(noise, reps)
    noise = noise[:clean.shape[0]]
    ps = power(clean)
    pn = power(noise)
    target_pn = ps / (10.0 ** (snr_db / 10.0))
    alpha = math.sqrt(max(target_pn, 1e-12) / pn)
    return (noise * alpha).astype(np.float32)

def load_nstdb_tracks(nstdb_dir: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Carica BW e MA dal NSTDB (records canonici 'bw', 'ma'), ritorna (bw, ma, fs)."""
    bw, fs_bw = None, None
    ma, fs_ma = None, None

    def _rd(name: str):
        p = os.path.join(nstdb_dir, name)
        sig, fs = rd_sig(p)
        return sig, fs

    sig_bw, fs_bw = _rd("bw")
    sig_ma, fs_ma = _rd("ma")

    if abs(fs_bw - fs_ma) > 1e-3:
        # portiamo entrambi a fs_bw per coerenza
        sig_ma = resample_linear(sig_ma, fs_ma, fs_bw)
        fs_ma = fs_bw

    return sig_bw, sig_ma, fs_bw


# ------------------ SNR binning ------------------

def snr_to_bin(snr: float, edges: List[float]) -> int:
    """
    edges = [e0, e1, ...] (in dB). Restituisce bin 1..(len(edges)+1) per snr presente.
    Esempio con edges=[0,3,6]:  (-inf,0)->1; [0,3)->2; [3,6)->3; [6,+inf)->4.
    """
    for i, thr in enumerate(edges):
        if snr < thr:
            return 1 + i
    return 1 + len(edges)  # top bin


# ------------------ Generazione core ------------------

def build_split(
    rec_paths: List[str],
    mit_dir: str,
    bw_track: np.ndarray,
    ma_track: np.ndarray,
    fs_nstdb: float,
    *,
    L: int,
    fs_target: float,
    n_samples: int,
    ratios: Dict[str, float],
    snr_dom_grid: List[float],
    snr_mix_base: Tuple[float, float],
    snr_jitter: float,
    snr_absent_value: float,
    snr_bin_edges: List[float],
    p_clean: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera (sig, y_present, y_snr_db, y_snr_bin) per uno split.
    - y_present: (N,3) in {0,1}
    - y_snr_db: (N,3) dB, absent -> snr_absent_value
    - y_snr_bin: (N,3) uint8, absent -> 0, altrimenti 1..B
    """
    # Pre-carico i record scelti in RAM per velocità
    ecg_bank: List[np.ndarray] = []
    for rid in rec_paths:
        sig, fs_rec = rd_sig(os.path.join(mit_dir, rid))
        if abs(fs_rec - fs_target) > 1e-6:
            sig = resample_linear(sig, fs_rec, fs_target)
        ecg_bank.append(sig.astype(np.float32))

    # Porta NSTDB a fs_target se serve
    if abs(fs_nstdb - fs_target) > 1e-6:
        bw_track = resample_linear(bw_track, fs_nstdb, fs_target)
        ma_track = resample_linear(ma_track, fs_nstdb, fs_target)

    N = int(n_samples)
    sig_out      = np.zeros((N, L), dtype=np.float32)
    present_out  = np.zeros((N, 3), dtype=np.uint8)
    snrdb_out    = np.full((N, 3), snr_absent_value, dtype=np.float32)
    snrbin_out   = np.zeros((N, 3), dtype=np.uint8)

    # normalizza ratios (single/double/triple)
    p_single = max(0.0, ratios.get("single", 0.6))
    p_double = max(0.0, ratios.get("double", 0.3))
    p_triple = max(0.0, ratios.get("triple", 0.1))
    tot = p_single + p_double + p_triple
    if tot <= 0:
        p_single, p_double, p_triple = 0.6, 0.3, 0.1
        tot = 1.0
    p_single /= tot; p_double /= tot; p_triple /= tot

    for i in range(N):
        # scegli finestra clean
        ecg = rng.choice(ecg_bank)
        clean = slice_random(ecg, L).astype(np.float32)
        clean = zscore_win(clean)

        # decide combinazione
        r = rng.uniform()
        if r < p_clean:
            # clean puro -> tutto assente
            noisy = clean.copy()
            present = np.array([0,0,0], dtype=np.uint8)
            snr_db = np.array([snr_absent_value]*3, dtype=np.float32)
        else:
            r2 = rng.uniform()
            if r2 < p_single:
                k = rng.integers(0, 3)  # un tipo
                sel = [k]
            elif r2 < p_single + p_double:
                # due tipi distinti
                sel = rng.choice([0,1,2], size=2, replace=False).tolist()
            else:
                sel = [0,1,2]

            noisy = clean.copy()
            present = np.zeros((3,), dtype=np.uint8)
            snr_db = np.full((3,), snr_absent_value, dtype=np.float32)

            # SNR di base
            base = float(rng.choice(snr_dom_grid))
            for k in sel:
                # jitter per rendere “naturali” i mix
                snr_k = max(0.0, base + rng.uniform(-snr_jitter, snr_jitter))
                if k == IDX["BW"]:
                    n = slice_random(bw_track, L)
                elif k == IDX["MA"]:
                    n = slice_random(ma_track, L)
                else:  # PLI
                    n = gen_pli(L, fs_target, rng)

                n_scaled = scale_noise_to_snr(clean, n, snr_k)
                noisy = noisy + n_scaled
                present[k] = 1
                snr_db[k] = snr_k

        # z-score finale (non altera l’SNR relativo)
        noisy = zscore_win(noisy)

        # binning per tipo (0=absent)
        snr_bin = np.zeros((3,), dtype=np.uint8)
        for k in range(3):
            if present[k]:
                snr_bin[k] = snr_to_bin(float(snr_db[k]), snr_bin_edges)
            else:
                snr_bin[k] = 0

        sig_out[i]     = noisy
        present_out[i] = present
        snrdb_out[i]   = snr_db
        snrbin_out[i]  = snr_bin

    return sig_out, present_out, snrdb_out, snrbin_out


# ------------------ Entry point ------------------

def main():
    ap = argparse.ArgumentParser(description="ECG Noise Estimation - Data Generator")
    ap.add_argument("--mit_dir", type=str, default="./data/mit-bih")
    ap.add_argument("--nstdb_dir", type=str, default="./data/noise_stress_test")
    ap.add_argument("--out", type=str, default="./data/bin/estimator_raw.h5")
    ap.add_argument("--L", type=int, default=1024, help="lunghezza finestra")
    ap.add_argument("--fs", type=float, default=360.0, help="fs target dopo resampling")
    ap.add_argument("--n_train", type=int, default=16000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--p_clean", type=float, default=0.05, help="quota di finestre clean (tutti i rumori assenti)")
    ap.add_argument("--p_single", type=float, default=0.55, help="quota 1 rumore")
    ap.add_argument("--p_double", type=float, default=0.30, help="quota 2 rumori")
    ap.add_argument("--p_triple", type=float, default=0.10, help="quota 3 rumori")
    ap.add_argument("--snr_dom_grid", type=str, default="0,3,6,9,12", help="grid dB per singoli/mix (base)")
    ap.add_argument("--snr_mix_jitter", type=float, default=1.5, help="+/- jitter (dB) attorno alla base")
    ap.add_argument("--snr_absent_value", type=float, default=-1.0, help="valore sentinel per SNR assente")
    ap.add_argument("--snr_bin_edges", type=str, default="0,3,6,9,12,18", help="soglie (dB) per i bin; 0=absent")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not _WFDB_OK:
        raise RuntimeError("wfdb non disponibile. Installa con: pip install wfdb")

    set_repro(args.seed)
    ensure_dir(os.path.dirname(args.out))

    # records & split per record (no leakage)
    recs = list_mit_records(args.mit_dir)
    if not recs:
        raise RuntimeError(f"Nessun record MIT-BIH trovato in {args.mit_dir}")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(recs)
    ntr = int(0.7 * len(recs)); nva = int(0.15 * len(recs))
    split_recs = {
        "train": recs[:ntr],
        "val":   recs[ntr:ntr+nva],
        "test":  recs[ntr+nva:]
    }

    # NSTDB tracks
    bw_track, ma_track, fs_nstdb = load_nstdb_tracks(args.nstdb_dir)

    # parsing soglie/griglie
    dom_grid = [float(x) for x in args.snr_dom_grid.split(",") if x.strip()!=""]
    bin_edges = [float(x) for x in args.snr_bin_edges.split(",") if x.strip()!=""]
    ratios = {"single": args.p_single, "double": args.p_double, "triple": args.p_triple}

    # mapping bin -> intervalli (string) per metadata
    # es: edges=[0,3,6] => bin1=(-inf,0), bin2=[0,3), bin3=[3,6), bin4=[6,inf)
    def bin_def(edges: List[float]) -> Dict[int, str]:
        out = {0: "absent"}
        cur_low = float("-inf")
        for i, thr in enumerate(edges):
            out[i+1] = f"[{cur_low:.1f},{thr:.1f})"
            cur_low = thr
        out[len(edges)+1] = f"[{cur_low:.1f},+inf)"
        return out

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    with h5py.File(args.out, "w") as h5:
        for split in ("train", "val", "test"):
            print(f"[{split}] records={len(split_recs[split])} target_N={counts[split]}")
            sig, yp, ydb, ybin = build_split(
                rec_paths=split_recs[split],
                mit_dir=args.mit_dir,
                bw_track=bw_track,
                ma_track=ma_track,
                fs_nstdb=fs_nstdb,
                L=args.L,
                fs_target=args.fs,
                n_samples=counts[split],
                ratios=ratios,
                snr_dom_grid=dom_grid,
                snr_mix_base=(min(dom_grid), max(dom_grid)),
                snr_jitter=float(args.snr_mix_jitter),
                snr_absent_value=float(args.snr_absent_value),
                snr_bin_edges=bin_edges,
                p_clean=float(args.p_clean),
                rng=rng,
            )

            g = h5.create_group(split)
            # chunking sensato per prestazioni
            chN = min(512, sig.shape[0]); chC = 3; chL = min(512, sig.shape[1])
            g.create_dataset("sig", data=sig, dtype=np.float32,
                             chunks=(chN, chL), compression="gzip", compression_opts=4, shuffle=True)
            g.create_dataset("y_present", data=yp, dtype=np.uint8,
                             chunks=(chN, chC), compression="gzip", compression_opts=4, shuffle=True)
            g.create_dataset("y_snr_db", data=ydb, dtype=np.float32,
                             chunks=(chN, chC), compression="gzip", compression_opts=4, shuffle=True)
            g.create_dataset("y_snr_bin", data=ybin, dtype=np.uint8,
                             chunks=(chN, chC), compression="gzip", compression_opts=4, shuffle=True)

            g.attrs["fs"] = float(args.fs)
            g.attrs["L"] = int(args.L)
            g.attrs["snr_absent_value"] = float(args.snr_absent_value)
            g.attrs["snr_bin_edges"] = np.array(bin_edges, dtype=np.float32)
            g.attrs["bin_def_json"] = json.dumps(bin_def(bin_edges))
            g.attrs["noise_order"] = json.dumps(NOISE_NAMES)

        # meta top-level
        h5.attrs["generator"] = "noise_estimator_v1_fullgas"
        h5.attrs["fs_target"] = float(args.fs)
        h5.attrs["window_L"] = int(args.L)
        h5.attrs["snr_dom_grid"] = np.array(dom_grid, dtype=np.float32)
        h5.attrs["snr_mix_jitter"] = float(args.snr_mix_jitter)
        h5.attrs["ratios_json"] = json.dumps({"clean": args.p_clean, **ratios})
        h5.attrs["noise_order"] = json.dumps(NOISE_NAMES)

    print("Wrote HDF5 ->", args.out)


if __name__ == "__main__":
    main()
