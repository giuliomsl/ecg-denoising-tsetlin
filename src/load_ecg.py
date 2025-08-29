# ============================================
# File: load_ecg.py
# Pipeline step: (source) -> data_generator
# ============================================
"""
Loader minimale e robusto per MIT-BIH.
Fornisce `iter_clean_records(mitdb_dir)` che restituisce (record_id:str, fs:int, signal:np.ndarray float32).
Non applica filtri né normalizzazioni: il data generator farà z-score per finestra e, se serve, il resampling a 360 Hz.

Per semplicità **non** usiamo `config.py`: i percorsi si passano come argomenti agli script successivi.
"""
from __future__ import annotations
import os
import argparse
from typing import Iterable, Tuple

import numpy as np

try:
    import wfdb  # `pip install wfdb`
except Exception as e:
    wfdb = None

__all__ = ["iter_clean_records", "list_record_bases"]


def list_record_bases(mitdb_dir: str) -> list[str]:
    """Restituisce la lista dei path base (senza estensione) dei record trovati.
    Cerca ricorsivamente file `.hea` e ritorna il path senza estensione.
    """
    bases: list[str] = []
    for root, _, files in os.walk(mitdb_dir):
        for f in files:
            if f.endswith(".hea"):
                base = os.path.join(root, os.path.splitext(f)[0])
                bases.append(base)
    bases.sort()
    return bases


def iter_clean_records(mitdb_dir: str, pick_channel: int = 0) -> Iterable[Tuple[str, int, np.ndarray]]:
    """Iteratore dei record MIT-BIH.

    Yields:
        (record_id, fs, signal_float32)
    Note:
        - Non esegue resampling né filtri.
        - Se `wfdb` non è installato/al path mancano i file, solleva RuntimeError.
    """
    if wfdb is None:
        raise RuntimeError("wfdb non installato. Esegui: pip install wfdb")

    bases = list_record_bases(mitdb_dir)
    if not bases:
        raise RuntimeError(f"Nessun file .hea trovato in: {mitdb_dir}")

    for base in bases:
        try:
            sig_arr, fields = wfdb.rdsamp(base)
            sig = sig_arr.astype(np.float32)
            fs = int(fields['fs'])
            if sig.ndim == 2:
                if pick_channel < 0 or pick_channel >= sig.shape[1]:
                    # fallback al primo canale
                    ch = 0
                else:
                    ch = pick_channel
                x = sig[:, ch]
            else:
                x = sig.astype(np.float32)
            record_id = os.path.basename(base)
            yield record_id, fs, x
        except Exception as e:
            # salta record problematici ma prosegui
            print(f"[load_ecg] Skip '{base}': {e}")
            continue


def _main():
    ap = argparse.ArgumentParser(description="Quick MIT-BIH loader test")
    ap.add_argument("--mitdb_dir", type=str, default="denoising_ecg/data/mit-bih")
    ap.add_argument("--max", type=int, default=3, help="max record da stampare")
    args = ap.parse_args()

    count = 0
    for rid, fs, x in iter_clean_records(args.mitdb_dir):
        print(f"record={rid} fs={fs} len={len(x)}")
        count += 1
        if count >= args.max:
            break
    if count == 0:
        print("Nessun record valido trovato.")


if __name__ == "__main__":
    _main()
