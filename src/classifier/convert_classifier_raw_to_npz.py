#!/usr/bin/env python3
"""
Converte il dataset HDF5 "raw" del classifier in tre file NPZ compatibili con
il preprocess multicanale (preprocess_ctm_multichannel.py):

Input HDF5 (default data/bin/classifier_raw.h5): gruppi {train,val,test}
  - sig (N,L) float32  -> X_noisy
  - y   (N,)  uint32   -> y_class
  - opzionali: y_multi (N,3) int32, snr_db (N,3) float32, rid (N,), start_idx (N,)

Output NPZ in data/consolidated_data/classifier_data:
  - train_classifier_data.npz
  - validation_classifier_data.npz   (mappa "val" -> "validation")
  - test_classifier_data.npz

Contenuti NPZ minimi:
  - X_noisy (N,L) float32
  - y_class (N,)  uint32
Facoltativi (se presenti nell'HDF5):
  - y_multi, snr_db, rid, start_idx
"""

import os
import argparse
import json
import h5py
import numpy as np


def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def convert(src_h5: str, out_dir: str) -> None:
    ensure_dir(out_dir)
    split_map = {"train": "train", "val": "validation", "test": "test"}
    with h5py.File(src_h5, "r") as h5:
        # salva meta se presente
        meta_json = h5.attrs.get("meta_json", None)
        if meta_json is not None:
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                f.write(meta_json if isinstance(meta_json, str) else meta_json.decode("utf-8"))

        for g in ("train", "val", "test"):
            if g not in h5:
                print(f"[WARN] Gruppo '{g}' non trovato, salto")
                continue
            grp = h5[g]
            X = grp["sig"][...].astype(np.float32)
            y = grp["y"][...].astype(np.uint32)
            payload = {
                "X_noisy": X,
                "y_class": y,
            }
            # opzionali se esistono
            if "y_multi" in grp:
                payload["y_multi"] = grp["y_multi"][...].astype(np.int32)
            if "snr_db" in grp:
                payload["snr_db"] = grp["snr_db"][...].astype(np.float32)
            if "rid" in grp:
                try:
                    payload["rid"] = grp["rid"][...].astype(str)
                except Exception:
                    payload["rid"] = grp["rid"][...]
            if "start_idx" in grp:
                payload["start_idx"] = grp["start_idx"][...].astype(np.int64)

            split_name = split_map[g]
            out_npz = os.path.join(out_dir, f"{split_name}_classifier_data.npz")
            np.savez_compressed(out_npz, **payload)
            print(f"[WRITE] {out_npz}  X={X.shape}  y={y.shape}")


def main():
    ap = argparse.ArgumentParser(description="Converti HDF5 raw del classifier in NPZ consolidati")
    ap.add_argument("--src_h5", type=str, default="./data/bin/classifier_raw.h5")
    ap.add_argument("--out_dir", type=str, default="./data/consolidated_data/classifier_data")
    args = ap.parse_args()

    src_h5 = os.path.abspath(args.src_h5)
    out_dir = os.path.abspath(args.out_dir)
    print(f"[CONVERT] src_h5={src_h5}\n          -> out_dir={out_dir}")
    convert(src_h5, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
