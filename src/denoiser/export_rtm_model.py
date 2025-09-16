#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_rtm_model.py — Convert .joblib to portable bundle using get_state() or save().
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joblib", required=True)
    ap.add_argument("--out-stem", required=True)
    ap.add_argument("--clauses", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--s", type=float, default=None)
    args = ap.parse_args()

    import joblib
    obj = joblib.load(str(Path(args.joblib)))
    model = obj.get("model", None) if isinstance(obj, dict) else obj
    if model is None:
        raise SystemExit("joblib non valido: manca 'model'.")

    clauses = args.clauses or getattr(model, "number_of_clauses", None)
    T = args.T or getattr(model, "T", None)
    s = args.s or getattr(model, "s", None)
    if clauses is None or T is None or s is None:
        raise SystemExit("Specifica --clauses/--T/--s (non trovati nell'oggetto).")

    stem = Path(args.out_stem); stem.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "get_state"):
        state = model.get_state()
        np.savez_compressed(str(stem.with_suffix(".state.npz")), **state)
        stem.with_suffix(".meta.json").write_text(json.dumps({"format":"state","clauses":int(clauses),"T":int(T),"s":float(s)}, indent=2))
        print(f"[OK] Esportato stato -> {stem.with_suffix('.state.npz').name}")
    elif hasattr(model, "save"):
        model.save(str(stem.with_suffix(".rtm")))
        stem.with_suffix(".meta.json").write_text(json.dumps({"format":"rtm","clauses":int(clauses),"T":int(T),"s":float(s)}, indent=2))
        print(f"[OK] Esportato .rtm -> {stem.with_suffix('.rtm').name}")
    else:
        raise SystemExit("pyTsetlinMachine non espone get_state() né save().")

    src_stem = Path(args.joblib).with_suffix("")
    for ext in (".yscaler.json", ".pre.json"):
        src = src_stem.with_suffix(ext)
        if src.exists():
            dst = stem.with_suffix(ext); dst.write_text(src.read_text())
            print(f"[OK] Copiato {ext}")

if __name__ == "__main__":
    main()
