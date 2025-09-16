#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rtm.py — Robust training for Regression Tsetlin Machine with portable save.
Saves:
  - <stem>.state.npz  (model.get_state())
  - <stem>.meta.json  (clauses, T, s, env, dataset attrs)
  - <stem>.yscaler.json (mu, sigma, clip) for target de-normalization
  - <stem>.env.json   (library versions) for provenance
Optionally also writes metrics CSV (val/train MAE/RMSE per epoch).

Requires: pyTsetlinMachine, numpy, h5py, scikit-learn, tqdm, joblib (optional)
"""
from __future__ import annotations
import argparse, json, math, os, sys, platform, time
from pathlib import Path
from typing import Optional, Dict
import numpy as np, h5py
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

try:
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
except Exception as e:
    raise SystemExit("Install pyTsetlinMachine: pip install pyTsetlinMachine\\n" + str(e))

def set_seed(seed: Optional[int] = None):
    if seed is None: return
    np.random.seed(int(seed))

def robust_fit_scale(y: np.ndarray, method: str = "mad"):
    if method == "mad":
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med))) + 1e-9
        sigma = mad / 0.6745
        return med, sigma
    elif method == "std":
        mu = float(np.mean(y))
        sigma = float(np.std(y) + 1e-9)
        return mu, sigma
    else:
        raise ValueError("Unknown norm method")

def robust_transform(y: np.ndarray, mu: float, sigma: float, clip: float = 3.0):
    z = (y - mu) / (sigma if sigma != 0.0 else 1.0)
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)

def evaluate(model: RegressionTsetlinMachine, X: np.ndarray, y: np.ndarray, batch: int) -> Dict[str, float]:
    preds = []
    n = X.shape[0]
    for i in range(0, n, batch):
        preds.append(model.predict(X[i:i+batch]))
    p = np.concatenate(preds).astype(np.float32)
    mae = float(mean_absolute_error(y, p))
    rmse = float(math.sqrt(mean_squared_error(y, p)))
    return {"mae": mae, "rmse": rmse}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="H5 file with train_X/train_y, val_X/val_y")
    ap.add_argument("--model-out", required=True, help="Output stem or .joblib (stem is recommended)")
    ap.add_argument("--clauses", type=int, default=10000)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--s", type=float, default=2.75)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--inner-epochs", type=int, default=3, help="epochs per mini-batch (incremental=True)")
    ap.add_argument("--batch", type=int, default=10000, help="mini-batch size for training")
    ap.add_argument("--eval-batch", type=int, default=50000, help="batch size for evaluation")
    ap.add_argument("--train-eval-cap", type=int, default=200000, help="cap for train eval (faster)")
    ap.add_argument("--patience", type=int, default=10, help="early stop patience (epochs without val MAE improvement)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics-csv", type=str, default=None)
    ap.add_argument("--norm-method", choices=["mad","std"], default="mad")
    ap.add_argument("--norm-clip", type=float, default=3.0)
    ap.add_argument("--train-cap", type=int, default=None, help="Optional cap on number of training rows (for quick runs)")
    args = ap.parse_args()

    set_seed(args.seed)

    # Normalize model_out to stem (no extension). We'll write .state.npz/.meta.json there.
    outp = Path(args.model_out)
    if outp.suffix in (".joblib", ".rtm", ".state", ".npz", ".json"):
        stem = outp.with_suffix("")
    else:
        stem = outp

    # Load data
    with h5py.File(str(Path(args.data)), "r") as f:
        Xtr = f["train_X"][:].astype(np.uint8, copy=False)
        ytr = f["train_y"][:].astype(np.float32, copy=False)
        Xva = f["val_X"][:].astype(np.uint8, copy=False)
        yva = f["val_y"][:].astype(np.float32, copy=False)
        # Try pull dataset attrs for provenance
        ds_attrs = {}
        for k in ("window","stride","encoder","bits","levels","include_deriv","fs","prefilter"):
            if k in f.attrs:
                try:
                    v = f.attrs[k]
                    if isinstance(v, (bytes, bytearray)):
                        v = v.decode("utf-8")
                    ds_attrs[k] = v if not isinstance(v, np.ndarray) else v.tolist()
                except Exception:
                    pass

    # Optional cap for quick runs
    if args.train_cap is not None and args.train_cap > 0:
        cap = min(int(args.train_cap), Xtr.shape[0])
        Xtr = Xtr[:cap]
        ytr = ytr[:cap]

    # Target normalization
    mu, sigma = robust_fit_scale(ytr, method=args.norm_method)
    ytr_n = robust_transform(ytr, mu, sigma, clip=args.norm_clip)
    yva_n = robust_transform(yva, mu, sigma, clip=args.norm_clip)
    print(f"[NORM] method={args.norm_method} mu={mu:.6f} sigma={sigma:.6f} clip=±{args.norm_clip:.1f}")

    # Model
    model = RegressionTsetlinMachine(int(args.clauses), int(args.T), float(args.s))

    # Training loop
    best_mae, best_ep = float("inf"), -1
    patience_left = int(args.patience)
    n_train = Xtr.shape[0]
    eval_cap = min(int(args.train_eval_cap), n_train)

    # Open metrics CSV if requested
    csv_f = None
    if args.metrics_csv:
        csv_f = open(args.metrics_csv, "w")
        csv_f.write("epoch,train_mae,train_rmse,val_mae,val_rmse,seconds\n"); csv_f.flush()

    import time
    for ep in range(1, int(args.epochs)+1):
        t0 = time.time()
        idx = np.arange(n_train)
        np.random.shuffle(idx)
        for i in tqdm(range(0, n_train, args.batch), desc=f"Epoch {ep}/{args.epochs}"):
            j = idx[i:i+args.batch]
            Xb, yb = Xtr[j], ytr_n[j]
            model.fit(Xb, yb, epochs=int(args.inner_epochs), incremental=True)

        tr = evaluate(model, Xtr[:eval_cap], ytr_n[:eval_cap], batch=args.eval_batch)
        va = evaluate(model, Xva, yva_n, batch=args.eval_batch)
        dt = time.time() - t0
        print(f"[E{ep}] {dt:.1f}s  train MAE={tr['mae']:.5f} RMSE={tr['rmse']:.5f}  val MAE={va['mae']:.5f} RMSE={va['rmse']:.5f}")

        if csv_f:
            csv_f.write(f"{ep},{tr['mae']:.6f},{tr['rmse']:.6f},{va['mae']:.6f},{va['rmse']:.6f},{dt:.2f}\n"); csv_f.flush()

        if va["mae"] < best_mae - 1e-6:
            best_mae, best_ep = va["mae"], ep
            if not hasattr(model, "get_state"):
                raise RuntimeError("pyTsetlinMachine RegressionTsetlinMachine lacks get_state() — update the package.")
            state = model.get_state()
            stem.parent.mkdir(parents=True, exist_ok=True)
            npz_path = stem.with_suffix(".state.npz")
            np.savez_compressed(str(npz_path), **state)
            meta = {"format": "state","clauses": int(args.clauses),"T": int(args.T),"s": float(args.s),
                    "best_val_mae": float(best_mae),"best_epoch": int(best_ep), "dataset_attrs": ds_attrs}
            stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
            stem.with_suffix(".yscaler.json").write_text(json.dumps({"mu": mu, "sigma": sigma, "clip": args.norm_clip}, indent=2))
            try:
                import sklearn, pyTsetlinMachine
                env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
                       "numpy": np.__version__, "sklearn": sklearn.__version__,
                       "pyTsetlinMachine": getattr(pyTsetlinMachine, "__version__", "unknown")}
                stem.with_suffix(".env.json").write_text(json.dumps(env, indent=2))
            except Exception:
                pass
            print(f"[SAVE] {npz_path.name} (+meta, yscaler, env)  val MAE {best_mae:.6f}")
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] no val improvement for {args.patience} epochs (best epoch {best_ep}).")
                break

    if csv_f: csv_f.close()

if __name__ == "__main__":
    main()
