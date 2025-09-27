#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rtm.py — Training robusto per Regression Tsetlin Machine con salvataggio portabile.

- Feature binarie int32 contigue (0/1) -> come nel run (1).
- Normalizzazione target: mad|std|pct (default: MAD, clip ±3).
- Diagnostica: check binarietà X, clipped_ratio del target, percentili.
- Salvataggio portabile via get_state() (dict o tuple/list) -> <stem>.state.npz + meta/scaler/env (JSON-safe).
- Early stopping, metrics CSV, evaluation batched.
- Opzione --train-cap per smoke test veloci.

Dipendenze: pyTsetlinMachine, numpy, h5py, scikit-learn, tqdm
"""
from __future__ import annotations
import argparse, json, math, platform, time
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# ---- pyTsetlinMachine ----
try:
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
except Exception as e:
    raise SystemExit(
        "pyTsetlinMachine non importabile. Installa/aggiorna: pip install -U pyTsetlinMachine\n"
        f"Dettagli: {e}"
    )

# ---------------- utils ----------------

def set_seed(seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(int(seed))

def robust_fit_scale(y: np.ndarray, method: str = "mad") -> tuple[float, float]:
    """Restituisce (mu, sigma) secondo il metodo scelto."""
    if method == "mad":
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med))) + 1e-9
        sigma = mad / 0.6745
        return med, sigma
    elif method == "std":
        mu = float(np.mean(y))
        sigma = float(np.std(y) + 1e-9)
        return mu, sigma
    elif method == "pct":
        lo, hi = np.percentile(y, [1.0, 99.0])
        if hi <= lo:
            mu = float(np.mean(y))
            sigma = float(np.std(y) + 1e-9)
        else:
            mu = float(0.5 * (lo + hi))
            sigma = float(0.5 * (hi - lo)) + 1e-9
        return mu, sigma
    else:
        raise ValueError("norm-method sconosciuto (mad|std|pct)")

def robust_transform(y: np.ndarray, mu: float, sigma: float, clip: Optional[float]) -> np.ndarray:
    z = (y - mu) / (sigma if sigma != 0.0 else 1.0)
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z.astype(np.float32, copy=False)

def batched_predict(model: RegressionTsetlinMachine, X: np.ndarray, batch: int) -> np.ndarray:
    out = []
    n = X.shape[0]
    for i in range(0, n, batch):
        out.append(model.predict(X[i:i + batch]))
    return np.concatenate(out).astype(np.float32, copy=False)

def evaluate(model: RegressionTsetlinMachine, X: np.ndarray, y: np.ndarray, batch: int) -> Dict[str, float]:
    p = batched_predict(model, X, batch=batch)
    mae = float(mean_absolute_error(y, p))
    rmse = float(math.sqrt(mean_squared_error(y, p)))
    return {"mae": mae, "rmse": rmse}

def evaluate_with_raw(model: RegressionTsetlinMachine,
                      X: np.ndarray,
                      y_norm: np.ndarray,
                      y_raw: np.ndarray,
                      mu: float,
                      sigma: float,
                      batch: int) -> Dict[str, float]:
    """
    Valuta sia nello spazio normalizzato (y_norm) sia in quello raw (ricostruendo p_raw = p* sigma + mu).
    Ritorna dizionario con chiavi mae, rmse, mae_raw, rmse_raw.
    """
    p_norm = batched_predict(model, X, batch=batch)
    mae = float(mean_absolute_error(y_norm, p_norm))
    rmse = float(math.sqrt(mean_squared_error(y_norm, p_norm)))
    p_raw = p_norm * (sigma if sigma != 0 else 1.0) + mu
    mae_raw = float(mean_absolute_error(y_raw, p_raw))
    rmse_raw = float(math.sqrt(mean_squared_error(y_raw, p_raw)))
    return {"mae": mae, "rmse": rmse, "mae_raw": mae_raw, "rmse_raw": rmse_raw}

def _to_jsonable(x: Any) -> Any:
    """Rende JSON-serializzabili scalari/array numpy, bytes, ecc."""
    import numpy as _np
    if isinstance(x, (bytes, bytearray)):
        try: return x.decode("utf-8")
        except Exception: return x.hex()
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(t) for t in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return x

def _decode_attr(v):
    # Converte gli attributi H5 in tipi Python puri
    return _to_jsonable(v)

def dataset_attrs_from_h5(f: h5py.File) -> Dict:
    keys = [
        "window", "stride", "encoder", "bits", "levels", "include_deriv",
        "fs",
        "prefilter_notch", "notch_freq", "notch_Q",
        "prefilter_hp", "hp_cutoff", "hp_order",
        "wavelet", "wavelet_level", "wavelet_mode",
    ]
    out = {}
    for k in keys:
        if k in f.attrs:
            out[k] = _decode_attr(f.attrs[k])
    return out

# ---- salvataggio portabile: supporta state come mapping o come sequenza ----

def save_rtm_state_portable(stem: Path,
                            model: RegressionTsetlinMachine,
                            meta_base: Dict,
                            y_scaler: Dict):
    """
    Salva lo stato in <stem>.state.npz + <stem>.meta.json (+ yscaler/env).
    Supporta get_state() che ritorna dict *oppure* tuple/list; in casi rari salva .state.npy pickled.
    """
    if not hasattr(model, "get_state"):
        raise RuntimeError("La tua pyTsetlinMachine non espone get_state(). Aggiorna il pacchetto.")

    state = model.get_state()
    meta = dict(meta_base)  # copia (poi JSON-safe)

    npz_path = stem.with_suffix(".state.npz")

    # Caso 1: mapping (dict-like)
    try:
        from collections.abc import Mapping
        is_mapping = isinstance(state, Mapping)
    except Exception:
        is_mapping = hasattr(state, "keys") and hasattr(state, "__getitem__")

    if is_mapping:
        # converti i valori a tipi numpy-friendly (array/scalari) prima di np.savez
        state_np = {}
        for k, v in state.items():
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            elif isinstance(v, (np.generic,)):
                v = np.array(v)
            state_np[str(k)] = v
        np.savez_compressed(str(npz_path), **state_np)
        meta["format"] = "state"
        meta["state_format"] = "mapping"
        meta["state_keys"] = list(state_np.keys())

    # Caso 2: sequenza (tuple/list)
    elif isinstance(state, (tuple, list)):
        payload = {}
        for i, v in enumerate(state):
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            elif isinstance(v, (np.generic,)):
                v = np.array(v)
            payload[f"s{i}"] = v
        np.savez_compressed(str(npz_path), **payload)
        meta["format"] = "state"
        meta["state_format"] = "sequence"
        meta["state_len"] = int(len(state))

    # Caso 3: altro (fallback sicuro pickled)
    else:
        np.save(str(stem.with_suffix(".state.npy")), state, allow_pickle=True)
        meta["format"] = "state"
        meta["state_format"] = "pickle_npy"

    # Scrivi meta + scaler + env (JSON-safe)
    stem.with_suffix(".meta.json").write_text(json.dumps(_to_jsonable(meta), indent=2))
    stem.with_suffix(".yscaler.json").write_text(json.dumps(_to_jsonable(y_scaler), indent=2))
    try:
        import sklearn, pyTsetlinMachine
        env = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "pyTsetlinMachine": getattr(pyTsetlinMachine, "__version__", "unknown"),
        }
        stem.with_suffix(".env.json").write_text(json.dumps(env, indent=2))
    except Exception:
        pass

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="H5 con train_X/train_y, val_X/val_y")
    ap.add_argument("--model-out", required=True, help="Stem di output (consigliato SENZA estensione)")

    # default allineati al run (1) ma ottimizzati
    ap.add_argument("--clauses", type=int, default=8000)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--s", type=float, default=2.75)  # Ripristino s originale per learning rate migliore

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--inner-epochs", type=int, default=3, help="epoche per mini-batch (incremental=True)")
    ap.add_argument("--batch", type=int, default=10000, help="mini-batch size")
    ap.add_argument("--eval-batch", type=int, default=50000, help="batch per valutazione/predict")
    ap.add_argument("--train-eval-cap", type=int, default=200000,
                    help="massimo di esempi del train usati per la valutazione (velocizza)")
    ap.add_argument("--train-cap", type=int, default=None,
                    help="se impostato, usa solo i primi N esempi del train (smoke test veloce)")
    ap.add_argument("--patience", type=int, default=10, help="early stopping (epoche senza miglioramento val MAE)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics-csv", type=str, default=None)
    ap.add_argument("--norm-method", choices=["mad", "std", "pct"], default="mad",
                    help="normalizzazione del target: mad|std|pct(1–99)")
    ap.add_argument("--norm-clip", type=float, default=5.0,
                    help="clip della target normalizzata (None per nessun clip) - default 5.0 per meno saturazione")
    ap.add_argument("--adaptive-lr", action="store_true",
                    help="Se True, riduce s se val MAE stagna per 3+ epoche")
    ap.add_argument("--eval-freq", type=int, default=1,
                    help="Frequenza valutazione (ogni N epoche) per accelerare training lungo")
    args = ap.parse_args()

    set_seed(args.seed)

    stem = Path(args.model_out).with_suffix("")  # normalizza a stem
    stem.parent.mkdir(parents=True, exist_ok=True)

    # Carica dati (INT32 contiguo! come nel run (1))
    with h5py.File(str(Path(args.data)), "r") as f:
        Xtr_full = np.ascontiguousarray(f["train_X"][:].astype(np.int32))
        ytr_full = f["train_y"][:].astype(np.float32, copy=False)
        Xva = np.ascontiguousarray(f["val_X"][:].astype(np.int32))
        yva = f["val_y"][:].astype(np.float32, copy=False)
        ds_attrs = dataset_attrs_from_h5(f)

    # Cap del train per smoke test (opzionale)
    if args.train_cap is not None:
        ncap = int(args.train_cap)
        Xtr = Xtr_full[:ncap]
        ytr = ytr_full[:ncap]
        print(f"[CAP] train ridotto a {Xtr.shape[0]} esempi per smoke test")
    else:
        Xtr, ytr = Xtr_full, ytr_full

    # Log dataset (stile run 1)
    print(f"[DATA] train={Xtr.shape} val={Xva.shape} test={(0, Xtr.shape[1])} features={Xtr.shape[1]}")

    # Diagnostica su X binaria
    sample_u = np.unique(Xtr[: min(2000, Xtr.shape[0])])
    if not np.all(np.isin(sample_u, [0, 1])):
        print(f"[WARN] train_X non è strettamente binario: { {int(x) for x in sample_u[:16]} } "
              f"(pyTsetlinMachine si aspetta 0/1). Verifica encoder/preprocess.")

    # Normalizzazione target + diagnostica clip (MAD, clip ±3 come nel run 1)
    mu, sigma = robust_fit_scale(ytr, method=args.norm_method)
    clip = args.norm_clip
    ytr_n = robust_transform(ytr, mu, sigma, clip)
    yva_n = robust_transform(yva, mu, sigma, clip)
    # diagnostica
    z = (ytr - mu) / (sigma if sigma != 0 else 1.0)
    clipped_ratio = float(np.mean(np.abs(z) >= (clip if clip is not None else 1e9)))
    print(f"[NORM] method={args.norm_method} mu={mu:.6f} sigma={sigma:.6f}"
          + ("" if clip is None else f" clip=±{clip:.1f}"))
    if clip is not None and clipped_ratio > 0.30:
        print(f"[HINT] clipped_ratio={clipped_ratio:.1%} → molti campioni saturano. "
              f"Per smoke test con pochi step prova --norm-method pct e/o --norm-clip 2.0")

    # Modello (come run (1) di default)
    model = RegressionTsetlinMachine(int(args.clauses), int(args.T), float(args.s))

    # Loop di training
    best_mae_raw, best_ep = float("inf"), -1  # Early stopping su MAE raw invece di normalizzata
    patience_left = int(args.patience)
    stagnant_epochs = 0  # Per adaptive lr
    current_s = float(args.s)  # Tracking per adaptive s
    n_train = Xtr.shape[0]
    eval_cap = min(int(args.train_eval_cap), n_train)

    # CSV metriche
    csv_f = None
    if args.metrics_csv:
        csv_f = open(args.metrics_csv, "w", buffering=1)
        csv_f.write("epoch,train_mae_norm,train_rmse_norm,train_mae_raw,train_rmse_raw,"
                    "val_mae_norm,val_rmse_norm,val_mae_raw,val_rmse_raw,seconds\n")

    try:
        for ep in range(1, int(args.epochs) + 1):
            t0 = time.time()
            idx = np.arange(n_train, dtype=np.int64)
            np.random.shuffle(idx)

            for i in tqdm(range(0, n_train, args.batch), desc=f"Epoch {ep}/{args.epochs}"):
                j = idx[i:i + args.batch]
                Xb, yb = Xtr[j], ytr_n[j]
                model.fit(Xb, yb, epochs=int(args.inner_epochs), incremental=True)

            # Valutazione (opzionalmente meno frequente per training lunghi)
            if ep % args.eval_freq == 0 or ep == args.epochs:
                tr = evaluate_with_raw(model, Xtr[:eval_cap], ytr_n[:eval_cap], ytr[:eval_cap],
                                       mu, sigma, batch=int(args.eval_batch))
                va = evaluate_with_raw(model, Xva, yva_n, yva, mu, sigma, batch=int(args.eval_batch))
                dt = time.time() - t0
                print(
                    f"[E{ep}] {dt:.1f}s  train MAE(norm)={tr['mae']:.5f} RMSE(norm)={tr['rmse']:.5f} "
                    f"MAE(raw)={tr['mae_raw']:.5f}  val MAE(norm)={va['mae']:.5f} RMSE(norm)={va['rmse']:.5f} "
                    f"MAE(raw)={va['mae_raw']:.5f}"
                    + (f" s={current_s:.2f}" if args.adaptive_lr else "")
                )

                if csv_f:
                    csv_f.write(
                        f"{ep},{tr['mae']:.6f},{tr['rmse']:.6f},{tr['mae_raw']:.6f},{tr['rmse_raw']:.6f},"
                        f"{va['mae']:.6f},{va['rmse']:.6f},{va['mae_raw']:.6f},{va['rmse_raw']:.6f},{dt:.2f}\n"
                    )

                # early-stopping + salvataggio best (PORTABILE) - basato su MAE raw
                if va["mae_raw"] < best_mae_raw - 1e-8:
                    best_mae_raw, best_ep = va["mae_raw"], ep
                    stagnant_epochs = 0  # Reset stagnation counter
                    meta_base = {
                        "clauses": int(args.clauses),
                        "T": int(args.T),
                        "s": float(current_s),  # Salva s corrente (può essere cambiato)
                        "best_val_mae": float(va["mae"]),  # norm per compatibilità
                        "best_val_mae_raw": float(best_mae_raw),  # questo è il criterio di early stop
                        "best_epoch": int(best_ep),
                        "dataset_attrs": _to_jsonable(ds_attrs),
                    }
                y_scaler = {"mu": float(mu), "sigma": float(sigma), "clip": clip}

                try:
                    save_rtm_state_portable(stem, model, meta_base, y_scaler)
                    print(
                        f"[SAVE] {stem.with_suffix('.state.npz').name} (+meta, yscaler, env)  "
                        f"val MAE_raw {best_mae_raw:.6f}"
                    )
                except Exception:
                    # fallback .rtm, se disponibile
                    try:
                        if hasattr(model, "save"):
                            model.save(str(stem.with_suffix(".rtm")))
                            meta = dict(meta_base); meta["format"] = "rtm"
                            stem.with_suffix(".meta.json").write_text(json.dumps(_to_jsonable(meta), indent=2))
                            stem.with_suffix(".yscaler.json").write_text(json.dumps(_to_jsonable(y_scaler), indent=2))
                            print(f"[SAVE] {stem.with_suffix('.rtm').name} (+meta, yscaler) — fallback")
                        else:
                            raise
                    except Exception as e2:
                        # ultima spiaggia: joblib
                        import joblib
                        joblib.dump({"model": model}, str(stem.with_suffix(".joblib")))
                        stem.with_suffix(".yscaler.json").write_text(json.dumps(_to_jsonable(y_scaler), indent=2))
                        print(
                            f"[SAVE] {stem.with_suffix('.joblib').name} (+yscaler) — fallback pickle [fragile]: {e2}"
                        )

                    patience_left = int(args.patience)
                else:
                    stagnant_epochs += 1
                    patience_left -= 1
                    
                    # Adaptive learning rate: riduci s se stagnante per 3+ epoche
                    if args.adaptive_lr and stagnant_epochs >= 3 and current_s > 2.0:
                        old_s = current_s
                        current_s = max(2.0, current_s - 0.1)  # Riduci s gradualmente, min 2.0
                        print(f"[ADAPTIVE] s: {old_s:.2f} -> {current_s:.2f} (stagnant={stagnant_epochs})")
                        stagnant_epochs = 0  # Reset dopo riduzione
                        
                    if patience_left <= 0:
                        print(
                            f"[EARLY STOP] nessun miglioramento val MAE_raw per {args.patience} epoche "
                            f"(best epoch {best_ep})."
                        )
                        break
            else:
                # Valutazione skippata, solo stampa progresso
                dt = time.time() - t0
                print(f"[E{ep}] {dt:.1f}s  (eval skipped, freq={args.eval_freq})")
    finally:
        if csv_f:
            csv_f.close()

if __name__ == "__main__":
    main()
