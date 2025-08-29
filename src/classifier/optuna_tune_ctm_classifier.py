# ============================================
# File: optuna_tune_ctm_classifier.py
# Pipeline step: HDF5 -> hyperparameter search -> best model
# ============================================
"""
Ricerca iperparametri con Optuna per il classificatore TM/CTM.
- Obiettivo: massimizzare macro-F1 su validation.
- Supporta backend conv2d (pyTsetlinMachine) e flat (fallback), selezionati via spazio di ricerca.
- Training in streaming HDF5 (no OOM), con early stopping a pazienza configurabile.
- Pruning intermedio con MedianPruner.

Esempio d'uso:

    # installa optuna se serve
    # pip install optuna

    python src/classifier/optuna_tune_ctm_classifier.py \
      --h5 denoising_ecg/data/bin/classifier_patches.h5 \
      --out_dir denoising_ecg/data/models/optuna_runs/ctm_search \
      --n_trials 30 --backend_space both --epochs 8 --patience 3

Salva:
- optuna.db (sqlite) + best_params.json
- best_metrics.json (val/test) e tm_classifier.pkl del best model (se picklable).
"""

import os
import json
import argparse
import importlib
import pickle
import random
from typing import Tuple, Optional

import numpy as np
import h5py
import optuna
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# --- dynamic imports (come nel trainer) ---

def _import_pyTM_flat():
    for mod, cls in (
        ("pyTsetlinMachine.tm", "MultiClassTsetlinMachine"),
        ("pyTsetlinMachine.tsetlinmachine", "MultiClassTsetlinMachine"),
    ):
        try:
            m = importlib.import_module(mod)
            return getattr(m, cls)
        except Exception:
            continue
    return None


def _import_pyTM_conv2d():
    for mod, cls in (
        ("pyTsetlinMachine.tm", "MultiClassConvolutionalTsetlinMachine2D"),
        ("pyTsetlinMachine.cTM", "MultiClassConvolutionalTsetlinMachine2D"),
        ("pyTsetlinMachine.ctm", "MultiClassConvolutionalTsetlinMachine2D"),
        ("pyTsetlinMachine.tsetlinmachine", "MultiClassConvolutionalTsetlinMachine2D"),
    ):
        try:
            m = importlib.import_module(mod)
            return getattr(m, cls)
        except Exception:
            continue
    return None


def _import_tmu():
    try:
        tm = importlib.import_module("tmu.models.classification.vanilla_classifier")
        conv = importlib.import_module("tmu.models.classification.conv_classifier")
        Flat = getattr(tm, "TMClassifier")
        Conv = getattr(conv, "ConvClassifier")
        return Flat, Conv
    except Exception:
        return None, None


# --- data utils ---

def _get_split(h5: h5py.File, split: str):
    return h5[f"{split}/X"], h5[f"{split}/y"]


def _to_tm_input(xs, conv: bool):
    if conv:
        Xin = xs.astype(np.uint32, copy=False)
    else:
        Xin = xs.reshape(xs.shape[0], -1).astype(np.uint32, copy=False)
    return np.ascontiguousarray(Xin)


def _evaluate_stream(model, Xds, yds, conv: bool, batch_size: int = 1024) -> Tuple[float, float]:
    n = int(yds.shape[0])
    y_true = []
    y_pred = []
    for i in range(0, n, batch_size):
        xs = Xds[i:i+batch_size]
        ys = yds[i:i+batch_size]
        Xin = _to_tm_input(xs, conv)
        yp = model.predict(Xin)
        y_true.append(ys)
        y_pred.append(yp)
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    acc = float(accuracy_score(yt, yp))
    f1 = float(f1_score(yt, yp, average="macro"))
    return acc, f1


# --- build & train ---

def _build_model(backend: str, C: int, L: int, clauses: int, T: int, s: float, patch_w: int, append_negated: bool):
    FlatTM = _import_pyTM_flat()
    ConvTM = _import_pyTM_conv2d()
    TMUFlat, TMUConv = _import_tmu()

    if backend == "conv":
        if ConvTM is None:
            raise RuntimeError("Conv2D backend non disponibile (pyTsetlinMachine Conv2D non importabile)")
        patch_dim = (C, min(patch_w, L))
        model = ConvTM(clauses, T, s, patch_dim=patch_dim, append_negated=append_negated)
        return model, True
    elif backend == "flat":
        if FlatTM is not None:
            model = FlatTM(clauses, T, s, append_negated=append_negated)
            return model, False
        elif TMUFlat is not None:
            model = TMUFlat(clauses, T, s, weighted_clauses=False)
            return model, False
        else:
            raise RuntimeError("Flat TM non disponibile (pyTsetlinMachine/tmu mancanti)")
    else:
        raise ValueError("backend sconosciuto")


def _train_eval(h5_path: str, params: dict, epochs: int, patience: int, batch_size: int, val_each: int, seed: int) -> Tuple[float, float, Optional[bytes], bool]:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    with h5py.File(h5_path, "r") as h5:
        Xtr, ytr = _get_split(h5, "train")
        Xva, yva = _get_split(h5, "val")
        C = int(Xtr.shape[1]); L = int(Xtr.shape[2])
        model, use_conv = _build_model(params["backend"], C, L, params["clauses"], params["T"], params["s"], params.get("patch_w", 63), params["append_negated"])
        n_train = int(ytr.shape[0])

        best_f1 = -1.0
        best_state = None
        best_epoch = -1
        patience_left = patience

        for epoch in range(1, epochs+1):
            for i in range(0, n_train, batch_size):
                xs = Xtr[i:i+batch_size]
                ys = ytr[i:i+batch_size]
                Xin = _to_tm_input(xs, use_conv)
                model.fit(Xin, ys, epochs=1, incremental=True)
            if epoch % val_each == 0:
                _, f1 = _evaluate_stream(model, Xva, yva, conv=use_conv, batch_size=batch_size)
                if f1 > best_f1:
                    best_f1 = f1; best_epoch = epoch
                    try:
                        best_state = pickle.dumps(model)
                    except Exception:
                        best_state = None
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
        return best_f1, float(best_epoch), best_state, use_conv


# --- Optuna objective ---

def _objective(trial: optuna.trial.Trial, h5_path: str, backend_space: str, max_epochs: int, patience: int, seed: int) -> float:
    # backend choice
    if backend_space == "both":
        backend = trial.suggest_categorical("backend", ["conv", "flat"])
    else:
        backend = backend_space
        trial.set_user_attr("backend", backend)

    append_negated = trial.suggest_categorical("append_negated", [True, False])
    clauses = trial.suggest_int("clauses", 800, 4000, step=200)
    s = trial.suggest_float("s", 3.0, 9.0, step=0.5)

    # T proporzionato alle clausole
    T_min = max(200, int(0.5 * clauses))
    T_max = int(2.0 * clauses)
    T = trial.suggest_int("T", T_min, T_max, step=100)

    # batch & patch
    if backend == "conv":
        batch = trial.suggest_categorical("batch_size", [512, 1024])
        patch_w = trial.suggest_categorical("patch_w", [31, 63, 127])
    else:
        batch = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
        patch_w = 63  # irrilevante per flat

    params = dict(backend=backend, clauses=clauses, T=T, s=s, patch_w=patch_w, append_negated=append_negated)

    # train breve con pruning (MedianPruner gestito a livello di study)
    best_f1, best_epoch, _, _ = _train_eval(h5_path, params, epochs=max_epochs, patience=patience, batch_size=batch, val_each=1, seed=seed)
    trial.report(best_f1, int(best_epoch) if best_epoch > 0 else 1)
    return best_f1


# --- main ---

def main():
    ap = argparse.ArgumentParser(description="Optuna tuning for TM/CTM classifier")
    ap.add_argument("--h5", type=str, default="denoising_ecg/data/bin/classifier_patches.h5")
    ap.add_argument("--out_dir", type=str, default="denoising_ecg/data/models/optuna_runs/ctm_search")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None, help="Timeout in secondi (opzionale)")
    ap.add_argument("--backend_space", choices=["both","conv","flat"], default="both")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--final_epochs", type=int, default=25)
    ap.add_argument("--final_patience", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # riduci rumore performance
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(k, "1")

    storage = f"sqlite:///{os.path.join(args.out_dir, 'optuna.db')}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(direction="maximize",
                                sampler=sampler,
                                pruner=pruner,
                                storage=storage,
                                study_name="tm_ctm_noisecls",
                                load_if_exists=True)

    study.optimize(lambda t: _objective(t, args.h5, args.backend_space, args.epochs, args.patience, args.seed),
                   n_trials=args.n_trials, timeout=args.timeout)

    print("Best trial:", study.best_trial.number)
    print("Best value (val macro-F1):", study.best_value)
    print("Best params:")
    for k,v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # salva best params
    best_params = study.best_trial.params.copy()
    best_params["backend"] = best_params.get("backend", ("conv" if args.backend_space in ("conv","both") else "flat"))
    with open(os.path.join(args.out_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # retrain finale con più epoche e salvataggio modello + metriche
    print("\n[FINAL TRAIN] Using best params…")
    batch = best_params.get("batch_size", 1024)
    f1, epoch, state, use_conv = _train_eval(args.h5, best_params,
                                             epochs=args.final_epochs,
                                             patience=args.final_patience,
                                             batch_size=batch, val_each=1,
                                             seed=args.seed)

    # ricrea modello e ricarica lo stato migliore (se possibile) per test finale
    with h5py.File(args.h5, "r") as h5:
        C = int(h5["train/X"].shape[1]); L = int(h5["train/X"].shape[2])
    model, use_conv = _build_model(best_params.get("backend","flat"), C, L,
                                   best_params["clauses"], best_params["T"], best_params["s"],
                                   best_params.get("patch_w",63), best_params["append_negated"])
    if state is not None:
        try:
            model = pickle.loads(state)
        except Exception:
            pass

    with h5py.File(args.h5, "r") as h5:
        Xva, yva = _get_split(h5, "val")
        Xte, yte = _get_split(h5, "test")
        va_acc, va_f1 = _evaluate_stream(model, Xva, yva, conv=use_conv, batch_size=batch)
        te_acc, te_f1 = _evaluate_stream(model, Xte, yte, conv=use_conv, batch_size=batch)

        # confusion test
        y_true = []; y_pred = []
        n_test = int(yte.shape[0])
        for i in range(0, n_test, batch):
            xs = Xte[i:i+batch]; ys = yte[i:i+batch]
            yp = model.predict(_to_tm_input(xs, use_conv))
            y_true.append(ys); y_pred.append(yp)
        yte_all = np.concatenate(y_true); ype_all = np.concatenate(y_pred)
        cm = confusion_matrix(yte_all, ype_all)

    # salva modello e metriche
    model_path = os.path.join(args.out_dir, "tm_classifier.pkl")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("Saved model ->", model_path)
    except Exception as e:
        print("Model not picklable:", e)

    metrics = {
        "backend": best_params.get("backend"),
        "val_acc": va_acc, "val_f1": va_f1,
        "test_acc": te_acc, "test_f1": te_f1,
        "best_params": best_params,
        "final_epochs": args.final_epochs,
    }
    with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), cm)
    print("Saved metrics ->", os.path.join(args.out_dir, "best_metrics.json"))


if __name__ == "__main__" and os.path.basename(__file__) == "optuna_tune_ctm_classifier.py":
    main()
