# File: src/classifier/train_ctm_classifier_multichannel.py
#!/usr/bin/env python3
"""
Addestra un Classificatore CTM Multi-Canale per identificare il rumore dominante.
"""
import os
import sys
import numpy as np
import time
import json
import pickle
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import argparse
import h5py

# Limita oversubscription dei thread (stabilità pyTM/TMU)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")

# --- Import e Setup (come prima) ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except Exception:
    pass

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
from src.config import CLASSIFIER_PIPELINE_PARAMS as PARAMS

# --- Backend utils (ripresi da trainer robusto) ---
import importlib

def _have_cuda() -> bool:
    return os.path.isdir("/usr/local/cuda") or os.path.exists("/proc/driver/nvidia/version")

def _import_tmu():
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
        return TMClassifier
    except Exception:
        return None

def _import_pyTM_conv2d():
    for mod in ("pyTsetlinMachine.tm", "pyTsetlinMachine.cTM",
                "pyTsetlinMachine.ctm", "pyTsetlinMachine.tsetlinmachine"):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "MultiClassConvolutionalTsetlinMachine2D")
        except Exception:
            pass
    return None

def _import_pyTM_flat():
    for mod in ("pyTsetlinMachine.tm", "pyTsetlinMachine.tsetlinmachine"):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "MultiClassTsetlinMachine")
        except Exception:
            pass
    return None

def _to_input(xs: np.ndarray, conv: bool) -> np.ndarray:
    if conv:
        return np.ascontiguousarray(xs.astype(np.uint32, copy=False))
    return np.ascontiguousarray(xs.reshape(xs.shape[0], -1).astype(np.uint32, copy=False))

def _eval_stream(model, X: np.ndarray, y: np.ndarray, conv: bool, batch: int = 2048):
    n = int(y.shape[0])
    y_true, y_pred = [], []
    for i in range(0, n, batch):
        xs = X[i:i + batch]
        yp = model.predict(_to_input(xs, conv))
        y_true.append(y[i:i + batch])
        y_pred.append(yp)
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    return float(accuracy_score(yt, yp)), float(f1_score(yt, yp, average="macro"))

# --- Configurazioni Directory ---
INPUT_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_ctm_multichannel')
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'ctm_classifier_multichannel')
LOGS_DIR_MC = os.path.join(LOGS_DIR, 'ctm_classifier_logs_multichannel')

# --- Helper ---
def now_ts():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dirs():
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR_MC, exist_ok=True)

# --- Funzioni Principali Aggiornate ---
def load_multichannel_data():
    h5_path = os.path.join(INPUT_DATA_DIR, "classifier_multichannel_data.h5")
    if not os.path.exists(h5_path):
        print(f"❌ ERRORE: File dati non trovato: {h5_path}. Esegui 'preprocess_ctm_multichannel.py'.")
        sys.exit(1)
    
    with h5py.File(h5_path, 'r') as hf:
        X_train = hf['train/X'][:]
        y_train = hf['train/y'][:]
        X_val = hf['validation/X'][:]
        y_val = hf['validation/y'][:]
        X_test = hf['test/X'][:]
        y_test = hf['test/y'][:]
    
    print("✅ Dati Multi-Canale caricati.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _try_build_tmu(TMClassifier, params: dict):
    sigs = [
        dict(params, append_negated=params.get("append_negated", False),
             weighted_clauses=params.get("weighted_clauses", True),
             type_i_ii_ratio=params.get("type_i_ii_ratio", 1.0)),
        dict(params, append_negated=params.get("append_negated", False),
             weighted_clauses=params.get("weighted_clauses", True)),
        dict(params, append_negated=params.get("append_negated", False)),
        dict(params),
    ]
    for p in sigs:
        try:
            return TMClassifier(**p)
        except Exception:
            continue
    return None

def evaluate_classifier(model, X, y, class_names, batch_size=512):
    n = X.shape[0]
    preds = []
    for i in range(0, n, batch_size):
        xb = np.ascontiguousarray(X[i:i+batch_size].astype(np.uint32, copy=False))
        preds.append(model.predict(xb))
    y_pred = np.concatenate(preds)
    acc = float(accuracy_score(y, y_pred))
    f1_macro = float(f1_score(y, y_pred, average='macro'))
    cm = confusion_matrix(y, y_pred, labels=list(range(len(class_names))))
    try:
        report = classification_report(y, y_pred, target_names=class_names, digits=4)
    except Exception:
        report = ""
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'confusion': cm,
        'report': report,
    }

def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(description='Train CTM Classifier (multicanale)')
    parser.add_argument('--epochs', type=int, default=int(PARAMS['trainer'].get('epochs', 40)))
    parser.add_argument('--patience', type=int, default=int(PARAMS['trainer'].get('patience', 8)))
    parser.add_argument('--val-eval-each', type=int, default=int(PARAMS['trainer'].get('val_eval_each', 1)))
    parser.add_argument('--batch-size', type=int, default=int(PARAMS['trainer']['model'].get('batch_size', 1024)))
    # quick-test & overrides
    parser.add_argument('--fast', action='store_true', help='Preset veloce: sottocampiona e riduce il modello')
    parser.add_argument('--max-train', type=int, default=None, help='Limite massimo campioni train')
    parser.add_argument('--max-val', type=int, default=None, help='Limite massimo campioni val')
    parser.add_argument('--max-test', type=int, default=None, help='Limite massimo campioni test')
    parser.add_argument('--clauses', type=int, default=None, help='Override number_of_clauses')
    parser.add_argument('--T', type=int, default=None, help='Override T')
    parser.add_argument('--s', type=float, default=None, help='Override s')
    parser.add_argument('--patch-w', type=int, default=None, help='Override patch width')
    parser.add_argument('--patch-h', type=int, default=None, help='Override patch height')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-subset', type=int, default=None, help='Usa solo i primi K campioni di validation per early stopping')
    parser.add_argument('--balance-train', action='store_true', help='Bilancia per classe i campioni usati in ogni epoca')
    parser.add_argument('--save-best', action='store_true', help='Pickle del modello al miglior F1 (può rallentare)')
    args = parser.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_multichannel_data()
    # Sottocampionamento per test veloci
    if args.fast:
        args.max_train = args.max_train or 1200
        args.max_val = args.max_val or 300
        args.max_test = args.max_test or 300
    if args.max_train is not None:
        X_train = X_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_val is not None:
        X_val = X_val[: args.max_val]
        y_val = y_val[: args.max_val]
    if args.max_test is not None:
        X_test = X_test[: args.max_test]
        y_test = y_test[: args.max_test]
    
    # Inferisci parametri dai dati (stack verticale: X.shape=(N, H, L))
    H_bins = int(X_train.shape[1])
    patch_w = int(PARAMS['trainer']['model'].get('patch_w', 31)) if args.patch_w is None else int(args.patch_w)
    patch_h = int(PARAMS['trainer']['model'].get('patch_h', min(64, H_bins))) if args.patch_h is None else int(args.patch_h)
    number_of_clauses = int(PARAMS['trainer']['model'].get('number_of_clauses', 1800)) if args.clauses is None else int(args.clauses)
    T = int(PARAMS['trainer']['model'].get('T', 800)) if args.T is None else int(args.T)
    s = float(PARAMS['trainer']['model'].get('s', 4.25)) if args.s is None else float(args.s)

    # Preset FAST: modello più piccolo e meno epoche
    if args.fast:
        number_of_clauses = min(number_of_clauses, 400)
        T = min(T, 200)
        s = float(s)
        patch_w = min(patch_w, 15)
        args.epochs = min(int(args.epochs), 5)
        args.patience = min(int(args.patience), 2)

    # Costruzione backend sicura (conv preferito, fallback flat)
    TMU = _import_tmu()
    PyConv = _import_pyTM_conv2d()
    PyFlat = _import_pyTM_flat()

    use_conv = False
    model = None
    platform = "CUDA" if _have_cuda() else "CPU"
    L = int(X_train.shape[2])

    # 1) TMU conv
    if TMU is not None:
        params = dict(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            platform=platform,
            patch_dim=(min(H_bins, patch_h), min(patch_w, L)),
            append_negated=False,
            weighted_clauses=True,
            type_i_ii_ratio=1.0,
        )
        model = _try_build_tmu(TMU, params)
        if model is not None:
            use_conv = True
            print(f"[INFO] Using TMU (conv) on {platform}")

    # 2) pyTM conv
    if model is None and PyConv is not None:
        try:
            model = PyConv(number_of_clauses, T, s, patch_dim=(min(H_bins, patch_h), min(patch_w, L)), append_negated=True)
            use_conv = True
            print("[INFO] Using pyTsetlinMachine Conv2D (CPU)")
        except Exception as e:
            print("[WARN] pyTM Conv2D init failed:", e)

    # 3) TMU flat
    if model is None and TMU is not None:
        params = dict(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            platform=platform,
            append_negated=False,
            weighted_clauses=True,
            type_i_ii_ratio=1.0,
        )
        model = _try_build_tmu(TMU, params)
        if model is not None:
            use_conv = False
            print(f"[INFO] Using TMU (flat) on {platform}")

    # 4) pyTM flat
    if model is None and PyFlat is not None:
        try:
            model = PyFlat(number_of_clauses, T, s, append_negated=False)
            use_conv = False
            print("[INFO] Falling back to pyTsetlinMachine Flat")
        except Exception as e:
            raise RuntimeError("No TM backend available: pyTM Flat init failed") from e

    # Warm-up su pochi campioni per classe
    y_tr_u = y_train.astype(np.uint32, copy=False)
    n_classes = int(np.max(y_tr_u)) + 1
    warm_idx = []
    for c in range(n_classes):
        idx = np.where(y_tr_u == c)[0]
        if idx.size:
            warm_idx.append(idx[:min(3, idx.size)])
    if warm_idx:
        warm_idx = np.concatenate(warm_idx)
        try:
            model.fit(_to_input(X_train[warm_idx], use_conv), y_tr_u[warm_idx], epochs=1, incremental=True)
            print(f"[WARMUP] Done on {warm_idx.size} samples across {n_classes} classes")
        except Exception as e:
            print("[WARN] Warm-up failed (continuing):", e)

    # Training loop con early stopping (batch streaming)
    np.random.seed(args.seed)
    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_left = int(args.patience)

    N = X_train.shape[0]
    # Indici per bilanciamento per epoca (opzionale)
    classes = np.unique(y_train)
    class_to_idx = {int(c): np.where(y_train == c)[0] for c in classes}
    epochs = int(args.epochs)

    print(f"🚀 Train start: epochs={epochs}, batch_size={args.batch_size}, patience={args.patience}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"[EPOCH {epoch:03d}] training on {N} samples (batch={args.batch_size})...")
        if args.balance_train:
            # prendi lo stesso numero per classe (min disponibile)
            per_class = [ids[np.random.permutation(ids.size)] for ids in class_to_idx.values() if ids.size > 0]
            min_n = int(min(arr.size for arr in per_class)) if per_class else 0
            sel = []
            for arr in per_class:
                sel.append(arr[:min_n])
            idx_epoch = np.concatenate(sel) if sel else np.arange(N)
            np.random.shuffle(idx_epoch)
        else:
            idx_epoch = np.random.permutation(N)

        total_batches = int(np.ceil(idx_epoch.size / args.batch_size))
        for bi in tqdm(range(0, idx_epoch.size, args.batch_size), total=total_batches, desc=f"E{epoch:03d} train", leave=False):
            idx = idx_epoch[bi:bi + args.batch_size]
            xs = X_train[idx]
            ys = y_train[idx]
            ys_u32 = np.asarray(ys, dtype=np.uint32)
            model.fit(_to_input(xs, use_conv), ys_u32, epochs=1, incremental=True)

        train_time = time.time() - t0
        v0 = time.time()
        if epoch % max(1, args.val_eval_each) == 0:
            if args.val_subset is not None and args.val_subset > 0:
                Xv = X_val[: args.val_subset]
                yv = y_val[: args.val_subset]
            else:
                Xv = X_val; yv = y_val
            acc, f1 = _eval_stream(model, Xv, yv.astype(np.uint32, copy=False), use_conv, batch=args.batch_size)
            val_time = time.time() - v0
            print(f"[E{epoch:02d}] val acc={acc:.4f} f1={f1:.4f} | train={train_time:.1f}s val={val_time:.1f}s")
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                if args.save_best:
                    try:
                        best_state = pickle.dumps(model)
                    except Exception:
                        best_state = None
                patience_left = int(args.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("⏹️  Early stopping attivato.")
                    break

    # Ripristina best
    if best_state is not None:
        try:
            ctm = pickle.loads(best_state)
            print(f"✅ Ripristinato best epoch {best_epoch} (val f1={best_f1:.4f})")
        except Exception:
            pass

    # Valutazione finale
    va_acc, va_f1 = _eval_stream(model, X_val, y_val.astype(np.uint32, copy=False), use_conv, batch=args.batch_size)
    te_acc, te_f1 = _eval_stream(model, X_test, y_test.astype(np.uint32, copy=False), use_conv, batch=args.batch_size)
    print("\n📊 TEST:")
    print(f"VAL acc={va_acc:.4f} f1={va_f1:.4f}")
    print(f"TEST acc={te_acc:.4f} f1={te_f1:.4f}")

    # Salvataggi
    run_id = now_ts()
    model_path = os.path.join(TRAINED_MODELS_DIR, f"ctm_classifier_multichannel_{run_id}.pkl")
    # Confusion + report
    y_true, y_pred = [], []
    for i in range(0, int(y_test.shape[0]), args.batch_size):
        xs = X_test[i:i + args.batch_size]
        yp = model.predict(_to_input(xs, use_conv))
        y_true.append(y_test[i:i + args.batch_size])
        y_pred.append(yp)
    yte_all = np.concatenate(y_true)
    ype_all = np.concatenate(y_pred)
    cm = confusion_matrix(yte_all, ype_all)

    # Salvataggi
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    np.save(os.path.join(TRAINED_MODELS_DIR, f"confusion_{run_id}.npy"), cm)
    with open(os.path.join(TRAINED_MODELS_DIR, f"metrics_{run_id}.json"), 'w') as f:
        json.dump({
            'val_best_f1': best_f1,
            'val_best_epoch': best_epoch,
            'val_accuracy': va_acc,
            'val_f1_macro': va_f1,
            'test_accuracy': te_acc,
            'test_f1_macro': te_f1,
            'params': {
                'number_of_clauses': number_of_clauses,
                'T': T,
                's': s,
                'patch_w': patch_w,
                'H_bins': H_bins,
                'epochs': int(args.epochs),
                'patience': int(args.patience),
                'batch_size': int(args.batch_size),
            }
        }, f, indent=2)
    print(f"💾 Modello salvato: {model_path}")

if __name__ == "__main__":
    main()