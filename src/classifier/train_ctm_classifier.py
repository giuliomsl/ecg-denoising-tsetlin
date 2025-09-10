# ============================================
# File: src/classifier/train_ctm_classifier.py
# Pipeline: preprocess -> train
# ============================================
"""
Classifier training (TMU-first, pyTM fallback) on HDF5 produced by preprocess.

Input HDF5 (by preprocess):
  - {train,val,test}/X : uint8 {0,1} shape (N, C, L)
  - {train,val,test}/y : uint32 shape (N,)

Backends priority:
  1) TMU TMClassifier (conv if patch_dim provided, else flat) on CUDA/CPU
  2) pyTsetlinMachine Conv2D
  3) pyTsetlinMachine Flat

Key features:
  - Robust TMU instantiation (tries multiple signatures, falls back if needed)
  - Warm-up on few samples per class (prevents weight-bank NoneType issues)
  - uint32 labels enforced end-to-end; remap non [0..K-1] if needed
  - Early stopping on macro-F1
  - Safe model pickling (best-epoch snapshot if supported)


Example:

python src/classifier/train_ctm_classifier.py \
  --h5 ./data/bin/classifier_patches.h5 \
  --out_dir ./data/models/ctm_classifier_tmu \
  --backend conv --clauses 2400 --T 1200 --s 7.0 --patch_w 63 \
  --append_negated --weighted_clauses --type_i_ii_ratio 1.0 \
  --epochs 30 --patience 8 --val_eval_each 1 --batch_size 4096


"""

from __future__ import annotations
import os
import json
import time
import argparse
import importlib
import pickle
import random
from typing import Tuple, Optional, Dict

import numpy as np
import h5py
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Limit thread oversubscription (important on BLAS-linked environments)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")


# --------------------- backend imports & utils ---------------------

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
    """Prepare X batch for backend: (N,C,L)->uint32 if conv, else flatten."""
    if conv:
        return np.ascontiguousarray(xs.astype(np.uint32, copy=False))
    return np.ascontiguousarray(xs.reshape(xs.shape[0], -1).astype(np.uint32, copy=False))


def _eval_stream(model, X: np.ndarray, y: np.ndarray, conv: bool, batch: int = 2048) -> Tuple[float, float]:
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


# -------------------------- TMU builder -----------------------------

def _try_build_tmu(TMClassifier, params: Dict) -> Optional[object]:
    """Try building TMU with decreasingly-ambitious signatures."""
    # 1) Most featured (may fail on some versions)
    sigs = [
        dict(params, append_negated=params.get("append_negated", False),
             weighted_clauses=params.get("weighted_clauses", True),
             type_i_ii_ratio=params.get("type_i_ii_ratio", 1.0)),
        dict(params, append_negated=params.get("append_negated", False),
             weighted_clauses=params.get("weighted_clauses", True)),
        dict(params, append_negated=params.get("append_negated", False)),
        dict(params),  # minimal
    ]
    for i, p in enumerate(sigs, 1):
        try:
            return TMClassifier(**p)
        except TypeError as e:
            # unsupported kwargs -> try a simpler signature
            continue
        except Exception:
            continue
    return None


# ----------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train TMU/pyTM ECG noise classifier")
    ap.add_argument("--h5", type=str, default="./data/bin/classifier_patches.h5")
    ap.add_argument("--out_dir", type=str, default="./data/models/ctm_classifier")

    # model & training
    ap.add_argument("--backend", choices=["auto", "conv", "flat"], default="auto")
    ap.add_argument("--clauses", type=int, default=2400)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--s", type=float, default=7.0)
    ap.add_argument("--patch_w", type=int, default=63)
    ap.add_argument("--append_negated", action="store_true", help="Enable negated literals duplication (if supported)")
    ap.add_argument("--weighted_clauses", action="store_true", help="Enable weighted clauses (TMU only, if supported)")
    ap.add_argument("--type_i_ii_ratio", type=float, default=1.0, help="TMU type I/II feedback ratio (if supported)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val_eval_each", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load data fully in RAM (fastest for TMU/pyTM)
    with h5py.File(args.h5, "r") as h5:
        Xtr = h5["train/X"][:]
        ytr = h5["train/y"][:]
        Xva = h5["val/X"][:]
        yva = h5["val/y"][:]
        Xte = h5["test/X"][:]
        yte = h5["test/y"][:]
    # ensure types & shape
    ytr = np.asarray(ytr, dtype=np.uint32).ravel()
    yva = np.asarray(yva, dtype=np.uint32).ravel()
    yte = np.asarray(yte, dtype=np.uint32).ravel()

    # remap labels if not [0..K-1]
    uniq = np.unique(ytr)
    mapping = None
    if not np.array_equal(uniq, np.arange(uniq.size, dtype=np.uint32)):
        mapping = {int(u): int(i) for i, u in enumerate(uniq.tolist())}
        remap = np.vectorize(lambda z: mapping[int(z)], otypes=[np.uint32])
        ytr = remap(ytr); yva = remap(yva); yte = remap(yte)
        print("[WARN] Labels remapped:", mapping)

    C = int(Xtr.shape[1]); L = int(Xtr.shape[2])
    n_classes = int(np.max(ytr)) + 1

    # backends
    TMU = _import_tmu()
    PyConv = _import_pyTM_conv2d()
    PyFlat = _import_pyTM_flat()

    use_conv = False
    model = None
    platform = "CUDA" if _have_cuda() else "CPU"

    # ---- 1) TMU conv (preferred when backend=auto/conv) ----
    if (args.backend in ("auto", "conv")) and TMU is not None:
        params = dict(
            number_of_clauses=args.clauses,
            T=args.T,
            s=args.s,
            platform=platform,
            patch_dim=(C, min(args.patch_w, L)),
            append_negated=args.append_negated,
            weighted_clauses=args.weighted_clauses,
            type_i_ii_ratio=args.type_i_ii_ratio,
        )
        model = _try_build_tmu(TMU, params)
        if model is not None:
            use_conv = True
            print(f"[INFO] Using TMU (conv) on {platform}")

    # ---- 2) pyTM Conv2D ----
    if model is None and (args.backend in ("auto", "conv")) and PyConv is not None:
        try:
            model = PyConv(args.clauses, args.T, args.s, patch_dim=(C, min(args.patch_w, L)),
                           append_negated=args.append_negated)
            use_conv = True
            print("[INFO] Using pyTsetlinMachine Conv2D (CPU)")
        except Exception as e:
            print("[WARN] pyTM Conv2D init failed:", e)

    # ---- 3) TMU flat ----
    if model is None and TMU is not None and args.backend in ("auto", "flat"):
        params = dict(
            number_of_clauses=args.clauses,
            T=args.T,
            s=args.s,
            platform=platform,
            append_negated=args.append_negated,
            weighted_clauses=args.weighted_clauses,
            type_i_ii_ratio=args.type_i_ii_ratio,
        )
        model = _try_build_tmu(TMU, params)
        if model is not None:
            use_conv = False
            print(f"[INFO] Using TMU (flat) on {platform}")

    # ---- 4) pyTM Flat ----
    if model is None and PyFlat is not None:
        try:
            model = PyFlat(args.clauses, args.T, args.s, append_negated=args.append_negated)
            use_conv = False
            print("[INFO] Falling back to pyTsetlinMachine Flat")
        except Exception as e:
            model = None
            print("[ERROR] pyTM Flat init failed:", e)

    if model is None:
        raise RuntimeError("No TM backend available (TMU/pyTM). Install TMU>=0.8.x or pyTsetlinMachine.")

    # ---- warm-up: few samples per class (prevents NoneType in weight banks on some TMU builds) ----
    warm_idx = []
    for c in range(n_classes):
        idx = np.where(ytr == c)[0]
        if idx.size:
            warm_idx.append(idx[:min(3, idx.size)])
    if warm_idx:
        warm_idx = np.concatenate(warm_idx)
        try:
            model.fit(_to_input(Xtr[warm_idx], use_conv), ytr[warm_idx], epochs=1, incremental=True)
            print(f"[WARMUP] Done on {warm_idx.size} samples across {n_classes} classes")
        except Exception as e:
            print("[WARN] Warm-up failed (continuing):", e)

    # ---------------- training loop ----------------
    best_f1, best_epoch, best_state = -1.0, -1, None
    patience_left = int(args.patience)
    N = int(ytr.shape[0])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(N)
        Xtr = Xtr[perm]; ytr = ytr[perm]

        # streaming batches
        for i in range(0, N, args.batch_size):
            xs = Xtr[i:i + args.batch_size]
            ys = ytr[i:i + args.batch_size]
            ys_u32 = np.asarray(ys, dtype=np.uint32)  # TMU requires uint32; harmless for pyTM
            model.fit(_to_input(xs, use_conv), ys_u32, epochs=1, incremental=True)

        epoch_time = time.time() - t0

        if epoch % args.val_eval_each == 0:
            acc, f1 = _eval_stream(model, Xva, yva, use_conv, batch=args.batch_size)
            print(f"[EPOCH {epoch:03d}] val acc={acc:.4f}  f1={f1:.4f}  | time={epoch_time:.1f}s")
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                try:
                    best_state = pickle.dumps(model)
                except Exception:
                    best_state = None
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("[INFO] Early stopping triggered.")
                    break

    if best_state is not None:
        try:
            model = pickle.loads(best_state)
            print(f"[INFO] Restored best epoch {best_epoch} (val F1={best_f1:.4f})")
        except Exception:
            pass

    # ---------------- final evaluation ----------------
    va_acc, va_f1 = _eval_stream(model, Xva, yva, use_conv, batch=args.batch_size)
    te_acc, te_f1 = _eval_stream(model, Xte, yte, use_conv, batch=args.batch_size)
    print(f"VAL:  acc={va_acc:.4f}  f1={va_f1:.4f}")
    print(f"TEST: acc={te_acc:.4f}  f1={te_f1:.4f}")

    # Confusion matrix & report
    y_true, y_pred = [], []
    for i in range(0, int(yte.shape[0]), args.batch_size):
        xs = Xte[i:i + args.batch_size]
        yp = model.predict(_to_input(xs, use_conv))
        y_true.append(yte[i:i + args.batch_size]); y_pred.append(yp)
    yte_all = np.concatenate(y_true); ype_all = np.concatenate(y_pred)
    cm = confusion_matrix(yte_all, ype_all)
    print("Confusion matrix (test):")
    print(cm)
    try:
        print(classification_report(yte_all, ype_all, digits=4))
    except Exception:
        pass

    # ---------------- save artifacts ----------------
    backend_tag = (
        f"tmu-{'conv' if use_conv else 'flat'}({ 'CUDA' if _have_cuda() else 'CPU' })"
        if _import_tmu() and isinstance(model, object) and "tmu" in type(model).__module__
        else ("pytm-conv" if use_conv else "pytm-flat")
    )

    metrics = {
        "backend": backend_tag,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "val_acc": va_acc, "val_f1": va_f1,
        "test_acc": te_acc, "test_f1": te_f1,
        "clauses": int(args.clauses), "T": int(args.T), "s": float(args.s),
        "patch_w": int(args.patch_w),
        "append_negated": bool(args.append_negated),
        "weighted_clauses": bool(args.weighted_clauses),
        "type_i_ii_ratio": float(args.type_i_ii_ratio),
        "batch_size": int(args.batch_size),
        "h5": os.path.abspath(args.h5),
        "best_epoch": int(best_epoch),
        "label_remap": (mapping if mapping is not None else None),
        "C": int(C), "L": int(L),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), cm)

    # Try to persist full model (may fail on some TMU builds)
    try:
        with open(os.path.join(args.out_dir, "tm_classifier.pkl"), "wb") as f:
            pickle.dump(model, f)
        print("[INFO] Saved model ->", os.path.join(args.out_dir, "tm_classifier.pkl"))
    except Exception as e:
        print("[WARN] Model not picklable:", e)


if __name__ == "__main__" and os.path.basename(__file__) == "train_ctm_classifier.py":
    main()
