#!/usr/bin/env python3
"""
Trainer CTM per dataset contrastivo (waveform + spectral).
Obiettivo: imparare bene i 3 rumori singoli prima, poi estendere ai mix.
Include:
- tqdm per avanzamento training
- timing per epoca e validazione
- salvataggio plot (acc/F1 per epoca) e matrice di confusione
"""
import os, sys, json, time
import numpy as np
import argparse
import h5py
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
except Exception:
    pass

from src.config import CLASSIFIER_PIPELINE_PARAMS as PARAMS

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'preprocessed_ctm_contrastive')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs', 'ctm_contrastive_logs')
os.makedirs(LOGS_DIR, exist_ok=True)

def _import_pyTM_conv2d():
    import importlib
    for mod in ("pyTsetlinMachine.tm", "pyTsetlinMachine.cTM", "pyTsetlinMachine.ctm", "pyTsetlinMachine.tsetlinmachine"):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "MultiClassConvolutionalTsetlinMachine2D")
        except Exception:
            pass
    return None

def _to_input(xs):
    return np.ascontiguousarray(xs.astype(np.uint32, copy=False))

def _eval(model, X, y, batch=2048):
    yt, yp = [], []
    for i in range(0, X.shape[0], batch):
        xb = _to_input(X[i:i+batch])
        yp.append(model.predict(xb))
        yt.append(y[i:i+batch])
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return float(accuracy_score(yt, yp)), float(f1_score(yt, yp, average='macro'))

def load_data(single_noise_only=False):
    path = os.path.join(INPUT_DIR, 'classifier_contrastive_data.h5')
    if not os.path.exists(path):
        raise SystemExit(f"Dati contrastivi non trovati: {path}")
    with h5py.File(path, 'r') as hf:
        Xtr = hf['train/X'][:]; ytr = hf['train/y'][:]
        Xva = hf['validation/X'][:]; yva = hf['validation/y'][:]
        Xte = hf['test/X'][:]; yte = hf['test/y'][:]
    if single_noise_only:
        # tieni classi 0..3 (escludi MIXED=4)
        keep = ytr < 4
        Xtr, ytr = Xtr[keep], ytr[keep]
        keep = yva < 4
        Xva, yva = Xva[keep], yva[keep]
        keep = yte < 4
        Xte, yte = Xte[keep], yte[keep]
    return (Xtr, ytr), (Xva, yva), (Xte, yte)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--clauses', type=int, default=1200)
    ap.add_argument('--T', type=int, default=600)
    ap.add_argument('--s', type=float, default=4.5)
    ap.add_argument('--patch-w', type=int, default=31)
    ap.add_argument('--single-noise', action='store_true', help='Escludi MIXED per training iniziale')
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()

    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_data(single_noise_only=args.single_noise)
    if args.fast:
        Xtr, ytr = Xtr[:2000], ytr[:2000]
        Xva, yva = Xva[:500], yva[:500]
        Xte, yte = Xte[:500], yte[:500]

    H, L = Xtr.shape[1], Xtr.shape[2]
    from importlib import import_module
    PyConv = _import_pyTM_conv2d()
    if PyConv is None:
        raise SystemExit("pyTsetlinMachine con MultiClassConvolutionalTsetlinMachine2D non disponibile")
    model = PyConv(args.clauses, args.T, args.s, patch_dim=(min(H, 48), min(args.patch_w, L)), append_negated=True)
    print("[INFO] Backend: pyTM Conv2D")

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    hist = { 'epoch': [], 'val_acc': [], 'val_f1': [], 'train_sec': [], 'val_sec': [] }

    best_f1 = -1.0; patience = args.patience
    N = Xtr.shape[0]
    print(f"🚀 Train start: epochs={args.epochs}, batch_size={args.batch_size}")
    for ep in range(1, args.epochs+1):
        ep_t0 = time.time()
        print(f"[EPOCH {ep:03d}] training on {N} samples (batch={args.batch_size})...")
        idx = np.random.permutation(N)
        total_batches = int(np.ceil(N / args.batch_size))
        for i in tqdm(range(0, N, args.batch_size), total=total_batches, desc=f"E{ep:03d} train", leave=False):
            j = idx[i:i+args.batch_size]
            model.fit(_to_input(Xtr[j]), ytr[j].astype(np.uint32), epochs=1, incremental=True)
        train_sec = time.time() - ep_t0
        v0 = time.time()
        acc, f1 = _eval(model, Xva, yva, batch=args.batch_size)
        val_sec = time.time() - v0
        print(f"[E{ep:02d}] val acc={acc:.4f} f1={f1:.4f} | train={train_sec:.1f}s val={val_sec:.1f}s")
        hist['epoch'].append(ep); hist['val_acc'].append(acc); hist['val_f1'].append(f1)
        hist['train_sec'].append(train_sec); hist['val_sec'].append(val_sec)
        if f1 > best_f1:
            best_f1 = f1; patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    acc_v, f1_v = _eval(model, Xva, yva, batch=args.batch_size)
    acc_t, f1_t = _eval(model, Xte, yte, batch=args.batch_size)
    print(f"\n📊 FINAL:")
    print(f"VAL acc={acc_v:.4f} f1={f1_v:.4f}")
    print(f"TEST acc={acc_t:.4f} f1={f1_t:.4f}")

    # Plot andamento val acc/f1
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        ax[0].plot(hist['epoch'], hist['val_acc'], marker='o'); ax[0].set_title('Val Accuracy'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Acc')
        ax[1].plot(hist['epoch'], hist['val_f1'], marker='o', color='tab:orange'); ax[1].set_title('Val F1 Macro'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('F1')
        fig.tight_layout()
        plot_path = os.path.join(LOGS_DIR, f"contrastive_metrics_{run_id}.png")
        fig.savefig(plot_path); plt.close(fig)
        print(f"📈 Plot salvato: {plot_path}")
    except Exception as e:
        print(f"[WARN] Plot metrics fallito: {e}")

    # Confusion matrix su test
    try:
        yt, yp = [], []
        for i in range(0, Xte.shape[0], args.batch_size):
            xb = _to_input(Xte[i:i+args.batch_size])
            yp.append(model.predict(xb))
            yt.append(yte[i:i+args.batch_size])
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        cm = confusion_matrix(yt, yp)
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        im = ax.imshow(cm, cmap='Blues'); fig.colorbar(im, ax=ax)
        ax.set_title('Confusion (test)'); ax.set_xlabel('Pred'); ax.set_ylabel('True')
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha='center', va='center', fontsize=8)
        fig.tight_layout()
        cm_path = os.path.join(LOGS_DIR, f"contrastive_cm_{run_id}.png")
        fig.savefig(cm_path); plt.close(fig)
        print(f"🧩 Confusion salvata: {cm_path}")
    except Exception as e:
        print(f"[WARN] Plot confusion fallito: {e}")

if __name__ == '__main__':
    main()
