#!/usr/bin/env python3
"""
Valutazione dei modelli CTM Multicanale salvati in models/ctm_classifier_multichannel.
Carica HDF5 multicanale, esegue predict su uno split (train/validation/test) e
stampa/salva metriche (accuracy, F1 macro, confusion matrix) e plot.
"""
import os, sys, json
import argparse
import numpy as np
import h5py
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
except Exception:
    PROJECT_ROOT = os.getcwd()

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'ctm_classifier_multichannel')
INPUT_DATA = os.path.join(DATA_DIR, 'preprocessed_ctm_multichannel', 'classifier_multichannel_data.h5')
OUT_DIR = os.path.join(PROJECT_ROOT, 'logs', 'ctm_classifier_logs_multichannel')

def _to_input(xs, conv=True):
    arr = np.ascontiguousarray(xs.astype(np.uint32, copy=False))
    return arr

def load_split(split='test'):
    if not os.path.exists(INPUT_DATA):
        raise SystemExit(f"Dati non trovati: {INPUT_DATA}")
    with h5py.File(INPUT_DATA, 'r') as hf:
        X = hf[f'{split}/X'][:]
        y = hf[f'{split}/y'][:]
    return X, y

def predict_in_batches(model, X, batch=2048):
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = _to_input(X[i:i+batch], conv=True)
        outs.append(model.predict(xb))
    return np.concatenate(outs)

def plot_confusion(cm, classes, title, out_png):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', type=str, default=MODELS_DIR)
    ap.add_argument('--split', type=str, default='test', choices=['train','validation','test'])
    ap.add_argument('--batch', type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = load_split(args.split)
    classes = ['CLEAN','BW','MA','PLI','MIXED']

    p = Path(args.models_dir)
    ckpts = sorted(p.glob('ctm_classifier_multichannel_*.pkl'))
    if not ckpts:
        raise SystemExit(f"Nessun modello trovato in {args.models_dir}")

    for ck in ckpts:
        with open(ck, 'rb') as f:
            model = pickle.load(f)
        ypred = predict_in_batches(model, X, batch=args.batch)
        acc = float(accuracy_score(y, ypred))
        f1 = float(f1_score(y, ypred, average='macro'))
        cm = confusion_matrix(y, ypred, labels=list(range(len(classes))))
        rep = classification_report(y, ypred, target_names=classes, digits=4)
        run_id = ck.stem.split('_')[-1]
        out_json = os.path.join(OUT_DIR, f'eval_{args.split}_{run_id}.json')
        out_png = os.path.join(OUT_DIR, f'cm_{args.split}_{run_id}.png')
        with open(out_json, 'w') as fp:
            json.dump({'model': ck.name, 'split': args.split, 'accuracy': acc, 'f1_macro': f1, 'report': rep}, fp, indent=2)
        plot_confusion(cm, classes, f'Confusion ({ck.name}) on {args.split}', out_png)
        print(f"{ck.name} | {args.split}: acc={acc:.4f} f1={f1:.4f} -> {out_json}")

if __name__ == '__main__':
    main()
