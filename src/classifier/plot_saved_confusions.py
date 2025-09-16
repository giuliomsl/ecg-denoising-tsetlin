#!/usr/bin/env python3
"""
Crea plot dalle confusion matrix e metriche già salvate in models/ctm_classifier_multichannel
senza caricare i modelli (evita segfault). Salva PNG in logs/ctm_classifier_logs_multichannel.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except Exception:
    PROJECT_ROOT = os.getcwd()

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'ctm_classifier_multichannel')
OUT_DIR = os.path.join(PROJECT_ROOT, 'logs', 'ctm_classifier_logs_multichannel')
os.makedirs(OUT_DIR, exist_ok=True)

def plot_cm(cm: np.ndarray, title: str, out_path: str, classes=None):
    if classes is None:
        classes = ['CLEAN','BW','MA','PLI','MIXED']
    plt.figure(figsize=(5.5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    # numbers
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    p = Path(MODELS_DIR)
    confs = sorted(p.glob('confusion_*.npy'))
    if not confs:
        print(f"Nessuna confusion_* trovata in {MODELS_DIR}")
        return
    for cf in confs:
        run_id = cf.stem.split('_')[-1]
        cm = np.load(str(cf))
        # Prova a leggere metriche
        metrics_path = p / f'metrics_{run_id}.json'
        subtitle = ''
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    met = json.load(f)
                va = met.get('val_accuracy'); vf = met.get('val_f1_macro')
                ta = met.get('test_accuracy'); tf = met.get('test_f1_macro')
                subtitle = f" | val F1={vf:.3f} acc={va:.3f} • test F1={tf:.3f} acc={ta:.3f}"
            except Exception:
                pass
        title = f"Confusion (run {run_id}){subtitle}"
        out_png = os.path.join(OUT_DIR, f'confusion_{run_id}_pretty.png')
        plot_cm(cm, title, out_png)
        print(f"Plot salvato: {out_png}")

if __name__ == '__main__':
    main()
