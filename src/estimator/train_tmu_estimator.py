# ============================================
# File: train_tmu_estimator.py
# Step 2: Noise Estimation (joint: presence + SNR-bin)
# ============================================
"""
Allena 3 modelli TMU (uno per ciascun rumore: BW, MA, PLI) su dati preprocessati
dallo script 'preprocess_estimator.py' (output HDF5 con X binario e target per tipo).

Formato HDF5 atteso (es. ./data/bin/estimator_patches.h5), gruppi: train/val/test
  - X          : (N, C, L) uint8 {0,1}   -> feature binarie pronte per TMU (conv/flat)
  - y_snr_bin  : (N, 3) uint8             -> per ciascun tipo: 0=assente, 1..K=bin SNR
  - (opz) y_present: (N, 3) uint8         -> ridondante: presenza (può essere inferita da y_snr_bin>0)
  - attrs utili su root o gruppi:
        noise_order: ["BW","MA","PLI"]    (fallback a questo ordine se assente)
        snr_bin_edges: [float,...]        (edges usati nel generator/preprocess)

Strategia "joint":
  - Un modello per tipo con classi {0,1,...,K}
  - Presenza = (pred != 0)
  - Bin SNR   = pred (1..K), valido solo se presenza==1

Backend & fallback:
  - Preferenza TMU TMClassifier (conv patch_dim=(C,patch_w)) su CUDA se disponibile; altrimenti CPU
  - Fallback pyTsetlinMachine Conv2D -> Flat

Metrica per early-stopping per tipo:
  - F1_presence (macro in binario: {assente,presente})
  - F1_snr (macro su classi 1..K, calcolata SOLO sui campioni con label >0)
  - score_combinato = 0.5 * F1_presence + 0.5 * F1_snr
Early stop quando non migliora per 'patience' valutazioni.

Output:
  - ./data/models/estimator_tmu/
        bw_model.pkl / ma_model.pkl / pli_model.pkl   (se picklable)
        metrics.json (complessive e per-tipo)
        confusion_bw.npy / confusion_ma.npy / confusion_pli.npy (joint: 0..K)
        class_report_*.txt (opzionale, se possibile)
"""

import os
import json
import argparse
import importlib
import pickle
import random
from typing import Dict, Tuple, List

import numpy as np
import h5py
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# Riduci oversubscription (stabilità su Colab/BLAS)
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")


# -------------------- Utils backend --------------------

def _have_cuda() -> bool:
    return os.path.isdir('/usr/local/cuda') or os.path.exists('/proc/driver/nvidia/version')

def _import_tmu():
    # Evita import di TMU su macOS/CPU per non generare warning pycuda
    if not _have_cuda():
        return None
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
        return TMClassifier
    except Exception:
        return None

def _import_pyTM_conv2d():
    for mod in ("pyTsetlinMachine.tm","pyTsetlinMachine.cTM","pyTsetlinMachine.ctm","pyTsetlinMachine.tsetlinmachine"):
        try:
            m = importlib.import_module(mod)
            return getattr(m, 'MultiClassConvolutionalTsetlinMachine2D')
        except Exception:
            pass
    return None

def _import_pyTM_flat():
    for mod in ("pyTsetlinMachine.tm","pyTsetlinMachine.tsetlinmachine"):
        try:
            m = importlib.import_module(mod)
            return getattr(m, 'MultiClassTsetlinMachine')
        except Exception:
            pass
    return None


# -------------------- IO helpers --------------------

def _to_input(xs: np.ndarray, conv: bool) -> np.ndarray:
    """Converte (N,C,L) -> uint32 contiguo (conv) oppure flatten (flat)."""
    if conv:
        return np.ascontiguousarray(xs.astype(np.uint32, copy=False))
    return np.ascontiguousarray(xs.reshape(xs.shape[0], -1).astype(np.uint32, copy=False))

def _noise_order_from_h5(h5: h5py.File) -> List[str]:
    # prova a leggere dall'attributo, altrimenti default
    for key in ("noise_order",):
        try:
            raw = h5.attrs.get(key, None)
            if raw is None: break
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            if isinstance(raw, str):
                # può essere JSON str o str(list)
                try:
                    v = json.loads(raw)
                    if isinstance(v, list) and len(v)==3:
                        return v
                except Exception:
                    pass
            if hasattr(raw, "tolist"):
                v = raw.tolist()
                if isinstance(v, list) and len(v)==3:
                    return v
        except Exception:
            pass
    return ["BW","MA","PLI"]

def _read_split(h5: h5py.File, split: str):
    X = h5[f"{split}/X"][:]
    # Preferisci dataset aggregato se presente, altrimenti costruisci da targets/binlabel/{bw,ma,pli}
    if f"{split}/y_snr_bin" in h5:
        ybin = h5[f"{split}/y_snr_bin"][:].astype(np.uint32, copy=False)  # (N,3)
    else:
        base = f"{split}/targets/binlabel"
        cols = []
        for name in ("bw", "ma", "pli"):
            cols.append(h5[f"{base}/{name}"][:].astype(np.uint32, copy=False))
        ybin = np.stack(cols, axis=1)
    return X, ybin


def _read_snr_edges(h5_path: str) -> List[float]:
    try:
        with h5py.File(h5_path, "r") as h:
            # preferisci meta_json root se presente
            mj = h.attrs.get("meta_json", None)
            if isinstance(mj, (bytes, bytearray)):
                mj = mj.decode("utf-8")
            if isinstance(mj, str):
                try:
                    meta = json.loads(mj)
                    if "snr_edges" in meta:
                        return [float(x) for x in meta["snr_edges"]]
                except Exception:
                    pass
            # fallback: prova attributo sul gruppo train
            if "train" in h:
                arr = h["train"].attrs.get("snr_bin_edges", None)
                if arr is not None:
                    try:
                        if hasattr(arr, "tolist"):
                            return [float(x) for x in arr.tolist()]
                    except Exception:
                        pass
    except Exception:
        pass
    return []


def _write_summary(out_dir: str, h5_path: str, types: List[str], metrics: Dict[str, Dict[str, Dict[str, float]]]):
    os.makedirs(out_dir, exist_ok=True)
    snr_edges = _read_snr_edges(h5_path)
    lines = []
    lines.append("Noise Estimator - Summary")
    lines.append("")
    if snr_edges:
        lines.append(f"SNR bins: edges={snr_edges} (K={len(snr_edges)-1})")
    lines.append("")
    lines.append("Per-type Test metrics:")
    for t in types:
        m = metrics.get("test", {}).get(t, {})
        fp = m.get("f1_presence", float('nan'))
        fs = m.get("f1_snr", float('nan'))
        cb = m.get("combined", float('nan'))
        lines.append(f"- {t}: F1pres={fp:.4f} F1snr={fs:.4f} COMB={cb:.4f}")
        lines.append(f"  confusion: confusion_{t.lower()}.npy")
    lines.append("")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))


# -------------------- Metrics --------------------

def _presence_from_bins(ybin: np.ndarray) -> np.ndarray:
    """ybin: (N,) 0..K -> presence binary (0/1)."""
    return (ybin > 0).astype(np.uint8)

def _f1_presence(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    # macro-F1 su {0,1}
    return float(f1_score(y_true_bin, y_pred_bin, average="macro"))

def _f1_snr_only_present(y_true_bins: np.ndarray, y_pred_bins: np.ndarray) -> float:
    """Macro-F1 sui soli campioni con label >0, considerando classi 1..K."""
    mask = (y_true_bins > 0)
    if not np.any(mask):
        return 0.0
    yt = y_true_bins[mask]
    yp = y_pred_bins[mask]
    # Remap in 0..K-1 per sklearn (facoltativo) – ma va bene anche 1..K
    # Qui usiamo labels uniche per stabilità
    uniq = np.unique(yt)
    return float(f1_score(yt, yp, labels=uniq, average="macro"))

def _acc_snr_only_present(y_true_bins: np.ndarray, y_pred_bins: np.ndarray) -> float:
    mask = (y_true_bins > 0)
    if not np.any(mask):
        return 0.0
    yt = y_true_bins[mask]
    yp = y_pred_bins[mask]
    return float(accuracy_score(yt, yp))


# -------------------- Train per tipo --------------------

def _build_model(C: int, L: int, args, TMU, PyConv, PyFlat):
    """Costruisce un classificatore (conv preferito)."""
    model = None
    use_conv = False
    have_cuda = _have_cuda()
    platform = 'CUDA' if have_cuda else 'CPU'

    # 1) Preferisci TMU conv SOLO se CUDA disponibile
    if have_cuda and args.backend in ('auto','conv') and TMU is not None:
        try:
            model = TMU(
                number_of_clauses=args.clauses,
                T=args.T,
                s=args.s,
                patch_dim=(C, min(args.patch_w, L)),
                platform=platform,
            )
            use_conv = True
            print(f"  > Using TMU conv on {platform}")
        except Exception as e:
            print("  ! TMU conv init failed:", e)

    # 2) pyTM conv
    if model is None and args.backend in ('auto','conv') and PyConv is not None:
        try:
            model = PyConv(args.clauses, args.T, args.s, patch_dim=(C, min(args.patch_w, L)))
            use_conv = True
            print("  > Using pyTsetlinMachine Conv2D (CPU)")
        except Exception as e:
            print("  ! pyTM Conv2D init failed:", e)

    # 3) TMU flat SOLO se CUDA disponibile
    if model is None and have_cuda and TMU is not None and args.backend in ('auto','flat'):
        try:
            model = TMU(number_of_clauses=args.clauses, T=args.T, s=args.s, platform=platform)
            use_conv = False
            print(f"  > Using TMU flat on {platform}")
        except Exception as e:
            print("  ! TMU flat init failed:", e)

    # 4) pyTM flat
    if model is None and PyFlat is not None:
        model = PyFlat(args.clauses, args.T, args.s)
        use_conv = False
        print("  > Falling back to pyTsetlinMachine Flat")

    if model is None:
        raise RuntimeError("Nessun backend TM disponibile (TMU/pyTM).")

    return model, use_conv


def _eval_joint(model, X: np.ndarray, ybins: np.ndarray, use_conv: bool, batch: int) -> Dict[str, float]:
    """Valuta un modello joint su dataset (1 tipo alla volta: ybins è (N,) 0..K)."""
    n = ybins.shape[0]
    preds = []
    for i in range(0, n, batch):
        xs = X[i:i+batch]
        yp = model.predict(_to_input(xs, use_conv))
        preds.append(yp.astype(np.uint32, copy=False))
    y_pred = np.concatenate(preds)

    # presence
    y_true_pres = _presence_from_bins(ybins)
    y_pred_pres = _presence_from_bins(y_pred)

    f1_pres = _f1_presence(y_true_pres, y_pred_pres)
    acc_pres = float(accuracy_score(y_true_pres, y_pred_pres))

    # snr solo present
    f1_snr = _f1_snr_only_present(ybins, y_pred)
    acc_snr = _acc_snr_only_present(ybins, y_pred)

    return {
        "f1_presence": f1_pres,
        "acc_presence": acc_pres,
        "f1_snr": f1_snr,
        "acc_snr": acc_snr,
        "combined": 0.5 * f1_pres + 0.5 * f1_snr,
    }, y_pred


def _train_one_type(tag: str, Xtr, ytr_bins, Xva, yva_bins, Xte, yte_bins, C, L, args, TMU, PyConv, PyFlat):
    print(f"\n=== Training type: {tag} ===")
    model, use_conv = _build_model(C, L, args, TMU, PyConv, PyFlat)

    # Warm-up minimo per stabilizzare le strutture interne
    # Prendiamo max 2 esempi per classe reale 0..K
    uniq = np.unique(ytr_bins)
    warm_idx = []
    for u in uniq:
        idx = np.where(ytr_bins == u)[0]
        if idx.size:
            warm_idx.append(idx[:min(2, idx.size)])
    if warm_idx:
        warm_idx = np.concatenate(warm_idx)
        model.fit(_to_input(Xtr[warm_idx], use_conv), ytr_bins[warm_idx].astype(np.uint32), epochs=1, incremental=True)

    best_score = -1.0
    best_state = None
    best_epoch = -1
    patience_left = args.patience

    N = int(ytr_bins.shape[0])
    n_batches = (N + args.batch_size - 1) // args.batch_size

    def _build_balanced_indices(y: np.ndarray, n_total: int, absent_max_ratio: float) -> np.ndarray:
        """Crea indici bilanciati sui bin SNR (0..K), con cap per la classe 0.
        Usa oversampling con replacement se necessario per i bin rari.
        Restituisce un array di lunghezza n_total.
        """
        rng = np.random.default_rng()
        uniq = np.unique(y)
        desired: Dict[int, int] = {}
        # quota per absent (0)
        if 0 in uniq:
            desired[0] = int(max(0.0, min(1.0, float(absent_max_ratio))) * n_total)
        else:
            desired[0] = 0
        # distribuzione uniforme sui presenti 1..K
        present_bins = [int(b) for b in np.unique(y[y > 0]).tolist()]
        if not present_bins:
            # tutto assente: fallback permutazione semplice
            return np.random.permutation(n_total)
        rest = n_total - desired[0]
        per_present = max(1, rest // len(present_bins))
        for b in present_bins:
            desired[b] = per_present
        # rimanenze -> distribuite sui presenti
        leftover = n_total - sum(desired.values())
        i = 0
        while leftover > 0 and present_bins:
            b = present_bins[i % len(present_bins)]
            desired[b] += 1
            leftover -= 1
            i += 1

        # sampling per classe
        pieces = []
        for b, cnt in desired.items():
            if cnt <= 0:
                continue
            pool = np.where(y == b)[0]
            if pool.size == 0:
                continue
            replace = cnt > pool.size
            take = rng.choice(pool, size=cnt, replace=replace)
            pieces.append(take)
        if not pieces:
            return np.random.permutation(n_total)
        perm = np.concatenate(pieces)
        rng.shuffle(perm)
        # se per qualche motivo abbiamo meno di n_total, riempi
        if perm.size < n_total:
            extra = rng.integers(0, y.shape[0], size=(n_total - perm.size,))
            perm = np.concatenate([perm, extra])
        return perm[:n_total]

    for epoch in range(1, args.epochs + 1):
        # shuffle/balance ogni epoca
        if getattr(args, "balance_bins", False):
            perm = _build_balanced_indices(ytr_bins, N, float(getattr(args, "absent_max_ratio", 0.4)))
        else:
            perm = np.random.permutation(N)
        Xtr = Xtr[perm]; ytr_bins = ytr_bins[perm]

        for bi in range(n_batches):
            i0 = bi * args.batch_size; i1 = min(N, i0 + args.batch_size)
            xs = Xtr[i0:i1]
            ys = ytr_bins[i0:i1].astype(np.uint32, copy=False)  # TMU vuole uint32
            model.fit(_to_input(xs, use_conv), ys, epochs=1, incremental=True)
            if bi % max(1, n_batches//5) == 0:
                pct = int(100*(bi+1)/n_batches)
                print(f"  - epoch {epoch:02d} | {tag} | batch {bi+1}/{n_batches} ({pct}%)")

        # Valutazione
        if (epoch % max(1, args.val_eval_each)) == 0:
            metrics, _ = _eval_joint(model, Xva, yva_bins, use_conv, args.batch_size)
            print(f"[{tag}] epoch {epoch:02d} | val: F1pres={metrics['f1_presence']:.4f} "
                  f"F1snr={metrics['f1_snr']:.4f} comb={metrics['combined']:.4f}")

            if metrics["combined"] > best_score:
                best_score = metrics["combined"]; best_epoch = epoch
                try:
                    best_state = pickle.dumps(model)
                except Exception:
                    best_state = None
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[{tag}] Early stopping.")
                    break

    # restore best
    if best_state is not None:
        try:
            model = pickle.loads(best_state)
            print(f"[{tag}] Restored best epoch {best_epoch} (val comb={best_score:.4f})")
        except Exception:
            pass

    # Final eval su val/test
    val_metrics, _ = _eval_joint(model, Xva, yva_bins, use_conv, args.batch_size)
    te_metrics, y_pred_test = _eval_joint(model, Xte, yte_bins, use_conv, args.batch_size)

    # Confusion (joint 0..K) su test
    cm = confusion_matrix(yte_bins, y_pred_test)
    try:
        cr = classification_report(yte_bins, y_pred_test, digits=4)
    except Exception:
        cr = ""

    return model, use_conv, val_metrics, te_metrics, cm, cr, best_epoch


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Train TMU Estimator (joint presence+SNR-bin per tipo)")
    ap.add_argument("--h5", type=str, default="./data/bin/estimator_patches.h5",
                    help="HDF5 preprocessato dall'estimator preprocessor")
    ap.add_argument("--out_dir", type=str, default="./data/models/estimator_tmu")
    ap.add_argument("--backend", choices=["auto","conv","flat"], default="auto")
    ap.add_argument("--clauses", type=int, default=2400)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--s", type=float, default=7.0)
    ap.add_argument("--patch_w", type=int, default=63)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val_eval_each", type=int, default=1, help="Valuta su val ogni N epoche")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--types", type=str, default="BW,MA,PLI",
                    help="Quali tipi allenare, lista separata da virgole tra BW,MA,PLI")
    # Bilanciamento SNR & rebin
    ap.add_argument("--balance_bins", action="store_true", help="Batch bilanciati sui bin SNR (cap per classe 0)")
    ap.add_argument("--absent_max_ratio", type=float, default=0.4, help="Quota massima di assenti (bin 0) per epoca/batch")
    ap.add_argument("--snr_rebin", type=int, default=0, help="Riclassifica i bin SNR positivi in K' livelli (K'>=2). 0=disabilitato")
    args = ap.parse_args()

    # seeds & dirs
    np.random.seed(args.seed); random.seed(args.seed); os.environ["PYTHONHASHSEED"]=str(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # backends
    TMU = _import_tmu()
    PyConv = _import_pyTM_conv2d()
    PyFlat = _import_pyTM_flat()

    # data
    with h5py.File(args.h5, "r") as h5:
        Xtr, ytr_bin_all = _read_split(h5, "train")
        Xva, yva_bin_all = _read_split(h5, "val")
        Xte, yte_bin_all = _read_split(h5, "test")
        # ordine tipi (fallback a BW,MA,PLI)
        noise_order = _noise_order_from_h5(h5)

    C = int(Xtr.shape[1]); L = int(Xtr.shape[2])
    print(f"[INFO] Shapes -> train={Xtr.shape} val={Xva.shape} test={Xte.shape} | C={C} L={L}")
    print(f"[INFO] Noise order: {noise_order}")

    # mappa indice per nome
    idx_map = {n:i for i,n in enumerate(noise_order)}

    # selezione tipi da argomento
    wanted = [t.strip().upper() for t in args.types.split(",") if t.strip()!=""]
    wanted = [t for t in wanted if t in idx_map]
    if not wanted:
        wanted = noise_order

    all_metrics = {"val":{}, "test":{}}
    best_epochs = {}
    cms = {}

    def _rebin_bins(y: np.ndarray, k_new: int) -> np.ndarray:
        if k_new is None or int(k_new) <= 1:
            return y
        y = y.copy()
        k_old = int(np.max(y))
        if k_old <= 1 or k_new >= k_old:
            return y
        pos = (y > 0)
        y_pos = y[pos].astype(np.float32)
        y[pos] = np.minimum(k_new, np.ceil(y_pos * (float(k_new) / float(k_old))).astype(np.uint32))
        return y.astype(np.uint32, copy=False)

    for tag in wanted:
        k = idx_map[tag]

        # Target joint per tipo k: vettore (N,) con 0..K_k
        ytr_k = ytr_bin_all[:, k].astype(np.uint32, copy=False)
        yva_k = yva_bin_all[:, k].astype(np.uint32, copy=False)
        yte_k = yte_bin_all[:, k].astype(np.uint32, copy=False)

        # Rebin opzionale dei bin SNR positivi
        if int(args.snr_rebin) > 1:
            K_orig = int(np.max(ytr_k))
            ytr_k = _rebin_bins(ytr_k, int(args.snr_rebin))
            yva_k = _rebin_bins(yva_k, int(args.snr_rebin))
            yte_k = _rebin_bins(yte_k, int(args.snr_rebin))
            print(f"[{tag}] SNR rebin: K_orig={K_orig} -> K'={int(args.snr_rebin)} (bin 0 invariato)")

        # Allena un modello per questo tipo
        model, use_conv, val_m, te_m, cm, cr, best_ep = _train_one_type(
            tag, Xtr, ytr_k, Xva, yva_k, Xte, yte_k, C, L, args, TMU, PyConv, PyFlat
        )

        # Salvataggi per-tipo
        cms[tag] = cm
        all_metrics["val"][tag] = val_m
        all_metrics["test"][tag] = te_m
        best_epochs[tag] = int(best_ep)

        # modello
        model_path = os.path.join(args.out_dir, f"{tag.lower()}_model.pkl")
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[{tag}] Saved model -> {model_path}")
        except Exception as e:
            print(f"[{tag}] Model not picklable: {e}")

        # confusion & report
        np.save(os.path.join(args.out_dir, f"confusion_{tag.lower()}.npy"), cm)
        if cr:
            with open(os.path.join(args.out_dir, f"class_report_{tag.lower()}.txt"), "w") as f:
                f.write(cr)

    # riassunto finale
    summary = {
        "backend": args.backend,
        "clauses": args.clauses, "T": args.T, "s": args.s,
        "patch_w": args.patch_w, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "types": wanted,
        "balance_bins": bool(args.balance_bins),
        "absent_max_ratio": float(args.absent_max_ratio),
        "snr_rebin": int(args.snr_rebin) if int(args.snr_rebin) > 0 else 0,
        "val_metrics": all_metrics["val"],
        "test_metrics": all_metrics["test"],
        "best_epochs": best_epochs,
        "h5": args.h5,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[DONE] Metrics saved to", os.path.join(args.out_dir, "metrics.json"))
    for tag in wanted:
        m = all_metrics["test"][tag]
        print(f"[TEST][{tag}] F1pres={m['f1_presence']:.4f} F1snr={m['f1_snr']:.4f} COMB={m['combined']:.4f}")

    # Summary file
    try:
        _write_summary(args.out_dir, args.h5, wanted, all_metrics)
        print("[DONE] Summary saved to", os.path.join(args.out_dir, "summary.txt"))
    except Exception as e:
        print("[WARN] Could not write summary:", e)


if __name__ == "__main__":
    main()
