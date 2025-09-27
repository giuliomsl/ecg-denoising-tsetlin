#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allena 3 Regression Tsetlin Machine indipendenti per stimare intensità [0,1] di BW / EMG / PLI.
Salva modelli come bundle portabile (.state.npz + .meta.json).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, h5py

try:
    from pyTsetlinMachine.tm import RegressionTsetlinMachine
except Exception as e:
    raise SystemExit("Install pyTsetlinMachine: pip install pyTsetlinMachine\n" + str(e))

def to_int32_bin(Xu8):
    return (Xu8 != 0).astype(np.int32, copy=False)

def y_forward(y, mode):
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    if mode == "identity": return y
    if mode == "sqrt":     return np.sqrt(y)
    if mode == "logit":
        eps = 1e-4
        z = y*(1-2*eps) + eps
        return np.log(z/(1.0-z)).astype(np.float32)
    raise ValueError("y-transform sconosciuta")

def save_state(stem: Path, model, meta_extra=None):
    state = model.get_state()
    meta = {"format":"state"}
    if isinstance(meta_extra, dict): meta.update(meta_extra or {})
    # mapping vs sequence
    from collections.abc import Mapping
    if isinstance(state, Mapping):
        payload = {str(k): (np.array(v) if isinstance(v, (list, tuple)) else v) for k,v in state.items()}
        np.savez_compressed(str(stem.with_suffix(".state.npz")), **payload)
        meta["state_format"] = "mapping"; meta["state_keys"] = list(payload.keys())
    elif isinstance(state, (tuple, list)):
        payload = {f"s{i}": (np.array(v) if isinstance(v, (list, tuple)) else v) for i,v in enumerate(state)}
        np.savez_compressed(str(stem.with_suffix(".state.npz")), **payload)
        meta["state_format"] = "sequence"; meta["state_len"] = len(payload)
    else:
        np.save(str(stem.with_suffix(".state.npy")), state, allow_pickle=True)
        meta["state_format"] = "pickle_npy"
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

def y_inverse(y_transformed, mode):
    """Trasformazione inversa per tornare alla scala originale [0,1]"""
    y_transformed = np.asarray(y_transformed, dtype=np.float32)
    if mode == "identity": 
        return y_transformed
    if mode == "sqrt":     
        return np.square(y_transformed)
    if mode == "logit":
        # Inversa: y = exp(z) / (1 + exp(z)) = 1 / (1 + exp(-z))
        return 1.0 / (1.0 + np.exp(-y_transformed))
    raise ValueError("y-transform sconosciuta")

def eval_mae(model, X, y_original, y_transformed, transform_mode, batch=50000, max_samples=None):
    """Calcola MAE nella scala originale [0,1] per vedere i veri progressi"""
    Xb = (X != 0).astype(np.int32, copy=False)
    
    # Limita campioni per velocizzare eval
    if max_samples and len(Xb) > max_samples:
        idx = np.random.choice(len(Xb), max_samples, replace=False)
        Xb = Xb[idx]
        y_original = y_original[idx]
    
    n = Xb.shape[0]; out=[]
    
    # Predici nella scala trasformata
    for i in range(0,n,batch): 
        out.append(model.predict(Xb[i:i+batch]).astype(np.float32))
    p_transformed = np.concatenate(out)
    
    # Trasforma indietro alla scala originale
    p_original = y_inverse(p_transformed, transform_mode)
    p_original = np.clip(p_original, 0.0, 1.0)  # Sicurezza: mantieni in [0,1]
    
    # MAE nella scala originale
    return float(np.mean(np.abs(p_original - y_original)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="H5 da build_dataset_explain.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--clauses", type=int, default=8000)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--s", type=float, default=2.6)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--inner-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--eval-batch", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--y-transform", choices=["identity","sqrt","logit"], default="identity",
                    help="Trasformazione del target in training; inverse in inference non serve perché l'output è [0,1].")
    ap.add_argument("--eval-cap", type=int, default=50000, help="Max campioni per eval (velocizza)")
    ap.add_argument("--parallel", action="store_true", help="Allena le 3 heads in parallelo")
    args = ap.parse_args()

    print("="*70)
    print("🧠 TRAINING REGRESSION TSETLIN MACHINE HEADS")
    print("="*70)
    
    np.random.seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"⚙️  Configurazione Training:")
    print(f"  • Dataset: {args.data}")
    print(f"  • Output: {args.outdir}")
    print(f"  • Architettura: {args.clauses:,} clauses, T={args.T}, s={args.s}")
    print(f"  • Training: {args.epochs} epochs x {args.inner_epochs} inner-epochs")
    print(f"  • Batch: train={args.batch:,}, eval={args.eval_batch:,}")
    print(f"  • Y-transform: {args.y_transform}")
    print(f"  • Eval cap: {args.eval_cap:,} campioni")
    print(f"  • Seed: {args.seed}")
    
    total_start_time = time.time()

    print(f"\n📁 Caricamento dataset...")
    load_start = time.time()
    
    with h5py.File(str(Path(args.data)), "r") as f:
        print(f"  🔍 Analizzando struttura dataset...")
        
        # Carica e converte features
        print(f"  📊 Caricando features...")
        Xtr_raw = f["train_X"][:]
        Xva_raw = f["val_X"][:]
        Xtr_bin = (Xtr_raw != 0).astype(np.int32, copy=False)
        Xva_bin = (Xva_raw != 0).astype(np.int32, copy=False)
        
        print(f"    • Train X: {Xtr_bin.shape} ({Xtr_bin.nbytes/(1024**2):.1f} MB)")
        print(f"    • Val X: {Xva_bin.shape} ({Xva_bin.nbytes/(1024**2):.1f} MB)")
        print(f"    • Sparsità train: {(Xtr_bin == 0).mean()*100:.1f}% zeri")
        
        # Carica target originali e trasformati
        print(f"  🎯 Caricando targets...")
        targets = {}
        for noise_type in ["bw", "emg", "pli"]:
            y_tr_orig = f[f"train_y_{noise_type}"][:].astype(np.float32)
            y_va_orig = f[f"val_y_{noise_type}"][:].astype(np.float32)
            y_tr_trans = y_forward(y_tr_orig, args.y_transform)
            y_va_trans = y_forward(y_va_orig, args.y_transform)
            
            # Statistiche target
            print(f"    • {noise_type.upper()}: train={y_tr_orig.shape[0]:,}, val={y_va_orig.shape[0]:,}")
            print(f"      - Originale: [{y_tr_orig.min():.3f}, {y_tr_orig.max():.3f}] μ={y_tr_orig.mean():.3f}")
            print(f"      - Trasformato({args.y_transform}): [{y_tr_trans.min():.3f}, {y_tr_trans.max():.3f}] μ={y_tr_trans.mean():.3f}")
            
            targets[noise_type] = {
                'train_orig': y_tr_orig, 'train_trans': y_tr_trans,
                'val_orig': y_va_orig, 'val_trans': y_va_trans
            }
    
    load_time = time.time() - load_start
    print(f"✅ Dataset caricato in {load_time:.1f}s")

    # Training delle 3 teste
    heads_results = {}
    
    for head_idx, (head, data) in enumerate(targets.items(), 1):
        print(f"\n" + "="*60)
        print(f"🧠 TRAINING HEAD {head_idx}/3: {head.upper()}")
        print(f"="*60)
        
        head_start_time = time.time()
        
        print(f"🔧 Inizializzando RTM...")
        model = RegressionTsetlinMachine(int(args.clauses), int(args.T), float(args.s))
        n = Xtr_bin.shape[0]; idx = np.arange(n)
        best_mae = float("inf")
        best_epoch = 0
        
        ytr_orig = data['train_orig']; ytr_trans = data['train_trans'] 
        yva_orig = data['val_orig']; yva_trans = data['val_trans']
        
        # Statistiche dettagliate del target
        print(f"📊 Statistiche Target {head.upper()}:")
        print(f"  • Training: {ytr_orig.shape[0]:,} campioni")
        print(f"  • Validation: {yva_orig.shape[0]:,} campioni")
        print(f"  • Range originale: [{ytr_orig.min():.4f}, {ytr_orig.max():.4f}]")
        print(f"  • Range trasformato: [{ytr_trans.min():.4f}, {ytr_trans.max():.4f}]")
        print(f"  • Media originale: {ytr_orig.mean():.4f} ± {ytr_orig.std():.4f}")
        print(f"  • Non-zero: {(ytr_orig > 0.01).mean()*100:.1f}%")
        
        # Stima training
        n_batches = (n + args.batch - 1) // args.batch
        total_updates = args.epochs * n_batches * args.inner_epochs
        print(f"📈 Piano training: {args.epochs} epochs x {n_batches:,} batches x {args.inner_epochs} inner = {total_updates:,} updates")
        
        # Training loop con monitoring dettagliato
        epoch_times = []
        train_maes = []
        val_maes = []
        
        for ep in range(1, int(args.epochs)+1):
            epoch_start = time.time()
            print(f"\n🔄 Epoch {ep}/{args.epochs}")
            
            # Shuffle e training
            np.random.shuffle(idx)
            batch_times = []
            
            for batch_idx, i in enumerate(range(0, n, args.batch), 1):
                batch_start = time.time()
                j = idx[i:i+args.batch]
                
                # Training sulla scala trasformata
                model.fit(Xtr_bin[j], ytr_trans[j], epochs=int(args.inner_epochs), incremental=True)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Progress ogni 10 batch o ultimo
                if batch_idx % 10 == 0 or i + args.batch >= n:
                    avg_batch_time = np.mean(batch_times[-10:])
                    remaining_batches = n_batches - batch_idx
                    eta_minutes = (remaining_batches * avg_batch_time) / 60
                    
                    print(f"  📦 Batch {batch_idx}/{n_batches} | "
                          f"Samples: {len(j):,} | "
                          f"Time: {batch_time:.1f}s | "
                          f"Avg: {avg_batch_time:.1f}s/batch | "
                          f"ETA: {eta_minutes:.1f}min")
            
            # Valutazione completa
            print(f"  🧪 Valutazione epoch {ep}...")
            eval_start = time.time()
            
            tr_mae = eval_mae(model, Xtr_bin, ytr_orig, ytr_trans, args.y_transform, 
                            batch=args.eval_batch, max_samples=args.eval_cap)
            va_mae = eval_mae(model, Xva_bin, yva_orig, yva_trans, args.y_transform, 
                            batch=args.eval_batch, max_samples=args.eval_cap)
            
            eval_time = time.time() - eval_start
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            train_maes.append(tr_mae)
            val_maes.append(va_mae)
            
            # Trend analysis
            improvement = ""
            if len(val_maes) > 1:
                delta = val_maes[-1] - val_maes[-2]
                if delta < 0:
                    improvement = f"↓ -{abs(delta):.4f}"
                else:
                    improvement = f"↑ +{delta:.4f}"
            
            print(f"  📊 Risultati Epoch {ep}:")
            print(f"    • Train MAE: {tr_mae:.4f}")
            print(f"    • Val MAE:   {va_mae:.4f} {improvement}")
            print(f"    • Tempo:     {epoch_time:.1f}s (training: {epoch_time-eval_time:.1f}s, eval: {eval_time:.1f}s)")
            
            # Salvataggio se miglioramento
            if va_mae < best_mae:
                best_mae = va_mae
                best_epoch = ep
                stem = outdir / f"rtm_intensity_{head}"
                save_state(stem, model, meta_extra={
                    "clauses": int(args.clauses), "T": int(args.T), "s": float(args.s),
                    "head": head, "y_transform": args.y_transform,
                    "best_epoch": ep, "best_val_mae": float(va_mae),
                    "train_mae": float(tr_mae)
                })
                print(f"    💾 NUOVO BEST! Salvato: {stem.with_suffix('.state.npz').name}")
            
            # Early stopping check
            if ep - best_epoch > 2:  # No improvement for 2+ epochs
                print(f"    ⏸️  Nessun miglioramento da {ep - best_epoch} epochs")
        
        # Riepilogo finale head
        head_time = time.time() - head_start_time
        avg_epoch_time = np.mean(epoch_times)
        
        print(f"\n✅ HEAD {head.upper()} COMPLETATO!")
        print(f"  • Best Val MAE: {best_mae:.4f} (epoch {best_epoch})")
        print(f"  • Tempo totale: {head_time:.1f}s ({head_time/60:.1f}min)")
        print(f"  • Tempo medio/epoch: {avg_epoch_time:.1f}s")
        print(f"  • Miglioramento: {train_maes[0] - best_mae:.4f} MAE")
        
        heads_results[head] = {
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'time': head_time,
            'final_train_mae': train_maes[-1] if train_maes else None
        }
    
    # Riepilogo finale completo
    total_time = time.time() - total_start_time
    
    print(f"\n" + "="*70)
    print(f"🎉 TRAINING COMPLETO!")
    print(f"="*70)
    
    print(f"⏱️  Timing Globale:")
    print(f"  • Tempo totale: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  • Caricamento: {load_time:.1f}s")
    print(f"  • Training: {total_time - load_time:.1f}s")
    
    print(f"\n📊 Risultati Finali:")
    best_overall = min(heads_results.values(), key=lambda x: x['best_mae'])
    worst_overall = max(heads_results.values(), key=lambda x: x['best_mae'])
    
    for head, results in heads_results.items():
        status = ""
        if results == best_overall:
            status = " 🥇 MIGLIORE"
        elif results == worst_overall and len(heads_results) > 1:
            status = " 🔴 DA MIGLIORARE"
        
        print(f"  • {head.upper():3}: MAE={results['best_mae']:.4f} "
              f"(epoch {results['best_epoch']}, {results['time']:.0f}s){status}")
    
    print(f"\n📁 Modelli salvati in: {outdir}/")
    print(f"  • rtm_intensity_bw.state.npz + .meta.json")
    print(f"  • rtm_intensity_emg.state.npz + .meta.json") 
    print(f"  • rtm_intensity_pli.state.npz + .meta.json")
    
    # Salva summary globale
    summary = {
        'training_completed': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_time_seconds': total_time,
        'config': {
            'clauses': args.clauses, 'T': args.T, 's': args.s,
            'epochs': args.epochs, 'inner_epochs': args.inner_epochs,
            'batch': args.batch, 'y_transform': args.y_transform
        },
        'results': heads_results,
        'best_overall_mae': best_overall['best_mae'],
        'avg_mae': np.mean([r['best_mae'] for r in heads_results.values()])
    }
    
    summary_file = outdir / "training_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\n📋 Summary salvato: {summary_file}")
    
    print(f"\n🚀 Ready per inference! Use i modelli .state.npz per predizioni.")

if __name__ == "__main__":
    main()
