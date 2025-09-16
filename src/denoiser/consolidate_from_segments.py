#!/usr/bin/env python3
from __future__ import annotations
import argparse, numpy as np, h5py, importlib.util
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
except Exception:  # fallback se tqdm non è installato
    def tqdm(x, **kwargs):
        return x

def _create_ds(h: h5py.File, name: str, shape, dtype, compression: str, compression_level: int, shuffle: bool, chunk_rows: int | None):
    comp = None if compression.lower() == 'none' else compression
    comp_opts = compression_level if (comp == 'gzip') else None
    chunks = True
    if isinstance(shape, tuple) and len(shape) == 2 and chunk_rows is not None:
        # Chunk per righe per ottimizzare IO sequenziale in training
        chunks = (min(max(1, int(chunk_rows)), int(shape[0])), int(shape[1]))
    return h.create_dataset(
        name, shape=shape, dtype=dtype,
        compression=comp, compression_opts=comp_opts,
        shuffle=bool(shuffle), chunks=chunks
    )

def save_h5(outp: Path, splits: list[tuple[str, tuple[np.ndarray, np.ndarray] | None]],
            fs: float | None, segment_length: int | None,
            compression: str = 'gzip', compression_level: int = 4, shuffle: bool = False,
            chunk_rows: int | None = 4096):
    """Scrive gli split in HDF5 (in RAM -> HDF5) e aggiunge metadati utili."""
    outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(outp), "w") as h:
        # Metadati globali
        if fs is not None:
            h.attrs['fs'] = float(fs)
        if segment_length is not None:
            h.attrs['segment_length'] = int(segment_length)
        h.attrs['created_at'] = datetime.utcnow().isoformat() + 'Z'
        h.attrs['tool'] = 'consolidate_from_segments.py'

        for name, pair in splits:
            if pair is None:
                continue
            noisy, clean = pair
            noisy = np.asarray(noisy, dtype=np.float32)
            clean = np.asarray(clean, dtype=np.float32)
            assert noisy.shape == clean.shape, f"Shape mismatch {name}: {noisy.shape} vs {clean.shape}"
            assert noisy.ndim == 2, f"Atteso (N,L) per {name}, got {noisy.shape}"
            N, L = noisy.shape
            # dataset preallocati e scrittura a blocchi
            dsn = _create_ds(h, f"{name}_noisy", shape=(N, L), dtype=np.float32,
                             compression=compression, compression_level=compression_level, shuffle=shuffle,
                             chunk_rows=chunk_rows)
            dsc = _create_ds(h, f"{name}_clean", shape=(N, L), dtype=np.float32,
                             compression=compression, compression_level=compression_level, shuffle=shuffle,
                             chunk_rows=chunk_rows)
            bs = max(1, min(int(chunk_rows or 4096), N))  # blocchi per righe
            for i in tqdm(range(0, N, bs), desc=f"write:{name}", leave=False):
                j = slice(i, min(i+bs, N))
                dsn[j] = noisy[j]
                dsc[j] = clean[j]
            h.attrs[f'n_samples_{name}'] = int(N)
            # Segment length di fallback se non passato
            if 'segment_length' not in h.attrs:
                h.attrs['segment_length'] = int(L)
    print(f"[OK] wrote {outp}")

def mode_npz(npz_path: Path, outp: Path, fs: float | None, segment_length: int | None,
             compression: str, compression_level: int, shuffle: bool, val_name: str,
             limit_train: int | None, limit_val: int | None, limit_test: int | None,
             chunk_rows: int | None):
    d = np.load(str(npz_path))
    def getpair(prefix):
        n = d.get(f"{prefix}_noisy", None); c = d.get(f"{prefix}_clean", None)
        if n is None or c is None: return None
        return (np.asarray(n, dtype=np.float32), np.asarray(c, dtype=np.float32))
    train = getpair("train"); val = getpair("val"); test = getpair("test")
    # limiti opzionali
    if train is not None and limit_train:
        n = min(limit_train, train[0].shape[0]); train = (train[0][:n], train[1][:n])
    if val is not None and limit_val:
        n = min(limit_val, val[0].shape[0]); val = (val[0][:n], val[1][:n])
    if test is not None and limit_test:
        n = min(limit_test, test[0].shape[0]); test = (test[0][:n], test[1][:n])
    # calcola segment_length se assente
    if segment_length is None:
        for pair in (train, val, test):
            if pair is not None:
                segment_length = int(pair[0].shape[-1])
                break
    splits = [("train", train), (val_name, val), ("test", test)]
    save_h5(outp, splits, fs, segment_length, compression, compression_level, shuffle, chunk_rows)

def mode_folders(root: Path, outp: Path, fs: float | None, segment_length: int | None,
                 compression: str, compression_level: int, shuffle: bool, val_name: str,
                 limit_train: int | None, limit_val: int | None, limit_test: int | None,
                 chunk_rows: int | None, provenance: bool):
    def list_aligned(split: str):
        dn = (root / split / "noisy"); dc = (root / split / "clean")
        if not dn.exists() or not dc.exists():
            return None
        no = {p.stem: p for p in dn.glob("*.npy")}
        cl = {p.stem: p for p in dc.glob("*.npy")}
        keys = sorted(set(no.keys()) & set(cl.keys()))
        if not keys:
            return None
        unmatched_noisy = sorted(set(no.keys()) - set(cl.keys()))
        unmatched_clean = sorted(set(cl.keys()) - set(no.keys()))
        return [(no[k], cl[k]) for k in keys], unmatched_noisy, unmatched_clean

    outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(outp), "w") as h:
        if fs is not None: h.attrs['fs'] = float(fs)
        h.attrs['created_at'] = datetime.utcnow().isoformat() + 'Z'
        h.attrs['tool'] = 'consolidate_from_segments.py'
        for split_name_src, split_name_dst, lim in (("train","train", limit_train),("val",val_name, limit_val),("test","test", limit_test)):
            got = list_aligned(split_name_src)
            if not got:
                continue
            pairs, unmatched_noisy, unmatched_clean = got
            if unmatched_noisy:
                print(f"[WARN] {split_name_src}: {len(unmatched_noisy)} noisy senza match")
            if unmatched_clean:
                print(f"[WARN] {split_name_src}: {len(unmatched_clean)} clean senza match")
            if lim:
                pairs = pairs[:min(lim, len(pairs))]
            # scopri N e L dal primo file
            x0 = np.load(str(pairs[0][0])).astype(np.float32)
            L = int(x0.shape[-1])
            if segment_length is None:
                h.attrs['segment_length'] = L
            N = len(pairs)
            dsn = _create_ds(h, f"{split_name_dst}_noisy", shape=(N, L), dtype=np.float32,
                             compression=compression, compression_level=compression_level, shuffle=shuffle,
                             chunk_rows=chunk_rows)
            dsc = _create_ds(h, f"{split_name_dst}_clean", shape=(N, L), dtype=np.float32,
                             compression=compression, compression_level=compression_level, shuffle=shuffle,
                             chunk_rows=chunk_rows)
            provenance_rows: list[dict] = []
            for i, (pn, pc) in enumerate(tqdm(pairs, desc=f"write:{split_name_dst}", leave=False)):
                xn = np.asarray(np.load(str(pn)), dtype=np.float32)
                xc = np.asarray(np.load(str(pc)), dtype=np.float32)
                assert xn.shape == xc.shape == x0.shape, f"Mismatch shape {pn.name} vs {pc.name}"
                dsn[i] = xn; dsc[i] = xc
                if provenance:
                    provenance_rows.append({"noisy": pn.name, "clean": pc.name})
            h.attrs[f'n_samples_{split_name_dst}'] = int(N)
            if provenance:
                prov = {
                    "split": split_name_dst,
                    "rows": provenance_rows,
                    "unmatched_noisy": unmatched_noisy,
                    "unmatched_clean": unmatched_clean,
                }
                (outp.parent / f"{split_name_dst}_index.json").write_text(json_dumps_safe(prov))
    print(f"[OK] wrote {outp}")

def import_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    return mod

def mode_python(loader_path: Path, generator_path: Path, outp: Path, splits, L, fs,
                compression: str, compression_level: int, shuffle: bool, val_name: str,
                limit_train: int | None, limit_val: int | None, limit_test: int | None,
                chunk_rows: int | None):
    loader = import_module(loader_path); gen = import_module(generator_path)
    if not hasattr(loader, 'load_clean_segments'):
        raise SystemExit("loader-module must define load_clean_segments(split, segment_length, fs)")
    if not hasattr(gen, 'apply_noise'):
        raise SystemExit("generator-module must define apply_noise(clean_segments, split, fs)")
    def get(split):
        clean = loader.load_clean_segments(split=split, segment_length=L, fs=fs)
        noisy = gen.apply_noise(clean_segments=clean, split=split, fs=fs)
        return (np.asarray(noisy, dtype=np.float32), np.asarray(clean, dtype=np.float32))
    train = get("train") if "train" in splits else None
    val   = get("val")   if "val"   in splits else None
    test  = get("test")  if "test"  in splits else None
    if train is not None and limit_train:
        n = min(limit_train, train[0].shape[0]); train = (train[0][:n], train[1][:n])
    if val is not None and limit_val:
        n = min(limit_val, val[0].shape[0]); val = (val[0][:n], val[1][:n])
    if test is not None and limit_test:
        n = min(limit_test, test[0].shape[0]); test = (test[0][:n], test[1][:n])
    splits_list = [("train", train), (val_name, val), ("test", test)]
    save_h5(outp, splits_list, fs=fs, segment_length=L,
            compression=compression, compression_level=compression_level, shuffle=shuffle,
            chunk_rows=chunk_rows)

def json_dumps_safe(obj) -> str:
    try:
        import json
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--folders", type=str, default=None)
    ap.add_argument("--loader-module", type=str, default=None)
    ap.add_argument("--generator-module", type=str, default=None)
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--segment-length", type=int, default=None, help="Lunghezza segmento; se non dato, dedotta dai dati")
    ap.add_argument("--fs", type=float, default=None, help="Frequenza campionamento; se non dato, opzionale come metadato")
    ap.add_argument("--validation-name", type=str, default="val", choices=["val","validation"], help="Nome split di validazione nell'HDF5")
    ap.add_argument("--compression", type=str, default="gzip", choices=["gzip","lzf","none"], help="Compressione HDF5")
    ap.add_argument("--compression-level", type=int, default=4, help="Livello gzip (se usato)")
    ap.add_argument("--shuffle", action='store_true', help="Abilita shuffle filter su dataset HDF5")
    ap.add_argument("--chunk-rows", type=int, default=4096, help="Righe per chunk HDF5 (ottimizza IO)")
    ap.add_argument("--limit-train", type=int, default=None, help="Limite elementi train")
    ap.add_argument("--limit-val", type=int, default=None, help="Limite elementi val")
    ap.add_argument("--limit-test", type=int, default=None, help="Limite elementi test")
    ap.add_argument("--provenance", action='store_true', help="Scrive JSON con mappatura riga->file e liste di file spaiati (solo folders)")
    args = ap.parse_args()
    outp = Path(args.out)
    if args.npz:
        mode_npz(Path(args.npz), outp, fs=args.fs, segment_length=args.segment_length,
                 compression=args.compression, compression_level=args.compression_level,
                 shuffle=args.shuffle, val_name=args.validation_name,
                 limit_train=args.limit_train, limit_val=args.limit_val, limit_test=args.limit_test,
                 chunk_rows=args.chunk_rows); return
    if args.folders:
        mode_folders(Path(args.folders), outp, fs=args.fs, segment_length=args.segment_length,
                     compression=args.compression, compression_level=args.compression_level,
                     shuffle=args.shuffle, val_name=args.validation_name,
                     limit_train=args.limit_train, limit_val=args.limit_val, limit_test=args.limit_test,
                     chunk_rows=args.chunk_rows, provenance=args.provenance); return
    if args.loader_module and args.generator_module:
        mode_python(Path(args.loader_module), Path(args.generator_module), outp, args.splits,
                    args.segment_length if args.segment_length is not None else 1024,
                    args.fs if args.fs is not None else 360.0,
                    compression=args.compression, compression_level=args.compression_level,
                    shuffle=args.shuffle, val_name=args.validation_name,
                    limit_train=args.limit_train, limit_val=args.limit_val, limit_test=args.limit_test,
                    chunk_rows=args.chunk_rows); return
    raise SystemExit("Specify one of: --npz, --folders, or --loader-module + --generator-module")

if __name__ == "__main__":
    main()
