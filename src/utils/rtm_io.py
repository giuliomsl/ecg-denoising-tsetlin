# src/explain/rtm_io.py
# -*- coding: utf-8 -*-
"""
IO robusto per RegressionTsetlinMachine (pyTsetlinMachine):
- save_rtm_bundle(...): salva .sdef _compose_on_tdef _compose_on_template(seq_loaded, seq_tmpl):
    # crea deep-copy del template e CI COPIA i valori del "loaded"
    # DEBUG: stampa info per diagnosticare shape mismatch
    print(f"[DEBUG _compose_on_template]")
    print(f"  len(seq_loaded) = {len(seq_loaded)}")
    print(f"  len(seq_tmpl)   = {len(seq_tmpl)}")
    
    # Stampa shape di ogni array
    for i, arr in enumerate(seq_loaded):
        print(f"  loaded[{i}]: shape={arr.shape}, dtype={arr.dtype}, size={arr.size}")
    
    for i, arr in enumerate(seq_tmpl):
        print(f"  tmpl[{i}]:   shape={arr.shape}, dtype={arr.dtype}, size={arr.size}")
    
    composed = []
    for i, (a, t) in enumerate(zip(seq_loaded, seq_tmpl)):
        b = np.ascontiguousarray(t.copy())           # stessa shape/dtype del template
        if a.shape == b.shape:
            b[...] = a
        elif a.size == b.size:
            b[...] = a.reshape(b.shape)
        else:
            raise ValueError(f"State shape mismatch at index {i}: loaded {a.shape} vs tmpl {b.shape}")
        composed.append(b)
    return composedoaded, seq_tmpl):
    # crea deep-copy del template e CI COPIA i valori del "loaded"
    # DEBUG
    print(f"[DEBUG _compose] len(seq_loaded)={len(seq_loaded)}, len(seq_tmpl)={len(seq_tmpl)}")
    for i, (a, t) in enumerate(zip(seq_loaded, seq_tmpl)):
        print(f"[DEBUG _compose] Array {i}: loaded.shape={a.shape} tmpl.shape={t.shape}")
    
    composed = []
    for i, (a, t) in enumerate(zip(seq_loaded, seq_tmpl)):
        b = np.ascontiguousarray(t.copy())           # stessa shape/dtype del template
        if a.shape == b.shape:
            b[...] = a
        elif a.size == b.size:
            b[...] = a.reshape(b.shape)
        else:
            raise ValueError(f"State shape mismatch: loaded {a.shape} vs tmpl {b.shape}")
        composed.append(b)
    return composedeta.json + (opzionale) signature
- load_rtm_bundle_robust(...): carica con dummy-fit, dtype canonici, version guard
- assert_state_effective(...): test di efficacia post-caricamento
"""

from __future__ import annotations
import json, time, platform
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List

import numpy as np

# ========= DTYPE CANONICI =========

def _canon_arr_save(a: np.ndarray) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if a.dtype == np.bool_ or a.dtype.kind in ('i','u'):
        a = a.astype(np.int32, copy=False)   # int32: formato più “compatibile” per pyTM 0.6.6
    elif a.dtype.kind == 'f':
        a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)

def _canon_arr_load(a: np.ndarray) -> np.ndarray:
    """Contiguo, uint32/float32 (caricamento)."""
    if a.dtype == np.bool_ or a.dtype.kind in ('i', 'u'):
        a = a.astype(np.uint32, copy=False)
    elif a.dtype.kind == 'f':
        a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)

# --- SAVE: robusto e portabile ---
def _get_pkg_version(dist: str) -> str:
    try:
        import importlib.metadata as im
        return im.version(dist)
    except Exception:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(dist).version
        except Exception:
            return "unknown"

def save_rtm_bundle(model, meta: dict, stem_path: str, add_signature: bool = True, signature_seed: int = 1234):
    stem = Path(stem_path)
    state_p = stem.with_suffix(".state.npz")
    meta_p  = stem.with_suffix(".meta.json")

    state = model.get_state()
    if isinstance(state, (list, tuple)):
        arrs = [ _canon_arr_save(x) for x in state ]
        np.savez_compressed(state_p, **{f"s{i}":a for i,a in enumerate(arrs)})
        state_format = "sequence"; state_len = len(arrs)
        state_shapes = [ list(a.shape) for a in arrs ]
        state_dtypes = [ str(a.dtype) for a in arrs ]
    elif isinstance(state, dict):
        arrs = { k:_canon_arr_save(v) for k,v in state.items() }
        np.savez_compressed(state_p, **arrs)
        state_format = "mapping"; state_len = len(arrs)
        state_shapes = { k:list(v.shape) for k,v in arrs.items() }
        state_dtypes = { k:str(v.dtype)   for k,v in arrs.items() }
    else:
        arr = _canon_arr_save(np.asarray(state))
        np.savez_compressed(state_p, s0=arr)
        state_format = "sequence"; state_len = 1
        state_shapes = [ list(arr.shape) ]
        state_dtypes = [ str(arr.dtype) ]

    # meta safe + versioni + iperparam
    safe_meta = {}
    for k,v in meta.items():
        if isinstance(v, (np.integer,)):   safe_meta[k] = int(v)
        elif isinstance(v, (np.floating,)): safe_meta[k] = float(v)
        else:                               safe_meta[k] = v

    safe_meta.update({
        "pytm_version": _get_pkg_version("pyTsetlinMachine"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "save_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_format": state_format,
        "state_len": state_len,
        "state_shapes": state_shapes,
        "state_dtypes": state_dtypes,
    })
    # firma (pred su vettore sentinella binario) - uint32 per TMU
    if add_signature and "n_features" in safe_meta:
        rng = np.random.RandomState(signature_seed)
        n_features = int(safe_meta["n_features"])
        x_sig = (rng.rand(1, n_features) > 0.5).astype(np.uint32)
        try:
            y_sig = float(model.predict(x_sig).astype(np.float32).ravel()[0])
            safe_meta["signature_seed"] = int(signature_seed)
            safe_meta["signature_pred"] = float(y_sig)
        except Exception:
            pass

    meta_p.write_text(json.dumps(safe_meta, indent=2))
    print(f"[SAVE] {state_p.name} (+ {meta_p.name})")


# ========= CARICAMENTO =========


def _load_state_npz(path: Union[str, Path]):
    data = np.load(str(path))
    files = list(data.files)
    seq_keys = sorted([k for k in files if k.startswith("s") and k[1:].isdigit()],
                      key=lambda k: int(k[1:]))
    if seq_keys:
        arrs = [ _canon_arr_load(data[k]) for k in seq_keys ]
        return arrs, "sequence"
    else:
        arrs = { k:_canon_arr_load(data[k]) for k in files }
        return arrs, "mapping"

def _alloc_template(model, n_features: int):
    # allocazione strutture interne; epochs=1 ma poi **sovrascriviamo tutto** lo stato
    Xd = np.zeros((1, n_features), dtype=np.int32)
    yd = np.zeros((1,), dtype=np.float32)
    try:
        model.fit(Xd, yd, epochs=1, incremental=True)
    except TypeError:
        model.fit(Xd, yd, epochs=1)
    return model.get_state()

def _to_seq(x):
    if isinstance(x, dict):
        return [x[k] for k in sorted(x.keys())]
    return list(x)

def _match_shapes_like_template(seq_loaded, seq_tmpl):
    out = []
    for a, t in zip(seq_loaded, seq_tmpl):
        if a.shape != t.shape and a.size == t.size:
            out.append(np.ascontiguousarray(a.reshape(t.shape)))
        else:
            out.append(a)
    return out

def _compose_on_template(seq_loaded, seq_tmpl):
    # crea deep-copy del template e CI COPIA i valori del “loaded”
    composed = []
    for a, t in zip(seq_loaded, seq_tmpl):
        b = np.ascontiguousarray(t.copy())           # stessa shape/dtype del template
        if a.shape == b.shape:
            b[...] = a
        elif a.size == b.size:
            b[...] = a.reshape(b.shape)
        else:
            raise ValueError(f"State shape mismatch: loaded {a.shape} vs tmpl {b.shape}")
        composed.append(b)
    return composed


def load_rtm_bundle_strict(
    model_stem: Union[str, Path],
    RTM_CPU_class: Any,
    strict_version: bool = False,
) -> Tuple[Any, dict]:
    stem = Path(model_stem)
    state_p = stem.with_suffix(".state.npz")
    meta_p  = stem.with_suffix(".meta.json")
    if not state_p.exists() or not meta_p.exists():
        raise FileNotFoundError(f"Mancano file {state_p} o {meta_p}")

    meta = json.loads(meta_p.read_text())

    # opzionale: blocco versione
    if strict_version:
        from importlib.metadata import version, PackageNotFoundError
        try:
            cur = version("pyTsetlinMachine")
        except PackageNotFoundError:
            cur = "unknown"
        if str(meta.get("pytm_version", "")) != str(cur):
            raise RuntimeError(f"pyTsetlinMachine version mismatch: saved={meta.get('pytm_version')} current={cur}")

    clauses = int(meta["clauses"]); T = int(meta["T"]); s = float(meta["s"])
    wcl = bool(meta.get("weighted_clauses", True))
    nbits = int(meta.get("number_of_state_bits", 8))
    btp  = int(meta.get("boost_true_positive_feedback", 1))
    nfeat = int(meta["n_features"])

    # 1) istanzia modello “vergine”
    model = RTM_CPU_class(clauses, T, s,
                          boost_true_positive_feedback=btp,
                          number_of_state_bits=nbits,
                          weighted_clauses=wcl)

    # 2) carica dal disco e normalizza dtypes
    loaded, fmt = _load_state_npz(state_p)   # usa la tua _load_state_npz
    seq_loaded = _to_seq(loaded)             # lista
    seq_loaded = [_canon_arr_load(x) for x in seq_loaded]  # int32/float32 contigui

    # 3) alloca template interno (dummy-fit) e ottieni shape attese
    tmpl = _alloc_template(model, nfeat)
    seq_tmpl = _to_seq(tmpl)
    tmpl_sizes = [t.size for t in seq_tmpl]
    tmpl_shapes = [t.shape for t in seq_tmpl]

    # 4) identifica semanticamente cosa è "weights" e cosa è "TA"
    #    - weights_size atteso: clauses
    #    - TA_size atteso: clauses * (2*nfeat) o clauses * nfeat (se senza negati)
    expect_ta_sizes = {clauses * (2 * nfeat), clauses * nfeat}

    # mappa loaded -> slots del template
    mapped: List[np.ndarray] = [None] * len(seq_tmpl)  # type: ignore
    used = [False] * len(seq_loaded)

    def _choose_loaded_for_size(target_size: int, prefer_ta: bool = False):
        # 1) match esatto per size
        for j, a in enumerate(seq_loaded):
            if not used[j] and a.size == target_size:
                used[j] = True
                return a
        # 2) se è lo slot TA, prova size compatibili (ta)
        if prefer_ta:
            for j, a in enumerate(seq_loaded):
                if not used[j] and a.size in expect_ta_sizes:
                    used[j] = True
                    return a
        # 3) ultima spiaggia: il rimanente con size massima/minima
        cand = None
        best = -1 if prefer_ta else 10**18
        for j, a in enumerate(seq_loaded):
            if used[j]: continue
            if prefer_ta:
                if a.size > best:
                    best = a.size; cand = j
            else:
                if a.size < best:
                    best = a.size; cand = j
        if cand is not None:
            used[cand] = True
            return seq_loaded[cand]
        return None

    # policy: lo slot con size==clauses è quasi certamente i pesi; quello grande è il TA
    # ordina gli slot del template dal più piccolo al più grande così il primo dovrebbe essere "weights"
    slot_order = sorted(range(len(seq_tmpl)), key=lambda i: tmpl_sizes[i])

    for i in slot_order:
        t = seq_tmpl[i]
        t_size = t.size
        prefer_ta = (t_size not in (clauses,))  # se non è lo slot pesi, presumiamo TA
        a = _choose_loaded_for_size(t_size, prefer_ta=prefer_ta)
        if a is None:
            raise ValueError(f"Non trovo un array compatibile per lo slot {i} (size={t_size})")

        # reshape/copy sul template (stesso dtype/shape del backend)
        b = np.ascontiguousarray(t.copy())  # buffer target
        if a.size == b.size:
            b[...] = a.reshape(b.shape)
        else:
            # caso TA con size incompatibile ma semantica compatibile (es. nfeat vs 2*nfeat)
            # prova a ricostruire come (clauses, -1) e poi rimodellare
            if b.ndim == 2 and b.shape[0] == clauses:
                try:
                    aa = a.reshape(clauses, -1)
                    # se aa cols != b cols ma stessa size, il ramo sopra avrebbe preso; qui proviamo fallback
                    # rimappa per colonne se possibile
                    if aa.size == b.size:
                        b[...] = aa.reshape(b.shape)
                    else:
                        raise ValueError
                except Exception:
                    raise ValueError(f"State shape mismatch: loaded {a.shape} vs tmpl {b.shape}")
            else:
                raise ValueError(f"State shape mismatch: loaded {a.shape} vs tmpl {b.shape}")
        mapped[i] = b

    # 5) doppio set_state
    model.set_state(mapped)
    model.set_state(mapped)

    # 6) signature check (se presente) - uint32 per TMU
    if "signature_seed" in meta and "signature_pred" in meta:
        rng = np.random.RandomState(int(meta["signature_seed"]))
        x_sig = (rng.rand(1, nfeat) > 0.5).astype(np.uint32)
        y_hat = float(model.predict(x_sig).astype(np.float32).ravel()[0])
        ref   = float(meta["signature_pred"])
        if not (np.isfinite(y_hat) and abs(y_hat - ref) <= 1e-5):
            raise AssertionError(f"Signature mismatch: loaded={y_hat:.8f} saved={ref:.8f}")

    return model, meta

# ========= CHECK UTILI =========

def assert_state_effective(model: Any, X_ref: np.ndarray, tol: float = 1e-7) -> None:
    """
    Verifica che il modello NON sia 'vergine':
      - predizioni non costanti nulle
      - varianza del voto > 0
    """
    y = model.predict(X_ref).astype(np.float32).ravel()
    if not np.isfinite(y).all():
        raise AssertionError("Predizioni non finite dopo set_state")
    if np.allclose(y, 0.0, atol=tol):
        raise AssertionError("Predizioni tutte ~0: possibile stato non applicato")
    if np.var(y) < tol:
        raise AssertionError("Predizioni quasi costanti: controlla set_state")
