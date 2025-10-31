# Legacy Explainability Scripts

⚠️ **DEPRECATI - Solo per riferimento storico**

Questi script sono stati **sostituiti** dallo script unificato:
```
src/explainability/analyze_explainability.py
```

## 📜 Script Archiviati

### 1. `complete_explainability_analysis.py` (672 righe)
**Funzionalità:**
- Clause weights distribution + Gini coefficient
- Feature importance (permutation-based)
- Rules extraction da top clauses
- Pattern responses analysis
- Summary report completo

**Sostituito da:**
```bash
python -m src.explainability.analyze_explainability --analysis all
```

---

### 2. `explain_v7_complete.py` (319 righe)
**Funzionalità:**
- V7-specific weights analysis
- Random Forest feature importance (proxy per TMU)
- Confronto Bitplanes vs High-Frequency features

**Sostituito da:**
```bash
python -m src.explainability.analyze_explainability \
    --analysis features \
    --method rf_proxy
```

---

### 3. `explain_feature_importance.py` (470 righe)
**Funzionalità:**
- Ablation study (mask features)
- Permutation importance
- Gradient-free sensitivity analysis

**Sostituito da:**
```bash
python -m src.explainability.analyze_explainability \
    --analysis features \
    --method ablation  # oppure permutation
```

---

### 4. `explain_rules_extraction.py` (719 righe)
**Funzionalità:**
- Top-N clauses rules extraction
- Synthetic patterns testing
- Feature masking per inferenza
- Bitplane sensitivity analysis

**Sostituito da:**
```bash
python -m src.explainability.analyze_explainability \
    --analysis rules \
    --top-k 10
```

---

### 5. `explain_weights_simple.py` (456 righe)
**Funzionalità:**
- Weights distribution + Gini
- Top-K clauses identification
- Pareto analysis (80/20 rule)
- Sparsity metrics

**Sostituito da:**
```bash
python -m src.explainability.analyze_explainability \
    --analysis weights
```

---

## 🔄 Migrazione

### Da Script Legacy a Script Unificato

**Prima (5 comandi separati):**
```bash
python complete_explainability_analysis.py
python explain_v7_complete.py
python explain_feature_importance.py --method ablation
python explain_rules_extraction.py
python explain_weights_simple.py
```

**Ora (1 comando):**
```bash
python -m src.explainability.analyze_explainability \
    --analysis all \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw emg \
    --method rf_proxy
```

---

## 📊 Vantaggi dello Script Unificato

| Aspetto | Legacy (5 script) | Unificato (1 script) |
|---------|-------------------|----------------------|
| **Linee di codice** | ~2636 righe totali | 1164 righe |
| **Duplicazione** | Sì (compute_gini, load_model ripetuti) | No (funzioni riutilizzate) |
| **Entry point** | 5 script separati | 1 script con flag |
| **Output** | Formati diversi | Standardizzato (PNG + JSON) |
| **Documentazione** | Sparsa nei file | Guida integrata completa |
| **Manutenzione** | Difficile (5 file) | Semplice (1 file) |

---

## 🗄️ Perché Archiviati?

1. **Cronologia preservata**: Git mantiene tutta la storia di sviluppo
2. **Riferimento**: Utile per capire l'evoluzione del progetto
3. **Backup**: Se serve ripristinare funzionalità specifiche
4. **Tesi**: Mostra il processo di refactoring e miglioramento

---

## ⚠️ Nota Importante

**NON USARE** questi script per nuove analisi. Sono mantenuti solo per:
- Riferimento storico
- Compatibilità con vecchie analisi
- Documentazione del processo di sviluppo

Per **tutte le nuove analisi**, usare:
```bash
python -m src.explainability.analyze_explainability --help
```

---

## 📅 Timeline

- **2025-10-27/30**: Sviluppo script individuali
- **2025-10-31**: Unificazione in `analyze_explainability.py`
- **2025-10-31**: Archiviazione script legacy

---

## 🔗 Riferimenti

- **Script unificato**: `src/explainability/analyze_explainability.py`
- **Documentazione**: `src/explainability/README.md`
- **Pipeline**: `PIPELINE_V7_SUMMARY.md` (FASE 7)
