# Source Code Organization

Pipeline V7 organizzata per fasi funzionali.

## 📁 Struttura

```
src/
├── data_preparation/          # FASE 1: Preparazione Dati
│   ├── load_ecg.py           # Caricamento MIT-BIH records
│   └── generate_noisy_ecg.py # Generazione segmenti rumorosi
│
├── feature_engineering/       # FASE 2-3: Feature Engineering & Selection
│   ├── features.py           # Feature encoding (bitplanes, thermometer)
│   ├── advanced_features.py  # High-frequency features (HF)
│   ├── prepare_and_build_explain_dataset_v2.py  # Feature extraction
│   ├── analyze_bitplane_importance.py           # Analisi importanza
│   ├── create_v7_dataset.py                     # V7 selection (threshold)
│   └── create_v7b_union_dataset.py              # V7b selection (union)
│
├── training/                  # FASE 4: Training
│   ├── train_tmu_v7.py       # Training TMU V7
│   └── grid_search_v7_hyperparams.py  # Ottimizzazione hyperparameters
│
├── inference/                 # FASE 5: Inference & Calibration
│   ├── inference_v7.py       # Inference principale V7
│   ├── test_v7_optimized.py  # Test modello ottimizzato
│   ├── infer_tmu.py          # Inference TMU generica
│   ├── infer_tmu_calibrated.py        # Inference con calibrazione
│   ├── calibrate_heads_simple.py      # Calibrazione semplice
│   ├── calibrate_intensities.py       # Calibrazione intensità
│   └── generate_val_predictions.py    # Predizioni validation
│
├── evaluation/                # FASE 6: Evaluation & Analysis
│   ├── evaluate_v7.py        # Evaluation performance V7
│   ├── analyze_v7_calibration.py      # Analisi calibrazione
│   ├── analyze_v7_rules.py            # Analisi regole
│   ├── analyze_grid_search.py         # Analisi grid search
│   └── compare_calibration_methods.py # Confronto metodi calibrazione
│
├── explainability/            # FASE 7: Explainability Analysis
│   ├── complete_explainability_analysis.py  # Analisi completa
│   ├── explain_v7_complete.py               # Feature importance V7
│   ├── explain_feature_importance.py        # Importanza generale
│   ├── explain_rules_extraction.py          # Estrazione regole
│   └── explain_weights_simple.py            # Analisi pesi clausole
│
├── visualization/             # FASE 8: Visualization
│   ├── visualize_explainability_features_v7.py  # Overview per tesi
│   ├── visualize_v7_rules.py                    # Visualizza regole
│   └── visualize_denoising_with_explainability.py  # Demo pipeline
│
├── utils/                     # Utilities
│   ├── pattern_templates.py  # Template pattern
│   └── rtm_io.py            # I/O utilities
│
├── config.py                  # Configurazione globale
└── __init__.py               # Inizializzazione modulo src
```

---

## 🎯 Esecuzione Pipeline per Fase

### FASE 1: Data Preparation
```bash
python -m src.data_preparation.load_ecg
python -m src.data_preparation.generate_noisy_ecg
```

### FASE 2-3: Feature Engineering
```bash
python -m src.feature_engineering.prepare_and_build_explain_dataset_v2
python -m src.feature_engineering.analyze_bitplane_importance
python -m src.feature_engineering.create_v7_dataset --strategy th0.0005
```

### FASE 4: Training
```bash
python -m src.training.train_tmu_v7 --seed 42
python -m src.training.grid_search_v7_hyperparams
```

### FASE 5: Inference
```bash
python -m src.inference.inference_v7
python -m src.inference.test_v7_optimized
```

### FASE 6: Evaluation
```bash
python -m src.evaluation.evaluate_v7
python -m src.evaluation.analyze_v7_calibration
python -m src.evaluation.analyze_grid_search
```

### FASE 7: Explainability
```bash
python -m src.explainability.complete_explainability_analysis
python -m src.explainability.explain_v7_complete
```

### FASE 8: Visualization
```bash
python -m src.visualization.visualize_explainability_features_v7
python -m src.visualization.visualize_v7_rules
```

---

## 📊 Statistiche Codebase

| Category | Files | Lines (est.) |
|----------|-------|--------------|
| **Data Preparation** | 2 | ~800 |
| **Feature Engineering** | 6 | ~1500 |
| **Training** | 2 | ~600 |
| **Inference** | 7 | ~1200 |
| **Evaluation** | 5 | ~900 |
| **Explainability** | 5 | ~1400 |
| **Visualization** | 3 | ~700 |
| **Utils** | 2 | ~300 |

**Total:** 32 file Python, ~7400 righe di codice

---

## 🔧 Import Path Updates

Dopo la riorganizzazione, gli import sono stati aggiornati:

**Prima:**
```python
from load_ecg import iter_clean_records
from features import build_features
```

**Dopo:**
```python
from src.data_preparation.load_ecg import iter_clean_records
from src.feature_engineering.features import build_features
```

---

## ✅ Vantaggi Struttura Organizzata

1. **Chiarezza:** Ogni fase ha la sua cartella dedicata
2. **Manutenibilità:** Facile trovare e modificare script
3. **Scalabilità:** Semplice aggiungere nuove fasi
4. **Modularità:** Ogni fase è un modulo Python indipendente
5. **Professionalità:** Struttura tipica di progetti ML production

---

**Organizzazione:** Giulio Maselli  
**Data:** 31 Ottobre 2025  
**Version:** V7 Reorganized
