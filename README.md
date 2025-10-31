# ECG Denoising con Tsetlin Machine (V7 Pipeline)

Sistema white-box interpretabile per il denoising ECG basato su **Tsetlin Machine Regression** e feature selection data-driven.

## 🎯 Caratteristiche Principali

- **Modello:** TMU-Optimized (3000 clausole, T=700, s=7.0)
- **Performance:** r=0.6785 medio (BW: 0.6529, EMG: 0.7041)
- **Features:** 57 (45 bitplanes + 12 HF) - **riduzione 95%** vs baseline
- **Miglioramento:** +4.2% vs TMU-Baseline con 95% meno features
- **Interpretabilità:** White-box con regole logiche esplicite

---

## 📋 Pipeline Completa V7

### **FASE 1: Preparazione Dati**
```bash
# 1.1. Carica record MIT-BIH
python -m src.data_preparation.load_ecg

# 1.2. Genera segmenti rumorosi con metadati
python -m src.data_preparation.generate_noisy_ecg
```

### **FASE 2: Feature Extraction**
```bash
# Estrae 1092 features (1080 bitplanes + 12 HF)
python -m src.feature_engineering.prepare_and_build_explain_dataset_v2 \
  --input data/explain_input_dataset.h5 \
  --output data/explain_features_dataset_v2.h5 \
  --window 360 --stride 120
```

### **FASE 3: Feature Selection**
```bash
# Analizza importanza bitplanes
python -m src.feature_engineering.analyze_bitplane_importance

# OPZIONE A: V7 con threshold (57 features)
python -m src.feature_engineering.create_v7_dataset --strategy th0.0005

# OPZIONE B: V7b con union per-task (~80 features)
python -m src.feature_engineering.create_v7b_union_dataset --top-k 50
```

### **FASE 4: Training**
```bash
# 4.1. Training base V7
python -m src.training.train_tmu_v7 \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output models/tmu_v7_selected \
  --clauses 10000 --T 700 --s 3.5 \
  --epochs 10 --patience 3 --seed 42

# 4.2. Grid Search per ottimizzazione hyperparameters
python -m src.training.grid_search_v7_hyperparams \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output results/v7_grid_search \
  --seed 42 --patience 2 --max-epochs 10

# 4.3. Retrain con hyperparams ottimali (T=700, s=7.0)
python -m src.training.train_tmu_v7 \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output models/tmu_v7_optimized \
  --clauses 3000 --T 700 --s 7.0 \
  --epochs 10 --patience 3 --seed 42
```

### **FASE 5: Inference & Calibration**
```bash
# 5.1. Genera predizioni sul test set
python -m src.inference.inference_v7 \
  --models models/tmu_v7_selected \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output results/v7_predictions

# 5.2. Analizza impatto calibrazione isotonica
python -m src.evaluation.analyze_v7_calibration

# 5.3. Test modello ottimizzato
python -m src.inference.test_v7_optimized
```

### **FASE 6: Evaluation**
```bash
# 6.1. Confronto V7 vs V4
python -m src.evaluation.evaluate_v7 \
  --v7-predictions results/v7_predictions/v7_test_predictions.npz \
  --v4-predictions results/v4_test_predictions.npz \
  --output results/v7_evaluation

# 6.2. Analisi grid search
python -m src.evaluation.analyze_grid_search
```

### **FASE 7: Explainability Analysis**
```bash
# 7.1. Analisi completa white-box
python -m src.explainability.complete_explainability_analysis

# 7.2. Feature importance V7
python -m src.explainability.explain_v7_complete

# 7.3. Estrazione regole logiche
python -m src.explainability.explain_rules_extraction

# 7.4. Analisi pesi clausole
python -m src.explainability.explain_weights_simple

# 7.5. Analisi regole V7
python -m src.evaluation.analyze_v7_rules
```

### **FASE 8: Visualization**
```bash
# 8.1. Panoramica explainability per tesi
python -m src.visualization.visualize_explainability_features_v7

# 8.2. Visualizza regole estratte
python -m src.visualization.visualize_v7_rules

# 8.3. Demo pipeline denoising completo
python -m src.visualization.visualize_denoising_with_explainability
```

---

## 🚀 Quick Start

### Requisiti
```bash
# Python 3.11+
pip install -r requirements.txt
```

### Esecuzione Pipeline Completa
```bash
# 1. Preparazione dati (se non già fatto)
python -m src.data_preparation.generate_noisy_ecg
python -m src.feature_engineering.prepare_and_build_explain_dataset_v2

# 2. Feature selection V7
python -m src.feature_engineering.create_v7_dataset --strategy th0.0005

# 3. Training
python -m src.training.train_tmu_v7 --seed 42

# 4. Inference & Evaluation
python -m src.inference.inference_v7
python -m src.evaluation.evaluate_v7
python -m src.evaluation.analyze_v7_calibration

# 5. Explainability
python -m src.explainability.complete_explainability_analysis
python -m src.visualization.visualize_explainability_features_v7
```

---

## 📊 Risultati Chiave (Test Set)

### Performance
| Modello | Features | BW (r) | EMG (r) | Media (r) | Miglioramento |
|---------|----------|--------|---------|-----------|---------------|
| TMU-Baseline | 1092 | 0.6102 | 0.6917 | 0.6509 | baseline |
| TMU-V7 | 57 | 0.6529 | 0.7041 | **0.6785** | **+4.2%** |

### Explainability Insights
- **Feature Importance:** HF features 15-21× più importanti dei bitplanes
- **Clause Weights:** Gini coefficient ~0.30 (semi-localized)
- **Top-3 Features:** Rappresentano ~40-50% del potere predittivo
- **Pareto Analysis:** Top-20 clausole = ~15-20% influenza totale

### Efficiency
- **Training Time:** ~45 min (vs ~7h del baseline)
- **Inference Speed:** ~3× più veloce
- **Memory:** 95% riduzione

---

## 📁 Struttura Repository

```
.
├── README.md                          # Main documentation
├── FINAL_PROJECT_REPORT_IT.md        # Technical report
├── PIPELINE_V7_SUMMARY.md            # Quick reference
├── REPOSITORY_STATUS.md              # Repository health status
├── requirements.txt                   # Dependencies
│
└── src/                               # Source code organized by phase
    ├── README.md                      # Source organization guide
    ├── config.py                      # Global configuration
    │
    ├── data_preparation/              # FASE 1: Data Preparation
    │   ├── load_ecg.py               # MIT-BIH loader
    │   └── generate_noisy_ecg.py     # Noisy segment generator
    │
    ├── feature_engineering/           # FASE 2-3: Feature Engineering
    │   ├── features.py               # Feature encoding
    │   ├── advanced_features.py      # HF features
    │   ├── prepare_and_build_explain_dataset_v2.py
    │   ├── analyze_bitplane_importance.py
    │   ├── create_v7_dataset.py
    │   └── create_v7b_union_dataset.py
    │
    ├── training/                      # FASE 4: Training
    │   ├── train_tmu_v7.py
    │   └── grid_search_v7_hyperparams.py
    │
    ├── inference/                     # FASE 5: Inference & Calibration
    │   ├── inference_v7.py
    │   ├── test_v7_optimized.py
    │   ├── infer_tmu.py
    │   ├── infer_tmu_calibrated.py
    │   └── calibrate_*.py
    │
    ├── evaluation/                    # FASE 6: Evaluation
    │   ├── evaluate_v7.py
    │   ├── analyze_v7_calibration.py
    │   ├── analyze_v7_rules.py
    │   └── analyze_grid_search.py
    │
    ├── explainability/                # FASE 7: Explainability
    │   ├── complete_explainability_analysis.py
    │   ├── explain_v7_complete.py
    │   ├── explain_feature_importance.py
    │   └── explain_*.py
    │
    ├── visualization/                 # FASE 8: Visualization
    │   ├── visualize_explainability_features_v7.py
    │   ├── visualize_v7_rules.py
    │   └── visualize_denoising_with_explainability.py
    │
    └── utils/                         # Utilities
        ├── pattern_templates.py
        └── rtm_io.py
```

---

## 🔬 Metodologia

### Architettura TMU
- **Tipo:** Regression Tsetlin Machine (TMRegressor)
- **Clausole:** 3000 (ottimale da grid search)
- **Threshold (T):** 700
- **Specificity (s):** 7.0
- **Weighted Clauses:** True
- **Calibrazione:** Isotonic Regression (leak-free su validation set)

### Feature Engineering
**Bitplane Features (45):**
- Codifica thermometer 9 bit (vs 15 in baseline)
- Statistiche: mean, std, min, max, median
- Threshold selezione: 0.0005

**High-Frequency Features (12):**
- Banda EMG alta (80-150 Hz): mean, std, max power
- Banda PLI (45-55 Hz): mean, std, max power
- Banda EMG media (50-80 Hz): mean, std, max power
- Banda EMG bassa (20-50 Hz): mean, std, max power

### Pipeline Processing
1. **Windowing:** 512 samples, stride 256 (overlap 50%)
2. **Feature Extraction:** Per-window features
3. **TMU Prediction:** Intensità rumore [0,1]
4. **Calibration:** Isotonic regression
5. **Reconstruction:** Overlap-add per serie temporale
6. **Denoising:** Filtraggio adattivo guidato da predizioni

---

## 📖 Citazioni

Se usi questo codice, cita:

```bibtex
@mastersthesis{maselli2025ecg,
  title={Stima Interpretabile dell'Intensit\`a del Rumore per il Ripristino Adattivo del Segnale ECG con Tsetlin Machine},
  author={Maselli, Giulio},
  year={2025},
  school={Universit\`a degli Studi [Nome]}
}
```

---

## 📝 License

MIT License - vedi file LICENSE per dettagli

---

## 👤 Autore

**Giulio Maselli**  
📧 [email]  
🔗 [GitHub](https://github.com/giuliomsl/ecg-denoising-tsetlin)

---

## 🙏 Acknowledgments

- Dataset: MIT-BIH Arrhythmia Database, Noise Stress Test Database
- Framework: Tsetlin Machine (PyTsetlin)
- Ispirazione: Explainable AI per applicazioni medicali
