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
python src/load_ecg.py

# 1.2. Genera segmenti rumorosi con metadati
python src/generate_noisy_ecg.py
```

### **FASE 2: Feature Extraction**
```bash
# Estrae 1092 features (1080 bitplanes + 12 HF)
python src/explain/prepare_and_build_explain_dataset_v2.py \
  --input data/explain_input_dataset.h5 \
  --output data/explain_features_dataset_v2.h5 \
  --window 360 --stride 120
```

### **FASE 3: Feature Selection**
```bash
# Analizza importanza bitplanes
python analyze_bitplane_importance.py

# OPZIONE A: V7 con threshold (57 features)
python create_v7_dataset.py --strategy th0.0005

# OPZIONE B: V7b con union per-task (~80 features)
python src/create_v7b_union_dataset.py --top-k 50
```

### **FASE 4: Training**
```bash
# 4.1. Training base V7
python train_tmu_v7.py \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output models/tmu_v7_selected \
  --clauses 10000 --T 700 --s 3.5 \
  --epochs 10 --patience 3 --seed 42

# 4.2. Grid Search per ottimizzazione hyperparameters
python grid_search_v7_hyperparams.py \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output results/v7_grid_search \
  --seed 42 --patience 2 --max-epochs 10

# 4.3. Retrain con hyperparams ottimali (T=700, s=7.0)
python train_tmu_v7.py \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output models/tmu_v7_optimized \
  --clauses 3000 --T 700 --s 7.0 \
  --epochs 10 --patience 3 --seed 42
```

### **FASE 5: Inference & Calibration**
```bash
# 5.1. Genera predizioni sul test set
python inference_v7.py \
  --models models/tmu_v7_selected \
  --features data/explain_features_dataset_v7_th0.0005.h5 \
  --output results/v7_predictions

# 5.2. Analizza impatto calibrazione isotonica
python analyze_v7_calibration.py

# 5.3. Test modello ottimizzato
python test_v7_optimized.py
```

### **FASE 6: Evaluation**
```bash
# 6.1. Confronto V7 vs V4
python evaluate_v7.py \
  --v7-predictions results/v7_predictions/v7_test_predictions.npz \
  --v4-predictions results/v4_test_predictions.npz \
  --output results/v7_evaluation

# 6.2. Analisi grid search
python analyze_grid_search.py
```

### **FASE 7: Explainability Analysis**
```bash
# 7.1. Analisi completa white-box
python complete_explainability_analysis.py

# 7.2. Feature importance V7
python explain_v7_complete.py

# 7.3. Estrazione regole logiche
python explain_rules_extraction.py

# 7.4. Analisi pesi clausole
python explain_weights_simple.py

# 7.5. Analisi regole V7
python analyze_v7_rules.py
```

### **FASE 8: Visualization**
```bash
# 8.1. Panoramica explainability per tesi
python visualize_explainability_features_v7.py

# 8.2. Visualizza regole estratte
python visualize_v7_rules.py

# 8.3. Demo pipeline denoising completo
python visualize_denoising_with_explainability.py
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
python src/generate_noisy_ecg.py
python src/explain/prepare_and_build_explain_dataset_v2.py

# 2. Feature selection V7
python create_v7_dataset.py --strategy th0.0005

# 3. Training
python train_tmu_v7.py --seed 42

# 4. Inference & Evaluation
python inference_v7.py
python evaluate_v7.py
python analyze_v7_calibration.py

# 5. Explainability
python complete_explainability_analysis.py
python visualize_explainability_features_v7.py
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
├── README.md                          # Questo file
├── FINAL_PROJECT_REPORT_IT.md        # Report finale progetto
├── requirements.txt                   # Dipendenze Python
│
├── src/                               # Codice sorgente core
│   ├── load_ecg.py                   # Caricamento MIT-BIH
│   ├── generate_noisy_ecg.py         # Generazione segmenti rumorosi
│   ├── create_v7b_union_dataset.py   # Feature selection V7b
│   └── explain/                       # Modulo explainability
│       ├── features.py               # Encoding features (bitplanes, thermometer)
│       ├── advanced_features.py      # Features spettrali HF
│       └── prepare_and_build_explain_dataset_v2.py  # Feature extraction
│
├── create_v7_dataset.py              # Feature selection V7
├── analyze_bitplane_importance.py    # Analisi importanza bitplanes
│
├── train_tmu_v7.py                   # Training TMU V7
├── grid_search_v7_hyperparams.py     # Ottimizzazione hyperparameters
│
├── inference_v7.py                   # Inference test set
├── test_v7_optimized.py              # Test modello ottimizzato
├── evaluate_v7.py                    # Evaluation vs baseline
│
├── analyze_v7_calibration.py         # Analisi calibrazione
├── analyze_v7_rules.py               # Analisi regole
├── analyze_grid_search.py            # Analisi grid search
│
├── complete_explainability_analysis.py  # Explainability completa
├── explain_v7_complete.py            # Feature importance V7
├── explain_feature_importance.py     # Importanza generale
├── explain_rules_extraction.py       # Estrazione regole logiche
├── explain_weights_simple.py         # Analisi pesi
│
├── visualize_explainability_features_v7.py  # Viz per tesi
├── visualize_v7_rules.py             # Viz regole
└── visualize_denoising_with_explainability.py  # Demo pipeline
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
