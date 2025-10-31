# Pipeline V7: Quick Reference Guide

## 📊 **Performance Summary**

| Metric | Value |
|--------|-------|
| **Model** | TMU-Optimized (3000 clauses, T=700, s=7.0) |
| **Test r (avg)** | 0.6785 |
| **Test r (BW)** | 0.6529 |
| **Test r (EMG)** | 0.7041 |
| **Features** | 57 (45 BP + 12 HF) |
| **Reduction** | 95% (1092 → 57) |
| **Improvement** | +4.2% vs baseline |
| **Training time** | ~45 min |

---

## 🔄 **Pipeline Execution Order**

### FASE 1: Data Preparation
```bash
# Load MIT-BIH records
python -m src.data_preparation.load_ecg

# Generate noisy segments with metadata
python -m src.data_preparation.generate_noisy_ecg
```
**Output:** `data/segmented_signals/{train,validation,test}/`

---

### FASE 2: Feature Extraction (1092 features)
```bash
python -m src.feature_engineering.prepare_and_build_explain_dataset_v2 \
  --input data/explain_input_dataset.h5 \
  --output data/explain_features_dataset_v2.h5
```
**Output:** `data/explain_features_dataset_v2.h5` (1080 BP + 12 HF)

---

### FASE 3: Feature Selection (57 features)
```bash
# Analyze bitplane importance
python -m src.feature_engineering.analyze_bitplane_importance

# Create V7 dataset (threshold strategy)
python -m src.feature_engineering.create_v7_dataset --strategy th0.0005
```
**Output:** `data/explain_features_dataset_v7_th0.0005.h5` (45 BP + 12 HF)

**Alternative:** Union-per-task strategy (~80 features)
```bash
python -m src.feature_engineering.create_v7b_union_dataset --top-k 50
```

---

### FASE 4: Training
```bash
# Base training
python -m src.training.train_tmu_v7 --seed 42

# Grid search for optimal hyperparameters
python -m src.training.grid_search_v7_hyperparams

# Retrain with optimal hyperparams
python -m src.training.train_tmu_v7 \
  --clauses 3000 --T 700 --s 7.0 \
  --output models/tmu_v7_optimized
```
**Output:** 
- `models/tmu_v7_selected/` (base)
- `models/tmu_v7_optimized/` (grid-search optimized)

---

### FASE 5: Inference
```bash
python -m src.inference.inference_v7 \
  --models models/tmu_v7_optimized \
  --features data/explain_features_dataset_v7_th0.0005.h5
```
**Output:** `results/v7_predictions/v7_test_predictions.npz`

---

### FASE 6: Evaluation
```bash
# Performance comparison
python -m src.evaluation.evaluate_v7

# Calibration analysis
python -m src.evaluation.analyze_v7_calibration

# Grid search analysis
python -m src.evaluation.analyze_grid_search

# Test optimized model
python -m src.inference.test_v7_optimized
```
**Output:** 
- `results/v7_evaluation/`
- `plots/v7_calibration/`

---

### FASE 7: Explainability
```bash
# Complete white-box analysis
python -m src.explainability.complete_explainability_analysis

# V7-specific feature importance
python -m src.explainability.explain_v7_complete

# Extract logical rules
python -m src.explainability.explain_rules_extraction

# Analyze clause weights
python -m src.explainability.explain_weights_simple

# Analyze V7 rules
python -m src.evaluation.analyze_v7_rules
```
**Output:** 
- `results/explainability_analysis.json`
- `plots/explainability/`

---

### FASE 8: Visualization
```bash
# Overview for thesis (publication-ready)
python -m src.visualization.visualize_explainability_features_v7

# Visualize extracted rules
python -m src.visualization.visualize_v7_rules

# Complete denoising pipeline demo
python -m src.visualization.visualize_denoising_with_explainability
```
**Output:** `plots/explainability_features_v7/`

---

## 📁 **Key Files Reference**

### **Core Pipeline**
| File | Purpose |
|------|---------|
| `src/data_preparation/load_ecg.py` | Load MIT-BIH clean records |
| `src/data_preparation/generate_noisy_ecg.py` | Generate noisy segments |
| `src/feature_engineering/prepare_and_build_explain_dataset_v2.py` | Extract 1092 features |
| `src/feature_engineering/create_v7_dataset.py` | Select 57 features (threshold) |
| `src/training/train_tmu_v7.py` | Train TMU models |
| `src/inference/inference_v7.py` | Generate predictions |
| `src/evaluation/evaluate_v7.py` | Performance evaluation |

### **Optimization**
| File | Purpose |
|------|---------|
| `src/training/grid_search_v7_hyperparams.py` | Hyperparameter optimization |
| `src/evaluation/analyze_grid_search.py` | Analyze grid search results |
| `src/feature_engineering/analyze_bitplane_importance.py` | Bitplane importance analysis |

### **Explainability**
| File | Purpose |
|------|---------|
| `src/explainability/complete_explainability_analysis.py` | Complete white-box analysis |
| `src/explainability/explain_v7_complete.py` | V7 feature importance |
| `src/explainability/explain_rules_extraction.py` | Extract logical rules |
| `src/evaluation/analyze_v7_rules.py` | Analyze extracted rules |
| `src/evaluation/analyze_v7_calibration.py` | Calibration impact analysis |

### **Visualization**
| File | Purpose |
|------|---------|
| `src/visualization/visualize_explainability_features_v7.py` | Publication-ready overview |
| `src/visualization/visualize_v7_rules.py` | Rules visualization |
| `src/visualization/visualize_denoising_with_explainability.py` | Pipeline demo |

---

## 🎯 **Key Results**

### **Feature Importance (Top-5)**
**BW Task:**
1. HF_emg_high_mean (0.3847)
2. HF_pli_mean (0.1234)
3. HF_emg_mid_mean (0.0892)
4. HF_emg_low_std (0.0543)
5. BP_bit8_mean (0.0321)

**EMG Task:**
1. HF_emg_high_mean (0.4123)
2. HF_emg_mid_std (0.1567)
3. HF_emg_high_std (0.1234)
4. HF_emg_low_mean (0.0876)
5. BP_bit7_std (0.0432)

### **Explainability Metrics**
- **Gini Coefficient:** ~0.30 (semi-localized)
- **HF Dominance:** 15-21× more important than bitplanes
- **Top-3 Features:** ~40-50% of predictive power
- **Top-20 Clauses:** ~15-20% of total influence

---

## ⚠️ **Common Issues & Solutions**

### **Missing Data Files**
```bash
# Check if data exists
ls -lh data/*.h5

# If missing, regenerate from step 1
python src/generate_noisy_ecg.py
```

### **Memory Issues During Training**
```bash
# Reduce batch size or use fewer clauses
python train_tmu_v7.py --clauses 2000
```

### **Calibration Warnings**
- Normal if you see "out_of_bounds='clip'" in isotonic regression
- This ensures predictions stay in [0,1] range

---

## 📚 **Additional Resources**

- **Full Report:** `FINAL_PROJECT_REPORT_IT.md`
- **Requirements:** `requirements.txt`
- **Detailed README:** `README.md`

---

## 🚀 **One-Line Complete Pipeline**
```bash
# Run entire pipeline (assumes data is prepared)
python create_v7_dataset.py && \
python train_tmu_v7.py --seed 42 && \
python inference_v7.py && \
python evaluate_v7.py && \
python complete_explainability_analysis.py && \
python visualize_explainability_features_v7.py
```

---

**Last Updated:** 30 Ottobre 2025  
**Author:** Giulio Maselli
