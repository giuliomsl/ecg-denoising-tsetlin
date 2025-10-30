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

### 1️⃣ **DATA PREPARATION**
```bash
# Load MIT-BIH records
python src/load_ecg.py

# Generate noisy segments with metadata
python src/generate_noisy_ecg.py
```
**Output:** `data/segmented_signals/{train,validation,test}/`

---

### 2️⃣ **FEATURE EXTRACTION** (1092 features)
```bash
python src/explain/prepare_and_build_explain_dataset_v2.py \
  --input data/explain_input_dataset.h5 \
  --output data/explain_features_dataset_v2.h5
```
**Output:** `data/explain_features_dataset_v2.h5` (1080 BP + 12 HF)

---

### 3️⃣ **FEATURE SELECTION** (57 features)
```bash
# Analyze bitplane importance
python analyze_bitplane_importance.py

# Create V7 dataset (threshold strategy)
python create_v7_dataset.py --strategy th0.0005
```
**Output:** `data/explain_features_dataset_v7_th0.0005.h5` (45 BP + 12 HF)

**Alternative:** Union-per-task strategy (~80 features)
```bash
python src/create_v7b_union_dataset.py --top-k 50
```

---

### 4️⃣ **TRAINING**
```bash
# Base training
python train_tmu_v7.py --seed 42

# Grid search for optimal hyperparameters
python grid_search_v7_hyperparams.py

# Retrain with optimal hyperparams
python train_tmu_v7.py \
  --clauses 3000 --T 700 --s 7.0 \
  --output models/tmu_v7_optimized
```
**Output:** 
- `models/tmu_v7_selected/` (base)
- `models/tmu_v7_optimized/` (grid-search optimized)

---

### 5️⃣ **INFERENCE**
```bash
python inference_v7.py \
  --models models/tmu_v7_optimized \
  --features data/explain_features_dataset_v7_th0.0005.h5
```
**Output:** `results/v7_predictions/v7_test_predictions.npz`

---

### 6️⃣ **EVALUATION**
```bash
# Performance comparison
python evaluate_v7.py

# Calibration analysis
python analyze_v7_calibration.py

# Grid search analysis
python analyze_grid_search.py

# Test optimized model
python test_v7_optimized.py
```
**Output:** 
- `results/v7_evaluation/`
- `plots/v7_calibration/`

---

### 7️⃣ **EXPLAINABILITY**
```bash
# Complete white-box analysis
python complete_explainability_analysis.py

# V7-specific feature importance
python explain_v7_complete.py

# Extract logical rules
python explain_rules_extraction.py

# Analyze clause weights
python explain_weights_simple.py

# Analyze V7 rules
python analyze_v7_rules.py
```
**Output:** 
- `results/explainability_analysis.json`
- `plots/explainability/`

---

### 8️⃣ **VISUALIZATION**
```bash
# Overview for thesis (publication-ready)
python visualize_explainability_features_v7.py

# Visualize extracted rules
python visualize_v7_rules.py

# Complete denoising pipeline demo
python visualize_denoising_with_explainability.py
```
**Output:** `plots/explainability_features_v7/`

---

## 📁 **Key Files Reference**

### **Core Pipeline**
| File | Purpose |
|------|---------|
| `src/load_ecg.py` | Load MIT-BIH clean records |
| `src/generate_noisy_ecg.py` | Generate noisy segments |
| `src/explain/prepare_and_build_explain_dataset_v2.py` | Extract 1092 features |
| `create_v7_dataset.py` | Select 57 features (threshold) |
| `train_tmu_v7.py` | Train TMU models |
| `inference_v7.py` | Generate predictions |
| `evaluate_v7.py` | Performance evaluation |

### **Optimization**
| File | Purpose |
|------|---------|
| `grid_search_v7_hyperparams.py` | Hyperparameter optimization |
| `analyze_grid_search.py` | Analyze grid search results |
| `analyze_bitplane_importance.py` | Bitplane importance analysis |

### **Explainability**
| File | Purpose |
|------|---------|
| `complete_explainability_analysis.py` | Complete white-box analysis |
| `explain_v7_complete.py` | V7 feature importance |
| `explain_rules_extraction.py` | Extract logical rules |
| `analyze_v7_rules.py` | Analyze extracted rules |
| `analyze_v7_calibration.py` | Calibration impact analysis |

### **Visualization**
| File | Purpose |
|------|---------|
| `visualize_explainability_features_v7.py` | Publication-ready overview |
| `visualize_v7_rules.py` | Rules visualization |
| `visualize_denoising_with_explainability.py` | Pipeline demo |

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
