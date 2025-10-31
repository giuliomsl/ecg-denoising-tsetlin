# 📦 Repository Status

**Last Updated:** 30 Ottobre 2025  
**Version:** V7 Pipeline (Optimized)  
**Status:** ✅ Production Ready

---

## 📊 **Statistics**

| Metric | Value |
|--------|-------|
| **Total Files Tracked** | 45 |
| **Python Scripts** | 37 |
| **Documentation** | 3 (README, Report, Summary) |
| **Pipeline Stages** | 8 |
| **Model Performance** | r=0.6785 (+4.2% vs baseline) |
| **Feature Reduction** | 95% (1092 → 57) |

---

## 🎯 **What's Included**

### ✅ **Core Pipeline Scripts (37 Python files)**
- Data preparation (2)
- Feature extraction & selection (5)
- Training & optimization (2)
- Inference & calibration (3)
- Evaluation (3)
- Explainability analysis (5)
- Visualization (3)
- Supporting modules (14)

### 📚 **Documentation (3 files)**
- `README.md` - Full project documentation
- `FINAL_PROJECT_REPORT_IT.md` - Complete technical report
- `PIPELINE_V7_SUMMARY.md` - Quick reference guide

### ⚙️ **Configuration**
- `requirements.txt` - Python dependencies
- `.gitignore` - Optimized for V7 pipeline

---

## 🚫 **What's Excluded (via .gitignore)**

### **Heavyweight Files**
- ❌ `data/*.h5` - HDF5 datasets (regenerable)
- ❌ `models/*.pkl` - Trained models (regenerable)
- ❌ `results/*.npz` - Predictions (regenerable)
- ❌ `plots/*.png` - Visualizations (regenerable)
- ❌ `logs/` - Training logs

### **Archived/Obsolete Code**
- ❌ `archivio/` - Old experiments
- ❌ `RTM_EXPLAIN_PYTSETLIN/` - Previous version
- ❌ `sandbox_models/` - Experimental models
- ❌ `notebooks/` - Jupyter notebooks
- ❌ `src/classifier/` - Old classifier code
- ❌ `src/denoiser/` - Old denoiser code
- ❌ `src/estimator/` - Old estimator code

### **Environment & IDE**
- ❌ `venv/`, `venv311/` - Virtual environments
- ❌ `__pycache__/` - Python cache
- ❌ `.vscode/` - IDE settings
- ❌ `.DS_Store` - macOS metadata

---

## 🔄 **Regenerating Excluded Data**

All excluded heavyweight files can be regenerated:

### **Datasets**
```bash
# Step 1-3 of pipeline
python src/generate_noisy_ecg.py
python src/explain/prepare_and_build_explain_dataset_v2.py
python create_v7_dataset.py
```

### **Models**
```bash
# Step 4 of pipeline
python train_tmu_v7.py --seed 42
python grid_search_v7_hyperparams.py
```

### **Predictions & Plots**
```bash
# Steps 5-8 of pipeline
python inference_v7.py
python evaluate_v7.py
python complete_explainability_analysis.py
python visualize_explainability_features_v7.py
```

---

## 📁 **Repository Structure**

```
.
├── README.md                          # Main documentation
├── FINAL_PROJECT_REPORT_IT.md        # Technical report
├── PIPELINE_V7_SUMMARY.md            # Quick reference
├── REPOSITORY_STATUS.md              # This file
├── requirements.txt                   # Dependencies
│
├── src/                               # Source code
│   ├── load_ecg.py                   # MIT-BIH loader
│   ├── generate_noisy_ecg.py         # Noisy segment generator
│   ├── create_v7b_union_dataset.py   # V7b feature selection
│   └── explain/                       # Explainability module
│       ├── features.py               # Feature encoding
│       ├── advanced_features.py      # HF features
│       └── prepare_and_build_explain_dataset_v2.py
│
├── create_v7_dataset.py              # V7 feature selection
├── analyze_bitplane_importance.py    # Bitplane analysis
│
├── train_tmu_v7.py                   # TMU training
├── grid_search_v7_hyperparams.py     # Hyperparameter optimization
│
├── inference_v7.py                   # Test set inference
├── evaluate_v7.py                    # Performance evaluation
├── test_v7_optimized.py              # Optimized model test
│
├── analyze_v7_calibration.py         # Calibration analysis
├── analyze_v7_rules.py               # Rules analysis
├── analyze_grid_search.py            # Grid search analysis
│
├── complete_explainability_analysis.py  # Full explainability
├── explain_v7_complete.py            # V7 feature importance
├── explain_feature_importance.py     # General importance
├── explain_rules_extraction.py       # Extract logical rules
├── explain_weights_simple.py         # Clause weights
│
├── visualize_explainability_features_v7.py  # Thesis viz
├── visualize_v7_rules.py             # Rules visualization
└── visualize_denoising_with_explainability.py  # Pipeline demo
```

---

## 🚀 **Quick Start for New User**

```bash
# 1. Clone repository
git clone <repo-url>
cd denoising_ecg

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Prepare data (if not available)
python src/generate_noisy_ecg.py
python src/explain/prepare_and_build_explain_dataset_v2.py

# 4. Run complete pipeline
python create_v7_dataset.py && \
python train_tmu_v7.py --seed 42 && \
python inference_v7.py && \
python evaluate_v7.py && \
python complete_explainability_analysis.py && \
python visualize_explainability_features_v7.py
```

---

## 📊 **Expected Results**

After running the complete pipeline, you should have:

- ✅ `models/tmu_v7_selected/` - Trained models
- ✅ `results/v7_predictions/` - Test predictions
- ✅ `results/v7_evaluation/` - Performance metrics
- ✅ `plots/v7_calibration/` - Calibration analysis
- ✅ `plots/explainability/` - Feature importance & rules
- ✅ `plots/explainability_features_v7/` - Publication-ready plots

**Performance Metrics:**
- Test r (avg): 0.6785
- Test r (BW): 0.6529
- Test r (EMG): 0.7041
- Improvement: +4.2% vs baseline

---

## 🔍 **File Counts by Category**

| Category | Count | Examples |
|----------|-------|----------|
| **Data Prep** | 2 | `load_ecg.py`, `generate_noisy_ecg.py` |
| **Feature Engineering** | 5 | `prepare_and_build_*.py`, `features.py` |
| **Training** | 2 | `train_tmu_v7.py`, `grid_search_*.py` |
| **Inference** | 3 | `inference_v7.py`, `test_v7_optimized.py` |
| **Evaluation** | 3 | `evaluate_v7.py`, `analyze_*.py` |
| **Explainability** | 5 | `explain_*.py`, `complete_explainability_analysis.py` |
| **Visualization** | 3 | `visualize_*.py` |
| **Support Modules** | 14 | `src/explain/*.py` |
| **Documentation** | 3 | `README.md`, `*REPORT*.md` |
| **Config** | 2 | `requirements.txt`, `.gitignore` |

**Total:** 45 files

---

## ✅ **Repository Health**

- ✅ All pipeline scripts functional
- ✅ Documentation complete and updated
- ✅ .gitignore optimized (excludes regenerable files)
- ✅ No obsolete code tracked
- ✅ Reproducible with `requirements.txt`
- ✅ Publication-ready visualizations

---

## 📝 **Maintenance Notes**

### **Adding New Features**
1. Add script to appropriate pipeline stage
2. Update `.gitignore` to include (use `!filename.py`)
3. Update `README.md` and `PIPELINE_V7_SUMMARY.md`
4. Commit with descriptive message

### **Regenerating Data**
All data is regenerable from scripts - do not commit large binary files.

### **Version Control**
- Commit scripts and documentation
- Exclude models, datasets, plots (regenerable)
- Use semantic commit messages

---

**Author:** Giulio Maselli  
**Project:** ECG Denoising con Tsetlin Machine  
**Version:** V7 (Optimized)
