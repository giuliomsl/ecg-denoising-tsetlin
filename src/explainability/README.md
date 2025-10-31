# Explainability Module

Analisi white-box completa dei modelli Tsetlin Machine per ECG denoising.

## 🎯 Script Principale

### `analyze_explainability.py` - Unified Analysis Tool

**Script unificato** che sostituisce i 5 script individuali precedenti (ora archiviati in `archivio/explainability_legacy/`):
- ~~`complete_explainability_analysis.py`~~ → archiviato
- ~~`explain_v7_complete.py`~~ → archiviato
- ~~`explain_feature_importance.py`~~ → archiviato
- ~~`explain_rules_extraction.py`~~ → archiviato
- ~~`explain_weights_simple.py`~~ → archiviato

**Struttura attuale (pulita):**
```
src/explainability/
├── __init__.py
├── analyze_explainability.py  ← SCRIPT PRINCIPALE (1164 righe)
└── README.md                   ← Questa guida
```

## 📖 Quick Start

### 1. Analisi Completa (Raccomandata per Tesi)

```bash
python -m src.explainability.analyze_explainability \
    --analysis all \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw emg \
    --output plots/explainability/
```

**Output:**
- `weights_distribution_*.png` - Distribuzione pesi, Gini, Pareto
- `feature_importance_*.png` - Importanza features (top-20, BP vs HF)
- `rules_extraction_*.png` - Regole estratte da top-K clauses
- `pattern_responses_*.png` - Risposta a pattern sintetici
- `*.json` - Statistiche complete in formato machine-readable

### 2. Solo Pesi (Veloce, No Dataset Required)

```bash
python -m src.explainability.analyze_explainability \
    --analysis weights \
    --models models/tmu_v7/ \
    --tasks bw emg \
    --output plots/weights/
```

### 3. Feature Importance con Random Forest Proxy (Veloce)

```bash
python -m src.explainability.analyze_explainability \
    --analysis features \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw emg \
    --method rf_proxy \
    --output plots/features/
```

### 4. Rules Extraction (Top-15 Clauses)

```bash
python -m src.explainability.analyze_explainability \
    --analysis rules \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw \
    --top-k 15 \
    --output plots/rules/
```

## 🔧 Moduli di Analisi

### MODULE 1: Weights Analysis
**Cosa fa:** Analizza la distribuzione dei pesi delle clauses
- **Gini coefficient:** Misura di sparsità (0=uniforme, 1=sparso)
- **Effective rank:** Quante clauses contribuiscono attivamente
- **Pareto analysis:** Regola 80/20 (top-K% clauses → X% peso totale)
- **Top clauses:** Identificazione delle clauses dominanti

**Quando usarlo:**
- Capire se il modello usa tutte le clauses o solo poche dominanti
- Verificare interpretabilità (più sparso = più interpretabile)
- Confrontare BW vs EMG (pattern di attivazione diversi)

**Tempo:** ~10 secondi

### MODULE 2: Feature Importance
**Cosa fa:** Determina quali features sono importanti per le predizioni
- **Ablation:** Rimuove feature, misura calo prestazioni (accurato, lento)
- **Permutation:** Permuta feature, misura calo prestazioni (medio)
- **RF Proxy:** Random Forest come proxy TMU (veloce)

**Quando usarlo:**
- Verificare se le 57 features selezionate sono effettivamente usate
- Confrontare Bitplanes vs High-Frequency features
- Identificare features ridondanti/inutili

**Tempo:** 
- Ablation: ~5 min (limit=2000)
- RF Proxy: ~30 sec

### MODULE 3: Rules Extraction
**Cosa fa:** Estrae regole IF-THEN interpretabili dalle top-K clauses
- Identifica top-K clauses per peso
- Testa risposta su pattern controllati
- Traduce in regole human-readable

**Quando usarlo:**
- Spiegare decisioni del modello a non-tecnici
- Validare fisiologicamente le predizioni ECG
- Debugging (capire perché il modello sbaglia)

**Tempo:** ~2 minuti

### MODULE 4: Pattern Responses
**Cosa fa:** Analizza risposta del modello a pattern con diverse caratteristiche
- Pattern puliti, BW-only, EMG-only, mixed
- Sensibilità task-specific
- Visualizzazione attivazioni

**Quando usarlo:**
- Capire su quali pattern il modello performa bene/male
- Identificare bias nel modello
- Validare generalizzazione

**Tempo:** ~1 minuto

## 🚀 Esempi per Use Case

### Capitolo Tesi: "Interpretabilità del Modello"
```bash
python -m src.explainability.analyze_explainability \
    --analysis all \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw emg \
    --method rf_proxy \
    --top-k 15 \
    --output plots/thesis_chapter_explainability/
```

### Quick Check: "Le mie features sono usate?"
```bash
python -m src.explainability.analyze_explainability \
    --analysis features \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw \
    --method rf_proxy \
    --limit 5000
```

### Confronto V7 vs V4
```bash
# V7 (57 features)
python -m src.explainability.analyze_explainability \
    --analysis features \
    --models models/tmu_v7/ \
    --dataset data/explain_features_dataset_v7_th0.0005.h5 \
    --tasks bw emg \
    --method rf_proxy \
    --output plots/v7_explainability/

# V4 (1092 features)  
python -m src.explainability.analyze_explainability \
    --analysis features \
    --models models/tmu_v4/ \
    --dataset data/explain_features_dataset_v2.h5 \
    --tasks bw emg \
    --method rf_proxy \
    --output plots/v4_explainability/
```

## 📊 Interpretazione Risultati

### Gini Coefficient
- **0.00-0.10:** Molto distribuito (tutte clauses contribuiscono)
- **0.10-0.50:** Moderatamente sparso
- **0.50-1.00:** Molto sparso (poche clauses dominano)

**Per TMU:** Tipicamente 0.03-0.06 (distribuito, robusto)

### Effective Rank
- **90-100%:** Quasi tutte le clauses attive (complesso)
- **50-90%:** Molte clauses attive (bilanciato)
- **10-50%:** Poche clauses attive (semplice, interpretabile)

**Per TMU:** Tipicamente 60-80% (buon bilanciamento)

### Pareto Analysis
- **Top 10% → 50% peso:** Molto concentrato
- **Top 30% → 80% peso:** Moderatamente concentrato (TMU tipico)
- **Top 50% → 90% peso:** Distribuito uniformemente

### Feature Importance
- **BP > HF:** Bitplanes dominano (pattern temporali critici)
- **HF > BP:** High-frequency dominano (componenti spettrali critiche)
- **BP ≈ HF:** Bilanciato (V7 tipico: 2-3x ratio)

## 🔬 Performance Tips

1. **Per analisi rapide:** Usa `--method rf_proxy` (10x più veloce)
2. **Per accuratezza:** Usa `--method ablation` con `--limit 2000`
3. **Per debugging:** Analizza singolo task prima (`--tasks bw`)
4. **Per memoria:** Evita `--analysis all` con dataset grandi

## 📦 Output Files

Ogni modulo genera:
- **PNG plots:** Visualizzazioni high-res (300 DPI) per pubblicazione
- **JSON stats:** Statistiche machine-readable per analisi successive

### Struttura Output
```
plots/explainability/
├── weights_distribution_bw.png
├── weights_distribution_emg.png
├── weights_stats_bw.json
├── weights_stats_emg.json
├── feature_importance_bw_rf_proxy.png
├── feature_importance_emg_rf_proxy.png
├── feature_importance_stats_bw_rf_proxy.json
├── feature_importance_stats_emg_rf_proxy.json
├── rules_extraction_bw_top10.png
├── rules_extraction_emg_top10.png
├── rules_bw_top10.json
├── rules_emg_top10.json
├── pattern_responses_bw.png
├── pattern_responses_emg.png
├── pattern_responses_bw.json
├── pattern_responses_emg.json
└── explainability_results_all.json  # Summary completo
```

## 🛠️ Troubleshooting

### Errore: "Model has no accessible weights"
**Causa:** Modello non compatibile
**Soluzione:** Verifica che il modello sia TMURegressor con weight_bank

### Errore: "Dataset not found"
**Causa:** Path dataset errato
**Soluzione:** Verifica path con `ls data/explain_features_dataset_*.h5`

### Warning: "Limited to 2000 samples"
**Non è un errore!** È un'ottimizzazione per velocità. Per analisi complete usa `--limit 10000`

### Memoria insufficiente
**Soluzione:** Riduci `--limit` a 1000 o usa `--method rf_proxy`

## 📝 Note Importanti

1. **Backward Compatibility:** Gli script vecchi sono deprecati ma ancora funzionanti per retrocompatibilità
2. **Seed Reproducibility:** Usa sempre `--seed 42` per risultati riproducibili
3. **Multi-task:** Analizza sempre BW e EMG insieme per confronto
4. **Output Size:** Analisi completa genera ~50 MB di plot + 5 MB JSON

## 🔗 References

- **TMU Documentation:** https://github.com/cair/pyTsetlinMachine
- **Explainability Paper:** Berge et al. (2019) - "Using the Tsetlin Machine to Learn..."
- **Thesis Chapter:** Section 4.3 - "White-Box Interpretability Analysis"
