# File: src/config.py
# ===================================================================
# CONFIGURAZIONE CENTRALE PER LA PIPELINE DI DENOISING ECG
# ===================================================================
# Questo file contiene tutti i parametri modificabili per il progetto.
# È l'unica "fonte di verità" per tutti gli script.

# --- 1. PARAMETRI DI GENERAZIONE DATI ---
# Usati da `data_generator.py` e `data_generator_classifier.py`
# -------------------------------------------------------------------
SEGMENT_LENGTH = 1024
TARGET_SNR_RANGE_DB = (5, 20)

# ===================================================================
# === PIANO A: DENOISING CON SPECIALISTI RTM (REGRESSION) ===
# ===================================================================

# --- 2. PARAMETRI DI PREPROCESSING (per RTM) ---
# Usati da `preprocess_specialist.py`
# -------------------------------------------------------------------
PREPROCESSING_PARAMS = {
    'BW': {
        'window_size': 256,
        'n_thresholds': 16
    },
    'MA': {
        'window_size': 64,
        'n_thresholds': 16
    },
    'PLI': {
        'window_size': 72,
        'n_thresholds': 8
    }
}

# --- 3. IPERPARAMETRI DEI MODELLI (per RTM) ---
# Usati da `train_specialist.py`
# -------------------------------------------------------------------
MODEL_PARAMS = {
    'BW': {
        'number_of_clauses': 2000,
        'T': 800,
        's': 5.0
    },
    'MA': {
        'number_of_clauses': 4000,
        'T': 1500,
        's': 10.0
    },
    'PLI': {
        # Esempio di parametri ottimizzati (da sostituire con i risultati di Optuna)
        'number_of_clauses': 1300,
        'T': 575,
        's': 4.8
    }
}

# --- 4. PARAMETRI DI TRAINING (per RTM e CTM) ---
# Usati da entrambi gli script di training
# -------------------------------------------------------------------
TRAINING_PARAMS = {
    'total_epochs': 100,          # epoche totali (gestite a blocchi)
    'block_size': 10,             # epoche per blocco (monitoring meno frequente, più stabile)
    'validation_subset_size': 5000,
    'patience': 5                 # blocchi senza miglioramento prima di early stopping
}

# ===================================================================
# === PIANO B: CLASSIFICAZIONE DEL RUMORE CON CTM ===
# ===================================================================

# --- 5. PARAMETRI DI PREPROCESSING (per CTM) ---
# Usati da `preprocess_ctm_classifier.py`
# -------------------------------------------------------------------
CLASSIFIER_PREPROC_PARAMS = {
    # Numero di "livelli" o "pixel verticali" per rappresentare l'ampiezza.
    # Un valore più alto dà più risoluzione ma aumenta la dimensione dei dati.
    'amplitude_bins': 64,
    # Pooling temporale per ridurre L e la memoria: 1 = nessun pooling, 2 = dimezza L
    # NB: nella pipeline attuale il valore è letto da variabile d'ambiente (CTM_CLS_TEMPORAL_POOL),
    # questo campo è la raccomandazione di default per coerenza documentale.
    'temporal_pool': 2
}

# --- 6. IPERPARAMETRI DEL MODELLO (per CTM) ---
# Usati da `train_ctm_classifier.py`
# -------------------------------------------------------------------
CLASSIFIER_MODEL_PARAMS = {
    # PROFILO BASELINE PER CPU (equilibrio qualità/tempo; aumentare per risultati migliori)
    'number_of_clauses': 800,
    'T': 400,
    's': 3.0,
    'patch_dim': (4, 4)
}

# --- 7. CONFIGURAZIONE PIPELINE CTM CLASSIFIER ---
# Combinazione dei parametri per il classificatore CTM
# -------------------------------------------------------------------
CLASSIFIER_PIPELINE_PARAMS = {
    'preprocessor': CLASSIFIER_PREPROC_PARAMS,
    'trainer': {
        'model': CLASSIFIER_MODEL_PARAMS
    }
}

# --- 8. PARAMETRI GENERAZIONE DATI (per CTM Classifier) ---
# Usati da `data_generator_classifier.py` (attualmente supportati anche via env vars)
# -------------------------------------------------------------------
CLASSIFIER_DATA_PARAMS = {
    # Probabilità di campionare segmenti Clean (nessun rumore):
    'p_clean': 0.20,
    # SNR del rumore dominante (dB) rispetto al segnale pulito:
    'snr_dom_db_range': (0.0, 6.0),
    # Margine (dB) per i componenti non-dominanti rispetto al dominante:
    'snr_margin_db': 12.0,
    # SNR medio per la classe Mixed (componenti di energia simile):
    'snr_mixed_db_range': (8.0, 14.0),
    # Limite core su macOS per ridurre picchi RAM:
    'max_cores_macos': 2
}