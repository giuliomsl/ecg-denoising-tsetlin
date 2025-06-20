#!/usr/bin/env python3
"""
S5 RTM Preprocessing - Binarizzazione Corretta per RTM
===================================================

Preprocessa i segnali ECG per la RegressionTsetlinMachine producendo:
- INPUT BINARIO (0/1) tramite binarizzazione a quantili delle finestre rumorose
- TARGET CONTINUI (valori reali) del rumore da predire

Pipeline Corretta:
1. Carica segnali ECG puliti e rumorosi
2. Crea finestre di dimensione WINDOW_LENGTH
3. Binarizza le finestre rumorose usando quantili come soglie
4. Calcola il rumore continuo come differenza (rumoroso - pulito)

Output:
- X_train_rtm_BINARIZED_q<N>.npy: Finestre rumorose binarizzate (0/1)
- y_train_rtm_aggregated_noise.npy: Rumore continuo da predire (float)
- Analoghi per validation e test

Autore: Pipeline RTM Corretta
Data: Giugno 2025
"""

import os
import numpy as np
import glob # Importato per cercare i file
from sklearn.model_selection import train_test_split
from datetime import datetime
import yaml # Per salvare i metadati

# Configurazione
WINDOW_LENGTH = 1024  # Questa ora rappresenta la lunghezza del segmento caricato
# NUM_QUANTILES_FOR_BINARIZATION è definito più avanti
OVERLAP = 0 # L'overlap è gestito in generate_noisy_ecg.py, qui non serve
RANDOM_STATE = 42

# Paths
BASE_PATH = '/Users/giuliomsl/Desktop/Tesi/Progetto/denoising_ecg'
# DATA_PATH punta a segmented_signals, che è corretto
DATA_PATH = os.path.join(BASE_PATH, 'data', 'segmented_signals')
OUTPUT_PATH = os.path.join(BASE_PATH, 'data', 'samplewise')
METADATA_FILENAME = "metadata_rtm_preprocessing.yaml"


def ensure_directory(path):
    """Crea directory se non esiste"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✅ Directory creata: {path}")

def load_segmented_data_from_files(data_dir):
    """
    Carica tutti i segmenti .npy da data_dir.
    Associa i segmenti 'clean' e 'noisy' basandosi sul nome del file.
    """
    print(f"📂 Caricamento segmenti da: {data_dir}...")
    
    clean_segments = []
    noisy_segments = []
    segment_identifiers = [] # Per tracciabilità e split stratificato se necessario

    # Trova tutti i file _noisy.npy per iniziare
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "*_noisy.npy")))
    if not noisy_files:
        print(f"⚠️ Nessun file *_noisy.npy trovato in {data_dir}. Controlla il percorso e i file generati.")
        return [], [], []

    print(f"🔍 Trovati {len(noisy_files)} file *_noisy.npy.")

    for noisy_file_path in noisy_files:
        base_name = os.path.basename(noisy_file_path).replace("_noisy.npy", "")
        clean_file_path = os.path.join(data_dir, f"{base_name}_clean.npy")
        
        if os.path.exists(clean_file_path):
            try:
                noisy_segment = np.load(noisy_file_path)
                clean_segment = np.load(clean_file_path)
                
                # Verifica la lunghezza del segmento
                if noisy_segment.shape[0] == WINDOW_LENGTH and clean_segment.shape[0] == WINDOW_LENGTH:
                    noisy_segments.append(noisy_segment)
                    clean_segments.append(clean_segment)
                    segment_identifiers.append(base_name)
                else:
                    print(f"⚠️ Segmento {base_name} scartato: lunghezza non corrispondente ({noisy_segment.shape[0]} vs {WINDOW_LENGTH})")
            except Exception as e:
                print(f"❌ Errore durante il caricamento o processamento del segmento {base_name}: {e}")
        else:
            print(f"⚠️ File clean corrispondente non trovato per {noisy_file_path}, segmento ignorato.")
            
    if not noisy_segments or not clean_segments:
        print("❌ Nessun segmento caricato con successo. Interruzione.")
        return [], [], []

    print(f"📊 Caricati {len(noisy_segments)} segmenti rumorosi e {len(clean_segments)} segmenti puliti.")
    return np.array(noisy_segments), np.array(clean_segments), segment_identifiers

# La funzione create_windows non è più necessaria e può essere rimossa o commentata.
# def create_windows(signals, window_length, overlap):
#     ...

def calculate_noise_targets(noisy_windows, clean_windows):
    """
    Calcola il rumore continuo come target per RTM.
    Estrae solo il campione centrale di ogni finestra per la regressione RTM.
    
    Args:
        noisy_windows: Segmenti rumorosi (n_segments, segment_length)
        clean_windows: Segmenti puliti (n_segments, segment_length)
    
    Returns:
        noise_targets: Rumore continuo del campione centrale (n_segments,)
    """
    print("🎯 Calcolo target rumore continuo (campione centrale)...")
    if noisy_windows.shape != clean_windows.shape:
        print(f"❌ ERRORE: Shape di noisy_windows ({noisy_windows.shape}) e clean_windows ({clean_windows.shape}) non corrispondono!")
        raise ValueError("Le shape dei segmenti rumorosi e puliti devono essere identiche.")
    if noisy_windows.ndim != 2 or clean_windows.ndim != 2:
        print(f"❌ ERRORE: I dati di input per calculate_noise_targets devono essere 2D (n_segments, segment_length).")
        print(f"         Shape noisy: {noisy_windows.shape}, Shape clean: {clean_windows.shape}")
        raise ValueError("Input non validi per calculate_noise_targets.")

    # Calcola rumore per tutta la finestra/segmento
    noise_full_segments = noisy_windows - clean_windows
    
    # Estrai solo il campione centrale
    center_idx = noise_full_segments.shape[1] // 2
    noise_targets = noise_full_segments[:, center_idx]
    
    print(f"   📊 Segmenti di rumore completi: shape {noise_full_segments.shape}")
    print(f"   📊 Target rumore centrale (idx={center_idx}): shape {noise_targets.shape}")
    if noise_targets.size > 0:
        print(f"   📊 Range rumore centrale: [{noise_targets.min():.6f}, {noise_targets.max():.6f}]")
        print(f"   📊 Rumore centrale medio: {noise_targets.mean():.6f} ± {noise_targets.std():.6f}")
    else:
        print("   ⚠️ Nessun target di rumore calcolato (input vuoti?).")
    
    return noise_targets

def binarize_with_quantiles(data_to_binarize, num_quantiles, fit_thresholds_on=None):
    """
    Binarizza i dati usando quantili come soglie.
    Le soglie vengono calcolate su 'fit_thresholds_on' (tipicamente X_train) 
    e poi applicate a 'data_to_binarize'.
    
    Args:
        data_to_binarize: Array 2D (n_segments, segment_length) da binarizzare.
        num_quantiles: Numero di quantili per la binarizzazione.
        fit_thresholds_on: Array 2D (n_segments_train, segment_length) su cui calcolare le soglie.
                           Se None, calcola le soglie su 'data_to_binarize' stesso (non raccomandato per val/test).
    
    Returns:
        binary_data: Array binario (n_segments, segment_length * (num_quantiles - 1))
        thresholds_used: Le soglie utilizzate per la binarizzazione.
    """
    print(f"🔄 Binarizzazione con {num_quantiles} quantili...")
    
    if fit_thresholds_on is None:
        print("   ⚠️  Nessun dato fornito per il fit delle soglie, le calcolo su data_to_binarize.")
        fit_data_for_thresholds = data_to_binarize
    else:
        fit_data_for_thresholds = fit_thresholds_on

    if fit_data_for_thresholds.size == 0:
        print("❌ ERRORE: Dati per il fit delle soglie sono vuoti.")
        # Restituisce array vuoti con la dimensionalità attesa per evitare errori a cascata
        # La dimensionalità attesa per le feature binarizzate è segment_length * numero di soglie
        # se num_quantiles è il numero di intervalli, allora ci sono num_quantiles-1 soglie.
        # Se num_quantiles è il numero di punti (es. 100 punti per 99 soglie), allora è num_quantiles-1
        # La mia implementazione precedente usava num_quantiles+1 punti per avere num_quantiles soglie.
        # np.linspace(0, 1, num_quantiles + 1)[1:-1] -> num_quantiles-1 soglie
        # Se NUM_QUANTILES_FOR_BINARIZATION = 20, allora 19 soglie.
        expected_feature_dim = data_to_binarize.shape[1] * (num_quantiles -1) if num_quantiles > 1 else data_to_binarize.shape[1]
        return np.empty((data_to_binarize.shape[0], expected_feature_dim), dtype=np.int8), []


    all_values_for_thresholds = fit_data_for_thresholds.flatten()
    # Calcola num_quantiles-1 soglie per dividere i dati in num_quantiles "bins"
    quantile_points = np.linspace(0, 1, num_quantiles + 1)[1:-1] # Es. per 20 quantili, 19 soglie
    
    if len(quantile_points) == 0 and num_quantiles == 1: # Caso speciale: binarizzazione singola soglia (mediana)
        thresholds = np.array([np.median(all_values_for_thresholds)])
        print(f"   INFO: Usando 1 quantile, la soglia è la mediana.")
    elif len(quantile_points) == 0 and num_quantiles < 1:
        print(f"❌ ERRORE: num_quantiles ({num_quantiles}) deve essere >= 1.")
        expected_feature_dim = data_to_binarize.shape[1] # Nessuna binarizzazione
        return data_to_binarize.astype(np.int8), [] # Non binarizza
    else:
        thresholds = np.quantile(all_values_for_thresholds, quantile_points)
        thresholds = np.unique(thresholds) # Rimuove soglie duplicate che possono accadere con dati discreti

    num_thresholds = len(thresholds)
    print(f"   📊 Soglie calcolate ({num_thresholds} uniche) su fit_data: shape {fit_data_for_thresholds.shape}")
    if num_thresholds > 0:
        print(f"   📊 Range soglie: [{thresholds[0]:.4f}, {thresholds[-1]:.4f}]")
    else:
        print("   ⚠️ Nessuna soglia calcolata. La binarizzazione potrebbe non produrre l'output atteso.")
        # Se non ci sono soglie, ogni campione produrrà un vettore di zeri della lunghezza delle feature binarizzate
        # Questo può accadere se num_quantiles = 1 e quantile_points è vuoto, o se i dati sono costanti.
        # La feature binarizzata sarà data_to_binarize.shape[1] * num_thresholds
        # Se num_quantiles è 1, thresholds ha 1 elemento (mediana).
        # Quindi num_thresholds dovrebbe essere almeno 1 se num_quantiles >= 1.
        # L'unico caso in cui num_thresholds può essere 0 è se quantile_points è vuoto E num_quantiles != 1.
        # Questo accade se num_quantiles = 0. Gestito sopra.
        # Quindi, se arriviamo qui con num_thresholds = 0, è un bug.
        # Per sicurezza, se num_thresholds è 0, non binarizza.
        print(f"   ❌ ERRORE INTERNO: num_thresholds è 0 con num_quantiles={num_quantiles}. Non dovrebbe accadere.")
        return data_to_binarize.astype(np.int8), thresholds # Non binarizzare

    n_segments, segment_length = data_to_binarize.shape
    if n_segments == 0: # Se data_to_binarize è vuoto
        print("   ⚠️ data_to_binarize è vuoto. Restituisco array binario vuoto.")
        return np.empty((0, segment_length * num_thresholds), dtype=np.int8), thresholds

    # Ogni campione nel segmento viene confrontato con tutte le soglie.
    # Risultato: (n_segments, segment_length, num_thresholds)
    # Poi reshape a (n_segments, segment_length * num_thresholds)
    # Esempio: campione x, soglie [t1, t2]. Output binario: [x<=t1, x<=t2]
    
    # Inizializza l'array binario. La dimensione della feature è segment_length * numero di soglie
    binary_data = np.zeros((n_segments, segment_length * num_thresholds), dtype=np.int8)
    
    if num_thresholds == 0: # Nessuna soglia, nessuna feature binaria (o errore)
        print("   ⚠️ Nessuna soglia valida, la binarizzazione produrrà feature vuote se non gestita.")
        # Potrebbe essere meglio restituire data_to_binarize non modificato o un errore.
        # Per ora, restituisce zeri con la forma corretta se num_thresholds è 0.
        # Questo significa che la dimensione della seconda colonna di binary_data sarà 0.
        # Questo causerà problemi più avanti.
        # Se num_thresholds è 0, significa che num_quantiles era probabilmente <=1 e non gestito bene.
        # O che i dati erano tali che np.unique(thresholds) ha restituito un array vuoto (improbabile).
        # Se num_quantiles = 1, thresholds ha 1 elemento (mediana).
        # Quindi num_thresholds dovrebbe essere almeno 1 se num_quantiles >= 1.
        # L'unico caso in cui num_thresholds può essere 0 è se quantile_points è vuoto E num_quantiles != 1.
        # Questo accade se num_quantiles = 0. Gestito sopra.
        # Quindi, se arriviamo qui con num_thresholds = 0, è un bug.
        # Per sicurezza, se num_thresholds è 0, non binarizza.
        print(f"   ❌ ERRORE INTERNO: num_thresholds è 0 con num_quantiles={num_quantiles}. Non dovrebbe accadere.")
        return data_to_binarize.astype(np.int8), thresholds # Non binarizzare

    for i, segment in enumerate(data_to_binarize):
        encoded_segment = []
        for sample_in_segment in segment:
            # Confronta il campione con ogni soglia
            # (sample <= t1, sample <= t2, ...)
            encoded_sample = (sample_in_segment <= thresholds).astype(np.int8)
            encoded_segment.extend(encoded_sample)
        binary_data[i, :] = encoded_segment
    
    print(f"✅ Binarizzazione completata: output shape {binary_data.shape}")
    if binary_data.size > 0:
        print(f"   📊 Valori unici nel risultato: {np.unique(binary_data)}")
    else:
        print(f"   ⚠️ Dati binarizzati sono vuoti.")
    return binary_data, thresholds


def main():
    """Funzione principale"""
    script_start_time = datetime.now()
    print("🚀 S5 RTM Preprocessing - Caricamento Segmenti e Binarizzazione Corretta")
    print("=" * 70)
    
    # Crea directory di output
    ensure_directory(OUTPUT_PATH)
    
    # Carica segmenti direttamente
    # La funzione ora restituisce (noisy_segments, clean_segments, segment_identifiers)
    noisy_segments_all, clean_segments_all, _ = load_segmented_data_from_files(DATA_PATH)
    
    if noisy_segments_all.size == 0 or clean_segments_all.size == 0:
        print("❌ Nessun dato caricato. Interruzione del preprocessing.")
        return

    print(f"📊 Segmenti totali caricati: {noisy_segments_all.shape[0]}")
    print(f"   Shape segmenti rumorosi: {noisy_segments_all.shape}")
    print(f"   Shape segmenti puliti: {clean_segments_all.shape}")

    # Non c'è più bisogno di create_windows
    # print("🔄 Creazione finestre...") # Rimosso
    # clean_windows = create_windows(clean_signals, WINDOW_LENGTH, OVERLAP) # Rimosso
    # noisy_windows = create_windows(noisy_signals, WINDOW_LENGTH, OVERLAP) # Rimosso
    # Ora usiamo direttamente i segmenti caricati come "finestre"
    
    # Split train/val/test (usando i segmenti interi come campioni)
    # Dividiamo gli indici per mantenere l'associazione tra noisy e clean
    num_total_segments = noisy_segments_all.shape[0]
    indices = np.arange(num_total_segments)

    # Split: 60% train, 20% validation, 20% test
    train_indices, temp_indices = train_test_split(indices, test_size=0.4, random_state=RANDOM_STATE)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=RANDOM_STATE) # 0.5 di 0.4 è 0.2

    X_train_noisy_segments = noisy_segments_all[train_indices]
    y_train_clean_segments = clean_segments_all[train_indices]
    
    X_val_noisy_segments = noisy_segments_all[val_indices]
    y_val_clean_segments = clean_segments_all[val_indices]
    
    X_test_noisy_segments = noisy_segments_all[test_indices]
    y_test_clean_segments = clean_segments_all[test_indices]

    print("🔄 Dataset diviso:")
    print(f"   Training: {X_train_noisy_segments.shape[0]} segmenti")
    print(f"   Validation: {X_val_noisy_segments.shape[0]} segmenti")
    print(f"   Test: {X_test_noisy_segments.shape[0]} segmenti")

    # Normalizza i segmenti rumorosi prima della binarizzazione
    # Calcola media e std SOLO sul training set rumoroso
    train_noisy_mean = np.mean(X_train_noisy_segments)
    train_noisy_std = np.std(X_train_noisy_segments)

    print(f"\\\\n🔄 Normalizzazione dei segmenti rumorosi (usando media/std del training set):")
    print(f"   Media Training Set Rumoroso: {train_noisy_mean:.4f}")
    print(f"   Std Dev Training Set Rumoroso: {train_noisy_std:.4f}")

    if train_noisy_std == 0:
        print("   ⚠️ Attenzione: Deviazione standard del training set rumoroso è 0. La normalizzazione potrebbe non funzionare come previsto.")
        # In questo caso, la normalizzazione (X - media) / std darebbe errore o NaN.
        # Potremmo decidere di non normalizzare o di usare un piccolo epsilon per std.
        # Per ora, procediamo, ma binarize_with_quantiles potrebbe avere problemi se i dati sono costanti.
        X_train_noisy_normalized = X_train_noisy_segments - train_noisy_mean
        X_val_noisy_normalized = X_val_noisy_segments - train_noisy_mean
        X_test_noisy_normalized = X_test_noisy_segments - train_noisy_mean
    else:
        X_train_noisy_normalized = (X_train_noisy_segments - train_noisy_mean) / train_noisy_std
        X_val_noisy_normalized = (X_val_noisy_segments - train_noisy_mean) / train_noisy_std
        X_test_noisy_normalized = (X_test_noisy_segments - train_noisy_mean) / train_noisy_std
    
    print(f"   Shape X_train normalizzato: {X_train_noisy_normalized.shape}")
    if X_train_noisy_normalized.size > 0:
        print(f"   Range X_train normalizzato: [{X_train_noisy_normalized.min():.4f}, {X_train_noisy_normalized.max():.4f}]")


    # Calcola i target di rumore (1D, campione centrale) per ogni set
    y_train_noise_target = calculate_noise_targets(X_train_noisy_segments, y_train_clean_segments)
    y_val_noise_target = calculate_noise_targets(X_val_noisy_segments, y_val_clean_segments)
    y_test_noise_target = calculate_noise_targets(X_test_noisy_segments, y_test_clean_segments)

    # Binarizza gli input rumorosi (X) normalizzati
    # Le soglie dei quantili devono essere calcolate SOLO su X_train_noisy_normalized
    # e poi applicate a X_val_noisy_normalized e X_test_noisy_normalized.
    
    # NUM_QUANTILES_FOR_BINARIZATION deve essere definito, es. 20 o 100
    # Lo definisco qui per chiarezza, ma potrebbe essere un parametro globale o da config.
    NUM_QUANTILES = 20 # Esempio, allineare con le aspettative del training
    print(f"INFO: Numero di quantili per binarizzazione impostato a: {NUM_QUANTILES}")

    print("\\\\n--- Binarizzazione Training Set ---")
    X_train_binarized, train_thresholds = binarize_with_quantiles(
        X_train_noisy_normalized, # Usa i dati normalizzati
        NUM_QUANTILES, 
        fit_thresholds_on=X_train_noisy_normalized # Calcola e applica su dati normalizzati
    )
    print("\\\\n--- Binarizzazione Validation Set ---")
    X_val_binarized, _ = binarize_with_quantiles(
        X_val_noisy_normalized, # Usa i dati normalizzati
        NUM_QUANTILES, 
        fit_thresholds_on=X_train_noisy_normalized # Applica soglie da training (calcolate su dati normalizzati)
    )
    print("\\\\n--- Binarizzazione Test Set ---")
    X_test_binarized, _ = binarize_with_quantiles(
        X_test_noisy_normalized, # Usa i dati normalizzati
        NUM_QUANTILES, 
        fit_thresholds_on=X_train_noisy_normalized # Applica soglie da training (calcolate su dati normalizzati)
    )
    
    # Verifica delle dimensioni finali
    print("\\n📊 Dimensioni finali dei dati processati:")
    print(f"   X_train_binarized: {X_train_binarized.shape}, y_train_noise_target: {y_train_noise_target.shape}")
    print(f"   X_val_binarized: {X_val_binarized.shape}, y_val_noise_target: {y_val_noise_target.shape}")
    print(f"   X_test_binarized: {X_test_binarized.shape}, y_test_noise_target: {y_test_noise_target.shape}")

    # Salva i file processati
    print("\\n💾 Salvataggio file processati...")
    q_suffix = f"_q{NUM_QUANTILES}"
    
    np.save(os.path.join(OUTPUT_PATH, f"X_train_rtm_BINARIZED{q_suffix}.npy"), X_train_binarized)
    np.save(os.path.join(OUTPUT_PATH, f"y_train_rtm_aggregated_noise.npy"), y_train_noise_target)

    # Calcola e salva il massimo valore assoluto del target di rumore (per normalizzazione)
    y_train_max_abs_noise = np.max(np.abs(y_train_noise_target))
    np.save(os.path.join(OUTPUT_PATH, f"y_train_max_abs_noise.npy"), y_train_max_abs_noise)
    print(f"   y_train_max_abs_noise salvato: {y_train_max_abs_noise:.6f}")

    np.save(os.path.join(OUTPUT_PATH, f"X_validation_rtm_BINARIZED{q_suffix}.npy"), X_val_binarized)
    np.save(os.path.join(OUTPUT_PATH, f"y_validation_rtm_aggregated_noise.npy"), y_val_noise_target)
    np.save(os.path.join(OUTPUT_PATH, f"X_test_rtm_BINARIZED{q_suffix}.npy"), X_test_binarized)
    np.save(os.path.join(OUTPUT_PATH, f"y_test_rtm_aggregated_noise.npy"), y_test_noise_target)
    
    # Salva metadati importanti
    metadata = {
        "script_version": "s5_rtm_preprocessing_v3_segmented_load_normalized",
        "preprocessing_timestamp": script_start_time.isoformat(),
        "data_source_path": DATA_PATH,
        "output_path": OUTPUT_PATH,
        "window_length_segments": WINDOW_LENGTH,
        "num_quantiles_for_binarization": NUM_QUANTILES,
        "normalization_train_set_mean": train_noisy_mean,
        "normalization_train_set_std": train_noisy_std,
        "binarization_thresholds_from_train_set": train_thresholds.tolist() if isinstance(train_thresholds, np.ndarray) else train_thresholds,
        "random_state_split": RANDOM_STATE,
        "split_ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "num_train_samples": X_train_binarized.shape[0],
        "num_validation_samples": X_val_binarized.shape[0],
        "num_test_samples": X_test_binarized.shape[0],
        "X_train_shape": X_train_binarized.shape,
        "y_train_shape": y_train_noise_target.shape,
    }
    metadata_path = os.path.join(OUTPUT_PATH, f"metadata_BINARIZED{q_suffix}.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, indent=4)
        
    print(f"   Metadati salvati in: {metadata_path}")
    print("✅ Preprocessing completato con successo!")
    script_duration = datetime.now() - script_start_time
    print(f"⏱️ Tempo totale esecuzione script: {script_duration}")

if __name__ == "__main__":
    # Rimuovo la vecchia costante NUM_QUANTILES_FOR_BINARIZATION dall'inizio del file
    # e la definisco localmente in main() o la passo come argomento se necessario.
    # Per ora, è definita in main().
    main()
