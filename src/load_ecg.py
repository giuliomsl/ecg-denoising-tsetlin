# File: load_ecg.py (Aggiornato)

import wfdb
import os
import matplotlib.pyplot as plt
import sys # Per sys.exit in caso di errore grave
import errno
import numpy as np

# --- FLAG PER AMBIENTE ---
# Imposta a True se stai eseguendo su Google Colab, False se in locale
RUNNING_ON_COLAB = False # <--- MODIFICA QUESTA RIGA MANUALMENTE
# -------------------------

# --- Definizione Percorsi basata sul Flag ---
if RUNNING_ON_COLAB:
    print("INFO: load_ecg.py - Flag RUNNING_ON_COLAB impostato a True.")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
             print("INFO: load_ecg.py - Montaggio Google Drive...")
             drive.mount('/content/drive', force_remount=True)
             import time
             time.sleep(5) # Dai tempo a Drive di montarsi
        else:
             print("INFO: load_ecg.py - Google Drive già montato.")
    except ImportError:
         print("ERRORE: load_ecg.py - Impossibile importare google.colab. Assicurati di essere in Colab.")
         sys.exit("Errore critico: ambiente Colab non rilevato come previsto.")


    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/" # <--- Modifica se il tuo percorso base su Drive è diverso
    # Se cloni il repository direttamente in Colab (non da Drive)
    # REPO_NAME = "denoising_ecg"
    # PROJECT_ROOT = f"/content/{REPO_NAME}/" # Percorso del progetto clonato in Colab
    # Altrimenti, se il progetto è su Drive:
    PROJECT_ROOT = os.path.join(GDRIVE_BASE, "denoising_ecg_project_folder_name") # <--- Modifica col nome della tua cartella progetto su Drive

    DATA_DIR = os.path.join(PROJECT_ROOT, "data") # Assumendo che 'data' sia dentro la cartella del progetto su Drive
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")# Assumendo che 'models' sia dentro la cartella del progetto su Drive

else:
    print("INFO: load_ecg.py - Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    # Definisci i percorsi relativi o assoluti per l'ambiente locale
    # Assumiamo che questo script sia in una sottocartella 'src' e la root del progetto sia un livello sopra.
    # Se esegui dalla root del progetto, PROJECT_ROOT = "."
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR) # Un livello sopra 'src'
    # Se questo script è già nella root del progetto, allora PROJECT_ROOT = "."
    # PROJECT_ROOT = "." # Descommenta se esegui dalla root del progetto

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Costruisci percorsi specifici che verranno usati nel resto del progetto
ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
# NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/") # Potrebbe non essere più usato direttamente se generi segmenti
SEGMENTED_SIGNAL_DIR = os.path.join(DATA_DIR, "segmented_signals") # Output di generate_noisy_ecg.py
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samplewise") # Output di s5_rtm_preprocessing.py
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "rtm_denoiser") # Output di s6_rtm_train.py

# Nome file per la mappa delle soglie RTM (usato da s5_preprocessing e s6_train)
# Il nome effettivo del file includerà il numero di quantili, gestito in s5_preprocessing.
# Questo è solo il percorso della directory e un nome base.
RTM_THRESHOLD_MAP_BASE_FILENAME = "rtm_threshold_map" # Il suffisso _q<N>.pkl verrà aggiunto
RTM_THRESHOLD_MAP_DIR = SAMPLE_DATA_DIR # Le mappe soglie vanno con i dati campionati

print(f"INFO: load_ecg.py - PROJECT_ROOT: {PROJECT_ROOT}")
print(f"INFO: load_ecg.py - DATA_DIR: {DATA_DIR}")
print(f"INFO: load_ecg.py - MODEL_DIR: {MODEL_DIR}")
print(f"INFO: load_ecg.py - SEGMENTED_SIGNAL_DIR: {SEGMENTED_SIGNAL_DIR}")
print(f"INFO: load_ecg.py - SAMPLE_DATA_DIR: {SAMPLE_DATA_DIR}")
print(f"INFO: load_ecg.py - MODEL_OUTPUT_DIR: {MODEL_OUTPUT_DIR}")
print(f"INFO: load_ecg.py - RTM_THRESHOLD_MAP_DIR: {RTM_THRESHOLD_MAP_DIR}")


# Assicurati che le directory di output principali esistano
# Altre directory specifiche verranno create dagli script che le usano.
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SEGMENTED_SIGNAL_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST:
         print(f"⚠️  ATTENZIONE: load_ecg.py - Impossibile creare directory {e.filename}. "
               "Assicurati che i percorsi base siano accessibili e scrivibili.")
except NameError:
     pass

def list_records(ecg_database_dir=ECG_DIR):
    """Restituisce una lista di tutti i record disponibili nel dataset MIT-BIH."""
    try:
        files = [f.split(".")[0] for f in os.listdir(ecg_database_dir) if f.endswith(".dat")]
        return sorted(list(set(files)))
    except FileNotFoundError:
        print(f"❌ ERRORE: load_ecg.py - La directory ECG '{ecg_database_dir}' non è stata trovata.")
        return []
    except Exception as e:
        print(f"❌ ERRORE: load_ecg.py - Errore durante la lettura di '{ecg_database_dir}': {e}")
        return []

def load_ecg_record(record_name, ecg_database_dir=ECG_DIR):
    """Carica un record ECG e le sue annotazioni dal MIT-BIH."""
    record_path_base = os.path.join(ecg_database_dir, record_name)
    try:
        record = wfdb.rdrecord(record_path_base)
        annotations = wfdb.rdann(record_path_base, "atr") # Annotazioni di default 'atr'
        return record, annotations
    except FileNotFoundError:
        print(f"❌ ERRORE: load_ecg.py - File per il record '{record_name}' non trovati in '{ecg_database_dir}'.")
        print(f"   Verifica che esistano {record_name}.dat, {record_name}.hea, e {record_name}.atr")
        return None, None
    except Exception as e:
        print(f"❌ ERRORE: load_ecg.py - Durante il caricamento del record '{record_name}': {e}")
        return None, None

def plot_ecg_segment(record_name, start_sec=0, duration_sec=10, ecg_database_dir=ECG_DIR):
    """Plotta un segmento del segnale ECG con le annotazioni."""
    record, annotations = load_ecg_record(record_name, ecg_database_dir)
    if record is None:
        return

    # Assumiamo che il primo canale (indice 0) sia quello di interesse
    if record.p_signal.ndim > 1:
        signal = record.p_signal[:, 0]
    else:
        signal = record.p_signal

    fs = record.fs

    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)

    if end_sample > len(signal):
        end_sample = len(signal)
    if start_sample >= end_sample:
        print("ERRORE: load_ecg.py - Intervallo di start/duration non valido.")
        return

    segment = signal[start_sample:end_sample]
    time_axis = np.arange(len(segment)) / fs + start_sec

    beat_locs_samples = [s for s in annotations.sample if start_sample <= s < end_sample]
    # beat_labels = [annotations.symbol[i] for i, s in enumerate(annotations.sample) if start_sample <= s < end_sample]
    beat_locs_time = [(s - start_sample) / fs for s in beat_locs_samples]


    plt.figure(figsize=(15, 5)) # Aumentata dimensione per leggibilità
    plt.plot(time_axis, segment, label=f"ECG Lead (Canale 0)", color='blue')
    if beat_locs_time:
        plt.scatter([t + start_sec for t in beat_locs_time], # Aggiusta per l'asse temporale corretto
                    [segment[s_idx] for s_idx in range(len(segment)) if (start_sample + s_idx) in beat_locs_samples], # Modo più sicuro per ottenere ampiezze
                    color='red', marker='x', label="Battiti Annotati")

    plt.xlabel(f"Tempo (s) - Inizio da {start_sec}s")
    plt.ylabel("Ampiezza (mV)") # Assumendo mV, verifica unità da .hea
    plt.title(f"ECG Record {record_name} (Finestra: {start_sec}-{start_sec + duration_sec}s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n--- Test di load_ecg.py ---")
    available_records = list_records()
    if available_records:
        print(f"Record disponibili ({len(available_records)}): {available_records[:5]}...") # Mostra solo i primi 5
        # Plotta un segmento del primo record disponibile
        plot_ecg_segment(available_records[0], start_sec=10, duration_sec=5)
    else:
        print("Nessun record trovato. Verifica il percorso ECG_DIR.")

    print(f"\nVerifica percorsi definiti:")
    print(f"  ECG_DIR: {ECG_DIR} (Esiste: {os.path.exists(ECG_DIR)})")
    print(f"  NOISE_DIR: {NOISE_DIR} (Esiste: {os.path.exists(NOISE_DIR)})")
    print(f"  SEGMENTED_SIGNAL_DIR: {SEGMENTED_SIGNAL_DIR} (Esiste: {os.path.exists(SEGMENTED_SIGNAL_DIR)})")
    print(f"  SAMPLE_DATA_DIR: {SAMPLE_DATA_DIR} (Esiste: {os.path.exists(SAMPLE_DATA_DIR)})")
    print(f"  MODEL_OUTPUT_DIR: {MODEL_OUTPUT_DIR} (Esiste: {os.path.exists(MODEL_OUTPUT_DIR)})")
    # Il file specifico della mappa delle soglie verrà creato da s5_preprocessing