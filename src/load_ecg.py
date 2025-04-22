import wfdb
import os
import matplotlib.pyplot as plt
import sys
import errno

# --- FLAG PER AMBIENTE ---
# Imposta a True se stai eseguendo su Google Colab, False se in locale
RUNNING_ON_COLAB = False # <--- MODIFICA QUESTA RIGA MANUALMENTE
# -------------------------

# --- Definizione Percorsi basata sul Flag ---
if RUNNING_ON_COLAB:
    print("INFO: Flag RUNNING_ON_COLAB impostato a True.")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
             print("INFO: Montaggio Google Drive...")
             drive.mount('/content/drive', force_remount=True)
             import time
             time.sleep(5)
        else:
             print("INFO: Google Drive già montato.")
    except ImportError:
         print("ERRORE: Impossibile importare google.colab. Assicurati di essere in Colab.")
         # Potresti voler uscire qui o impostare percorsi di default
         exit()


    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/" # <--- Modifica se necessario
    REPO_NAME = "denoising_ecg"  # <--- Modifica col nome del tuo repo clonato
    PROJECT_ROOT = f"/content/{REPO_NAME}/" # Percorso del progetto clonato in Colab

    DATA_DIR = os.path.join(GDRIVE_BASE, "data")
    MODEL_DIR = os.path.join(GDRIVE_BASE, "models")

else:
    print("INFO: Flag RUNNING_ON_COLAB impostato a False (Ambiente Locale).")
    # Definisci i percorsi relativi o assoluti per l'ambiente locale
    # Assumiamo esecuzione dalla root del progetto per semplicità
    PROJECT_ROOT = "."
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Costruisci percorsi specifici che verranno usati nel resto dello script
ECG_DIR = os.path.join(DATA_DIR, "mit-bih/")
NOISE_DIR = os.path.join(DATA_DIR, "noise_stress_test/")
NOISY_ECG_DIR = os.path.join(DATA_DIR, "noisy_ecg/")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "samplewise/")
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "multi_tm_denoiser/")
BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

print(f"INFO: Usando DATA_DIR: {DATA_DIR}")
print(f"INFO: Usando MODEL_DIR: {MODEL_DIR}")

# Assicurati che le directory di output esistano (necessario crearle
# sia in locale che su Drive se non esistono già)
try:
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOISY_ECG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
except OSError as e:
     if e.errno != errno.EEXIST:
         print(f"⚠️ Attenzione: impossibile creare directory {e.filename}. Assicurati che il percorso base esista.")
except NameError: # Se errno non è importato
     pass # Ignora l'errore se non possiamo controllare errno

def list_records():
    """Restituisce una lista di tutti i record disponibili nel dataset MIT-BIH."""
    try:
        # Usa la variabile ECG_DIR definita sopra
        files = [f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")]
        # Rimuovi duplicati (se header/atr hanno lo stesso nome base) e ordina
        return sorted(list(set(files)))
    except FileNotFoundError:
        print(f"❌ Errore: La directory ECG '{ECG_DIR}' non è stata trovata.")
        return []
    except Exception as e:
        print(f"❌ Errore durante la lettura di '{ECG_DIR}': {e}")
        return []

def load_ecg(record_name):
    """Carica un record ECG e le sue annotazioni dal MIT-BIH"""
    # Usa la variabile ECG_DIR definita sopra
    record_path_base = os.path.join(ECG_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_path_base)
        # Le annotazioni di default hanno estensione 'atr'
        annotations = wfdb.rdann(record_path_base, "atr")
        return record, annotations
    except FileNotFoundError:
        print(f"❌ Errore: File per il record '{record_name}' non trovati in '{ECG_DIR}'.")
        print(f"   Verifica che esistano {record_name}.dat, {record_name}.hea, e {record_name}.atr")
        return None, None
    except Exception as e:
        print(f"❌ Errore durante il caricamento del record '{record_name}': {e}")
        return None, None

def plot_ecg(record_name, start=0, duration=10):
    """
    Plotta un segmento del segnale ECG con le annotazioni.
    
    - record_name: Nome del file ECG (es. "103")
    - start: Secondo di inizio della finestra da visualizzare
    - duration: Durata della finestra da visualizzare (in secondi)
    """
    record, annotations = load_ecg(record_name)
    signal = record.p_signal[:, 0]  # Prende il primo canale ECG
    fs = record.fs  # Frequenza di campionamento (360 Hz)

    # Definisce i campioni da visualizzare
    start_sample = int(start * fs)
    end_sample = int((start + duration) * fs)

    # Estrae il segmento del segnale
    segment = signal[start_sample:end_sample]

    # Filtra le annotazioni per la finestra selezionata
    beat_locs = [s for s in annotations.sample if start_sample <= s < end_sample]
    beat_labels = [annotations.symbol[i] for i, s in enumerate(annotations.sample) if start_sample <= s < end_sample]

    # Plot del segmento di ECG
    plt.figure(figsize=(12, 5))
    plt.plot(segment, label="ECG Lead 1", color='blue')
    plt.scatter([s - start_sample for s in beat_locs], [segment[s - start_sample] for s in beat_locs], color='red', marker='x', label="Battiti Cardiaci")
    plt.xlabel("Campioni")
    plt.ylabel("Ampiezza")
    plt.title(f"ECG Record {record_name} (Finestra: {start}-{start+duration}s)")
    plt.legend()
    plt.show()

# Test - Carica e mostra un segmento di un record ECG
if __name__ == "__main__":
    available_records = list_records()
    print("Record disponibili:", available_records)

    if available_records:
        plot_ecg(available_records[3], start=30, duration=10)  # Zoom sui secondi 30-40
