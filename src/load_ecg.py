import wfdb
import os
import matplotlib.pyplot as plt
import sys

# --- Rilevamento Ambiente e Definizione Percorsi ---

# Verifica se siamo in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("INFO: Rilevato ambiente Google Colab.")
    from google.colab import drive
    # Monta Google Drive se non già montato
    if not os.path.exists('/content/drive/MyDrive'):
         print("INFO: Montaggio Google Drive...")
         drive.mount('/content/drive')
         # Attendi un attimo per assicurarti che il mount sia completo
         import time
         time.sleep(5)
    else:
         print("INFO: Google Drive già montato.")

    # Definisci i percorsi base per Colab (ASSUMENDO la tua struttura su Drive)
    GDRIVE_BASE = "/content/drive/MyDrive/Tesi_ECG_Denoising/" # Modifica se necessario
    REPO_NAME = "TUO_REPO_NAME" # Il nome della cartella clonata da GitHub
    PROJECT_ROOT_COLAB = f"/content/{REPO_NAME}/" # Percorso del progetto clonato

    # Percorsi Dati su Drive
    DATA_DIR_COLAB = os.path.join(GDRIVE_BASE, "data")
    ECG_DIR = os.path.join(DATA_DIR_COLAB, "mit-bih/")
    NOISE_DIR = os.path.join(DATA_DIR_COLAB, "noise_stress_test/")
    NOISY_ECG_DIR = os.path.join(DATA_DIR_COLAB, "noisy_ecg/")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR_COLAB, "samplewise/")

    # Percorsi Output Modelli su Drive
    MODEL_OUTPUT_DIR = os.path.join(GDRIVE_BASE, "models/multi_tm_denoiser/")

    # Percorso per info binarizzazione su Drive
    BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

    # Assicurati che le directory di output esistano su Drive
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    # Potrebbe essere necessario creare anche NOISY_ECG_DIR e SAMPLE_DATA_DIR se
    # esegui anche i preprocessing su Colab la prima volta.
    os.makedirs(NOISY_ECG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

else:
    print("INFO: Rilevato ambiente Locale.")
    # Definisci i percorsi relativi per l'ambiente locale
    # Assumiamo che lo script sia eseguito dalla root del progetto
    # o che i percorsi relativi funzionino dalla posizione dello script.
    # Se esegui da src/, potresti dover usare '../data' etc.
    PROJECT_ROOT_LOCAL = "." # O specifica il percorso assoluto/relativo corretto
    DATA_DIR_LOCAL = os.path.join(PROJECT_ROOT_LOCAL, "data")

    ECG_DIR = os.path.join(DATA_DIR_LOCAL, "mit-bih/")
    NOISE_DIR = os.path.join(DATA_DIR_LOCAL, "noise_stress_test/")
    NOISY_ECG_DIR = os.path.join(DATA_DIR_LOCAL, "noisy_ecg/")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR_LOCAL, "samplewise/")
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT_LOCAL, "models/multi_tm_denoiser/")
    BIN_INFO_PATH = os.path.join(SAMPLE_DATA_DIR, "binarization_info.pkl")

    # Assicurati che le directory di output esistano localmente
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOISY_ECG_DIR, exist_ok=True) # Se generate_noisy_ecg è separato
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True) # Se preprocessing è separato

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
