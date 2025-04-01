import wfdb
import os
import matplotlib.pyplot as plt

# Percorso ai dati
DATA_DIR = "data/mit-bih/"

def list_records():
    """Restituisce una lista di tutti i record disponibili nel dataset."""
    files = [f.split(".")[0] for f in os.listdir(DATA_DIR) if f.endswith(".dat")]
    return sorted(set(files))

def load_ecg(record_name):
    """Carica un record ECG dal MIT-BIH"""
    record_path = os.path.join(DATA_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, "atr")  # Carica le annotazioni
    return record, annotations

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
