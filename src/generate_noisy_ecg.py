import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt

# Cartelle dei dati
ECG_DIR = "data/mit-bih/"
NOISE_DIR = "data/noise_stress_test/"
OUTPUT_DIR = "data/noisy_ecg/"

# Assicuriamoci che la cartella di output esista
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ecg(record_name, start=0, duration=10):
    """Carica un segnale ECG dal MIT-BIH e ne estrae una finestra temporale."""
    record_path = os.path.join(ECG_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    fs = record.fs  # Frequenza di campionamento (360 Hz per MIT-BIH)

    # Definiamo il range di campioni da estrarre
    start_sample = int(start * fs)
    end_sample = int((start + duration) * fs)

    # Estrarre solo il segmento richiesto
    ecg_signal = record.p_signal[start_sample:end_sample, 0]
    return ecg_signal, fs

def load_noise(noise_name, length):
    """Carica un segnale di rumore dal Noise Stress Test Database e lo adatta alla lunghezza richiesta."""
    noise_path = os.path.join(NOISE_DIR, noise_name)
    noise_record = wfdb.rdrecord(noise_path)
    noise_signal = noise_record.p_signal[:, 0]

    # Ridimensioniamo il rumore alla lunghezza desiderata
    noise_signal = np.resize(noise_signal, length)
    return noise_signal

def generate_pli(length, fs=360, freq=50, amplitude=0.1):
    """Genera un segnale di Powerline Interference (PLI) sintetico a 50Hz (che è la frequenza comune europea, 
    mentre quella del Nord America è 60Hz)."""
    t = np.arange(length) / fs  # Tempo in secondi
    pli_signal = amplitude * np.sin(2 * np.pi * freq * t)
    return pli_signal

def generate_noisy_ecg(ecg_signal, bw_signal, ma_signal, pli_signal, noise_levels=(0.3, 0.2, 0.05)):
    """Combina ECG pulito con i rumori, normalizzandoli."""
    bw_weight, ma_weight, pli_weight = noise_levels

    # Normalizziamo i rumori e li scaliamo
    bw_signal = (bw_signal / np.max(np.abs(bw_signal))) * bw_weight
    ma_signal = (ma_signal / np.max(np.abs(ma_signal))) * ma_weight
    pli_signal = (pli_signal / np.max(np.abs(pli_signal))) * pli_weight

    # Creiamo il segnale rumoroso
    noisy_ecg = ecg_signal + bw_signal + ma_signal + pli_signal
    
    return noisy_ecg

def calculate_snr(original_signal, noisy_signal):
    """Calcola il rapporto segnale-rumore (SNR) in decibel."""
    power_signal = np.mean(original_signal ** 2)
    power_noise = np.mean((original_signal - noisy_signal) ** 2)
    snr_db = 10 * np.log10(power_signal / power_noise)
    return snr_db

def save_noisy_ecg(ecg_name, noisy_signal, start, duration):
    """Salva l'ECG rumoroso in formato .npy"""
    output_path = os.path.join(OUTPUT_DIR, f"{ecg_name}_noisy_{start}-{start+duration}s.npy")
    np.save(output_path, noisy_signal)
    print(f"✅ Salvato: {output_path}")

def process_and_save_noisy_ecg(ecg_name, start=0, duration=10):
    """Genera e salva un segnale ECG rumoroso su una finestra temporale selezionata."""
    print(f"🔹 Processando {ecg_name} (finestra {start}-{start+duration}s)...")

    # Carica il segmento ECG
    ecg_signal, fs = load_ecg(ecg_name, start, duration)
    length = len(ecg_signal)

    # Carica i rumori e li adatta alla stessa lunghezza dell'ECG
    bw_signal = load_noise("bw", length)
    ma_signal = load_noise("ma", length)
    pli_signal = generate_pli(length, fs)  # Generiamo PLI sintetico

    # Genera l'ECG rumoroso
    noisy_signal = generate_noisy_ecg(ecg_signal, bw_signal, ma_signal, pli_signal)

    # Calcola il SNR
    snr_value = calculate_snr(ecg_signal, noisy_signal)
    print(f"📊 SNR per {ecg_name} (finestra {start}-{start+duration}s): {snr_value:.2f} dB")

    # Salva il segnale rumoroso
    save_noisy_ecg(ecg_name, noisy_signal, start, duration)

    # Plot per verifica
    plt.figure(figsize=(12, 5))
    plt.plot(ecg_signal, label="ECG Pulito", color='blue')
    plt.plot(noisy_signal, label="ECG Rumoroso", alpha=0.7, color='orange')
    plt.xlabel("Campioni")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.title(f"ECG Pulito vs Rumoroso - {ecg_name} (Finestra {start}-{start+duration}s) - SNR: {snr_value:.2f} dB")
    plt.show()

if __name__ == "__main__":
    # Lista dei file ECG disponibili
    ecg_files = sorted([f.split(".")[0] for f in os.listdir(ECG_DIR) if f.endswith(".dat")])

    # Generiamo segnali rumorosi per i primi 5 ECG (finestra 30-40s)
    for ecg_name in ecg_files[:5]:
        process_and_save_noisy_ecg(ecg_name, start=30, duration=10)
