import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# === Carica un segmento ECG preprocessato ===
record = "205"
data = np.load(f"data/processed/{record}_noisy.npy") 
signal = data[0][:, 0]  # Usa solo la prima feature grezza (es. "sopra mediana")

# === Usa il segnale analogico originale se disponibile ===
# Per ora usiamo la feature binarizzata come proxy visiva

# === Applichiamo quantizzazione adattiva ===
n_bins = 4
signal_reshaped = signal.reshape(-1, 1)

quantizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
quantized_levels = quantizer.fit_transform(signal_reshaped).astype(int).flatten()

# === Costruisci il thermometer encoding ===
thermo_encoded = np.array([[1 if i <= val else 0 for i in range(n_bins)] for val in quantized_levels])

# === Plot ===
plt.figure(figsize=(12, 6))

# --- Segnale binarizzato e livelli ---
plt.subplot(2, 1, 1)
plt.plot(signal, label="Segnale (feature 0)", color='black')
plt.scatter(np.arange(len(quantized_levels)), quantized_levels, label="Livello quantizzato", c=quantized_levels, cmap='viridis', s=10)
plt.title("Segnale binario + livelli quantizzati")
plt.xlabel("Campioni")
plt.ylabel("Livello")
plt.legend()

# --- Thermometer encoding (tacche binarie) ---
plt.subplot(2, 1, 2)
for i in range(n_bins):
    plt.plot(thermo_encoded[:, i] + i * 1.2, label=f"Thermo bin {i}")  # spostamento verticale per separare le linee
plt.title("Thermometer encoding (binari)")
plt.xlabel("Campioni")
plt.yticks([])
plt.legend()

plt.tight_layout()
plt.show()
