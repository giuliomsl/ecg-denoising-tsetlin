import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURAZIONE ===
RECORD_ID = "205"
NUM_FINESTRE = 3  # Quante finestre plottare
FEATURE_NAMES = ["Soglia", "Gradiente", "Quant1", "Quant2", "Quant3", "Quant4"]

# === PATH ===
BASE_PATH = "data/processed"
CLEAN_PATH = os.path.join(BASE_PATH, f"{RECORD_ID}_clean.npy")
NOISY_PATH = os.path.join(BASE_PATH, f"{RECORD_ID}_noisy.npy")

# === CARICAMENTO DATI ===
if not (os.path.exists(CLEAN_PATH) and os.path.exists(NOISY_PATH)):
    raise FileNotFoundError("⚠️ Uno dei file clean/noisy non è stato trovato.")

clean_data = np.load(CLEAN_PATH)
noisy_data = np.load(NOISY_PATH)

print(f"✅ Dati caricati: {RECORD_ID}_clean.npy e {RECORD_ID}_noisy.npy")
print(f"Forma dei dati: {clean_data.shape} (finestre, campioni, feature)")

# === ESTRATTO PRIMI CAMPIONI ===
print("\n🧾 Primi 10 campioni della prima finestra (clean):")
print(clean_data[0][:10])

print("\n🧾 Primi 10 campioni della prima finestra (noisy):")
print(noisy_data[0][:10])

# === DISTRIBUZIONE BINARIA NELLA PRIMA FINESTRA ===
print("\n📊 Distribuzione binaria nella PRIMA FINESTRA:")
num_features = clean_data.shape[2]
for f in range(num_features):
    name = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f"Feature {f}"
    clean_0, clean_1 = np.sum(clean_data[0][:, f] == 0), np.sum(clean_data[0][:, f] == 1)
    noisy_0, noisy_1 = np.sum(noisy_data[0][:, f] == 0), np.sum(noisy_data[0][:, f] == 1)
    print(f"{name:<10} | Clean: 0 → {clean_0}, 1 → {clean_1} | Noisy: 0 → {noisy_0}, 1 → {noisy_1}")

# === PLOT ===
for i in range(min(NUM_FINESTRE, clean_data.shape[0])):
    # --- Plot separati per ogni feature ---
    fig, axs = plt.subplots(num_features, 1, figsize=(12, 2 * num_features), sharex=True)
    for f in range(num_features):
        label = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f"Feature {f}"
        axs[f].plot(clean_data[i][:, f], label="Clean", color='green')
        axs[f].plot(noisy_data[i][:, f], label="Noisy", color='red', linestyle='--', alpha=0.7)
        axs[f].set_ylabel(label)
        axs[f].set_ylim(-0.1, 1.1)
        axs[f].set_yticks([0, 1])
        axs[f].legend(loc="upper right")
        axs[f].grid(True, linestyle="--", alpha=0.4)

    axs[-1].set_xlabel("Campioni")
    fig.suptitle(f"📊 Plot separati - Record {RECORD_ID} - Finestra {i}")
    plt.tight_layout()
    plt.show()

print("\n✅ Visualizzazione completata!")
