
import numpy as np
import pickle
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from ecg_dataset import ECGDenoisingDataset
import os
from ecg_utils import plot_signal_comparison

# === CONFIG ===
RECORD_IDS = ["100", "101", "102", "103", "104"]  # Record IDs da usare
PROCESSED_DIR = "data/processed/"
TARGET_BIT_INDEX = 0  # Quale bit del segnale pulito vogliamo predire
OUTPUT_MODEL_PATH = "models/tm_bitwise_model.npy"

CLAUSES = 100
T = 15
S = 3.9
EPOCHS = 30

# === Load binarized data ===
print("🔹 Caricamento finestre binarizzate...")

dataset = ECGDenoisingDataset(PROCESSED_DIR, RECORD_IDS)
X_raw, y_full = dataset.get_data()

# Controllo che il numero di finestre coincida
assert X_raw.shape[0] == y_full.shape[0], f"Incoerenza: {X_raw.shape[0]} finestre X vs {y_full.shape[0]} finestre y"

# Reshape per la TM
X = X_raw.reshape(X_raw.shape[0], -1)  # Flatten: (n_finestre, 1024 * n_features)

# Estraiamo il bit target e calcoliamo il voto di maggioranza per finestra
y = y_full[:, :, TARGET_BIT_INDEX]  # (n_finestre, 1024)
y_majority = (np.mean(y, axis=1) > 0.5).astype(int)  # (n_finestre,)

print("✅ Dati pronti:")
print("X shape:", X.shape)
print("y shape:", y_majority.shape)

# === TM training ===
print("⚙️ Inizializzazione MultiClassTsetlinMachine...")
tm = MultiClassTsetlinMachine(CLAUSES, T, S)

print("🚀 Training per", EPOCHS, "epoche...")
for epoch in range(EPOCHS):
    tm.fit(X, y_majority, epochs=1)
    acc = 100 * (tm.predict(X) == y_majority).mean()
    print(f"🧠 Epoca {epoch+1}: Accuratezza = {acc:.2f}%")

# === Salva modello ===
os.makedirs("models", exist_ok=True)
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(tm.get_state(), f)

# === Visualizza esempio ===
idx = 0
pred = tm.predict(X[idx].reshape(1, -1))[0]
label = y_majority[idx]
print(f"🔍 Finestra {idx}: Predetto = {pred}, Reale = {label}")