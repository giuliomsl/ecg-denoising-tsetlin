import numpy as np
import os
import pickle
from pyTsetlinMachine.tm import RegressionTsetlinMachine

# Percorsi
X_test_path = 'data/samplewise/X_test_rtm_BINARIZED_q20.npy'
y_test_path = 'data/samplewise/y_test_rtm_aggregated_noise.npy'
y_max_path = 'data/samplewise/y_train_max_abs_noise.npy'
model_path = 'models/rtm_denoiser/rtm_denoiser_c10k_T1k_s3_Ynorm_aggregated_T1000_btpf1_clauses10000_state_bits8_s3.0_q20_BEST.state'

# Configurazione RTM
config = {
    'number_of_clauses': 10000,
    'T': 1000,
    's': 3.0,
    'boost_true_positive_feedback': 1,
    'number_of_state_bits': 8
}

# Caricamento dati
X = np.load(X_test_path)
y = np.load(y_test_path)
y_max = np.load(y_max_path)

# Caricamento modello
rtm = RegressionTsetlinMachine(**config)
# Workaround: fit a 0 epoche per inizializzare la struttura interna
rtm.fit(X[:1], y[:1], epochs=0)
with open(model_path, 'rb') as f:
    rtm.set_state(pickle.load(f))

# Predizione (output normalizzato)
y_pred_norm = rtm.predict(X)
# De-normalizzazione
y_pred = y_pred_norm * y_max

# MSE finale
mse = np.mean((y_pred - y) ** 2)
print(f"Test MSE (denormalizzato): {mse:.6f}")

# Mostra anche la baseline
mse_baseline = np.mean(y ** 2)
print(f"Baseline MSE (zero): {mse_baseline:.6f}")

# Miglioramento percentuale
improvement = (mse_baseline - mse) / mse_baseline * 100
print(f"Miglioramento percentuale rispetto alla baseline: {improvement:.2f}%")
