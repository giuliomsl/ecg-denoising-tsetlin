## Struttura del progetto

```
ecg-denoising-tsetlin/
│
├── src/                      # Codice sorgente
│   ├── load_ecg.py
│   ├── generate_noisy_ecg.py
│   ├── preprocessing.py
│   ├── binarization.py
│   ├── ecg_dataset.py
│   ├── ecg_utils.py
│   ├── train_tm.py
│   └── ...
│
├── config/                   # Configurazioni YAML
├── data/                     # (Ignorata da Git) Contiene segnali .npy, MIT-BIH, rumore
├── models/                   # (Ignorata) Salvataggio dei modelli
├── requirements.txt
├── .gitignore
└── README.md
```


# denoising_ecg

Progetto per il denoising del segnale ECG con Tsetlin Machines.

1) load_ecg: caricamento e visualizzazione segnali reali puliti

2) generate_noisy_ecg.py: Aggiunta rumore sintetico ai segnali puliti (BW, MA, PLI a 50Hz) e calcola SNR

3) preprocessing.py + binarization.py: Segmenta il segnale in X campioni (1024 al momento, circa 3 secondi) e lo binarizza in varie modalità (trovare la più efficiente!!!)

4) check_preprocessed.py + visualize_thermometer_encoding.py: Mostra le feature binarie e fa il confronto tra segnale noisy/clean

5) ecg_dataset.py: Carica dataset binarizzati pronti per addestrare e/o valutare TM

6) ecg_utils.py: Calcolo SNR, normalizzazione, plot

# ecg-denoising-tsetlin
4c5d37317d411302f794e38dfe8dc327fa66c152
