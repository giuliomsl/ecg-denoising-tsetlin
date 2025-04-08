# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import errno
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    # Assicurati che l'import sia corretto basato sul file tm.py fornito
    from pyTsetlinMachine.tm import MultiClassTsetlinMachine
except ImportError:
    print("Errore: Libreria pyTsetlinMachine o classe MultiClassTsetlinMachine non trovata.")
    exit()
except NameError:
     print("Errore: Nome MultiClassTsetlinMachine non definito nel modulo importato.")
     exit()


class MultiLabelTsetlinMachine:
    """
    Gestisce un insieme di MultiClassTsetlinMachine (usate per compiti binari)
    per classificazione multi-label (Approccio Binary Relevance), con supporto
    per validazione e salvataggio dei migliori stati.
    """
    def __init__(self, n_labels, n_clauses=100, T=15, s=3.9, **kwargs):
        if not isinstance(n_labels, int) or n_labels <= 0:
            raise ValueError("n_labels deve essere un intero positivo.")

        self.n_labels = n_labels
        self.tm_params = {
            "number_of_clauses": n_clauses,
            "T": T,
            "s": s,
            **kwargs # Passa altri parametri come number_of_state_bits, ecc.
        }
        # Lista per contenere gli stati finali/migliori addestrati
        self.trained_states = [None] * n_labels
        # Cache per istanze TM usate nella predizione (per evitare dummy fit multipli)
        self._tms_predict_instances = {}
        print(f"✅ Manager inizializzato per {self.n_labels} etichette.")
        print(f"   Parametri TM per ogni label: {self.tm_params}")
        print(f"   (Userà MultiClassTsetlinMachine per ogni compito binario)")

    def _create_tm_instance(self):
        """Crea una nuova istanza di MultiClassTsetlinMachine."""
        try:
            # Passa i parametri conservati
            # Nota: MultiClassTsetlinMachine potrebbe avere parametri leggermente diversi
            #       da TsetlinMachine base, assicurati che tm_params sia compatibile.
            #       In particolare, 'indexed' è un parametro di MultiClassTsetlinMachine.
            #       Se non lo passi in kwargs, potresti volerlo aggiungere qui.
            params_for_mc = self.tm_params.copy()
            if 'indexed' not in params_for_mc:
                 params_for_mc['indexed'] = True # Default comune, verifica se appropriato
            return MultiClassTsetlinMachine(**params_for_mc)
        except TypeError as e:
            print(f"❌ Errore inizializzazione MultiClassTsetlinMachine: {e}")
            print(f"   Parametri forniti: {self.tm_params}")
            print(f"   Verifica che i parametri siano validi per MultiClassTsetlinMachine.")
            raise
        except NameError:
             print("❌ Errore: Classe MultiClassTsetlinMachine non definita.")
             raise

    def fit(self, X_train, Y_train, epochs=100, X_val=None, Y_val=None, use_best_state=True, verbose=True):
        """
        Addestra una MultiClassTsetlinMachine (per compito binario) per ogni etichetta.
        (Logica interna quasi identica a prima, ma usa _create_tm_instance che
         ora crea MultiClassTsetlinMachine)
        """
        print(f"\n⚙️ Inizio Addestramento ({epochs} epoche per label)...")
        start_total_time = time.time()

        # --- Validazione Input ---
        # ... (come prima) ...
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("Incoerenza campioni X_train / Y_train.")
        if Y_train.shape[1] != self.n_labels:
            raise ValueError(f"Y_train ha {Y_train.shape[1]} colonne, attese {self.n_labels}.")

        use_validation = X_val is not None and Y_val is not None
        if use_validation:
            if X_val.shape[0] != Y_val.shape[0]:
                raise ValueError("Incoerenza campioni X_val / Y_val.")
            if Y_val.shape[1] != self.n_labels:
                raise ValueError(f"Y_val ha {Y_val.shape[1]} colonne, attese {self.n_labels}.")
            if verbose: print("   (Modalità con validazione attiva)")
        elif use_best_state:
             if verbose: print("   (Attenzione: use_best_state=True ma dati di validazione non forniti. Verrà salvato l'ultimo stato.)")
             use_best_state = False

        total_training_time = 0
        self.trained_states = [None] * self.n_labels

        for i in range(self.n_labels):
            label_name = f"Label_{i}"
            if verbose:
                print(f"\n--- Addestramento per '{label_name}' ({i+1}/{self.n_labels}) ---")
            start_label_time = time.time()

            y_train_label_i = Y_train[:, i].astype(np.uint32)
            y_val_label_i = Y_val[:, i].astype(np.uint32) if use_validation else None

            tm = self._create_tm_instance()
            # Non serve salvare l'istanza qui se non la riusiamo tra epoche

            best_val_acc = -1.0
            best_state_for_label = None
            last_valid_state = None # Per salvare l'ultimo stato se non c'è validazione

            for epoch in range(epochs):
                try:
                    tm.fit(X_train, y_train_label_i, epochs=1)
                except Exception as e:
                    print(f"❌ Errore fit epoca {epoch+1} label {i}: {e}")
                    break # Interrompe per questa label

                # --- Valutazione Epoca ---
                acc_train = -1.0
                acc_val = -1.0
                current_state = None
                try:
                    if verbose:
                        acc_train = accuracy_score(y_train_label_i, tm.predict(X_train))
                    if use_validation:
                        acc_val = accuracy_score(y_val_label_i, tm.predict(X_val))
                    current_state = tm.get_state() # Ottieni stato dopo fit
                    last_valid_state = current_state # Aggiorna ultimo stato valido
                except Exception as e:
                    print(f"⚠️ Errore valutazione/get_state epoca {epoch+1} label {i}: {e}")
                    current_state = None # Non possiamo usare questo stato

                # Stampa progresso
                if verbose:
                    if use_validation:
                        print(f"  Epoch {epoch+1}/{epochs}: Train Acc={acc_train:.4f}, Val Acc={acc_val:.4f}", end="")
                    else:
                        print(f"  Epoch {epoch+1}/{epochs}: Train Acc={acc_train:.4f}", end="")

                # Aggiorna stato migliore
                if use_validation and current_state is not None:
                    if acc_val > best_val_acc:
                        best_val_acc = acc_val
                        best_state_for_label = current_state
                        if verbose: print(" *Best*")
                    else:
                         if verbose: print("")
                elif verbose:
                     print("")

            # --- Fine Epoche per Label i ---
            if use_best_state and best_state_for_label is not None:
                self.trained_states[i] = best_state_for_label
                if verbose: print(f"  -> Stato migliore salvato (Val Acc: {best_val_acc:.4f})")
            elif last_valid_state is not None: # Salva l'ultimo stato valido ottenuto
                 self.trained_states[i] = last_valid_state
                 if verbose: print(f"  -> Stato finale salvato (Train Acc: {acc_train:.4f})")
            else:
                 self.trained_states[i] = None
                 if verbose: print(f"  ❌ Nessuno stato valido salvato per label {i}.")

            end_label_time = time.time()
            elapsed_label = end_label_time - start_label_time
            total_training_time += elapsed_label
            if verbose: print(f"  Tempo per '{label_name}': {elapsed_label:.2f}s")

        end_total_time = time.time()
        print(f"\n✅ Addestramento completato. Tempo totale: {total_training_time:.2f}s")


    def predict(self, X):
        """
        Prevede le etichette per i dati X usando gli stati TM salvati.
        Include un dummy fit per inizializzare le istanze MultiClassTM.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("L'input X deve essere un array NumPy.")
        if X.ndim != 2:
             raise ValueError(f"L'input X deve essere 2D (n_samples, n_features), ottenuto {X.ndim}D.")
        if X.shape[0] == 0:
             print("⚠️ Attenzione: Input X per predict è vuoto. Restituito array vuoto.")
             return np.zeros((0, self.n_labels), dtype=np.uint8)

        print(f"\n🔍 Predizione Multi-Label su {X.shape[0]} campioni...")
        start_pred_time = time.time()
        predictions = np.zeros((X.shape[0], self.n_labels), dtype=np.uint8)

        # --- Dummy data per inizializzazione ---
        n_features = X.shape[1]
        # Crea dummy data solo se necessario (almeno una TM da inizializzare)
        X_dummy = None
        y_dummy = None

        for i in range(self.n_labels):
            state = self.trained_states[i]
            if state is not None:
                # Crea/Recupera istanza TM per predizione
                # Usiamo una cache per evitare di ricreare e fare dummy fit ogni volta
                if i not in self._tms_predict_instances or self._tms_predict_instances[i] is None:
                    tm_eval = self._create_tm_instance()
                    # --- Esegui Dummy Fit ---
                    if X_dummy is None: # Crea dummy data solo la prima volta
                         X_dummy = np.zeros((2, n_features), dtype=np.uint8)
                         y_dummy = np.array([0, 1], dtype=np.uint32) # Classi 0 e 1
                    try:
                         # Fit minimale per inizializzare strutture interne
                         tm_eval.fit(X_dummy, y_dummy, epochs=0) # epochs=0 o 1
                         self._tms_predict_instances[i] = tm_eval # Salva nella cache
                         # print(f"   (Dummy fit eseguito per istanza predizione label {i})") # Debug
                    except Exception as e:
                         print(f"❌ Errore durante dummy fit per label {i}: {e}")
                         self._tms_predict_instances[i] = None # Marca come non utilizzabile
                         continue # Salta la predizione per questa label
                else:
                     tm_eval = self._tms_predict_instances[i] # Recupera dalla cache

                # --- Esegui Predizione Reale ---
                if tm_eval is not None:
                    try:
                        tm_eval.set_state(state) # Imposta lo stato addestrato
                        predictions[:, i] = tm_eval.predict(X).astype(np.uint8)
                    except Exception as e:
                        print(f"❌ Errore predizione label {i}: {e}. Output sarà 0.")
                        # predictions[:, i] rimane 0
            else:
                print(f"⚠️ Stato non disponibile per label {i}. Output sarà 0.")

        end_pred_time = time.time()
        print(f"✅ Predizione completata in {end_pred_time - start_pred_time:.2f}s.")
        return predictions

    # --- Metodi get_trained_states, set_trained_states, save, load ---
    #     (Possono rimanere come nella versione precedente)
    def get_trained_states(self):
        """Restituisce la lista degli stati TM addestrati (migliori o finali)."""
        return self.trained_states

    def set_trained_states(self, states):
        """
        Imposta manualmente gli stati addestrati. Utile dopo il caricamento.
        Verifica che il numero di stati corrisponda a n_labels.
        """
        if not isinstance(states, list) or len(states) != self.n_labels:
            raise ValueError(f"Input 'states' deve essere una lista di lunghezza {self.n_labels}.")
        self.trained_states = states
        # Resetta la cache delle istanze di predizione perché gli stati sono cambiati
        self._tms_predict_instances = {}
        print(" Stati TM impostati manualmente (cache predizione resettata).")

    def save(self, filepath):
        """Salva i parametri e gli stati addestrati in un file .pkl"""
        # Assicurati che la directory esista
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except OSError as e:
            # Ignora errore se la directory esiste già, altrimenti solleva eccezione
            if e.errno != errno.EEXIST:
                print(f"❌ Errore creazione directory per {filepath}: {e}")
                raise

        data_to_save = {
            'n_labels': self.n_labels,
            'tm_params': self.tm_params,
            'trained_states': self.trained_states
        }
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data_to_save, f)
            print(f"💾 Manager Multi-label TM salvato in {filepath}")
        except Exception as e:
            print(f"❌ Errore durante salvataggio in {filepath}: {e}")

    @classmethod
    def load(cls, filepath):
        """Carica parametri e stati da un file .pkl e restituisce una nuova istanza."""
        print(f"📂 Caricamento Manager Multi-label TM da {filepath}...")
        try:
            with open(filepath, "rb") as f:
                data_loaded = pickle.load(f)

            if not all(k in data_loaded for k in ['n_labels', 'tm_params', 'trained_states']):
                 raise ValueError("File .pkl non contiene i dati attesi.")

            # Crea nuova istanza con i parametri caricati
            # Passa tm_params come kwargs a __init__
            instance = cls(n_labels=data_loaded['n_labels'], **data_loaded['tm_params'])
            instance.set_trained_states(data_loaded['trained_states'])
            print("✅ Manager caricato con successo.")
            return instance
        except FileNotFoundError:
            print(f"❌ Errore: File non trovato - {filepath}")
            raise
        except Exception as e:
            print(f"❌ Errore durante caricamento da {filepath}: {e}")
            raise

# --- Esempio Utilizzo (invariato, ma ora usa MultiClassTM internamente) ---
if __name__ == "__main__":
     # ... (l'esempio di utilizzo rimane lo stesso di prima) ...
    print("\n--- Esempio Utilizzo MultiLabelTsetlinMachine ---")
    noise_labels_example = ['BW', 'MA', 'PLI']
    tm_parameters_example = {
        "number_of_clauses": 200,
        "T": 150,
        "s": 2.5,
        "number_of_state_bits": 8,
        "boost_true_positive_feedback": 1,
        "indexed": True # Parametro specifico di MultiClassTsetlinMachine
    }
    n_labels = len(noise_labels_example)
    print("\nGenerazione dati fittizi...")
    n_samples_train = 500
    n_samples_val = 100
    n_features = 50
    X_train = np.random.randint(0, 2, size=(n_samples_train, n_features), dtype=np.uint8)
    y_train = np.random.randint(0, 2, size=(n_samples_train, n_labels), dtype=np.uint8)
    X_val = np.random.randint(0, 2, size=(n_samples_val, n_features), dtype=np.uint8)
    y_val = np.random.randint(0, 2, size=(n_samples_val, n_labels), dtype=np.uint8)
    print("Dati fittizi generati.")

    manager = MultiLabelTsetlinMachine(n_labels, **tm_parameters_example)
    manager.fit(X_train, y_train, epochs=3, X_val=X_val, y_val=y_val, use_best_state=True, verbose=True)

    model_path = "models/mltm_manager_example_mc.pkl" # Nome file diverso
    manager.save(model_path)

    manager_loaded = MultiLabelTsetlinMachine.load(model_path)
    predictions = manager_loaded.predict(X_val)

    print("\nEsempio Predizioni (prime 5):")
    print(f"Etichette: {noise_labels_example}")
    for i in range(min(5, n_samples_val)):
        print(f"Campione {i}: Reale={y_val[i]}, Predetto={predictions[i]}")

    print("\nValutazione Accuratezza (per label sul set di validazione):")
    for i in range(manager_loaded.n_labels):
        label_name = f"Label_{i}"
        if i < len(noise_labels_example): label_name = noise_labels_example[i]
        label_accuracy = accuracy_score(y_val[:, i], predictions[:, i])
        print(f"  Accuratezza per '{label_name}': {label_accuracy:.4f}")