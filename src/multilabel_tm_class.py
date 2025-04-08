# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import time
import errno # Per gestione errori directory
from sklearn.metrics import accuracy_score
# Rimuovi import train_test_split se non usato qui
# from sklearn.model_selection import train_test_split

# Import per parallelizzazione
from joblib import Parallel, delayed
import traceback # Utile per debug errori in processi paralleli

try:
    # Usa MultiClassTsetlinMachine come base per ogni compito binario
    from pyTsetlinMachine.tm import MultiClassTsetlinMachine
except ImportError:
    print("Errore: Libreria pyTsetlinMachine o classe MultiClassTsetlinMachine non trovata.")
    exit()
except NameError:
     print("Errore: Nome MultiClassTsetlinMachine non definito nel modulo importato.")
     exit()


# --- Funzione Helper per Training Parallelo ---
#     Definita fuori dalla classe per facilitare il pickling da parte di joblib

def _train_single_tm_job(label_index, label_name, X_train, y_train_label_i,
                         X_val, y_val_label_i, tm_params, epochs,
                         use_best_state, class_instance_creator, verbose_job):
    """
    Funzione eseguita da ogni processo parallelo per addestrare una singola TM.

    Args:
        label_index (int): Indice dell'etichetta.
        label_name (str): Nome dell'etichetta (per logging).
        X_train, y_train_label_i: Dati di training per questa etichetta.
        X_val, y_val_label_i: Dati di validazione (possono essere None).
        tm_params (dict): Parametri per creare l'istanza TM.
        epochs (int): Numero di epoche.
        use_best_state (bool): Se salvare il miglior stato da validazione.
        class_instance_creator (function): Funzione che crea un'istanza TM.
        verbose_job (bool): Se stampare output dettagliato da questo job.

    Returns:
        tuple: (int, object): Indice dell'etichetta e lo stato TM addestrato (o None).
    """
    if verbose_job: print(f"[Job {label_index}] Avvio training per '{label_name}'...")
    start_time = time.time()
    tm = None # Inizializza a None
    try:
        # Crea istanza TM usando la funzione passata
        tm = class_instance_creator()

        best_val_acc = -1.0
        best_state_for_label = None
        last_valid_state = None
        acc_train_last = -1.0 # Per log finale se no validazione

        for epoch in range(epochs):
            # Fit
            tm.fit(X_train, y_train_label_i, epochs=1)
            current_state = tm.get_state() # Ottieni stato subito
            last_valid_state = current_state

            # Valutazione (se applicabile)
            if X_val is not None:
                acc_val = accuracy_score(y_val_label_i, tm.predict(X_val))
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_state_for_label = current_state
                    # Stampa meno verbosa in parallelo
                    if verbose_job and epoch % 5 == 0: # Stampa ogni 5 epoche
                         print(f"  [Job {label_index}] Ep {epoch+1} New Best Val: {best_val_acc:.4f}")
            # Calcola acc train per log finale (se no validazione)
            elif epoch == epochs - 1 and verbose_job:
                 acc_train_last = accuracy_score(y_train_label_i, tm.predict(X_train))


        # Decide quale stato restituire
        final_state = None
        log_msg = ""
        if use_best_state and best_state_for_label is not None:
            final_state = best_state_for_label
            log_msg = f"(Best Val Acc: {best_val_acc:.4f})"
        elif last_valid_state is not None:
            final_state = last_valid_state
            log_msg = f"(Last State, Train Acc: {acc_train_last:.4f})"
        else:
             log_msg = "(Nessuno stato valido ottenuto!)"

        end_time = time.time()
        if verbose_job: print(f"[Job {label_index}] Fine training per '{label_name}' in {end_time - start_time:.2f}s {log_msg}")
        return label_index, final_state

    except Exception as e:
         # Cattura eccezioni nel processo figlio e le riporta
         print(f"❌ ERRORE nel job parallelo per label {label_index} ('{label_name}'): {e}")
         # Stampa traceback completo per debug
         traceback.print_exc()
         return label_index, None # Restituisce None per indicare fallimento


# --- Classe Manager Principale ---

class MultiLabelTsetlinMachine: # Rinominata come richiesto
    """
    Gestisce un insieme di MultiClassTsetlinMachine (usate per compiti binari)
    per classificazione multi-label (Approccio Binary Relevance), con supporto
    per validazione, salvataggio dei migliori stati e training parallelo.
    """
    def __init__(self, n_labels, n_clauses=100, T=15, s=3.9, **kwargs):
        if not isinstance(n_labels, int) or n_labels <= 0:
            raise ValueError("n_labels deve essere un intero positivo.")

        self.n_labels = n_labels
        self.tm_params = {
            "number_of_clauses": n_clauses,
            "T": T,
            "s": s,
            **kwargs
        }
        self.trained_states = [None] * n_labels
        self._tms_predict_instances = {} # Cache per predizione
        print(f"✅ Manager inizializzato per {self.n_labels} etichette.")
        print(f"   Parametri TM per ogni label: {self.tm_params}")
        print(f"   (Userà MultiClassTsetlinMachine per ogni compito binario)")

    def _create_tm_instance(self):
        """Crea una nuova istanza di MultiClassTsetlinMachine."""
        # Questa funzione viene passata ai job paralleli
        try:
            params_for_mc = self.tm_params.copy()
            if 'indexed' not in params_for_mc: params_for_mc['indexed'] = True
            # Rimuovi n_jobs se presente, joblib gestisce il parallelismo esterno
            params_for_mc.pop('n_jobs', None)
            return MultiClassTsetlinMachine(**params_for_mc)
        except Exception as e:
            print(f"❌ Errore _create_tm_instance: {e}")
            raise

    def fit(self, X_train, Y_train, epochs=100, X_val=None, Y_val=None,
        use_best_state=True, n_jobs=-1, verbose=True, verbose_parallel=False):
        """
        Addestra le TM binarie in parallelo usando joblib.

        Args:
            X_train, Y_train: Dati di training.
            epochs (int): Epoche per ogni TM.
            X_val, Y_val: Dati di validazione (opzionale).
            use_best_state (bool): Salva miglior stato da validazione.
            n_jobs (int): Numero di processi paralleli per joblib (-1 usa tutti i core).
            verbose (bool): Stampa messaggi principali del manager.
            verbose_parallel (bool): Stampa messaggi dettagliati da ogni job parallelo.
        """
        if verbose: print(f"\n⚙️ Inizio Addestramento Parallelo ({epochs} epoche per label, n_jobs={n_jobs})...")
        start_total_time = time.time()

        # --- Validazione Input ---
        if X_train.shape[0] != Y_train.shape[0]: raise ValueError("Incoerenza campioni X_train / Y_train.")
        if Y_train.shape[1] != self.n_labels: raise ValueError(f"Y_train ha {Y_train.shape[1]} colonne, attese {self.n_labels}.")
        use_validation = X_val is not None and Y_val is not None
        if use_validation:
            if X_val.shape[0] != Y_val.shape[0]: raise ValueError("Incoerenza campioni X_val / Y_val.")
            if Y_val.shape[1] != self.n_labels: raise ValueError(f"Y_val ha {Y_val.shape[1]} colonne, attese {self.n_labels}.")
            if verbose: print("   (Modalità con validazione attiva)")
        elif use_best_state:
             if verbose: print("   (Attenzione: use_best_state=True ma dati di validazione non forniti. Verrà salvato l'ultimo stato.)")
             use_best_state = False

        # --- Preparazione Task Paralleli ---
        tasks = []
        # Creiamo una lista di etichette/nomi per logging, se non li abbiamo
        label_names = [f"Label_{i}" for i in range(self.n_labels)]

        for i in range(self.n_labels):
            y_train_label_i = Y_train[:, i].astype(np.uint32)
            y_val_label_i = Y_val[:, i].astype(np.uint32) if use_validation else None
            tasks.append(
                delayed(_train_single_tm_job)(
                    i, label_names[i], X_train, y_train_label_i,
                    X_val, y_val_label_i, self.tm_params, epochs,
                    use_best_state, self._create_tm_instance, verbose_parallel
                )
            )

        # --- Esecuzione Parallela ---
        # Prefer backend 'loky' (default) o 'multiprocessing' per processi reali
        # verbose=5 o 10 dà output da joblib stesso
        if verbose: print(f"   Avvio di {len(tasks)} job paralleli...")
        results = Parallel(n_jobs=n_jobs, verbose=(5 if verbose else 0))(tasks)

        # --- Raccolta Risultati ---
        self.trained_states = [None] * self.n_labels # Inizializza/Resetta
        successful_trainings = 0
        for result in results:
            if result is not None: # Se il job non ha fallito
                label_index, final_state = result
                if final_state is not None:
                    self.trained_states[label_index] = final_state
                    successful_trainings += 1
                else:
                    if verbose: print(f"⚠️ Training fallito o nessuno stato valido per label {label_index}.")
            else:
                 # Questo non dovrebbe accadere se _train_single_tm_job ritorna sempre una tupla
                 if verbose: print("⚠️ Ricevuto risultato None da un job parallelo.")


        end_total_time = time.time()
        total_elapsed = end_total_time - start_total_time
        if verbose:
            print(f"\n✅ Addestramento completato.")
            print(f"   Tempo totale (wall clock): {total_elapsed:.2f}s")
            print(f"   Addestrati con successo stati per {successful_trainings}/{self.n_labels} etichette.")
            if successful_trainings < self.n_labels:
                 print("   ATTENZIONE: Alcuni training per etichetta potrebbero essere falliti.")


    def predict(self, X, verbose=False):
        """
        Prevede le etichette per i dati X usando gli stati TM salvati.
        Include un dummy fit per inizializzare le istanze MultiClassTM.
        """
        # ... (Logica di predict quasi identica a prima, ma usa _create_tm_instance) ...
        if not isinstance(X, np.ndarray): raise TypeError("Input X deve essere NumPy array.")
        if X.ndim != 2: raise ValueError(f"Input X deve essere 2D, ottenuto {X.ndim}D.")
        if X.shape[0] == 0: return np.zeros((0, self.n_labels), dtype=np.uint8)

        if verbose:
            print(f"\n🔍 Predizione Multi-Label su {X.shape[0]} campioni...")
        start_pred_time = time.time()
        predictions = np.zeros((X.shape[0], self.n_labels), dtype=np.uint8)

        # Dummy data per inizializzazione (se necessario)
        n_features = X.shape[1]
        X_dummy = None
        y_dummy = None

        for i in range(self.n_labels):
            state = self.trained_states[i]
            if state is not None:
                # Usa cache per istanze di predizione
                if i not in self._tms_predict_instances or self._tms_predict_instances[i] is None:
                    tm_eval = self._create_tm_instance()
                    if X_dummy is None: # Crea solo se serve
                         X_dummy = np.zeros((2, n_features), dtype=np.uint8)
                         y_dummy = np.array([0, 1], dtype=np.uint32)
                    try:
                         tm_eval.fit(X_dummy, y_dummy, epochs=0)
                         self._tms_predict_instances[i] = tm_eval
                    except Exception as e:
                         print(f"❌ Errore dummy fit per predizione label {i}: {e}")
                         self._tms_predict_instances[i] = None
                         continue
                else:
                     tm_eval = self._tms_predict_instances[i]

                # Esegui Predizione
                if tm_eval is not None:
                    try:
                        tm_eval.set_state(state)
                        predictions[:, i] = tm_eval.predict(X).astype(np.uint8)
                    except Exception as e:
                        print(f"❌ Errore predizione label {i}: {e}. Output sarà 0.")
            else:
                if verbose: print(f"⚠️ Stato non disponibile per label {i}. Output sarà 0.")

        end_pred_time = time.time()
        if verbose: print(f"✅ Predizione completata in {end_pred_time - start_pred_time:.2f}s.")
        return predictions

    # --- Metodi get_trained_states, set_trained_states, save, load ---
    #     (Possono rimanere come nella versione precedente)
    def get_trained_states(self):
        return self.trained_states

    def set_trained_states(self, states):
        if not isinstance(states, list) or len(states) != self.n_labels:
            raise ValueError(f"Input 'states' deve essere una lista di lunghezza {self.n_labels}.")
        self.trained_states = states
        self._tms_predict_instances = {} # Resetta cache predizione
        print(" Stati TM impostati manualmente.")

    def save(self, filepath):
        """Salva i parametri e gli stati addestrati in un file .pkl"""
        data_to_save = {
            'n_labels': self.n_labels,
            'tm_params': self.tm_params,
            'trained_states': self.trained_states
        }
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

# --- Esempio di Utilizzo (invariato) ---
if __name__ == "__main__":
    # ... (l'esempio di utilizzo rimane lo stesso) ...
    print("\n--- Esempio Utilizzo MultiLabelTsetlinMachine (Parallelo) ---")
    noise_labels_example = ['BW', 'MA', 'PLI']
    tm_parameters_example = {
        "number_of_clauses": 200, "T": 150, "s": 2.5,
        "number_of_state_bits": 8, "boost_true_positive_feedback": 1,
        "indexed": True
    }
    n_labels = len(noise_labels_example)
    print("\nGenerazione dati fittizi...")
    n_samples_train = 500; n_samples_val = 100; n_features = 50
    X_train = np.random.randint(0, 2, size=(n_samples_train, n_features), dtype=np.uint8)
    y_train = np.random.randint(0, 2, size=(n_samples_train, n_labels), dtype=np.uint8)
    X_val = np.random.randint(0, 2, size=(n_samples_val, n_features), dtype=np.uint8)
    y_val = np.random.randint(0, 2, size=(n_samples_val, n_labels), dtype=np.uint8)
    print("Dati fittizi generati.")

    manager = MultiLabelTsetlinMachine(n_labels, **tm_parameters_example)
    # Addestra in parallelo (es. usando tutti i core)
    manager.fit(X_train, y_train, epochs=5, X_val=X_val, Y_val=y_val, # <-- Usa Y_val
                use_best_state=True, n_jobs=-1, verbose=True, verbose_parallel=False)

    model_path = "models/parallel_tm_manager_example.pkl"
    manager.save(model_path)
    manager_loaded = MultiLabelTsetlinMachine.load(model_path)
    predictions = manager_loaded.predict(X_val)

    print("\nEsempio Predizioni (prime 5):")
    print(f"Etichette: {noise_labels_example}")
    for i in range(min(5, n_samples_val)): print(f"Campione {i}: Reale={y_val[i]}, Predetto={predictions[i]}")

    print("\nValutazione Accuratezza (per label sul set di validazione):")
    for i in range(manager_loaded.n_labels):
        label_name = f"Label_{i}";
        if i < len(noise_labels_example): label_name = noise_labels_example[i]
        label_accuracy = accuracy_score(y_val[:, i], predictions[:, i])
        print(f"  Accuratezza per '{label_name}': {label_accuracy:.4f}")