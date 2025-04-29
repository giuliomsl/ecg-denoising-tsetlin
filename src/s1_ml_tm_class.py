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
    (Implementazione come nella versione precedente)
    """
    if verbose_job: print(f"[Job {label_index}] Avvio training per '{label_name}'...")
    start_time = time.time()
    tm = None # Inizializza a None
    try:
        tm = class_instance_creator()
        best_val_acc = -1.0
        best_state_for_label = None
        last_valid_state = None
        acc_train_last = -1.0

        for epoch in range(epochs):
            tm.fit(X_train, y_train_label_i, epochs=1)
            # --- INIZIO MODIFICA: Gestione Stato MultiClassTM ---
            try:
                current_state_full = tm.get_state() # Ottiene lista di tuple [(cls0), (cls1)]
                # Verifica che lo stato sia nel formato atteso
                if isinstance(current_state_full, list) and len(current_state_full) >= 2:
                     last_valid_state = current_state_full # Salva lo stato completo
                else:
                     print(f"⚠️ [Job {label_index}] Formato stato TM inatteso: {type(current_state_full)}. Stato non salvato.")
                     last_valid_state = None # Non usare questo stato
                     # Potrebbe essere utile interrompere qui se lo stato è sempre errato
                     # break
            except Exception as e_getstate:
                 print(f"⚠️ [Job {label_index}] Errore tm.get_state() epoca {epoch+1}: {e_getstate}")
                 last_valid_state = None # Non fidarti dello stato
                 # break # Potrebbe essere meglio interrompere qui

            # --- FINE MODIFICA ---

            if X_val is not None and last_valid_state is not None: # Valuta solo se abbiamo uno stato valido
                try:
                    acc_val = accuracy_score(y_val_label_i, tm.predict(X_val))
                    if acc_val > best_val_acc:
                        best_val_acc = acc_val
                        best_state_for_label = last_valid_state # Salva lo stato completo migliore
                        if verbose_job and epoch % 5 == 0:
                             print(f"  [Job {label_index}] Ep {epoch+1} New Best Val: {best_val_acc:.4f}")
                except Exception as e_val:
                     print(f"⚠️ [Job {label_index}] Errore valutazione epoca {epoch+1}: {e_val}")
                     # Non aggiornare best_state se la valutazione fallisce

            elif epoch == epochs - 1 and verbose_job and last_valid_state is not None:
                 try:
                    acc_train_last = accuracy_score(y_train_label_i, tm.predict(X_train))
                 except Exception as e_train_acc:
                      print(f"⚠️ [Job {label_index}] Errore calcolo acc_train finale: {e_train_acc}")


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
        # Restituisce lo stato completo (lista di tuple) o None
        return label_index, final_state

    except Exception as e:
         print(f"❌ ERRORE nel job parallelo per label {label_index} ('{label_name}'): {e}")
         traceback.print_exc()
         return label_index, None


# --- Classe Manager Principale ---

class MultiLabelTsetlinMachine:
    """
    Gestisce un insieme di MultiClassTsetlinMachine (usate per compiti binari)
    per classificazione multi-label (Approccio Binary Relevance), con supporto
    per validazione, salvataggio dei migliori stati e training parallelo.
    """
    def __init__(self, n_labels, n_clauses=100, T=15, s=3.9, **kwargs):
        # ... (init come prima) ...
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
        # ... (come prima) ...
        try:
            params_for_mc = self.tm_params.copy()
            if 'indexed' not in params_for_mc: params_for_mc['indexed'] = True
            params_for_mc.pop('n_jobs', None)
            return MultiClassTsetlinMachine(**params_for_mc)
        except Exception as e:
            print(f"❌ Errore _create_tm_instance: {e}")
            raise

    def fit(self, X_train, Y_train, epochs=100, X_val=None, Y_val=None,
        use_best_state=True, n_jobs=-1, verbose=True, verbose_parallel=False):
        """
        Addestra le TM binarie in parallelo usando joblib.
        (Implementazione come nella versione precedente, usa _train_single_tm_job)
        """
        # ... (validazione input come prima) ...
        if verbose: print(f"\n⚙️ Inizio Addestramento Parallelo ({epochs} epoche per label, n_jobs={n_jobs})...")
        start_total_time = time.time()
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

        tasks = []
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
        if verbose: print(f"   Avvio di {len(tasks)} job paralleli...")
        results = Parallel(n_jobs=n_jobs, verbose=(5 if verbose else 0))(tasks)

        self.trained_states = [None] * self.n_labels
        successful_trainings = 0
        for result in results:
            if result is not None:
                label_index, final_state = result
                if final_state is not None:
                    # --- INIZIO MODIFICA: Verifica formato stato ---
                    # Verifica che final_state sia una lista di tuple come atteso da MultiClassTM
                    if isinstance(final_state, list) and len(final_state) >= 2 and isinstance(final_state[0], tuple) and isinstance(final_state[1], tuple):
                        self.trained_states[label_index] = final_state
                        successful_trainings += 1
                    else:
                         if verbose: print(f"⚠️ Stato restituito per label {label_index} ha formato inatteso: {type(final_state)}. Non salvato.")
                    # --- FINE MODIFICA ---
                else:
                    if verbose: print(f"⚠️ Training fallito o nessuno stato valido per label {label_index}.")
            else:
                 if verbose: print("⚠️ Ricevuto risultato None da un job parallelo.")

        end_total_time = time.time()
        total_elapsed = end_total_time - start_total_time
        if verbose:
            print(f"\n✅ Addestramento completato.")
            print(f"   Tempo totale (wall clock): {total_elapsed:.2f}s")
            print(f"   Addestrati con successo stati per {successful_trainings}/{self.n_labels} etichette.")
            if successful_trainings < self.n_labels: print("   ATTENZIONE: Alcuni training per etichetta potrebbero essere falliti.")


    def predict(self, X, verbose=False):
        """
        Prevede le etichette per i dati X usando gli stati TM salvati.
        Include un dummy fit e stampe di debug aggiuntive.
        """
        if not isinstance(X, np.ndarray): raise TypeError("Input X deve essere NumPy array.")
        if X.ndim != 2: raise ValueError(f"Input X deve essere 2D, ottenuto {X.ndim}D.")
        if X.shape[0] == 0: return np.zeros((0, self.n_labels), dtype=np.uint8)

        if verbose: print(f"\n🔍 Predizione Multi-Label su {X.shape[0]} campioni...")
        start_pred_time = time.time()
        predictions = np.zeros((X.shape[0], self.n_labels), dtype=np.uint8)
        n_features = X.shape[1]
        X_dummy = None
        y_dummy = None

        # --- INIZIO BLOCCO DEBUG STATI ---
        print(f"DEBUG (Predict): Lunghezza self.trained_states: {len(self.trained_states)}")
        # Stampa tipo e struttura di ogni stato prima del loop
        for idx, st in enumerate(self.trained_states):
             print(f"DEBUG (Predict): Stato per label {idx}: Tipo={type(st)}", end="")
             if isinstance(st, list):
                 print(f", Lunghezza={len(st)}", end="")
                 if len(st) > 0: print(f", Tipo[0]={type(st[0])}", end="")
                 if len(st) > 1: print(f", Tipo[1]={type(st[1])}", end="")
             print() # Nuova riga
        # --- FINE BLOCCO DEBUG STATI ---

        for i in range(self.n_labels):
            state = self.trained_states[i] # Stato per la label i

            # --- INIZIO BLOCCO DEBUG DETTAGLIATO PER LABEL SPECIFICA (es. i=6) ---
            # if i == 6: # Attiva questo blocco se l'errore persiste per label 6
            #     print(f"\nDEBUG DETTAGLIATO per Label {i}:")
            #     print(f"  Stato recuperato (self.trained_states[{i}]): Tipo={type(state)}")
            #     if isinstance(state, list):
            #         print(f"  Lunghezza stato: {len(state)}")
            #         for j, item in enumerate(state):
            #              print(f"    Elemento {j}: Tipo={type(item)}")
            #              if isinstance(item, tuple):
            #                   print(f"      Lunghezza tupla: {len(item)}")
            #                   # Stampa tipo e shape degli array nella tupla (se sono numpy array)
            #                   if len(item) >= 2 and isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
            #                        print(f"        Weights: Tipo={type(item[0])}, Shape={item[0].shape}, Dtype={item[0].dtype}")
            #                        print(f"        TA States: Tipo={type(item[1])}, Shape={item[1].shape}, Dtype={item[1].dtype}")
            #              elif isinstance(item, np.ndarray):
            #                   print(f"      Shape array: {item.shape}, Dtype={item.dtype}")
            #     elif state is None:
            #          print("  Stato è None.")
            #     else:
            #          print(f"  Stato ha tipo inatteso: {type(state)}")
            # --- FINE BLOCCO DEBUG DETTAGLIATO ---

            if state is not None:
                # Verifica aggiuntiva formato stato prima di usarlo
                if not (isinstance(state, list) and len(state) >= 2 and isinstance(state[0], tuple) and isinstance(state[1], tuple)):
                     print(f"⚠️ Formato stato non valido per label {i}. Tipo: {type(state)}. Output sarà 0.")
                     continue # Salta la predizione per questa label

                # Usa cache per istanze di predizione
                if i not in self._tms_predict_instances or self._tms_predict_instances[i] is None:
                    tm_eval = self._create_tm_instance()
                    if X_dummy is None:
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
                        # --- INIZIO BLOCCO DEBUG set_state ---
                        # print(f"DEBUG: Chiamata set_state per label {i}...")
                        tm_eval.set_state(state) # Passa lo stato completo (lista di 2 tuple)
                        # print(f"DEBUG: set_state per label {i} completato.")
                        # --- FINE BLOCCO DEBUG set_state ---

                        # --- INIZIO BLOCCO DEBUG predict ---
                        # print(f"DEBUG: Chiamata predict per label {i}...")
                        pred_result = tm_eval.predict(X)
                        # print(f"DEBUG: predict per label {i} completato. Shape output: {pred_result.shape}")
                        # --- FINE BLOCCO DEBUG predict ---

                        predictions[:, i] = pred_result.astype(np.uint8)

                    # Cattura specificamente IndexError
                    except IndexError as e_idx:
                         print(f"❌ Errore Indice durante predizione/set_state label {i}: {e_idx}. Output sarà 0.")
                         traceback.print_exc() # Stampa traceback completo
                    except Exception as e:
                        print(f"❌ Errore generico predizione label {i}: {e}. Output sarà 0.")
            else:
                if verbose: print(f"⚠️ Stato non disponibile per label {i}. Output sarà 0.")

        end_pred_time = time.time()
        if verbose: print(f"✅ Predizione completata in {end_pred_time - start_pred_time:.2f}s.")
        return predictions

    # --- Metodi get_trained_states, set_trained_states, save, load ---
    # (Invariati rispetto alla versione precedente)
    def get_trained_states(self):
        return self.trained_states

    def set_trained_states(self, states):
        if not isinstance(states, list) or len(states) != self.n_labels:
            raise ValueError(f"Input 'states' deve essere una lista di lunghezza {self.n_labels}.")
        self.trained_states = states
        self._tms_predict_instances = {}
        print(" Stati TM impostati manualmente.")

    def save(self, filepath):
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
    # ... (resto dell'esempio come prima) ...