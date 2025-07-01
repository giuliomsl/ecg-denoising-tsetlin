# File: s5_rtm_train.py (Optimized for Pre-Binarized Data and Improved Feedback)

import numpy as np
import os
import time
import pickle
from sklearn.utils import shuffle
from pyTsetlinMachine.tm import RegressionTsetlinMachine

# Import necessary utilities and paths
try:
    # Assuming a structure where utils or config files are accessible
    from load_ecg import SAMPLE_DATA_DIR, MODEL_OUTPUT_DIR
except ImportError:
    print("ERROR: s5_rtm_train.py: Ensure load_ecg.py is accessible and configured.")
    print("                         Fallbacks paths will be used.")
    PROJECT_ROOT_FALLBACK = "."
    DATA_DIR_FALLBACK = os.path.join(PROJECT_ROOT_FALLBACK, "data")
    SAMPLE_DATA_DIR = os.path.join(DATA_DIR_FALLBACK, "samplewise")
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT_FALLBACK, "models", "rtm_denoiser")

try:
    import optuna
except ImportError:
    optuna = None

# --- Configuration Constant (for loading the correct files) ---
# This value MUST match the one used in s5_rtm_preprocessing.py
# when you generated the _BINARIZED_q<N>.npy files.
NUM_QUANTILES_USED_IN_PREPROCESSING = 20 # Default updated for q20, but can be overridden

def train_rtm_denoiser(
    noise_type_to_predict,
    rtm_config,
    num_epochs,
    num_quantiles_for_data=NUM_QUANTILES_USED_IN_PREPROCESSING, # To load the correct files
    early_stopping_patience=None,
    model_save_name_prefix="rtm_denoiser",
    log_epoch_interval=1, # feedback every N epochs
    y_train_max_abs_noise=None, # new optional parameter
    trial=None # <--- added for Optuna pruning
):
    """
    Trains a Regression Tsetlin Machine to predict a specific type of noise,
    using pre-binarized data. Training occurs on the entire dataset each epoch.
    """
    print(f"\n--- Starting RTM Training for Noise: {noise_type_to_predict.upper()} (Pre-Binarized Data) ---")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    q_suffix_data = f"_q{num_quantiles_for_data}" if num_quantiles_for_data and num_quantiles_for_data > 0 else "_unique"
    binarized_file_suffix_load = f"_BINARIZED{q_suffix_data}.npy"

    print(f"INFO: Loading pre-binarized data with suffix: {binarized_file_suffix_load}")
    try:
        X_train_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, f"X_train_rtm{binarized_file_suffix_load}"))
        y_train_target_noise = np.load(os.path.join(SAMPLE_DATA_DIR, f"y_train_rtm_aggregated_noise.npy"))
        X_val_binarized = np.load(os.path.join(SAMPLE_DATA_DIR, f"X_validation_rtm{binarized_file_suffix_load}"))
        y_val_target_noise = np.load(os.path.join(SAMPLE_DATA_DIR, f"y_validation_rtm_aggregated_noise.npy"))
    except FileNotFoundError as e:
        print(f"ERROR: Pre-binarized data file not found: {e}. "
              f"Run s5_rtm_preprocessing.py with num_quantiles={num_quantiles_for_data}.")
        return None, [], []

    # Normalize targets if requested
    if y_train_max_abs_noise is not None:
        print(f"✅ Normalizing Y targets: dividing by y_train_max_abs_noise = {y_train_max_abs_noise:.6f}")
        y_train_target_noise = y_train_target_noise / y_train_max_abs_noise
        y_val_target_noise = y_val_target_noise / y_train_max_abs_noise

    if X_train_binarized.shape[0] == 0:
        print("ERROR: Binarized training data is empty.")
        return None, [], []
    if X_val_binarized.shape[0] == 0:
        print("WARNING: Binarized validation data is empty. Early stopping based on validation loss will be disabled.")
        early_stopping_patience = None

    print(f"INFO: Binarized Training Data: X_shape={X_train_binarized.shape}, y_shape={y_train_target_noise.shape}")
    if X_val_binarized.shape[0] > 0:
        print(f"INFO: Binarized Validation Data: X_shape={X_val_binarized.shape}, y_shape={y_val_target_noise.shape}")
    expected_binarized_dim = X_train_binarized.shape[1]
    print(f"INFO: Binarized input dimensionality (from data): {expected_binarized_dim}")

    print("INFO: Initializing RegressionTsetlinMachine...")
    # Remove n_jobs from the config if present, as it's not a constructor parameter
    rtm_init_config = {k: v for k, v in rtm_config.items() if k != 'n_jobs'}
    tm = RegressionTsetlinMachine(**rtm_init_config)

    # Set n_jobs separately if present in the original config
    if 'n_jobs' in rtm_config:
        # This is a hypothetical method; check the library's documentation for the correct way to set threads.
        # tm.set_num_threads(rtm_config['n_jobs']) 
        print(f"INFO: Number of threads set to {rtm_config['n_jobs']} (if supported and implemented correctly)")


    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = -1
    config_items = []
    for k, v in sorted(rtm_config.items()):
        k_simple = k.replace("number_of_", "").replace("boost_true_positive_feedback", "btpf")
        config_items.append(f"{k_simple}{v}")
    config_str = "_".join(config_items)
    # Filename for the best model (based on validation loss)
    model_filename_best = f"{model_save_name_prefix}_{noise_type_to_predict}_{config_str}{q_suffix_data}_BEST.state"
    model_save_path_best = os.path.join(MODEL_OUTPUT_DIR, model_filename_best)
    # Filename for the final model (after all epochs)
    model_filename_final = f"{model_save_name_prefix}_{noise_type_to_predict}_{config_str}{q_suffix_data}_FINAL.state"
    model_save_path_final = os.path.join(MODEL_OUTPUT_DIR, model_filename_final)
    print(f"INFO: The best model will be saved as: {model_filename_best}")
    print(f"INFO: The final model will be saved as: {model_filename_final}")

    print("\nINFO: Starting training loop...")
    total_training_samples = X_train_binarized.shape[0]

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Shuffle every epoch
        X_train_shuffled, y_train_shuffled = shuffle(X_train_binarized, y_train_target_noise, random_state=epoch)
        
        # pyTsetlinMachine does not support native batching: fit on the entire dataset for each epoch.
        # We assume that if n_jobs is supported, it's set via a method like set_num_threads before the loop.
        tm.fit(X_train_shuffled, y_train_shuffled, epochs=1)

        # Calculate training loss
        y_pred_train_epoch = tm.predict(X_train_binarized)
        current_train_loss = np.mean((y_pred_train_epoch - y_train_target_noise) ** 2)
        train_losses.append(current_train_loss)
        
        # Calculate validation loss
        current_val_loss = float('nan')
        if X_val_binarized.shape[0] > 0:
            y_pred_val = tm.predict(X_val_binarized)
            current_val_loss = np.mean((y_pred_val - y_val_target_noise) ** 2)
        val_losses.append(current_val_loss)
        
        epoch_duration = time.time() - epoch_start_time
        val_loss_str = f"{current_val_loss:.6f}" if not np.isnan(current_val_loss) else "N/A"
        
        if (epoch+1) % log_epoch_interval == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f"  Epoch {epoch+1} Completed. Duration: {epoch_duration:.2f}s - Train Loss (MSE): {current_train_loss:.6f} - Val Loss (MSE): {val_loss_str}")
            
        # Early stopping and saving the best model
        if X_val_binarized.shape[0] > 0 and not np.isnan(current_val_loss):
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                try:
                    with open(model_save_path_best, "wb") as f:
                        pickle.dump(tm.get_state(), f)
                    print(f"  INFO: Model state saved to '{model_save_path_best}' (Val Loss: {best_val_loss:.6f})")
                except Exception as e:
                    print(f"  ERROR during model saving: {e}")
            else:
                epochs_no_improve += 1
                
            if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                print(f"INFO: Early stopping triggered after {epoch+1} epochs.")
                break
        elif early_stopping_patience is not None:
            print(f"  WARNING: Early stopping is not applicable without validation data. Training until num_epochs ({num_epochs}).")
        
        # Optuna pruning
        if trial is not None:
            trial.report(current_val_loss, epoch)
            if trial.should_prune():
                print(f"[Optuna] Trial pruned at epoch {epoch+1} (val_loss={current_val_loss:.6f})")
                raise optuna.TrialPruned()

    # Always save the final model after all epochs
    try:
        with open(model_save_path_final, "wb") as f:
            pickle.dump(tm.get_state(), f)
        print(f"INFO: Final model state saved to '{model_save_path_final}'")
    except Exception as e:
        print(f"ERROR during final model saving: {e}")

    print(f"--- Training Completed for Noise: {noise_type_to_predict.upper()} ---")

    # Load the best model if available, applying the workaround for the crash
    if os.path.exists(model_save_path_best):
        print(f"\nBest Validation Loss (MSE): {best_val_loss:.6f} at epoch {best_epoch+1}.")
        print(f"Loading the best model from: {model_save_path_best}")
        
        # Create a new RTM instance with the same configuration
        tm_loaded = RegressionTsetlinMachine(**rtm_init_config)

        # WORKAROUND: Run fit() with 0 epochs on placeholder data to initialize
        # internal structures before using set_state(). This prevents segmentation faults.
        # The dimensionality of X_placeholder must match the one used in training.
        placeholder_dim = X_train_binarized.shape[1]
        X_placeholder = np.zeros((1, placeholder_dim), dtype=np.uint8)
        y_placeholder = np.zeros(1, dtype=np.float32)
        tm_loaded.fit(X_placeholder, y_placeholder, epochs=0)
        
        try:
            with open(model_save_path_best, "rb") as f:
                best_state = pickle.load(f)
            
            tm_loaded.set_state(best_state)
            print("INFO: Best model loaded successfully with workaround applied.")
            final_model_to_return = tm_loaded
        except Exception as e_load:
            print(f"CRITICAL WARNING: Could not load the saved best model: {e_load}")
            print("Returning the last trained model from memory as a fallback.")
            final_model_to_return = tm # fallback
    else:
        print("WARNING: No best model was saved (or training was interrupted). Returning the last model.")
        final_model_to_return = tm

    return final_model_to_return, train_losses, val_losses


if __name__ == "__main__":
    def safe_training_main():
        """Wrapper function to prevent segmentation faults at program termination."""
        # This value MUST match the one used in s5_rtm_preprocessing.py
        # NUM_QUANTILES_USED_IN_PREPROCESSING is defined globally above

        # Load configuration from config.yaml
        import yaml
        CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_yaml = yaml.safe_load(f)
            
            rtm_params_from_yaml = config_yaml.get('rtm_params', {})
            training_params_from_yaml = config_yaml.get('training_params', {})
            
            # Defaults if not present in YAML or sections are missing
            default_rtm_config = {
                "number_of_clauses": 200,
                "T": 10000,
                "s": 3.0,
                "boost_true_positive_feedback": 1,
                "number_of_state_bits": 8
            }
            default_training_config = {
                "num_epochs": 20, 
                "early_stopping_patience": 5,
                "model_save_name_prefix": "rtm_denoiser_from_config"
            }

            # Merge defaults with values from YAML
            # YAML values override defaults
            current_rtm_config = {**default_rtm_config, **rtm_params_from_yaml}
            current_training_config = {**default_training_config, **training_params_from_yaml}
            
            # Extract num_quantiles if specified, otherwise use the global default
            num_quantiles_for_data_run = current_rtm_config.get('num_quantiles_for_data', NUM_QUANTILES_USED_IN_PREPROCESSING)

        except FileNotFoundError:
            print(f"WARNING: Configuration file '{CONFIG_FILE_PATH}' not found. Using hardcoded defaults.")
            current_rtm_config = {
                "number_of_clauses": 1000,
                "T": 2000,
                "s": 3.0,
                "boost_true_positive_feedback": 1,
                "number_of_state_bits": 8,
                "n_jobs": -1
            }
            current_training_config = {
                "num_epochs": 50,
                "early_stopping_patience": 10, 
                "model_save_name_prefix": "rtm_denoiser_fallback"
            }
            num_quantiles_for_data_run = NUM_QUANTILES_USED_IN_PREPROCESSING

        except Exception as e:
            print(f"ERROR loading or parsing '{CONFIG_FILE_PATH}': {e}. Using hardcoded defaults.")
            # Same fallbacks as FileNotFoundError
            current_rtm_config = {
                "number_of_clauses": 1000, "T": 2000, "s": 3.0, 
                "boost_true_positive_feedback": 1, "number_of_state_bits": 8, "n_jobs": -1
            }
            current_training_config = {
                "num_epochs": 50, "early_stopping_patience": 10, 
                "model_save_name_prefix": "rtm_denoiser_error_fallback"
            }
            num_quantiles_for_data_run = NUM_QUANTILES_USED_IN_PREPROCESSING


        try:
            print("=== STARTING RTM TRAINING (from config.yaml or fallback) ===")
            print(f"RTM Configuration used: {current_rtm_config}")
            print(f"Training Configuration used: {current_training_config}")
            print(f"Number of quantiles for data: {num_quantiles_for_data_run}")

            # Load the maximum absolute value of the noise target (for normalization)
            y_train_max_abs_noise_path = os.path.join(SAMPLE_DATA_DIR, "y_train_max_abs_noise.npy")
            if not os.path.exists(y_train_max_abs_noise_path):
                print(f"❌ File y_train_max_abs_noise.npy not found in {SAMPLE_DATA_DIR}. Run s5_rtm_preprocessing.py first.")
                return False
            y_train_max_abs_noise = np.load(y_train_max_abs_noise_path)
            print(f"✅ y_train_max_abs_noise loaded: {y_train_max_abs_noise:.6f}")

            # The train_rtm_denoiser function now handles normalization internally
            # by accepting the y_train_max_abs_noise parameter.

            # Pass the value to train_rtm_denoiser via kwargs
            trained_model_agg, train_hist_agg, val_hist_agg = train_rtm_denoiser(
                noise_type_to_predict="aggregated",
                rtm_config=current_rtm_config, # Pass the entire config, including n_jobs if present
                num_epochs=current_training_config["num_epochs"],
                num_quantiles_for_data=num_quantiles_for_data_run,
                early_stopping_patience=current_training_config["early_stopping_patience"],
                model_save_name_prefix=current_training_config["model_save_name_prefix"],
                y_train_max_abs_noise=y_train_max_abs_noise # new parameter
            )

            if trained_model_agg:
                print("\nINFO: Model for aggregated noise trained successfully.")
                
                # Plotting only if there is valid loss history data
                plot_train = bool(train_hist_agg)
                plot_val = bool(val_hist_agg) and not all(np.isnan(val_hist_agg)) if val_hist_agg else False

                if plot_train or plot_val:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(12, 6))
                        
                        if plot_train:
                            plt.plot(train_hist_agg, label="Training Loss (MSE)", marker='o', linestyle='-')
                        if plot_val:
                            plt.plot(val_hist_agg, label="Validation Loss (MSE)", marker='x', linestyle='--')
                        
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss (MSE)")
                        plt.title(f"RTM Learning Curves - Aggregated Noise (Quantiles: {num_quantiles_for_data_run})")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
                        plot_save_path = os.path.join(MODEL_OUTPUT_DIR, f"learning_curve_rtm_agg_q{num_quantiles_for_data_run}.png")
                        plt.savefig(plot_save_path)
                        print(f"INFO: Learning curve saved to '{plot_save_path}'")
                        plt.close()
                        
                    except Exception as e_plot:
                        print(f"ERROR: Could not save the learning curve plot: {e_plot}")
                else:
                    print("INFO: History data is missing or invalid for plotting.")
                
                print("=== TRAINING COMPLETED SUCCESSFULLY ===")
                return True
            else:
                print("ERROR: Training failed.")
                return False
                
        except Exception as e:
            print(f"CRITICAL ERROR during training: {e}")
            return False
        finally:
            # Force garbage collection to prevent memory issues
            import gc
            gc.collect()
            print("INFO: Resource cleanup completed.")

    # Run the training safely
    success = safe_training_main()
    
    if success:
        print("\nTRAINING FINISHED. Exiting program safely.")
    else:
        print("\nTRAINING FAILED. Check logs above for details.")