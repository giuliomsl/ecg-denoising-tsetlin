import os
import numpy as np

class ECGDenoisingDataset:
    def __init__(self, processed_dir="data/processed/", record_ids=None):
        """
        Loads the preprocessed binned data (noisy and clean).
        """
        self.processed_dir = processed_dir
        self.record_ids = record_ids or self._detect_available_records()

        self.X_noisy, self.X_clean = self._load_all()

    def _detect_available_records(self):
        files = os.listdir(self.processed_dir)
        record_ids = sorted({f.split("_")[0] for f in files if f.endswith("_noisy.npy")})
        return record_ids

    def _load_all(self):
        X_noisy_list = []
        X_clean_list = []

        for rid in self.record_ids:
            noisy_path = os.path.join(self.processed_dir, f"{rid}_noisy.npy")
            clean_path = os.path.join(self.processed_dir, f"{rid}_clean.npy")

            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                X_noisy = np.load(noisy_path)
                X_clean = np.load(clean_path)

                X_noisy_list.append(X_noisy)
                X_clean_list.append(X_clean)
            else:
                print(f"⚠️ Missing files for record {rid}, ignored.")

        return np.concatenate(X_noisy_list), np.concatenate(X_clean_list)

    def get_data(self):
        """Returns (X_noisy, X_clean) as a tuple"""
        return self.X_noisy, self.X_clean

    def summary(self):
        print("✅ Dataset loaded")
        print(f"Records used: {self.record_ids}")
        print(f"Shape X_noisy: {self.X_noisy.shape}")
        print(f"Shape X_clean: {self.X_clean.shape}")


class ECGSamplewiseDataset:
    def __init__(self, sample_dir="data/samplewise/"):
        self.X_path = os.path.join(sample_dir, "X_train_samples.npy")
        self.y_path = os.path.join(sample_dir, "y_train_samples.npy")

        if not os.path.exists(self.X_path) or not os.path.exists(self.y_path):
            raise FileNotFoundError("⚠️ Sample files not found. Please run preprocessing first.")

        self.X = np.load(self.X_path)
        self.y = np.load(self.y_path)

    def get_data(self):
        return self.X, self.y

    def summary(self):
        print("✅ Sample-wise dataset loaded.")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
