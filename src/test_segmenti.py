from ecg_dataset import ECGDenoisingDataset
ds = ECGDenoisingDataset("data/processed/", ["100", "101", "102", "103", "104"])
X, y = ds.get_data()
print(X.shape, y.shape)
