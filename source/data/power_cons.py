import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset


def load_powercons(path: Path):
    df = pd.read_csv(path, sep='\t', header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y).astype(np.int64)
    return X, y_encoded, le.classes_


class PowerConsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
