"""
Data loading utilities for video sequence datasets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib


class VideoSequenceDataset(Dataset):
    """
    Dataset class for loading video sequences from preprocessed .npy files.
    """

    def __init__(self, features, labels=None, transform=None):
        """
        Args:
            features: Numpy array of shape (num_videos, num_frames, feature_dim)
            labels: Numpy array of shape (num_videos,) or None for test data
            transform: Optional transform to apply to features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]

        if self.transform:
            features = self.transform(features)

        if self.labels is not None:
            return features, self.labels[idx].unsqueeze(0)
        else:
            return features


def load_preprocessed_data(features_dir, dataset_percentage=1.0, random_state=42):
    """
    Load preprocessed feature sequences from .npy files.

    Args:
        features_dir: Directory containing the .npy files
        dataset_percentage: Percentage of training data to use
        random_state: Random seed for reproducibility

    Returns:
        X_train_sequences: Training features
        X_test_sequences: Test features
        y_train: Training labels
        scaler: Fitted StandardScaler object
    """

    percentage_str = str(int(dataset_percentage * 100))
    train_features_file = os.path.join(
        features_dir, f"X_train_sequences_{percentage_str}pct.npy"
    )
    test_features_file = os.path.join(features_dir, "X_test_sequences.npy")
    scaler_file = os.path.join(
        features_dir, f"scaler_attention_{percentage_str}pct.joblib"
    )

    print(f"Loading training features from {train_features_file}")
    X_train_sequences = np.load(train_features_file)
    print(f"Loading test features from {test_features_file}")
    X_test_sequences = np.load(test_features_file)

    data_base_path = os.path.expanduser("./data-nexar/")
    df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df["id"] = df["id"].astype(str).str.zfill(5)

    if dataset_percentage < 1.0:
        df = df.sample(frac=dataset_percentage, random_state=random_state).reset_index(
            drop=True
        )

    y_train = df["target"].values

    if os.path.exists(scaler_file):
        print(f"Loading scaler from {scaler_file}")
        scaler = joblib.load(scaler_file)
    else:
        print("Creating new scaler...")
        scaler = StandardScaler()
        num_videos, num_frames, num_features = X_train_sequences.shape
        X_train_reshaped = X_train_sequences.reshape(-1, num_features)
        scaler.fit(X_train_reshaped)
        joblib.dump(scaler, scaler_file)
        print(f"Scaler saved to {scaler_file}")

    print("Scaling features...")
    X_train_scaled = scale_sequences(X_train_sequences, scaler)
    X_test_scaled = scale_sequences(X_test_sequences, scaler)

    print(f"Training sequences shape: {X_train_scaled.shape}")
    print(f"Test sequences shape: {X_test_scaled.shape}")
    print(f"Training labels shape: {y_train.shape}")

    return X_train_scaled, X_test_scaled, y_train, scaler


def scale_sequences(sequences, scaler):
    """
    Apply scaling to sequence data.

    Args:
        sequences: Numpy array of shape (num_videos, num_frames, num_features)
        scaler: Fitted StandardScaler object

    Returns:
        scaled_sequences: Scaled sequences with same shape as input
    """
    num_videos, num_frames, num_features = sequences.shape

    sequences_reshaped = sequences.reshape(-1, num_features)
    sequences_scaled = scaler.transform(sequences_reshaped)

    return sequences_scaled.reshape(num_videos, num_frames, num_features)


def create_data_loaders(
    X_train,
    X_test,
    y_train,
    test_size=0.2,
    batch_size=32,
    random_state=42,
    num_workers=4,
):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        test_size: Fraction of training data to use for validation
        batch_size: Batch size for DataLoaders
        random_state: Random seed for train/val split
        num_workers: Number of workers for DataLoaders

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        stratify=y_train,
        random_state=random_state,
    )

    train_dataset = VideoSequenceDataset(X_tr, y_tr)
    val_dataset = VideoSequenceDataset(X_val, y_val)
    test_dataset = VideoSequenceDataset(X_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
