"""
Extract video IDs of misclassified samples (False Negatives and False Positives)
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import sys


save_dir = "results/video_id_check"
os.makedirs(save_dir, exist_ok=True)
current_dir = os.path.dirname(os.path.abspath(__file__))
codebase_dir = os.path.join(current_dir, "..")
sys.path.append(codebase_dir)

from module_hierarchical_transformer import HierarchicalTransformer
from utils.data_loader import VideoSequenceDataset


def get_misclassified_by_type(threshold=0.5):
    """Get two separate lists: False Negatives and False Positives"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_base_path = os.path.expanduser("./data-nexar/")
    df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df["id"] = df["id"].astype(str).str.zfill(5)

    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, stratify=train_val_df["target"], random_state=42
    )

    print(f"Test set size: {len(test_df)}")

    features_dir = "features/ablation_cache"
    X_train = np.load(os.path.join(features_dir, "train_rgb_efficientnet_features.npy"))
    X_test = np.load(os.path.join(features_dir, "test_rgb_efficientnet_features.npy"))

    flow_train = np.load(
        os.path.join(features_dir, "train_flow_efficientnet_features.npy")
    )
    flow_test = np.load(
        os.path.join(features_dir, "test_flow_efficientnet_features.npy")
    )

    flow_train_padded = np.pad(flow_train, ((0, 0), (1, 0), (0, 0)), mode="constant")
    flow_test_padded = np.pad(flow_test, ((0, 0), (1, 0), (0, 0)), mode="constant")

    X_train_combined = np.concatenate([X_train, flow_train_padded], axis=2)
    X_test_combined = np.concatenate([X_test, flow_test_padded], axis=2)

    num_train_videos, num_frames, num_features = X_train_combined.shape
    X_train_reshaped = X_train_combined.reshape(-1, num_features)

    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)

    X_test_reshaped = X_test_combined.reshape(-1, num_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(
        len(test_df), num_frames, num_features
    )

    model = HierarchicalTransformer(
        input_dim=num_features,
        d_model=512,
        num_heads=8,
        num_layers=3,
        d_ff=1024,
        max_seq_len=32,
        dropout=0.3,
    ).to(device)

    weights_path = (
        "results/ablation_studies/efficientnet_hierarchical_with_flow/best_model.pth"
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Model weights loaded from: {weights_path}")

    test_dataset = VideoSequenceDataset(X_test_scaled, test_df["target"].values)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    predictions = []
    labels = []

    with torch.no_grad():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy().flatten())
            labels.extend(target.cpu().numpy().flatten())

    predictions = np.array(predictions)
    labels = np.array(labels)
    predicted_binary = (predictions > threshold).astype(int)
    true_binary = labels.astype(int)

    false_negative_mask = (true_binary == 1) & (predicted_binary == 0)
    false_negative_indices = np.where(false_negative_mask)[0]
    crash_not_detected = test_df.iloc[false_negative_indices]["id"].tolist()

    false_positive_mask = (true_binary == 0) & (predicted_binary == 1)
    false_positive_indices = np.where(false_positive_mask)[0]
    crash_predicted_wrongly = test_df.iloc[false_positive_indices]["id"].tolist()

    print(
        f"\nCrash occurred but NOT detected (False Negatives): {len(crash_not_detected)}"
    )
    print(f"Video IDs: {crash_not_detected}")

    print(
        f"\nCrash NOT occurred but predicted (False Positives): {len(crash_predicted_wrongly)}"
    )
    print(f"Video IDs: {crash_predicted_wrongly}")

    with open(os.path.join(save_dir, "crash_not_detected_ids.txt"), "w") as f:
        for video_id in crash_not_detected:
            f.write(f"{video_id}\n")

    with open(os.path.join(save_dir, "crash_predicted_wrongly_ids.txt"), "w") as f:
        for video_id in crash_predicted_wrongly:
            f.write(f"{video_id}\n")

    print(f"\nSaved to:")
    print(f"- {save_dir}/crash_not_detected_ids.txt ({len(crash_not_detected)} videos)")
    print(
        f"- {save_dir}/crash_predicted_wrongly_ids.txt ({len(crash_predicted_wrongly)} videos)"
    )

    return crash_not_detected, crash_predicted_wrongly


if __name__ == "__main__":
    false_negatives, false_positives = get_misclassified_by_type()
