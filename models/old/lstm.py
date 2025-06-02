#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2  # Kept for potential future use, though not directly used for pre-extracted features
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Ensure joblib is installed for saving/loading scaler: pip install joblib
import joblib  # For saving/loading scaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# torchvision models and transforms are not strictly needed if features are pre-extracted
# from torchvision import models, transforms
# from torchvision.models import Inception_V3_Weights

# Set local cache directory (less relevant if not downloading models here)
os.environ["TORCH_HOME"] = "./cache"
os.makedirs("./cache", exist_ok=True)

# --- Configuration ---
DATASET_PERCENTAGE = 1.0  # Percentage of the dataset used

# Feature Config (Update if your InceptionV3 feature dimension is different)
NUM_FRAMES = 32  # As per your latest experiments
FEAT_DIM = 2048  # From InceptionV3 (pooling='avg' on the base model output per frame)

# Model Architecture Config
LSTM_HIDDEN_DIM = 512  # Size of LSTM hidden state
MLP_HIDDEN_DIMS = [32]  # Hidden dims for MLP after LSTM
DROPOUT_RATE = 0.2

# Training Config
BATCH_SIZE = 64
NUM_EPOCHS = 50  # Start with fewer epochs for initial testing
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Learning Rate Scheduler Config
LR_PATIENCE = 10
LR_FACTOR = 0.1

SAVE_METRICS_CSV = True
PRINT_FREQUENCY = 5  # Print every epoch for initial debugging

# --- File Paths for Saved Data (Adjust for 32-frame features) ---
percentage_str = str(int(DATASET_PERCENTAGE * 100))
FEATURES_BASE_DIR = (
    "features/cnn-32-frame/"  # NEW: Separate folder for these features/models
)
os.makedirs(FEATURES_BASE_DIR, exist_ok=True)

# !!! IMPORTANT: Update these paths to your actual 32-frame feature files !!!
X_TRAIN_FEATURES_FILE = (
    f"features/cnn-32-frame/X_train_sequences_{percentage_str}pct.npy"  # EXAMPLE PATH
)
X_TEST_FEATURES_FILE = "features/cnn-32-frame/X_test_sequences.npy"  # EXAMPLE PATH
SCALER_FILE = os.path.join(FEATURES_BASE_DIR, f"scaler_lstm_{percentage_str}pct.joblib")

MODEL_SAVE_PATH = os.path.join(
    FEATURES_BASE_DIR, f"best_lstm_model_{percentage_str}pct.pth"
)
METRICS_CSV_PATH = os.path.join(
    FEATURES_BASE_DIR, f"training_metrics_lstm_{percentage_str}pct.csv"
)
PLOT_FILENAME = os.path.join(
    FEATURES_BASE_DIR, f"training_curves_lstm_{percentage_str}pct.png"
)
HISTORY_FILENAME = os.path.join(
    FEATURES_BASE_DIR, f"training_history_lstm_{percentage_str}pct.npy"
)


# Device Config
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load CSVs & pad IDs
print("Loading CSV data...")
data_base_path = os.path.expanduser("./data-nexar/")
try:
    df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_base_path, "test.csv"))
except FileNotFoundError:
    print(f"ERROR: CSV files not found at {data_base_path}. Please check paths.")
    exit()

df["id"] = df["id"].astype(str).str.zfill(5)
df_test["id"] = df_test["id"].astype(str).str.zfill(5)

if DATASET_PERCENTAGE < 1.0:
    print(f"Sampling {DATASET_PERCENTAGE * 100}% of the training dataset...")
    df = df.sample(frac=DATASET_PERCENTAGE, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    print(f"Training set reduced to {len(df)} samples")

print(f"Total Train Videos: {len(df)}")
print(f"Total Test Videos:  {len(df_test)}")

# ========================================
# FEATURE LOADING (Assumes features are pre-extracted)
# ========================================
print("Loading pre-extracted 32-frame feature sequences...")
if os.path.exists(X_TRAIN_FEATURES_FILE) and os.path.exists(X_TEST_FEATURES_FILE):
    X_train_sequences = np.load(X_TRAIN_FEATURES_FILE)
    X_test_sequences = np.load(X_TEST_FEATURES_FILE)
    print("Feature sequences loaded successfully.")
    print(
        f"Training sequences shape: {X_train_sequences.shape}"
    )  # Expected: (num_videos, NUM_FRAMES, FEAT_DIM)
    print(f"Test sequences shape: {X_test_sequences.shape}")
else:
    print(f"ERROR: Pre-computed feature files not found. Please check paths:")
    print(f"Expected training features at: {X_TRAIN_FEATURES_FILE}")
    print(f"Expected test features at: {X_TEST_FEATURES_FILE}")
    print("Please extract features first or correct the paths in the script.")
    exit()

if X_train_sequences.shape[1] != NUM_FRAMES:
    print(
        f"Warning: Loaded training features have {X_train_sequences.shape[1]} frames, but NUM_FRAMES is {NUM_FRAMES}."
    )
if X_train_sequences.shape[2] != FEAT_DIM:
    print(
        f"Warning: Loaded training features have dimension {X_train_sequences.shape[2]}, but FEAT_DIM is {FEAT_DIM}."
    )


# ========================================
# DATA PREPARATION (FOR SEQUENCES)
# =================_=======================
print("Scaling feature sequences...")
num_train_videos, num_frames_per_video, num_features = X_train_sequences.shape
num_test_videos = X_test_sequences.shape[0]

# Reshape for scaling: (num_videos * num_frames, num_features)
X_train_reshaped = X_train_sequences.reshape(-1, num_features)

if os.path.exists(SCALER_FILE):
    print(f"Loading scaler from {SCALER_FILE}...")
    scaler = joblib.load(SCALER_FILE)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
else:
    print("Fitting new scaler...")
    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler fitted and saved to {SCALER_FILE}.")

# Reshape back to sequence: (num_videos, num_frames, num_features)
X_train_scaled_sequences = X_train_scaled_reshaped.reshape(
    num_train_videos, num_frames_per_video, num_features
)

# Scale test data
X_test_reshaped = X_test_sequences.reshape(-1, num_features)
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
X_test_scaled_sequences = X_test_scaled_reshaped.reshape(
    num_test_videos, num_frames_per_video, num_features
)

y = df["target"].values

X_tr_seq, X_val_seq, y_tr, y_val = train_test_split(
    X_train_scaled_sequences,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ========================================
# MODEL DEFINITION (LSTM-BASED CLASSIFIER)
# ========================================
class SequenceDataset(Dataset):
    def __init__(self, features_sequence, labels):
        self.features = torch.FloatTensor(features_sequence)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Ensure label is (1,) for BCELoss
        return self.features[idx], self.labels[idx].unsqueeze(-1)


class LSTMCollisionClassifier(nn.Module):
    def __init__(
        self,
        input_dim=FEAT_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        mlp_hidden_dims=None,
        dropout=DROPOUT_RATE,
        num_lstm_layers=1,
    ):
        super(LSTMCollisionClassifier, self).__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = MLP_HIDDEN_DIMS

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,  # expects input: (batch, seq_len, feature)
            dropout=dropout
            if num_lstm_layers > 1
            else 0,  # add dropout between LSTM layers if multiple
            bidirectional=True,
        )  # Starting with unidirectional

        mlp_layers = []
        # Input to MLP is the output of LSTM (lstm_hidden_dim)
        # If bidirectional, it would be lstm_hidden_dim * 2
        prev_mlp_dim = lstm_hidden_dim * 2
        for hidden_dim_mlp in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_mlp_dim, hidden_dim_mlp),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim_mlp),
                ]
            )
            prev_mlp_dim = hidden_dim_mlp

        mlp_layers.append(nn.Linear(prev_mlp_dim, 1))
        mlp_layers.append(nn.Sigmoid())
        self.classifier_mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_sequence):  # x_sequence: (batch_size, seq_len, feature_dim)
        # lstm_out: (batch, seq_len, lstm_hidden_dim) - all hidden states for all time steps
        # self.hidden: ( (num_layers * num_directions, batch, lstm_hidden_dim), (num_layers * num_directions, batch, lstm_hidden_dim) ) - (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x_sequence)

        # We typically use the output of the last time step for classification.
        # If batch_first=True, lstm_out is (batch, seq_len, hidden_dim).
        # So, last_hidden_state is lstm_out[:, -1, :]
        last_time_step_output = lstm_out[:, -1, :]

        # Alternatively, for a unidirectional LSTM, h_n[-1] (for the last layer) should be the same as lstm_out[:, -1, :]
        # If bidirectional, h_n would be (num_layers*2, batch, hidden_dim), and you might concatenate h_n[-1] and h_n[-2].
        # For simplicity, using lstm_out[:, -1, :] is robust.

        output = self.classifier_mlp(last_time_step_output)
        return output


# ========================================
# TRAINING SETUP
# ========================================
train_dataset_seq = SequenceDataset(X_tr_seq, y_tr)
val_dataset_seq = SequenceDataset(X_val_seq, y_val)

train_loader_seq = DataLoader(train_dataset_seq, batch_size=BATCH_SIZE, shuffle=True)
val_loader_seq = DataLoader(val_dataset_seq, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMCollisionClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
)  # mode='max' for AUC


# ========================================
# TRAINING FUNCTIONS
# ========================================
def train_epoch_lstm(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features_seq, labels in train_loader:  # Use tqdm here if preferred
        features_seq, labels = features_seq.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features_seq)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()
        total_loss += loss.item() * features_seq.size(0)
    return total_loss / len(train_loader.dataset)


def validate_lstm(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for features_seq, labels in val_loader:
            features_seq, labels = features_seq.to(device), labels.to(device)
            outputs = model(features_seq)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features_seq.size(0)
            all_predictions.extend(outputs.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    all_predictions_np = np.array(all_predictions).ravel()
    all_true_labels_np = np.array(all_true_labels).ravel()

    # Handle cases where a class might be missing in a batch for AUC
    if len(np.unique(all_true_labels_np)) < 2:
        auc_score = 0.5  # Or handle as an error/warning
        print(
            f"Warning: Only one class present in validation set for this epoch. AUC set to {auc_score}."
        )
    else:
        auc_score = roc_auc_score(all_true_labels_np, all_predictions_np)
    return avg_loss, auc_score


# ========================================
# TRAINING LOOP
# ========================================
train_losses = []
val_losses = []
val_aucs = []
learning_rates = []
best_val_auc = 0.0
metrics_data = []

print("Starting LSTM Model training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch_lstm(model, train_loader_seq, criterion, optimizer, device)
    val_loss, val_auc = validate_lstm(model, val_loader_seq, criterion, device)
    current_lr = optimizer.param_groups[0]["lr"]

    scheduler.step(val_auc)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    learning_rates.append(current_lr)
    metrics_data.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "learning_rate": current_lr,
            "dataset_percentage": DATASET_PERCENTAGE,
        }
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Epoch {epoch + 1}: New best model saved with Val AUC: {val_auc:.4f}")

    if (epoch + 1) % PRINT_FREQUENCY == 0:
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Best Val AUC: {best_val_auc:.4f}"
        )

# ========================================
# EVALUATION AND VISUALIZATION
# ========================================
if SAVE_METRICS_CSV:
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"Training metrics saved → {METRICS_CSV_PATH}")

print(f"Loading best model from {MODEL_SAVE_PATH} for final evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
final_val_loss, final_val_auc = validate_lstm(model, val_loader_seq, criterion, device)
print(f"\nFinal Validation ROC-AUC (LSTM Model from best epoch): {final_val_auc:.4f}")

# Plotting
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title(f"Loss ({percentage_str}% data, LSTM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(val_aucs, label="Validation AUC", color="green")
plt.title(f"Validation AUC ({percentage_str}% data, LSTM)")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.subplot(1, 3, 3)
plt.plot(learning_rates, label="Learning Rate", color="orange")
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.tight_layout()
plt.savefig(PLOT_FILENAME)
print(f"Training curves saved → {PLOT_FILENAME}")
plt.show()  # Uncomment if you want plots to display interactively

# ========================================
# INFERENCE AND SUBMISSION
# ========================================
test_dataset_seq = SequenceDataset(
    X_test_scaled_sequences, np.zeros(len(X_test_scaled_sequences))
)  # Dummy labels
test_loader_seq = DataLoader(test_dataset_seq, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
all_test_predictions = []
with torch.no_grad():
    for features_seq, _ in tqdm(test_loader_seq, desc="Predicting on test set"):
        features_seq = features_seq.to(device)
        outputs = model(features_seq)
        all_test_predictions.extend(outputs.cpu().numpy())

all_test_predictions_np = np.array(all_test_predictions).ravel()
submission = pd.DataFrame({"id": df_test["id"], "score": all_test_predictions_np})
submission_filename = os.path.join(
    FEATURES_BASE_DIR, f"submission_lstm_{percentage_str}pct.csv"
)
submission.to_csv(submission_filename, index=False)
print(f"Submission file written → {submission_filename}")

# Save training history
history_data = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_aucs": val_aucs,
    "learning_rates": learning_rates,
    "best_val_auc": best_val_auc,
    "final_val_auc": final_val_auc,
    "dataset_percentage": DATASET_PERCENTAGE,
    "num_frames": NUM_FRAMES,
}
np.save(HISTORY_FILENAME, history_data)
print(f"Training history saved → {HISTORY_FILENAME}")

print(f"\n🎯 LSTM MODEL TRAINING SUMMARY")
print(f"Dataset used: {DATASET_PERCENTAGE * 100}% ({len(df)} samples)")
print(f"Frames per video: {NUM_FRAMES}")
print(f"Best validation AUC: {best_val_auc:.4f}")
print(f"Final validation AUC (from best model): {final_val_auc:.4f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Metrics saved to: {METRICS_CSV_PATH}")
print(f"Submission saved to: {submission_filename}")
