#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for F.softmax
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

os.environ["TORCH_HOME"] = "./cache"
os.makedirs("./cache", exist_ok=True)

# --- Configuration ---
DATASET_PERCENTAGE = 1.0

NUM_FRAMES = 32
FEAT_DIM = 2048

# Model Architecture Config
LSTM_HIDDEN_DIM = 512
# --- NEW: Attention Dimension ---
ATTENTION_DIM = 256  # Hyperparameter for the attention mechanism
# ---
MLP_HIDDEN_DIMS = [32]
DROPOUT_RATE = 0.2
NUM_LSTM_LAYERS = 1  # Added for clarity, can be > 1

# Training Config
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5  # You might need to re-tune this
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42

LR_PATIENCE = 10
LR_FACTOR = 0.1

SAVE_METRICS_CSV = True
PRINT_FREQUENCY = 5

percentage_str = str(int(DATASET_PERCENTAGE * 100))
# --- Modified FEATURES_BASE_DIR and filenames for LSTM with Attention ---
FEATURES_BASE_DIR = "features/lstm_attention_32frame/"
os.makedirs(FEATURES_BASE_DIR, exist_ok=True)

X_TRAIN_FEATURES_FILE = f"features/cnn-32-frame/X_train_sequences_{percentage_str}pct.npy"  # Using existing 32-frame features
X_TEST_FEATURES_FILE = (
    "features/cnn-32-frame/X_test_sequences.npy"  # Using existing 32-frame features
)
SCALER_FILE = os.path.join(
    FEATURES_BASE_DIR, f"scaler_lstm_attention_{percentage_str}pct.joblib"
)

MODEL_SAVE_PATH = os.path.join(
    FEATURES_BASE_DIR, f"best_lstm_attention_model_{percentage_str}pct.pth"
)
METRICS_CSV_PATH = os.path.join(
    FEATURES_BASE_DIR, f"training_metrics_lstm_attention_{percentage_str}pct.csv"
)
PLOT_FILENAME = os.path.join(
    FEATURES_BASE_DIR, f"training_curves_lstm_attention_{percentage_str}pct.png"
)
HISTORY_FILENAME = os.path.join(
    FEATURES_BASE_DIR, f"training_history_lstm_attention_{percentage_str}pct.npy"
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

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

print("Loading pre-extracted 32-frame feature sequences...")
if os.path.exists(X_TRAIN_FEATURES_FILE) and os.path.exists(X_TEST_FEATURES_FILE):
    X_train_sequences = np.load(X_TRAIN_FEATURES_FILE)
    X_test_sequences = np.load(X_TEST_FEATURES_FILE)
    print("Feature sequences loaded successfully.")
    print(f"Training sequences shape: {X_train_sequences.shape}")
    print(f"Test sequences shape: {X_test_sequences.shape}")
else:
    print(f"ERROR: Pre-computed feature files not found. Please check paths:")
    print(f"Expected training features at: {X_TRAIN_FEATURES_FILE}")
    print(f"Expected test features at: {X_TEST_FEATURES_FILE}")
    exit()

if X_train_sequences.shape[1] != NUM_FRAMES:
    print(
        f"Warning: Loaded training features have {X_train_sequences.shape[1]} frames, but NUM_FRAMES is {NUM_FRAMES}."
    )  #
if X_train_sequences.shape[2] != FEAT_DIM:
    print(
        f"Warning: Loaded training features have dimension {X_train_sequences.shape[2]}, but FEAT_DIM is {FEAT_DIM}."
    )  #

print("Scaling feature sequences...")
num_train_videos, num_frames_per_video, num_features = X_train_sequences.shape  #
num_test_videos = X_test_sequences.shape[0]  #

X_train_reshaped = X_train_sequences.reshape(-1, num_features)  #

if os.path.exists(SCALER_FILE):
    print(f"Loading scaler from {SCALER_FILE}...")  #
    scaler = joblib.load(SCALER_FILE)  #
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)  #
else:
    print("Fitting new scaler...")  #
    scaler = StandardScaler()  #
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)  #
    joblib.dump(scaler, SCALER_FILE)  #
    print(f"Scaler fitted and saved to {SCALER_FILE}.")  #

X_train_scaled_sequences = X_train_scaled_reshaped.reshape(
    num_train_videos, num_frames_per_video, num_features
)  #

X_test_reshaped = X_test_sequences.reshape(-1, num_features)  #
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)  #
X_test_scaled_sequences = X_test_scaled_reshaped.reshape(
    num_test_videos, num_frames_per_video, num_features
)  #

y = df["target"].values  #

X_tr_seq, X_val_seq, y_tr, y_val = train_test_split(
    X_train_scaled_sequences,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,  #
)


class SequenceDataset(Dataset):  #
    def __init__(self, features_sequence, labels):  #
        self.features = torch.FloatTensor(features_sequence)  #
        self.labels = torch.FloatTensor(labels)  #

    def __len__(self):  #
        return len(self.features)  #

    def __getitem__(self, idx):  #
        return self.features[idx], self.labels[idx].unsqueeze(-1)  #


# --- NEW: Attention Module Definition ---
class AttentionModule(nn.Module):
    def __init__(self, lstm_output_dim, attention_dim):
        super(AttentionModule, self).__init__()
        # Linear layer to transform LSTM outputs to attention_dim
        self.W_att = nn.Linear(lstm_output_dim, attention_dim, bias=False)
        # Linear layer to compute attention scores (vector v_att)
        self.v_att = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_outputs):  # lstm_outputs: (batch, seq_len, lstm_output_dim)
        # lstm_outputs is the sequence of hidden states from all time steps

        # Project lstm_outputs to attention_dim
        # u_t = tanh(W_att * h_t)
        u_t = torch.tanh(self.W_att(lstm_outputs))  # (batch, seq_len, attention_dim)

        # Compute scores for each time step
        # e_t = v_att * u_t
        scores = self.v_att(u_t).squeeze(-1)  # (batch, seq_len)

        # Compute attention weights (alpha) by applying softmax
        alpha = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Compute context vector (weighted sum of lstm_outputs)
        # context = sum(alpha_t * h_t)
        # alpha.unsqueeze(1): (batch, 1, seq_len)
        # lstm_outputs:       (batch, seq_len, lstm_output_dim)
        # bmm result:         (batch, 1, lstm_output_dim) -> squeeze to (batch, lstm_output_dim)
        context = torch.bmm(alpha.unsqueeze(1), lstm_outputs).squeeze(1)
        return (
            context,
            alpha,
        )  # Return context vector and attention weights (for inspection)


# --- MODIFIED: LSTM Classifier with Attention ---
class LSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim=FEAT_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        mlp_hidden_dims=None,
        dropout=DROPOUT_RATE,
        num_lstm_layers=NUM_LSTM_LAYERS,
    ):  # num_lstm_layers was hardcoded to 1 before
        super(LSTMAttentionClassifier, self).__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = MLP_HIDDEN_DIMS  #

        self.num_directions = 2  # Since bidirectional=True
        self.lstm = nn.LSTM(
            input_size=input_dim,  #
            hidden_size=lstm_hidden_dim,  #
            num_layers=num_lstm_layers,  #
            batch_first=True,  #
            dropout=dropout if num_lstm_layers > 1 else 0,  #
            bidirectional=True,
        )  #

        lstm_output_dim = (
            lstm_hidden_dim * self.num_directions
        )  # (derived, as prev_mlp_dim = lstm_hidden_dim * 2)
        self.attention = AttentionModule(lstm_output_dim, attention_dim)

        mlp_layers = []  #
        # Input to MLP is the context vector from attention, which has lstm_output_dim
        prev_mlp_dim = lstm_output_dim  # This remains the same as lstm_hidden_dim * 2
        for hidden_dim_mlp in mlp_hidden_dims:  #
            mlp_layers.extend(
                [  #
                    nn.Linear(prev_mlp_dim, hidden_dim_mlp),  #
                    nn.ReLU(),  #
                    nn.Dropout(dropout),  #
                    nn.BatchNorm1d(hidden_dim_mlp),  #
                ]
            )
            prev_mlp_dim = hidden_dim_mlp  #

        mlp_layers.append(nn.Linear(prev_mlp_dim, 1))  #
        mlp_layers.append(nn.Sigmoid())  #
        self.classifier_mlp = nn.Sequential(*mlp_layers)  #
        self.attention_weights = None  # To store attention weights for inspection

    def forward(self, x_sequence):  # x_sequence: (batch_size, seq_len, feature_dim)
        # lstm_out: (batch, seq_len, lstm_hidden_dim * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x_sequence)  #

        # Pass all LSTM outputs to the attention mechanism
        context_vector, self.attention_weights = self.attention(lstm_out)

        # Pass the context vector through the MLP
        output = self.classifier_mlp(
            context_vector
        )  # (context_vector replaces last_time_step_output)
        return output


# --- Update model initialization ---
model = LSTMAttentionClassifier().to(device)  #
criterion = nn.BCELoss()  #
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)  #
# --- Ensure scheduler monitors the correct metric based on its mode ---
# If mode="min" for loss, step with val_loss. If mode="max" for AUC, step with val_auc.
# The original script had scheduler.step(val_auc) but mode="min". Let's assume val_loss is monitored or mode is changed.
# For this example, let's assume we want to monitor val_loss and use mode="min".
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
)  # (changed from val_auc to val_loss if mode is min)

train_dataset_seq = SequenceDataset(X_tr_seq, y_tr)  #
val_dataset_seq = SequenceDataset(X_val_seq, y_val)  #
train_loader_seq = DataLoader(train_dataset_seq, batch_size=BATCH_SIZE, shuffle=True)  #
val_loader_seq = DataLoader(val_dataset_seq, batch_size=BATCH_SIZE, shuffle=False)  #


# Training functions remain the same, just ensure they are called with the new model.
# (train_epoch_lstm and validate_lstm can be reused, let's rename them for clarity if needed, but they are generic)
def train_epoch_model(
    model, train_loader, criterion, optimizer, device
):  # Renamed for clarity
    model.train()  #
    total_loss = 0  #
    for features_seq, labels in train_loader:  #
        features_seq, labels = features_seq.to(device), labels.to(device)  #
        optimizer.zero_grad()  #
        outputs = model(features_seq)  #
        loss = criterion(outputs, labels)  #
        loss.backward()  #
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
        optimizer.step()  #
        total_loss += loss.item() * features_seq.size(0)  #
    return total_loss / len(train_loader.dataset)  #


def validate_model(model, val_loader, criterion, device):  # Renamed for clarity
    model.eval()  #
    total_loss = 0  #
    all_predictions = []  #
    all_true_labels = []  #
    with torch.no_grad():  #
        for features_seq, labels in val_loader:  #
            features_seq, labels = features_seq.to(device), labels.to(device)  #
            outputs = model(features_seq)  #
            loss = criterion(outputs, labels)  #
            total_loss += loss.item() * features_seq.size(0)  #
            all_predictions.extend(outputs.cpu().numpy())  #
            all_true_labels.extend(labels.cpu().numpy())  #

    avg_loss = total_loss / len(val_loader.dataset)  #
    all_predictions_np = np.array(all_predictions).ravel()  #
    all_true_labels_np = np.array(all_true_labels).ravel()  #

    if len(np.unique(all_true_labels_np)) < 2:  #
        auc_score = 0.5  #
        print(
            f"Warning: Only one class present in validation set for this epoch. AUC set to {auc_score}."
        )  #
    else:
        auc_score = roc_auc_score(all_true_labels_np, all_predictions_np)  #
    return avg_loss, auc_score  #


train_losses = []  #
val_losses = []  #
val_aucs = []  #
learning_rates = []  #
best_val_metric = 0.0  # Can be best AUC or best (lowest) loss depending on strategy
# Let's stick to best_val_auc for model saving, but step scheduler with val_loss
best_val_auc_for_saving = 0.0


metrics_data = []  #

print("Starting LSTM with Attention Model training...")
for epoch in range(NUM_EPOCHS):  #
    train_loss = train_epoch_model(
        model, train_loader_seq, criterion, optimizer, device
    )  #
    val_loss, val_auc = validate_model(model, val_loader_seq, criterion, device)  #
    current_lr = optimizer.param_groups[0]["lr"]  #

    scheduler.step(val_loss)  # Step scheduler with validation loss as mode="min"

    train_losses.append(train_loss)  #
    val_losses.append(val_loss)  #
    val_aucs.append(val_auc)  #
    learning_rates.append(current_lr)  #
    metrics_data.append(
        {  #
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,  #
            "val_auc": val_auc,
            "learning_rate": current_lr,
            "dataset_percentage": DATASET_PERCENTAGE,  #
        }
    )

    if val_auc > best_val_auc_for_saving:  # (using a separate var for clarity)
        best_val_auc_for_saving = val_auc  #
        torch.save(model.state_dict(), MODEL_SAVE_PATH)  #
        print(f"Epoch {epoch + 1}: New best model saved with Val AUC: {val_auc:.4f}")  #

    if (epoch + 1) % PRINT_FREQUENCY == 0:  #
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Best Val AUC for Saving: {best_val_auc_for_saving:.4f}"
        )  #

if SAVE_METRICS_CSV:  #
    metrics_df = pd.DataFrame(metrics_data)  #
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)  #
    print(f"Training metrics saved → {METRICS_CSV_PATH}")  #

print(f"Loading best model from {MODEL_SAVE_PATH} for final evaluation...")  #
model.load_state_dict(torch.load(MODEL_SAVE_PATH))  #
final_val_loss, final_val_auc = validate_model(
    model, val_loader_seq, criterion, device
)  #
print(
    f"\nFinal Validation ROC-AUC (LSTM w/ Attention Model from best epoch): {final_val_auc:.4f}"
)  #

plt.figure(figsize=(18, 5))  #
plt.subplot(1, 3, 1)  #
plt.plot(train_losses, label="Training Loss")  #
plt.plot(val_losses, label="Validation Loss")  #
plt.title(f"Loss ({percentage_str}% data, LSTM w/ Attention)")  #
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)  #
plt.subplot(1, 3, 2)  #
plt.plot(val_aucs, label="Validation AUC", color="green")  #
plt.title(f"Validation AUC ({percentage_str}% data, LSTM w/ Attention)")  #
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)  #
plt.subplot(1, 3, 3)  #
plt.plot(learning_rates, label="Learning Rate", color="orange")  #
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.yscale("log")  #
plt.tight_layout()  #
plt.savefig(PLOT_FILENAME)
print(f"Training curves saved → {PLOT_FILENAME}")  #
plt.show()  #

test_dataset_seq = SequenceDataset(
    X_test_scaled_sequences, np.zeros(len(X_test_scaled_sequences))
)  #
test_loader_seq = DataLoader(test_dataset_seq, batch_size=BATCH_SIZE, shuffle=False)  #

model.eval()  #
all_test_predictions = []  #
with torch.no_grad():  #
    for features_seq, _ in tqdm(test_loader_seq, desc="Predicting on test set"):  #
        features_seq = features_seq.to(device)  #
        outputs = model(features_seq)  #
        all_test_predictions.extend(outputs.cpu().numpy())  #

all_test_predictions_np = np.array(all_test_predictions).ravel()  #
submission = pd.DataFrame({"id": df_test["id"], "score": all_test_predictions_np})  #
submission_filename = os.path.join(
    FEATURES_BASE_DIR, f"submission_lstm_attention_{percentage_str}pct.csv"
)  #
submission.to_csv(submission_filename, index=False)  #
print(f"Submission file written → {submission_filename}")  #

history_data = {  #
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_aucs": val_aucs,  #
    "learning_rates": learning_rates,
    "best_val_auc_for_saving": best_val_auc_for_saving,
    "final_val_auc": final_val_auc,  # (updated key)
    "dataset_percentage": DATASET_PERCENTAGE,
    "num_frames": NUM_FRAMES,  #
}
np.save(HISTORY_FILENAME, history_data)  #
print(f"Training history saved → {HISTORY_FILENAME}")  #

print(f"\n🎯 LSTM w/ ATTENTION MODEL TRAINING SUMMARY")  #
print(f"Dataset used: {DATASET_PERCENTAGE * 100}% ({len(df)} samples)")  #
print(f"Frames per video: {NUM_FRAMES}")  #
print(f"Best validation AUC for saving: {best_val_auc_for_saving:.4f}")  # (updated key)
print(f"Final validation AUC (from best model): {final_val_auc:.4f}")  #
print(f"Model saved to: {MODEL_SAVE_PATH}")  #
print(f"Metrics saved to: {METRICS_CSV_PATH}")  #
print(f"Submission saved to: {submission_filename}")  #
