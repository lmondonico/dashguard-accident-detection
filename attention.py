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

import torch
import torch.nn as nn
import torch.nn.functional as F # NEW: For F.softmax
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

# Set local cache directory for pretrained models
os.environ["TORCH_HOME"] = "./cache"
os.makedirs("./cache", exist_ok=True)

# --- Configuration ---
DATASET_PERCENTAGE = 1.0  # Percentage of the dataset used

# Feature Extraction Config
NUM_FRAMES = 16
FRAME_SIZE = (299, 299)
FEAT_DIM = 2048 # From InceptionV3 (pooling='avg' on the base model output per frame)

# Model Architecture Config (for the MLP part after attention)
# NEW: Attention dimension
ATTENTION_DIM = 128 # Hyperparameter for the attention mechanism
HIDDEN_DIMS = [128, 32] # Hidden dims for MLP after attention
DROPOUT_RATE = 0.4 # MODIFIED for potentially deeper MLP part

# Training Config
BATCH_SIZE = 32 # MODIFIED: Often smaller batches work better with sequence models / attention
NUM_EPOCHS = 100 # MODIFIED: Attention models might train for different durations
LEARNING_RATE = 1e-5 # MODIFIED: Might need different LR for attention
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Learning Rate Scheduler Config
LR_PATIENCE = 10
LR_FACTOR = 0.5

SAVE_METRICS_CSV = True
PRINT_FREQUENCY = 5 # MODIFIED

# --- File Paths for Saved Data ---
percentage_str = str(int(DATASET_PERCENTAGE * 100))
FEATURES_BASE_DIR = "features/attention/" # NEW: Separate folder for these features
os.makedirs(FEATURES_BASE_DIR, exist_ok=True)
X_TRAIN_FEATURES_FILE = os.path.join(FEATURES_BASE_DIR, f"X_train_sequences_{percentage_str}pct.npy")
X_TEST_FEATURES_FILE = os.path.join(FEATURES_BASE_DIR, "X_test_sequences.npy")
SCALER_FILE = os.path.join(FEATURES_BASE_DIR, f"scaler_attention_{percentage_str}pct.joblib")
MODEL_SAVE_PATH = f"best_attention_model_{percentage_str}pct.pth"
METRICS_CSV_PATH = f"training_metrics_attention_{percentage_str}pct.csv"
PLOT_FILENAME = f"training_curves_attention_{percentage_str}pct.png"
HISTORY_FILENAME = f"training_history_attention_{percentage_str}pct.npy"


# Device Config
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Optional: Configure TensorFlow for GPU memory growth (Not strictly needed for PyTorch)
# For PyTorch, memory management is generally handled differently.

# Load CSVs & pad IDs
print("Loading CSV data...")
data_base_path = os.path.expanduser('./data/nexar-collision-prediction/')
try:
    df       = pd.read_csv(os.path.join(data_base_path, 'train.csv'))
    df_test  = pd.read_csv(os.path.join(data_base_path, 'test.csv'))
except FileNotFoundError:
    print(f"ERROR: CSV files not found at {data_base_path}. Please check paths.")
    exit()

df["id"] = df["id"].astype(str).str.zfill(5)
df_test["id"] = df_test["id"].astype(str).str.zfill(5)

if DATASET_PERCENTAGE < 1.0:
    print(f"Sampling {DATASET_PERCENTAGE * 100}% of the training dataset...")
    df = df.sample(frac=DATASET_PERCENTAGE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Training set reduced to {len(df)} samples")

train_dir = os.path.join(data_base_path, 'train/')
test_dir = os.path.join(data_base_path, 'test/')
print(f"Total Train Videos: {len(df)}")
print(f"Total Test Videos:  {len(df_test)}")

# ========================================
# FEATURE EXTRACTION (MODIFIED)
# ========================================
def extract_frames(path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *size, 3), dtype=np.uint8)
    step = max(total_video_frames // num_frames, 1)
    frames_list = []
    for i in range(num_frames):
        frame_pos = i * step
        if frame_pos >= total_video_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, size)
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames_list) if frames_list else np.empty((0, *size, 3), dtype=np.uint8)

print("Loading InceptionV3 base model for feature extraction...")
base_model_fe = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
base_model_fe.fc = nn.Identity()
base_model_fe = base_model_fe.to(device)
base_model_fe.eval()

preprocess_fe = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE), # InceptionV3 expects 299x299
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_feature_sequences(video_ids, video_folder, num_target_frames=NUM_FRAMES):
    video_features_list = []
    for video_id in tqdm(video_ids, desc=f"Extracting feature sequences from {video_folder}"):
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        frames_np = extract_frames(video_path, num_frames=num_target_frames)
        current_num_frames = frames_np.shape[0]

        if current_num_frames == 0:
            print(f"Warning: Video {video_id} had 0 frames. Appending sequence of zeros.")
            video_features_list.append(np.zeros((num_target_frames, FEAT_DIM), dtype=np.float32))
            continue

        # Preprocess frames
        batch_frames_list = []
        for frame_idx in range(frames_np.shape[0]):
            processed_frame = preprocess_fe(frames_np[frame_idx])
            batch_frames_list.append(processed_frame)
        
        batch_tensor = torch.stack(batch_frames_list).to(device)

        with torch.no_grad():
            # Output of base_model_fe will be (current_num_frames, FEAT_DIM)
            frame_features_tensor = base_model_fe(batch_tensor)
        
        frame_features_np = frame_features_tensor.cpu().numpy()

        # Pad if fewer than num_target_frames were extracted and processed
        if current_num_frames < num_target_frames:
            print(f"Warning: Video {video_id} processed {current_num_frames}/{num_target_frames} frames. Padding features with zeros.")
            padding = np.zeros((num_target_frames - current_num_frames, FEAT_DIM), dtype=np.float32)
            frame_features_np = np.vstack((frame_features_np, padding))
        
        video_features_list.append(frame_features_np)
        
    return np.array(video_features_list, dtype=np.float32)

if os.path.exists(X_TRAIN_FEATURES_FILE) and os.path.exists(X_TEST_FEATURES_FILE):
    print(f"Loading pre-computed feature sequences from {X_TRAIN_FEATURES_FILE} and {X_TEST_FEATURES_FILE}...")
    X_train_sequences = np.load(X_TRAIN_FEATURES_FILE)
    X_test_sequences = np.load(X_TEST_FEATURES_FILE)
    print("Feature sequences loaded successfully.")
else:
    print("Pre-computed feature sequences not found. Starting feature extraction...")
    X_train_sequences = get_feature_sequences(df["id"], train_dir)
    np.save(X_TRAIN_FEATURES_FILE, X_train_sequences)
    print(f"Saved training feature sequences to {X_TRAIN_FEATURES_FILE}")

    X_test_sequences = get_feature_sequences(df_test["id"], test_dir)
    np.save(X_TEST_FEATURES_FILE, X_test_sequences)
    print(f"Saved test feature sequences to {X_TEST_FEATURES_FILE}")

if X_train_sequences.size > 0:
    print(f"Training sequences shape: {X_train_sequences.shape}") # (num_videos, NUM_FRAMES, FEAT_DIM)
    print(f"Test sequences shape: {X_test_sequences.shape}")
else:
    print("Error: Feature sequences are empty.")
    exit()

# ========================================
# DATA PREPARATION (FOR SEQUENCES)
# ========================================
print("Scaling feature sequences...")
num_train_videos, num_frames_per_video, num_features = X_train_sequences.shape
num_test_videos = X_test_sequences.shape[0]

X_train_reshaped = X_train_sequences.reshape(-1, num_features)

if os.path.exists(SCALER_FILE):
    print(f"Loading scaler from {SCALER_FILE}...")
    scaler = pd.read_pickle(SCALER_FILE)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
else:
    print("Fitting new scaler...")
    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    pd.to_pickle(scaler, SCALER_FILE)
    print(f"Scaler fitted and saved to {SCALER_FILE}.")

X_train_scaled_sequences = X_train_scaled_reshaped.reshape(num_train_videos, num_frames_per_video, num_features)

X_test_reshaped = X_test_sequences.reshape(-1, num_features)
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
X_test_scaled_sequences = X_test_scaled_reshaped.reshape(num_test_videos, num_frames_per_video, num_features)

y = df["target"].values

X_tr_seq, X_val_seq, y_tr, y_val = train_test_split(
    X_train_scaled_sequences, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ========================================
# MODEL DEFINITION (ATTENTION-BASED CLASSIFIER - NEW)
# ========================================
class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(AttentionMechanism, self).__init__()
        self.W = nn.Linear(feature_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, features): # features: (batch_size, seq_len, feature_dim)
        e = torch.tanh(self.W(features))    # (batch_size, seq_len, attention_dim)
        scores = self.v(e).squeeze(2)       # (batch_size, seq_len)
        alpha = F.softmax(scores, dim=1)    # (batch_size, seq_len)
        
        # context = torch.sum(alpha.unsqueeze(2) * features, dim=1) # Element-wise multiplication and sum
        # More efficient way using batch matrix multiplication:
        # alpha.unsqueeze(1) gives (batch_size, 1, seq_len)
        # features is (batch_size, seq_len, feature_dim)
        # bmm result is (batch_size, 1, feature_dim), then squeeze
        context = torch.bmm(alpha.unsqueeze(1), features).squeeze(1) # (batch_size, feature_dim)
        return context, alpha


class AttentionCollisionClassifier(nn.Module):
    def __init__(self, input_dim=FEAT_DIM, attention_dim=ATTENTION_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT_RATE):
        super(AttentionCollisionClassifier, self).__init__()
        self.attention = AttentionMechanism(input_dim, attention_dim)
        
        mlp_layers = []
        prev_mlp_dim = input_dim # Input to MLP is the context vector from attention
        for hidden_dim_mlp in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_mlp_dim, hidden_dim_mlp),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim_mlp),
            ])
            prev_mlp_dim = hidden_dim_mlp
        
        mlp_layers.append(nn.Linear(prev_mlp_dim, 1))
        mlp_layers.append(nn.Sigmoid())
        self.classifier_mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_sequence): # x_sequence: (batch_size, seq_len, feature_dim)
        context_vector, self.attention_weights = self.attention(x_sequence) # Store weights for potential inspection
        output = self.classifier_mlp(context_vector)
        return output

class SequenceDataset(Dataset): # RENAMED from FeaturesDataset
    def __init__(self, features_sequence, labels):
        self.features = torch.FloatTensor(features_sequence) # Should be (num_samples, seq_len, feat_dim)
        self.labels = torch.FloatTensor(labels) # Should be (num_samples,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1) # Ensure label is (1,) for BCELoss


# ========================================
# TRAINING SETUP (FOR ATTENTION MODEL)
# ========================================
train_dataset_seq = SequenceDataset(X_tr_seq, y_tr)
val_dataset_seq = SequenceDataset(X_val_seq, y_val)

train_loader_seq = DataLoader(train_dataset_seq, batch_size=BATCH_SIZE, shuffle=True)
val_loader_seq = DataLoader(val_dataset_seq, batch_size=BATCH_SIZE, shuffle=False)

model = AttentionCollisionClassifier().to(device) # Using default params from config
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR) # mode='max' for AUC

# ========================================
# TRAINING FUNCTIONS (train_epoch, validate - adapted for sequence model)
# ========================================
def train_epoch_attention(model, train_loader, criterion, optimizer, device): # Renamed
    model.train()
    total_loss = 0
    for features_seq, labels in train_loader:
        features_seq, labels = features_seq.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features_seq) # .squeeze() is not needed if labels are (batch,1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features_seq.size(0)
    return total_loss / len(train_loader.dataset)

def validate_attention(model, val_loader, criterion, device): # Renamed
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
    # Ensure predictions and labels are 1D arrays for roc_auc_score
    all_predictions_np = np.array(all_predictions).ravel()
    all_true_labels_np = np.array(all_true_labels).ravel()
    auc_score = roc_auc_score(all_true_labels_np, all_predictions_np)
    return avg_loss, auc_score

# ========================================
# TRAINING LOOP (FOR ATTENTION MODEL)
# ========================================
train_losses = []
val_losses = []
val_aucs = []
learning_rates = []
best_auc = 0.0 # Initialize best_auc
metrics_data = []

print("Starting Attention Model training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch_attention(model, train_loader_seq, criterion, optimizer, device)
    val_loss, val_auc = validate_attention(model, val_loader_seq, criterion, device)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Note: ReduceLROnPlateau steps on a metric, not epoch loss usually.
    # For AUC, mode should be 'max'.
    scheduler.step(val_auc) 

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    learning_rates.append(current_lr)
    metrics_data.append({
        "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
        "val_auc": val_auc, "learning_rate": current_lr, "dataset_percentage": DATASET_PERCENTAGE,
    })

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Epoch {epoch+1}: New best model saved with Val AUC: {val_auc:.4f}")

    if (epoch + 1) % PRINT_FREQUENCY == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Best Val AUC: {best_auc:.4f}")

# ========================================
# EVALUATION AND VISUALIZATION (FOR ATTENTION MODEL)
# ========================================
if SAVE_METRICS_CSV:
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"Training metrics saved → {METRICS_CSV_PATH}")

print(f"Loading best model from {MODEL_SAVE_PATH} for final evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
_, final_auc = validate_attention(model, val_loader_seq, criterion, device) # Get final AUC from best model
print(f"\nFinal Validation ROC-AUC (Attention Model): {final_auc:.4f}")

# Plotting (similar to before)
plt.figure(figsize=(18, 5)) # Adjusted figure size
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title(f'Loss ({percentage_str}% data)')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(val_aucs, label='Validation AUC', color='green')
plt.title(f'Validation AUC ({percentage_str}% data)')
plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
plt.subplot(1, 3, 3)
plt.plot(learning_rates, label='Learning Rate', color='orange')
plt.title('Learning Rate Schedule'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.legend(); plt.grid(True); plt.yscale('log')
plt.tight_layout()
plt.savefig(PLOT_FILENAME); print(f"Training curves saved → {PLOT_FILENAME}")
plt.show()

# ========================================
# INFERENCE AND SUBMISSION (FOR ATTENTION MODEL)
# ========================================
X_test_tensor = torch.FloatTensor(X_test_scaled_sequences)
test_dataset_seq = SequenceDataset(X_test_scaled_sequences, np.zeros(len(X_test_scaled_sequences))) # Dummy labels
test_loader_seq = DataLoader(test_dataset_seq, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
all_test_predictions = []
with torch.no_grad():
    for features_seq, _ in test_loader_seq:
        features_seq = features_seq.to(device)
        outputs = model(features_seq)
        all_test_predictions.extend(outputs.cpu().numpy())

all_test_predictions_np = np.array(all_test_predictions).ravel()
submission = pd.DataFrame({'id': df_test['id'], 'score': all_test_predictions_np})
submission_filename = f"submission_attention_{percentage_str}pct.csv"
submission.to_csv(submission_filename, index=False)
print(f"Written → {submission_filename}")

# Save training history
history_data = {
    "train_losses": train_losses, "val_losses": val_losses, "val_aucs": val_aucs,
    "learning_rates": learning_rates, "best_auc": best_auc, "final_auc": final_auc,
    "dataset_percentage": DATASET_PERCENTAGE,
}
np.save(HISTORY_FILENAME, history_data)
print(f"Training history saved → {HISTORY_FILENAME}")

print(f"\n🎯 ATTENTION MODEL TRAINING SUMMARY")
print(f"Dataset used: {DATASET_PERCENTAGE*100}% ({len(df)} samples)")
print(f"Best validation AUC: {best_auc:.4f}")
print(f"Final validation AUC (from best model): {final_auc:.4f}")