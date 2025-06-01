"""
Baseline script for the Nexar Collision Prediction Challenge.
"""

##############################################

DATASET_PERCENTAGE = 1  # Percentage of the dataset used (0.1 = 10%, 1.0 = 100%)

# Feature Extraction Config
NUM_FRAMES = 32  # Frames to extract per video
FRAME_SIZE = (299, 299)  # Frame size (InceptionV3 expects 299x299)
FEAT_DIM = 2048  # Feature dimension (InceptionV3 features before final FC layer)

# Model Architecture Config
HIDDEN_DIMS = [128]
DROPOUT_RATE = 0.3

# Training Config
BATCH_SIZE = 64
NUM_EPOCHS = 1000
WEIGHT_DECAY = 1e-2  # L2 reg weight decay
TEST_SIZE = 0.35  # Validation split size
RANDOM_STATE = 42

# OneCycleLR Config - Only essential inputs
INITIAL_LR = 1e-8  # Starting learning rate
MAX_LR = 5e-7  # Maximum learning rate (peak)
FINAL_LR = 5e-9  # Final learning rate at the end
PCT_START = 0.2  # Percentage of training spent increasing LR

SAVE_METRICS_CSV = True
PRINT_FREQUENCY = 20  # Print training progress every N epochs
PLOT_UPDATE_FREQUENCY = 5  # Update plots every N epochs

##############################################

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

# Enable interactive plotting
plt.ion()

# Set local cache directory for pretrained models
os.environ["TORCH_HOME"] = "./cache"
os.makedirs("./cache", exist_ok=True)

# Device Config
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load CSVs & pad IDs
df = pd.read_csv("./data/nexar-collision-prediction/train.csv")
df_test = pd.read_csv("./data/nexar-collision-prediction/test.csv")
df["id"] = df["id"].astype(str).str.zfill(5)
df_test["id"] = df_test["id"].astype(str).str.zfill(5)

# Sample dataset based on percentage
if DATASET_PERCENTAGE < 1.0:
    print(f"Sampling {DATASET_PERCENTAGE * 100}% of the training dataset...")
    df = df.sample(frac=DATASET_PERCENTAGE, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    print(f"Training set reduced from full size to {len(df)} samples")

# Define video folders & filenames
train_dir = "./data/nexar-collision-prediction/train/"
test_dir = "./data/nexar-collision-prediction/test/"

df["train_videos"] = df["id"] + ".mp4"
df_test["test_videos"] = df_test["id"] + ".mp4"

# Quick sanity prints
print(f"Total Train Videos: {len(df['train_videos'])}")
print(f"Total Test Videos:  {len(df_test['test_videos'])}")

# ========================================
# FEATURE EXTRACTION
# ========================================


# Frame sampling helper
def extract_frames(path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.empty((0, *size, 3), dtype=np.uint8)
    step = max(total // num_frames, 1)
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else np.empty((0, *size, 3), dtype=np.uint8)


# Load InceptionV3 model
base_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
base_model.fc = nn.Identity()  # Remove final classification layer
base_model = base_model.to(device)
base_model.eval()

# Define preprocessing transforms
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_features(ids, folder):
    feats = []
    for vid in tqdm(ids, desc=f"Extracting from {folder}"):
        path = os.path.join(folder, f"{vid}.mp4")
        frames = extract_frames(path, size=FRAME_SIZE)
        if frames.size == 0:
            feats.append(np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        # Preprocess frames
        batch = []
        for frame in frames:
            processed_frame = preprocess(frame)
            batch.append(processed_frame)

        batch_tensor = torch.stack(batch).to(device)

        with torch.no_grad():
            features = base_model(batch_tensor)
            # Average features across frames
            avg_features = features.mean(dim=0).cpu().numpy()

        feats.append(avg_features)

    return np.vstack(feats)


# Extract features
percentage_str = str(int(DATASET_PERCENTAGE * 100))
X_TRAIN_FEATURES_FILE = f"features/baseline/X_train_full_features_{percentage_str}pct.npy"
X_TEST_FEATURES_FILE = "features/baseline/X_test_features.npy"

if os.path.exists(X_TRAIN_FEATURES_FILE) and os.path.exists(X_TEST_FEATURES_FILE):
    print(
        f"Loading pre-computed features from {X_TRAIN_FEATURES_FILE} and {X_TEST_FEATURES_FILE}..."
    )
    X_train_full = np.load(X_TRAIN_FEATURES_FILE)
    X_test = np.load(X_TEST_FEATURES_FILE)
    print("Features loaded successfully.")
else:
    print("Pre-computed features not found. Starting feature extraction...")
    os.makedirs("features", exist_ok=True)

    X_train_full = get_features(df["id"], train_dir)
    np.save(X_TRAIN_FEATURES_FILE, X_train_full)
    print(f"Saved training features to {X_TRAIN_FEATURES_FILE}")

    X_test = get_features(df_test["id"], test_dir)
    np.save(X_TEST_FEATURES_FILE, X_test)
    print(f"Saved test features to {X_TEST_FEATURES_FILE}")

# ========================================
# DATA PREPARATION
# ========================================

# Scale & split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_full)
y = df["target"].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ========================================
# MODEL DEFINITION
# ========================================


class CollisionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT_RATE):
        super(CollisionClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ========================================
# TRAINING SETUP
# ========================================

# Create datasets and dataloaders
train_dataset = FeaturesDataset(X_tr, y_tr)
val_dataset = FeaturesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = CollisionClassifier(input_dim=FEAT_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

# Calculate total steps for OneCycleLR
total_steps = NUM_EPOCHS * len(train_loader)

# Initialize OneCycleLR scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    total_steps=total_steps,
    pct_start=PCT_START,
    div_factor=MAX_LR / INITIAL_LR,
    final_div_factor=INITIAL_LR / FINAL_LR,
    anneal_strategy="cos",
)

# ========================================
# LIVE PLOTTING SETUP
# ========================================

# Setup live plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Training Progress ({int(DATASET_PERCENTAGE * 100)}% of data)", fontsize=16
)

# Initialize empty lists and lines
train_losses = []
val_losses = []
val_aucs = []
learning_rates = []
epochs = []

# Setup plot lines
(line1,) = ax1.plot([], [], "b-", label="Training Loss")
(line2,) = ax1.plot([], [], "r-", label="Validation Loss")
(line3,) = ax2.plot([], [], "g-", label="Validation AUC")
(line4,) = ax3.plot([], [], "orange", label="Learning Rate")

# Setup axes
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend(frameon=False)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC")
ax2.set_title("Validation AUC")
ax2.legend(frameon=False)

ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("OneCycleLR Schedule")
ax3.set_yscale("log")
ax3.legend(frameon=False)

plt.tight_layout()


def update_plots():
    """Update the live plots with current data"""
    # Update data
    line1.set_data(epochs, train_losses)
    line2.set_data(epochs, val_losses)
    line3.set_data(epochs, val_aucs)
    line4.set_data(epochs, learning_rates)

    # Update axes limits
    if epochs:
        ax1.set_xlim(0, max(epochs) + 1)
        ax2.set_xlim(0, max(epochs) + 1)
        ax3.set_xlim(0, max(epochs) + 1)

        # Loss plot - more breathing room
        if train_losses and val_losses:
            all_losses = train_losses + val_losses
            ax1.set_ylim(min(all_losses) * 0.9, max(all_losses) * 1.05)

        # AUC plot - more breathing room
        if val_aucs:
            ax2.set_ylim(min(val_aucs) * 0.9, max(val_aucs) * 1.1)

        # LR plot - more breathing room
        if learning_rates:
            ax3.set_ylim(min(learning_rates) * 0.5, max(learning_rates) * 2.0)

    # Refresh the plot
    fig.canvas.draw()
    fig.canvas.flush_events()


# ========================================
# TRAINING FUNCTIONS
# ========================================


# Training loop with loss tracking and metrics collection
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # OneCycleLR needs to be stepped after each batch

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    auc_score = roc_auc_score(true_labels, predictions)

    return avg_loss, auc_score, predictions


# ========================================
# TRAINING LOOP WITH LIVE PLOTTING
# ========================================

# Training loop with metrics collection and live plotting
best_auc = 0
metrics_data = []

print("Starting training with live plotting...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, device
    )
    val_loss, val_auc, _ = validate(model, val_loader, criterion, device)

    # Get current learning rate
    current_lr = optimizer.param_groups[0]["lr"]

    # Store metrics
    epochs.append(epoch + 1)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    learning_rates.append(current_lr)

    # Collect metrics for CSV
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

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), f"best_model_{percentage_str}pct.pth")

    if (epoch + 1) % PRINT_FREQUENCY == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )
        print(f"Learning Rate: {current_lr:.6f}, Best Val AUC: {best_auc:.4f}")

    # Update plots
    if (epoch + 1) % PLOT_UPDATE_FREQUENCY == 0:
        update_plots()

# Final plot update
update_plots()

# ========================================
# EVALUATION AND VISUALIZATION
# ========================================

# Save metrics to CSV
if SAVE_METRICS_CSV:
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = f"training_metrics_{percentage_str}pct.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training metrics saved → {metrics_csv_path}")

# Load best model for final evaluation
model.load_state_dict(torch.load(f"best_model_{percentage_str}pct.pth"))
_, final_auc, val_predictions = validate(model, val_loader, criterion, device)
print(f"\nFinal Validation ROC-AUC: {final_auc:.4f}")

# Save the final plot
plot_filename = f"training_curves_{percentage_str}pct.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Training curves saved → {plot_filename}")

# Keep the plot window open
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the window open

# ========================================
# INFERENCE AND SUBMISSION
# ========================================

# Inference & submission
X_test_scaled = scaler.transform(X_test)
test_dataset = FeaturesDataset(X_test_scaled, np.zeros(len(X_test_scaled)))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
test_predictions = []
with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        outputs = model(features).squeeze()
        test_predictions.extend(outputs.cpu().numpy())

submission = pd.DataFrame({"id": df_test["id"], "score": test_predictions})
submission_filename = f"submission_{percentage_str}pct.csv"
submission.to_csv(submission_filename, index=False)
print(f"Written → {submission_filename}")

# Save training history
history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_aucs": val_aucs,
    "learning_rates": learning_rates,
    "best_auc": best_auc,
    "dataset_percentage": DATASET_PERCENTAGE,
}
history_filename = f"training_history_{percentage_str}pct.npy"
np.save(history_filename, history)
print(f"Training history saved → {history_filename}")

# Print final summary
print(f"\n🎯 TRAINING SUMMARY")
print(f"Dataset used: {int(DATASET_PERCENTAGE * 100)}% ({len(df)} samples)")
print(f"Best validation AUC: {best_auc:.4f}")
print(f"Final validation AUC: {final_auc:.4f}")
if SAVE_METRICS_CSV:
    print(f"Metrics saved to: {metrics_csv_path}")
