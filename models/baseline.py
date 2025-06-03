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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

DATASET_PERCENTAGE = 1.0
NUM_FRAMES = 32
FRAME_SIZE = (299, 299)
FEAT_DIM = 2048
HIDDEN_DIMS = [128]
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
NUM_EPOCHS = 1000
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.35
RANDOM_STATE = 42
INITIAL_LR = 1e-8
MAX_LR = 5e-7
FINAL_LR = 5e-9
PCT_START = 0.2
SAVE_METRICS_CSV = True
PRINT_FREQUENCY = 20
PLOT_UPDATE_FREQUENCY = 5

RESULTS_DIR = "results/baseline"
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
FEATURES_ROOT_DIR = "features/baseline"
INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
UNIFORM_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "uniform_flow_frames")

for directory in [
    RESULTS_DIR,
    WEIGHTS_DIR,
    FEATURES_ROOT_DIR,
    INTERMEDIATE_FRAMES_DIR,
    UNIFORM_FRAMES_DIR,
    "./cache",
]:
    os.makedirs(directory, exist_ok=True)

os.environ["TORCH_HOME"] = "./cache"
plt.ion()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

df = pd.read_csv("./data-nexar/train.csv")
df_test = pd.read_csv("./data-nexar/test.csv")
df["id"] = df["id"].astype(str).str.zfill(5)
df_test["id"] = df_test["id"].astype(str).str.zfill(5)

if DATASET_PERCENTAGE < 1.0:
    print(f"Sampling {DATASET_PERCENTAGE * 100}% of the training dataset...")
    df = df.sample(frac=DATASET_PERCENTAGE, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    print(f"Training set reduced to {len(df)} samples")

train_dir = "./data-nexar/train/"
test_dir = "./data-nexar/test/"
df["train_videos"] = df["id"] + ".mp4"
df_test["test_videos"] = df_test["id"] + ".mp4"
print(f"Total Train Videos: {len(df['train_videos'])}")
print(f"Total Test Videos:  {len(df_test['test_videos'])}")


def extract_frames(path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((num_frames, *size, 3), dtype=np.uint8)

    frames_list = []
    frame_indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frame_indices = np.clip(frame_indices, 0, total - 1)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            if frames_list:
                frames_list.append(frames_list[-1].copy())
            else:
                frames_list.append(np.zeros(size + (3,), dtype=np.uint8))
            continue
        frame = cv2.resize(frame, size)
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1].copy())
        else:
            frames_list.append(np.zeros(size + (3,), dtype=np.uint8))
    return np.stack(frames_list[:num_frames])


base_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
base_model.fc = nn.Identity()
base_model.aux_logits = False
base_model = base_model.to(device)
base_model.eval()

preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_features(ids, folder, uniform_frames_storage_dir):
    feats = []
    for vid in tqdm(ids, desc=f"Extracting from {folder}"):
        intermediate_frames_file = os.path.join(
            uniform_frames_storage_dir, f"{vid}_frames.npy"
        )
        frames_np = None

        if os.path.exists(intermediate_frames_file):
            frames_np = np.load(intermediate_frames_file)
            if frames_np.shape[0] != NUM_FRAMES:
                frames_np = None

        if frames_np is None:
            path = os.path.join(folder, f"{vid}.mp4")
            frames_np = extract_frames(path, num_frames=NUM_FRAMES, size=FRAME_SIZE)
            np.save(intermediate_frames_file, frames_np)

        if frames_np.size == 0 or frames_np.shape[0] < NUM_FRAMES:
            print(
                f"Warning: Not enough frames for {vid} ({frames_np.shape[0] if frames_np is not None else 0}), using zeros."
            )
            feats.append(np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        batch = torch.stack([preprocess(frame) for frame in frames_np]).to(device)
        with torch.no_grad():
            features_output = base_model(batch)
            avg_features = features_output.mean(dim=0).cpu().numpy()
        feats.append(avg_features)
    return np.vstack(feats)


percentage_str = str(int(DATASET_PERCENTAGE * 100))
X_TRAIN_FEATURES_FILE = os.path.join(
    FEATURES_ROOT_DIR, f"X_train_full_features_{percentage_str}pct_N{NUM_FRAMES}.npy"
)
X_TEST_FEATURES_FILE = os.path.join(
    FEATURES_ROOT_DIR, f"X_test_features_N{NUM_FRAMES}.npy"
)

if os.path.exists(X_TRAIN_FEATURES_FILE) and os.path.exists(X_TEST_FEATURES_FILE):
    print(f"Loading pre-computed aggregated features...")
    X_train_full = np.load(X_TRAIN_FEATURES_FILE)
    X_test = np.load(X_TEST_FEATURES_FILE)
else:
    print("Aggregated features not found. Generating...")
    X_train_full = get_features(df["id"], train_dir, UNIFORM_FRAMES_DIR)
    np.save(X_TRAIN_FEATURES_FILE, X_train_full)
    print(f"Saved training aggregated features to {X_TRAIN_FEATURES_FILE}")
    X_test = get_features(df_test["id"], test_dir, UNIFORM_FRAMES_DIR)
    np.save(X_TEST_FEATURES_FILE, X_test)
    print(f"Saved test aggregated features to {X_TEST_FEATURES_FILE}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_full)
y = df["target"].values
X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)


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
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
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


train_dataset = FeaturesDataset(X_tr, y_tr)
val_dataset = FeaturesDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CollisionClassifier(input_dim=FEAT_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
total_steps = NUM_EPOCHS * len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    total_steps=total_steps,
    pct_start=PCT_START,
    div_factor=MAX_LR / INITIAL_LR if INITIAL_LR > 0 else 25,  # Avoid division by zero
    final_div_factor=INITIAL_LR / FINAL_LR
    if FINAL_LR > 0
    else 1e4,  # Avoid division by zero
    anneal_strategy="cos",
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Baseline Training Progress ({percentage_str}% data, N{NUM_FRAMES})", fontsize=16
)
train_losses, val_losses, val_aucs, learning_rates, epochs_list = [], [], [], [], []
(line1,) = ax1.plot([], [], "b-", label="Training Loss")
(line2,) = ax1.plot([], [], "r-", label="Validation Loss")
(line3,) = ax2.plot([], [], "g-", label="Validation AUC")
(line4,) = ax3.plot([], [], "orange", label="Learning Rate")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss")
ax1.legend(frameon=False)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC")
ax2.set_title("Validation AUC")
ax2.legend(frameon=False)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("LR Schedule")
ax3.set_yscale("log")
ax3.legend(frameon=False)
plt.tight_layout()


def update_plots():
    line1.set_data(epochs_list, train_losses)
    line2.set_data(epochs_list, val_losses)
    line3.set_data(epochs_list, val_aucs)
    line4.set_data(epochs_list, learning_rates)
    if epochs_list:
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, max(epochs_list) + 1)
        if train_losses and val_losses:
            ax1.set_ylim(
                min(train_losses + val_losses) * 0.9,
                max(train_losses + val_losses) * 1.05,
            )
        if val_aucs:
            ax2.set_ylim(
                min(val_aucs) * 0.9 if min(val_aucs) > 0 else -0.1,
                max(val_aucs) * 1.1 if max(val_aucs) < 1 else 1.0,
            )
        if learning_rates:
            ax3.set_ylim(min(learning_rates) * 0.5, max(learning_rates) * 2.0)
    fig.canvas.draw()
    fig.canvas.flush_events()


def train_epoch_baseline(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_baseline(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(val_loader)
    try:
        auc = roc_auc_score(true_labels, predictions)
    except ValueError:
        print("AUC error in validation")
        auc = 0.0
    return avg_loss, auc


best_auc = 0.0
metrics_data = []
final_auc = 0.0
print("Starting baseline training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch_baseline(
        model, train_loader, criterion, optimizer, scheduler, device
    )
    val_loss, val_auc = validate_baseline(model, val_loader, criterion, device)
    current_lr = optimizer.param_groups[0]["lr"]
    epochs_list.append(epoch + 1)
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
            "lr": current_lr,
        }
    )

    MODEL_SAVE_PATH = os.path.join(
        WEIGHTS_DIR, f"best_model_baseline_{percentage_str}pct_N{NUM_FRAMES}.pth"
    )
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    if (epoch + 1) % PRINT_FREQUENCY == 0:
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.2e} | Best AUC: {best_auc:.4f}"
        )
    if (epoch + 1) % PLOT_UPDATE_FREQUENCY == 0:
        update_plots()
update_plots()

METRICS_CSV_PATH = os.path.join(
    RESULTS_DIR, f"training_metrics_baseline_{percentage_str}pct_N{NUM_FRAMES}.csv"
)
PLOT_FILENAME = os.path.join(
    RESULTS_DIR, f"training_curves_baseline_{percentage_str}pct_N{NUM_FRAMES}.png"
)
SUBMISSION_FILENAME = os.path.join(
    RESULTS_DIR, f"submission_baseline_{percentage_str}pct_N{NUM_FRAMES}.csv"
)
HISTORY_FILENAME = os.path.join(
    RESULTS_DIR, f"training_history_baseline_{percentage_str}pct_N{NUM_FRAMES}.npy"
)
MODEL_SAVE_PATH = os.path.join(
    WEIGHTS_DIR, f"best_model_baseline_{percentage_str}pct_N{NUM_FRAMES}.pth"
)  # Re-affirm for loading

if SAVE_METRICS_CSV:
    pd.DataFrame(metrics_data).to_csv(METRICS_CSV_PATH, index=False)
    print(f"Metrics saved to {METRICS_CSV_PATH}")

if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(f"Loaded best model from {MODEL_SAVE_PATH}")
    _, final_auc = validate_baseline(model, val_loader, criterion, device)
    print(f"\nFinal Validation ROC-AUC (best model): {final_auc:.4f}")
else:
    print(f"Best model path not found: {MODEL_SAVE_PATH}")

plt.savefig(PLOT_FILENAME, dpi=300, bbox_inches="tight")
print(f"Plot saved to {PLOT_FILENAME}")
plt.ioff()
plt.show()

X_test_scaled = scaler.transform(X_test)
test_dataset = FeaturesDataset(X_test_scaled, np.zeros(len(X_test_scaled)))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model.eval()
test_predictions = []
with torch.no_grad():
    for features_batch, _ in test_loader:
        features_batch = features_batch.to(device)
        outputs = model(features_batch)
        test_predictions.extend(outputs.cpu().numpy().flatten())
submission = pd.DataFrame({"id": df_test["id"], "score": test_predictions})
submission.to_csv(SUBMISSION_FILENAME, index=False)
print(f"Submission saved to {SUBMISSION_FILENAME}")

history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_aucs": val_aucs,
    "lr": learning_rates,
    "best_auc": best_auc,
    "final_auc": final_auc,
    "dataset_percentage": DATASET_PERCENTAGE,
    "num_frames": NUM_FRAMES,
}
np.save(HISTORY_FILENAME, history)
print(f"History saved to {HISTORY_FILENAME}")

summary_path = os.path.join(
    RESULTS_DIR, f"training_summary_baseline_{percentage_str}pct_N{NUM_FRAMES}.txt"
)
with open(summary_path, "w") as f:
    f.write("BASELINE MLP TRAINING SUMMARY\n=============================\n")
    f.write(f"Timestamp: {pd.Timestamp.now()}\n")
    f.write(
        f"Dataset Percentage: {DATASET_PERCENTAGE * 100}%\nNumber of Videos (train sample): {len(df)}\n"
    )
    f.write(
        f"Number of Frames per Video: {NUM_FRAMES}\nFeature Dimension (InceptionV3): {FEAT_DIM}\n"
    )
    f.write(f"Hidden Dimensions: {HIDDEN_DIMS}\nDropout Rate: {DROPOUT_RATE}\n")
    f.write(
        f"Epochs: {NUM_EPOCHS}\nBatch Size: {BATCH_SIZE}\nOptimizer: Adam, Weight Decay: {WEIGHT_DECAY}\n"
    )
    f.write(
        f"Scheduler: OneCycleLR (Initial: {INITIAL_LR}, Max: {MAX_LR}, Final: {FINAL_LR})\n"
    )
    f.write(
        f"Best Validation AUC: {best_auc:.4f}\nFinal Validation AUC (loaded best model): {final_auc:.4f}\n\n"
    )
    f.write("PATHS:\n")
    f.write(f"  Aggregated CNN Features: {FEATURES_ROOT_DIR}\n")
    f.write(f"  Intermediate Uniform Frames: {UNIFORM_FRAMES_DIR}\n")
    f.write(f"  Results: {RESULTS_DIR}\n")
    f.write(f"  Weights: {WEIGHTS_DIR}\n")
    f.write(f"  Model Saved: {MODEL_SAVE_PATH}\n")
    f.write(f"  Plot: {PLOT_FILENAME}\n")
    f.write(f"  Metrics CSV: {METRICS_CSV_PATH}\n")
    f.write(f"  History Numpy: {HISTORY_FILENAME}\n")
    f.write(f"  Submission: {SUBMISSION_FILENAME}\n")
print(f"Training summary saved to {summary_path}")
print(
    f"\nBASELINE MLP SUMMARY\nDataset: {percentage_str}% ({len(df)} samples), Frames: {NUM_FRAMES}\nBest AUC: {best_auc:.4f}, Final AUC: {final_auc:.4f}"
)
