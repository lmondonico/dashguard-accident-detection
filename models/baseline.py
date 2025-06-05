"""
Baseline model for collision detection using InceptionV3 features.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
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
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-2
FINAL_TEST_SIZE = 0.1  # 10% for final test
CV_FOLDS = 5
RANDOM_STATE = 42
INITIAL_LR = 1e-8
MAX_LR = 5e-7
FINAL_LR = 5e-9
PCT_START = 0.2
SAVE_METRICS_CSV = True
PRINT_FREQUENCY = 10

RESULTS_DIR = "results/baseline_cv"
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
FEATURES_ROOT_DIR = "features/baseline_cv"
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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

df = pd.read_csv("./data-nexar/train.csv")
df["id"] = df["id"].astype(str).str.zfill(5)

if DATASET_PERCENTAGE < 1.0:
    print(f"Sampling {DATASET_PERCENTAGE * 100}% of the training dataset...")
    df = df.sample(frac=DATASET_PERCENTAGE, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    print(f"Training set reduced to {len(df)} samples")

train_dir = "./data-nexar/train/"
df["train_videos"] = df["id"] + ".mp4"
print(f"Total Train Videos: {len(df['train_videos'])}")

cv_data, final_test_data = train_test_split(
    df, test_size=FINAL_TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE
)
cv_data = cv_data.reset_index(drop=True)
final_test_data = final_test_data.reset_index(drop=True)

print(f"CV data: {len(cv_data)} samples")
print(f"Final test data: {len(final_test_data)} samples")


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


# Load InceptionV3 model
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


def get_features(ids, folder, uniform_frames_storage_dir, split_name=""):
    feats = []
    for vid in tqdm(ids, desc=f"Extracting from {folder} ({split_name})"):
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


def load_or_extract_features(data_split, split_name):
    """Load features if they exist, otherwise extract them"""
    percentage_str = str(int(DATASET_PERCENTAGE * 100))
    features_file = os.path.join(
        FEATURES_ROOT_DIR,
        f"X_{split_name}_features_{percentage_str}pct_N{NUM_FRAMES}.npy",
    )

    if os.path.exists(features_file):
        print(f"Loading existing {split_name} features from {features_file}")
        return np.load(features_file)
    else:
        print(f"Extracting {split_name} features...")
        features = get_features(
            data_split["id"], train_dir, UNIFORM_FRAMES_DIR, split_name
        )
        np.save(features_file, features)
        print(f"Saved {split_name} features to {features_file}")
        return features


def check_existing_models():
    """Check if all fold models already exist"""
    existing_models = []
    for fold in range(1, CV_FOLDS + 1):
        model_path = os.path.join(WEIGHTS_DIR, f"best_model_fold_{fold}.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                existing_models.append(
                    {
                        "fold": fold,
                        "model_path": model_path,
                        "val_auc": checkpoint.get("val_auc", 0.0),
                        "scaler": checkpoint.get("scaler", None),
                        "best_val_auc": checkpoint.get("val_auc", 0.0),
                        "fold_metrics": [],
                    }
                )
                print(
                    f"Found existing model for fold {fold} with AUC: {checkpoint.get('val_auc', 0.0):.4f}"
                )
            except Exception as e:
                print(f"Warning: Could not load model for fold {fold}: {e}")
                return None
        else:
            return None

    if len(existing_models) == CV_FOLDS:
        print(f"Found all {CV_FOLDS} trained models. Skipping training.")
        return existing_models
    else:
        return None


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
    return avg_loss, auc, predictions


def run_cross_validation():
    """Run 5-fold cross-validation"""

    existing_models = check_existing_models()
    if existing_models is not None:
        return existing_models, None, None

    print("Loading/extracting features for CV data...")
    X_cv = load_or_extract_features(cv_data, "cv")
    y_cv = cv_data["target"].values

    print("Loading/extracting features for final test data...")
    X_final_test = load_or_extract_features(final_test_data, "final_test")
    y_final_test = final_test_data["target"].values

    cv_results = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"\n=== FOLD {fold + 1}/{CV_FOLDS} ===")

        model_path = os.path.join(WEIGHTS_DIR, f"best_model_fold_{fold + 1}.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                print(
                    f"Found existing model for fold {fold + 1} with AUC: {checkpoint.get('val_auc', 0.0):.4f}"
                )
                cv_results.append(
                    {
                        "fold": fold + 1,
                        "best_val_auc": checkpoint.get("val_auc", 0.0),
                        "model_path": model_path,
                        "scaler": checkpoint.get("scaler", None),
                        "fold_metrics": [],
                    }
                )
                continue
            except Exception as e:
                print(
                    f"Warning: Could not load existing model for fold {fold + 1}: {e}"
                )
                print("Will retrain this fold.")

        X_train_fold, X_val_fold = X_cv[train_idx], X_cv[val_idx]
        y_train_fold, y_val_fold = y_cv[train_idx], y_cv[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        train_dataset = FeaturesDataset(X_train_scaled, y_train_fold)
        val_dataset = FeaturesDataset(X_val_scaled, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = CollisionClassifier(input_dim=FEAT_DIM).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
        )

        total_steps = NUM_EPOCHS * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=MAX_LR,
            total_steps=total_steps,
            pct_start=PCT_START,
            div_factor=MAX_LR / INITIAL_LR if INITIAL_LR > 0 else 25,
            final_div_factor=INITIAL_LR / FINAL_LR if FINAL_LR > 0 else 1e4,
            anneal_strategy="cos",
        )

        best_auc = 0.0
        fold_metrics = []

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch_baseline(
                model, train_loader, criterion, optimizer, scheduler, device
            )
            val_loss, val_auc, _ = validate_baseline(
                model, val_loader, criterion, device
            )
            current_lr = optimizer.param_groups[0]["lr"]

            fold_metrics.append(
                {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "lr": current_lr,
                }
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "scaler": scaler,
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "val_auc": val_auc,
                    },
                    model_path,
                )

            if (epoch + 1) % PRINT_FREQUENCY == 0:
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"Best AUC: {best_auc:.4f}"
                )

        cv_results.append(
            {
                "fold": fold + 1,
                "best_val_auc": best_auc,
                "model_path": model_path,
                "scaler": scaler,
                "fold_metrics": fold_metrics,
            }
        )

        print(f"Fold {fold + 1} completed. Best AUC: {best_auc:.4f}")

    return cv_results, X_final_test, y_final_test


def evaluate_final_test(cv_results, X_final_test, y_final_test):
    """Evaluate on final test set using ensemble of CV models"""

    if X_final_test is None:
        print("Loading final test features for evaluation...")
        X_final_test = load_or_extract_features(final_test_data, "final_test")
        y_final_test = final_test_data["target"].values

    all_predictions = []

    for fold_result in cv_results:
        model = CollisionClassifier(input_dim=FEAT_DIM).to(device)
        checkpoint = torch.load(
            fold_result["model_path"], map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = fold_result["scaler"]

        X_test_scaled = scaler.transform(X_final_test)

        test_dataset = FeaturesDataset(X_test_scaled, y_final_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        fold_predictions = []

        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                outputs = model(features)
                fold_predictions.extend(outputs.cpu().numpy().flatten())

        all_predictions.append(fold_predictions)

    ensemble_predictions = np.mean(all_predictions, axis=0)
    final_auc = roc_auc_score(y_final_test, ensemble_predictions)

    return final_auc, ensemble_predictions


def save_cv_results(cv_results, final_auc):
    """Save cross-validation results"""
    percentage_str = str(int(DATASET_PERCENTAGE * 100))

    all_metrics = []
    for result in cv_results:
        fold_metrics = result.get("fold_metrics", [])
        all_metrics.extend(fold_metrics)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(
            RESULTS_DIR, f"cv_metrics_baseline_{percentage_str}pct_N{NUM_FRAMES}.csv"
        )
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Detailed metrics saved to: {metrics_path}")
    else:
        metrics_path = "No detailed metrics (models were pre-trained)"
        print("No detailed metrics to save (models were pre-trained)")

    fold_summary = pd.DataFrame(
        [
            {
                "fold": result["fold"],
                "best_val_auc": result["best_val_auc"],
            }
            for result in cv_results
        ]
    )

    fold_summary_path = os.path.join(
        RESULTS_DIR, f"cv_fold_summary_{percentage_str}pct_N{NUM_FRAMES}.csv"
    )
    fold_summary.to_csv(fold_summary_path, index=False)

    fold_aucs = [result["best_val_auc"] for result in cv_results]

    summary_path = os.path.join(
        RESULTS_DIR, f"cv_summary_baseline_{percentage_str}pct_N{NUM_FRAMES}.txt"
    )

    with open(summary_path, "w") as f:
        f.write("5-FOLD CROSS-VALIDATION RESULTS\n\n")
        f.write(f"Dataset: {DATASET_PERCENTAGE * 100}% of training data\n")
        f.write(f"Samples: CV={len(cv_data)}, Test={len(final_test_data)}\n")

        f.write("\nResults:\n")
        for i, result in enumerate(cv_results):
            f.write(f"Fold {result['fold']}: AUC = {result['best_val_auc']:.4f}\n")

        f.write(f"\nMean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n")
        f.write(f"Final Test AUC: {final_auc:.4f}\n")

        f.write(f"\nOutput files:\n")
        f.write(f"- Metrics: {os.path.basename(metrics_path)}\n")
        f.write(f"- Summary: {os.path.basename(fold_summary_path)}\n")

    print(f"Results saved to {summary_path}")
    print(f"CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Final Test AUC: {final_auc:.4f}")

    return metrics_path, fold_summary_path, summary_path


print("Starting 5-fold cross-validation...")
cv_results, X_final_test, y_final_test = run_cross_validation()

print("\nEvaluating on final test set...")
final_auc, final_predictions = evaluate_final_test(
    cv_results, X_final_test, y_final_test
)

print("\nSaving results...")
save_cv_results(cv_results, final_auc)

print(f"\n CROSS-VALIDATION COMPLETED")
fold_aucs = [result["best_val_auc"] for result in cv_results]
print(f"CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"Final Test AUC: {final_auc:.4f}")
