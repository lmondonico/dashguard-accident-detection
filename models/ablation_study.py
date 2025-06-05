"""
Ablation study script for comparing different model architectures and configurations
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import timm
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
codebase_dir = os.path.join(current_dir, "..")
sys.path.append(codebase_dir)

from module_hierarchical_transformer import HierarchicalTransformer
from utils.data_loader import VideoSequenceDataset
from torch.utils.data import DataLoader


class Config:
    DATASET_PERCENTAGE = 1.0
    NUM_FRAMES = 32
    FRAME_SIZE = (300, 300)

    EFFICIENTNET_RGB_FEAT_DIM = 1536 + 512
    EFFICIENTNET_FLOW_FEAT_DIM = 1536 + 512
    INCEPTION_RGB_FEAT_DIM = 2048
    INCEPTION_FLOW_FEAT_DIM = 2048

    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 3
    D_FF = 1024
    MAX_SEQ_LEN = 32
    DROPOUT = 0.3

    BATCH_SIZE = 16
    NUM_EPOCHS = 40
    LEARNING_RATE = 8e-7
    WEIGHT_DECAY = 1e-3
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    MAX_LR = 8e-7
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 100

    PRINT_FREQUENCY = 5

    BASE_DIR = "results/ablation_studies"
    FEATURES_CACHE_DIR = "features/ablation_cache"
    INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
    UNIFORM_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "uniform_flow_frames")
    TEMPORAL_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "crash_rgb_frames")

    def __init__(self):
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.FEATURES_CACHE_DIR, exist_ok=True)


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=True, features_only=True
        )

        dummy_input = torch.randn(1, 3, 300, 300)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        self.scale_dims = [f.shape[1] for f in features[-3:]]
        self.adaptive_pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(1) for _ in range(len(self.scale_dims))]
        )

        total_dim = sum(self.scale_dims)
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, 1536),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
        )

        self.motion_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            scale_features = self.backbone(x)

        pooled_features = []
        for i, features_map in enumerate(scale_features[-3:]):
            pooled = self.adaptive_pools[i](features_map).flatten(1)
            pooled_features.append(pooled)

        combined_backbone = torch.cat(pooled_features, dim=1)
        backbone_features = self.feature_fusion(combined_backbone)
        motion_features = self.motion_extractor(x)

        return torch.cat([backbone_features, motion_features], dim=1)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        self.backbone.fc = nn.Identity()
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)


class FCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        pooled = x.transpose(1, 2)
        pooled = self.global_pool(pooled).squeeze(-1)
        return self.network(pooled)


def normalize_flow_for_cnn(flow_field):
    dx = flow_field[..., 0]
    dy = flow_field[..., 1]

    norm_dx = ((dx - dx.min()) / (dx.max() - dx.min() + 1e-8) * 255).astype(np.uint8)
    norm_dy = ((dy - dy.min()) / (dy.max() - dy.min() + 1e-8) * 255).astype(np.uint8)

    flow_img = np.zeros((*flow_field.shape[:2], 3), dtype=np.uint8)
    flow_img[..., 0] = norm_dx
    flow_img[..., 1] = norm_dy

    return flow_img


def load_preprocessed_frames(video_ids, frames_dir):
    frames_list = []
    for video_id in video_ids:
        frames_file = os.path.join(frames_dir, f"{video_id}_frames.npy")
        if os.path.exists(frames_file):
            frames = np.load(frames_file)
            frames_list.append(frames)
        else:
            # Create dummy frames if file doesn't exist
            frames_list.append(np.zeros((32, 300, 300, 3), dtype=np.uint8))
    return frames_list


def get_feature_cache_path(config, split_name, feature_type, backbone):
    """Generate cache file path for features"""
    cache_filename = f"{split_name}_{feature_type}_{backbone}_features.npy"
    return os.path.join(config.FEATURES_CACHE_DIR, cache_filename)


def extract_features(
    frames_list, feature_extractor, preprocess, device, is_flow=False, cache_path=None
):
    # Check if cached features exist
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting {'flow' if is_flow else 'RGB'} features...")
    features_list = []

    for frames in tqdm(frames_list, desc="Extracting features"):
        if frames.shape[0] == 0:
            feat_dim = (
                2048
                if isinstance(feature_extractor, InceptionFeatureExtractor)
                else 2048
            )
            target_length = 31 if is_flow else 32
            features_list.append(np.zeros((target_length, feat_dim), dtype=np.float32))
            continue

        if is_flow:
            flow_sequence = []
            if frames.shape[0] >= 2:
                import cv2

                for i in range(1, len(frames)):
                    prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
                    curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    flow_img = normalize_flow_for_cnn(flow)
                    flow_sequence.append(flow_img)

                if flow_sequence:
                    batch_inputs = [preprocess(flow_img) for flow_img in flow_sequence]
                else:
                    features_list.append(np.zeros((31, 2048), dtype=np.float32))
                    continue
            else:
                features_list.append(np.zeros((31, 2048), dtype=np.float32))
                continue
        else:
            batch_inputs = [preprocess(frame) for frame in frames]

        if batch_inputs:
            batch_tensor = torch.stack(batch_inputs).to(device)
            with torch.no_grad():
                features = feature_extractor(batch_tensor).cpu().numpy()

            target_length = 31 if is_flow else 32
            if features.shape[0] < target_length:
                padding = np.zeros(
                    (target_length - features.shape[0], features.shape[1]),
                    dtype=np.float32,
                )
                features = np.vstack([features, padding])
            elif features.shape[0] > target_length:
                features = features[:target_length]

            features_list.append(features)
        else:
            target_length = 31 if is_flow else 32
            feat_dim = features.shape[1] if "features" in locals() else 2048
            features_list.append(np.zeros((target_length, feat_dim), dtype=np.float32))

    features_array = np.array(features_list, dtype=np.float32)

    if cache_path:
        print(f"Caching features to {cache_path}")
        np.save(cache_path, features_array)

    return features_array


def load_or_extract_features(config, split_dfs, backbone="efficientnet", use_flow=True):
    """Load cached features or extract them if not available"""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if backbone == "inception":
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        feature_extractor = InceptionFeatureExtractor().to(device)
    else:
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(config.FRAME_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        feature_extractor = EfficientNetFeatureExtractor().to(device)

    feature_extractor.eval()

    all_features = {}
    split_names = ["train", "val", "test"]

    for split_name, df_split in zip(split_names, split_dfs):
        print(f"\nProcessing {split_name} features...")

        rgb_cache_path = get_feature_cache_path(config, split_name, "rgb", backbone)

        print(f"Loading RGB frames for {split_name}...")
        rgb_frames = load_preprocessed_frames(
            df_split["id"], config.TEMPORAL_FRAMES_DIR
        )

        rgb_features = extract_features(
            rgb_frames,
            feature_extractor,
            preprocess,
            device,
            is_flow=False,
            cache_path=rgb_cache_path,
        )

        if use_flow:
            flow_cache_path = get_feature_cache_path(
                config, split_name, "flow", backbone
            )

            print(f"Loading flow frames for {split_name}...")
            flow_frames = load_preprocessed_frames(
                df_split["id"], config.UNIFORM_FRAMES_DIR
            )

            flow_features = extract_features(
                flow_frames,
                feature_extractor,
                preprocess,
                device,
                is_flow=True,
                cache_path=flow_cache_path,
            )

            flow_padded = np.pad(
                flow_features, ((0, 0), (1, 0), (0, 0)), mode="constant"
            )
            combined_features = np.concatenate([rgb_features, flow_padded], axis=2)
        else:
            combined_features = rgb_features

        all_features[split_name] = combined_features
        print(f"{split_name} features shape: {combined_features.shape}")

    return all_features["train"], all_features["val"], all_features["test"]


def plot_training_curves(
    train_losses, val_losses, val_aucs, learning_rates, experiment_name, exp_dir
):
    """Create training progress plots similar to the baseline example"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{experiment_name} Training Progress", fontsize=16)

    epochs = list(range(1, len(train_losses) + 1))

    axes[0].plot(epochs, train_losses, "b-", label="Training Loss")
    axes[0].plot(epochs, val_losses, "r-", label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, val_aucs, "g-", label="Validation AUC")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend(frameon=False)

    axes[2].plot(epochs, learning_rates, "orange", label="Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(exp_dir, "training_curves.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix(y_true, y_pred, experiment_name, exp_dir, threshold=0.5):
    """Create confusion matrix with counts and percentages"""

    y_pred_binary = (np.array(y_pred) > threshold).astype(int)
    y_true_binary = np.array(y_true).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    cm_percent = cm.astype("float") / cm.sum() * 100

    labels = np.array(
        [
            [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]
    )

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({"font.size": 14})

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["No Collision", "Collision"],
        yticklabels=["No Collision", "Collision"],
        annot_kws={"size": 16},
    )
    plt.title(f"{experiment_name} - Test Set Confusion Matrix", fontsize=18)
    plt.ylabel("True Label", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(
        os.path.join(exp_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    return cm, cm_percent


def run_experiment(
    experiment_name, model_constructor, use_flow=True, backbone="efficientnet"
):
    print(f"\n=== Running {experiment_name} ===")

    config = Config()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    exp_dir = os.path.join(config.BASE_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    results_file = os.path.join(exp_dir, "results.csv")
    if os.path.exists(results_file):
        print(f"Experiment {experiment_name} already completed. Loading results...")
        results_df = pd.read_csv(results_file)
        return results_df.iloc[0].to_dict()

    data_base_path = os.path.expanduser("./data-nexar/")
    df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df["id"] = df["id"].astype(str).str.zfill(5)

    if config.DATASET_PERCENTAGE < 1.0:
        df = df.sample(
            frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df["target"],
        random_state=config.RANDOM_STATE,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.VAL_SIZE,
        stratify=train_val_df["target"],
        random_state=config.RANDOM_STATE,
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train_combined, X_val_combined, X_test_combined = load_or_extract_features(
        config, [train_df, val_df, test_df], backbone=backbone, use_flow=use_flow
    )

    num_train_videos, num_frames, num_features = X_train_combined.shape
    X_train_reshaped = X_train_combined.reshape(-1, num_features)

    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(
        num_train_videos, num_frames, num_features
    )

    X_val_reshaped = X_val_combined.reshape(-1, num_features)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled_reshaped.reshape(len(val_df), num_frames, num_features)

    X_test_reshaped = X_test_combined.reshape(-1, num_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(
        len(test_df), num_frames, num_features
    )

    print(f"Feature shape: {X_train_scaled.shape}")

    train_dataset = VideoSequenceDataset(X_train_scaled, train_df["target"].values)
    val_dataset = VideoSequenceDataset(X_val_scaled, val_df["target"].values)
    test_dataset = VideoSequenceDataset(X_test_scaled, test_df["target"].values)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = model_constructor(input_dim=num_features).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    total_steps = config.NUM_EPOCHS * steps_per_epoch

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        total_steps=total_steps,
        pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
    )

    best_val_auc = 0.0
    train_losses, val_losses, val_aucs, learning_rates = [], [], [], []

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        val_predictions, val_labels = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_loss = total_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_predictions)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        learning_rates.append(current_lr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

        if (epoch + 1) % config.PRINT_FREQUENCY == 0:
            print(
                f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth")))
    model.eval()

    test_predictions, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            test_predictions.extend(outputs.cpu().numpy().flatten())
            test_labels.extend(labels.cpu().numpy().flatten())

    test_auc = roc_auc_score(test_labels, test_predictions)

    plot_training_curves(
        train_losses, val_losses, val_aucs, learning_rates, experiment_name, exp_dir
    )
    cm, cm_percent = plot_confusion_matrix(
        test_labels, test_predictions, experiment_name, exp_dir
    )

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results = {
        "experiment": experiment_name,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "input_dim": num_features,
        "use_flow": use_flow,
        "backbone": backbone,
        "confusion_matrix_tn": int(tn),
        "confusion_matrix_fp": int(fp),
        "confusion_matrix_fn": int(fn),
        "confusion_matrix_tp": int(tp),
        "confusion_matrix_tn_pct": float(cm_percent[0, 0]),
        "confusion_matrix_fp_pct": float(cm_percent[0, 1]),
        "confusion_matrix_fn_pct": float(cm_percent[1, 0]),
        "confusion_matrix_tp_pct": float(cm_percent[1, 1]),
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)

    print(f"Results: Val AUC = {best_val_auc:.4f}, Test AUC = {test_auc:.4f}")
    print(f"Confusion Matrix:")
    print(f"TN: {tn} ({cm_percent[0, 0]:.1f}%), FP: {fp} ({cm_percent[0, 1]:.1f}%)")
    print(f"FN: {fn} ({cm_percent[1, 0]:.1f}%), TP: {tp} ({cm_percent[1, 1]:.1f}%)")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return results


def main():
    config = Config()
    all_results = []

    # Ablation 1: Original (EfficientNet + Hierarchical Transformer + Flow)
    def efficientnet_hierarchical(input_dim):
        return HierarchicalTransformer(
            input_dim=input_dim,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            d_ff=config.D_FF,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
        )

    results1 = run_experiment(
        "efficientnet_hierarchical_with_flow",
        efficientnet_hierarchical,
        use_flow=True,
        backbone="efficientnet",
    )
    all_results.append(results1)

    # Ablation 2: InceptionV3 + Hierarchical Transformer + Flow
    def inception_hierarchical(input_dim):
        return HierarchicalTransformer(
            input_dim=input_dim,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            d_ff=config.D_FF,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
        )

    results2 = run_experiment(
        "inception_hierarchical_with_flow",
        inception_hierarchical,
        use_flow=True,
        backbone="inception",
    )
    all_results.append(results2)

    # Ablation 3: EfficientNet + FC + Flow
    results3 = run_experiment(
        "efficientnet_fc_with_flow",
        FCClassifier,
        use_flow=True,
        backbone="efficientnet",
    )
    all_results.append(results3)

    # Ablation 4: EfficientNet + Hierarchical Transformer (no flow)
    results4 = run_experiment(
        "efficientnet_hierarchical_no_flow",
        efficientnet_hierarchical,
        use_flow=False,
        backbone="efficientnet",
    )
    all_results.append(results4)

    consolidated_df = pd.DataFrame(all_results)
    consolidated_df.to_csv(
        os.path.join(config.BASE_DIR, "consolidated_results.csv"), index=False
    )

    print("\n=== ABLATION STUDY RESULTS ===")
    for result in all_results:
        print(
            f"{result['experiment']}: Val AUC = {result['best_val_auc']:.4f}, Test AUC = {result['test_auc']:.4f}"
        )

    consolidated_df = pd.DataFrame(all_results)
    consolidated_df.to_csv(
        os.path.join(config.BASE_DIR, "consolidated_results.csv"), index=False
    )

    print("\n=== ABLATION STUDY RESULTS ===")
    for result in all_results:
        print(f"\n{result['experiment']}:")
        print(
            f"Val AUC = {result['best_val_auc']:.4f}, Test AUC = {result['test_auc']:.4f}"
        )
        print(f"Test Accuracy = {result['test_accuracy']:.4f}")
        print(
            f"Confusion Matrix: TN={result['confusion_matrix_tn']} ({result['confusion_matrix_tn_pct']:.1f}%), "
            + f"FP={result['confusion_matrix_fp']} ({result['confusion_matrix_fp_pct']:.1f}%)"
        )
        print(
            f"FN={result['confusion_matrix_fn']} ({result['confusion_matrix_fn_pct']:.1f}%), "
            + f"TP={result['confusion_matrix_tp']} ({result['confusion_matrix_tp_pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
