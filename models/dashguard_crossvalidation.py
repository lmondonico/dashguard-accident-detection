"""
Cross-validation training script for the DashGuard model
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision import models, transforms
import timm
import warnings

warnings.filterwarnings(
    "ignore", message=".*Unexpected keys.*found while loading pretrained weights.*"
)

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

    CRASH_FOCUS_RATIO = 0.7
    TIME_WINDOW_SECONDS = 5.0

    BACKBONE = "efficientnet_b3"
    RGB_FEAT_DIM = 1536 + 512
    FLOW_FEAT_DIM = 1536 + 512
    COMBINED_FEAT_DIM = RGB_FEAT_DIM + FLOW_FEAT_DIM

    INPUT_DIM = COMBINED_FEAT_DIM
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
    FINAL_TEST_SIZE = 0.1  # 10% for final test
    CV_FOLDS = 5
    RANDOM_STATE = 42

    MAX_LR = 8e-7
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e3

    PRINT_FREQUENCY = 5

    RESULTS_DIR = "results/cv_efficientnet_multimodal"
    WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")

    INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
    UNIFORM_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "uniform_flow_frames")
    TEMPORAL_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "crash_rgb_frames")

    FLOW_FEATURES_DIR = "features/efficientnet_optical_flow"
    TEMPORAL_RGB_FEATURES_DIR = "features/efficientnet_crash_rgb"

    def __init__(self):
        for directory in [
            self.RESULTS_DIR,
            self.WEIGHTS_DIR,
            self.INTERMEDIATE_FRAMES_DIR,
            self.UNIFORM_FRAMES_DIR,
            self.TEMPORAL_FRAMES_DIR,
            self.FLOW_FEATURES_DIR,
            self.TEMPORAL_RGB_FEATURES_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    @property
    def percentage_str(self):
        return str(int(self.DATASET_PERCENTAGE * 100))


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, backbone_name="efficientnet_b3"):
        super(MultiScaleFeatureExtractor, self).__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
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

        final_features = torch.cat([backbone_features, motion_features], dim=1)

        return final_features


def extract_uniform_frames(video_path, num_frames=32, frame_size=(300, 300)):
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *frame_size, 3), dtype=np.uint8)

    frame_positions = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)

    frames_list = []
    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1])
        else:
            frames_list.append(np.zeros((*frame_size, 3), dtype=np.uint8))

    return np.stack(frames_list[:num_frames])


def extract_crash_frames(
    video_path,
    crash_time,
    num_frames=32,
    crash_focus_ratio=0.7,
    time_window=5.0,
    frame_size=(300, 300),
):
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *frame_size, 3), dtype=np.uint8)

    if pd.isna(crash_time) or crash_time < 0:
        frame_positions = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)
    else:
        crash_frame = int(crash_time * fps)
        crash_frame = max(0, min(crash_frame, total_video_frames - 1))

        crash_frames_count = int(num_frames * crash_focus_ratio)
        context_frames_count = num_frames - crash_frames_count

        window_frames = int(time_window * fps)
        start_crash_window = max(0, crash_frame - window_frames // 2)
        end_crash_window = min(total_video_frames, crash_frame + window_frames // 2)

        frame_positions = []

        if end_crash_window > start_crash_window:
            crash_positions = np.linspace(
                start_crash_window, end_crash_window, crash_frames_count, dtype=int
            )
            frame_positions.extend(crash_positions)

        if context_frames_count > 0:
            before_frames = context_frames_count // 2
            if start_crash_window > 0:
                before_positions = np.linspace(
                    0, start_crash_window, before_frames + 1, dtype=int
                )[:-1]
                frame_positions.extend(before_positions)

            after_frames = context_frames_count - before_frames
            if end_crash_window < total_video_frames:
                after_positions = np.linspace(
                    end_crash_window,
                    total_video_frames - 1,
                    after_frames + 1,
                    dtype=int,
                )[1:]
                frame_positions.extend(after_positions)

        frame_positions = sorted(list(set(frame_positions)))

    frames_list = []
    for pos in frame_positions[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1])
        else:
            frames_list.append(np.zeros((*frame_size, 3), dtype=np.uint8))

    return np.stack(frames_list[:num_frames])


def calculate_optical_flow(frames_sequence):
    flow_sequence = []
    if len(frames_sequence) < 2:
        return np.empty((0, *frames_sequence.shape[1:3], 2), dtype=np.float32)

    prev_gray = cv2.cvtColor(frames_sequence[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames_sequence)):
        current_gray = cv2.cvtColor(frames_sequence[i], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow_sequence.append(flow)
        prev_gray = current_gray

    return (
        np.stack(flow_sequence)
        if flow_sequence
        else np.empty((0, *frames_sequence.shape[1:3], 2), dtype=np.float32)
    )


def normalize_flow_for_cnn(flow_field):
    dx = flow_field[..., 0]
    dy = flow_field[..., 1]

    norm_dx = cv2.normalize(dx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_dy = cv2.normalize(dy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flow_img_3channel = np.zeros((*flow_field.shape[:2], 3), dtype=np.uint8)
    flow_img_3channel[..., 0] = norm_dx
    flow_img_3channel[..., 1] = norm_dy
    flow_img_3channel[..., 2] = 0

    return flow_img_3channel


class MultimodalFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.feature_extractor = MultiScaleFeatureExtractor(config.BACKBONE).to(
            self.device
        )
        self.feature_extractor.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(config.FRAME_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_and_save_frames(
        self,
        video_ids,
        video_folder,
        crash_times,
        branch_name,
        frames_dir,
        is_final_test=False,
    ):
        for i, video_id in enumerate(
            tqdm(video_ids, desc=f"Extracting {branch_name} frames")
        ):
            frames_file = os.path.join(frames_dir, f"{video_id}_frames.npy")

            if os.path.exists(frames_file):
                continue

            video_path = os.path.join(video_folder, f"{video_id}.mp4")

            if branch_name == "uniform_flow" or is_final_test:
                frames = extract_uniform_frames(
                    video_path, self.config.NUM_FRAMES, self.config.FRAME_SIZE
                )
            elif branch_name == "crash_rgb":
                crash_time = crash_times[i] if crash_times is not None else None

                if is_final_test:
                    frames = extract_uniform_frames(
                        video_path, self.config.NUM_FRAMES, self.config.FRAME_SIZE
                    )
                else:
                    frames = extract_crash_frames(
                        video_path,
                        crash_time,
                        self.config.NUM_FRAMES,
                        self.config.CRASH_FOCUS_RATIO,
                        self.config.TIME_WINDOW_SECONDS,
                        self.config.FRAME_SIZE,
                    )

            np.save(frames_file, frames)

    def process_branch_to_features(
        self, video_ids, frames_dir, output_file, is_flow=False
    ):
        if os.path.exists(output_file):
            print(f"Loading existing features from {output_file}")
            return np.load(output_file)

        print(f"Processing {'flow' if is_flow else 'RGB'} features...")
        features_list = []
        target_length = (
            self.config.NUM_FRAMES - 1 if is_flow else self.config.NUM_FRAMES
        )
        feat_dim = self.config.FLOW_FEAT_DIM if is_flow else self.config.RGB_FEAT_DIM

        for video_id in tqdm(
            video_ids, desc=f"Processing {'flow' if is_flow else 'RGB'}"
        ):
            frames_file = os.path.join(frames_dir, f"{video_id}_frames.npy")

            if not os.path.exists(frames_file):
                features_list.append(
                    np.zeros((target_length, feat_dim), dtype=np.float32)
                )
                continue

            frames = np.load(frames_file)

            if frames.shape[0] == 0:
                features_list.append(
                    np.zeros((target_length, feat_dim), dtype=np.float32)
                )
                continue

            if is_flow:
                if frames.shape[0] < 2:
                    features_list.append(
                        np.zeros((target_length, feat_dim), dtype=np.float32)
                    )
                    continue

                flow_sequence = calculate_optical_flow(frames)
                if flow_sequence.shape[0] == 0:
                    features_list.append(
                        np.zeros((target_length, feat_dim), dtype=np.float32)
                    )
                    continue

                batch_inputs = []
                for flow_field in flow_sequence:
                    flow_img = normalize_flow_for_cnn(flow_field)
                    processed_flow = self.preprocess(flow_img)
                    batch_inputs.append(processed_flow)
            else:
                batch_inputs = []
                for frame in frames:
                    processed_frame = self.preprocess(frame)
                    batch_inputs.append(processed_frame)

            if batch_inputs:
                batch_tensor = torch.stack(batch_inputs).to(self.device)
                with torch.no_grad():
                    features = self.feature_extractor(batch_tensor).cpu().numpy()

                if features.shape[0] < target_length:
                    padding = np.zeros(
                        (target_length - features.shape[0], feat_dim),
                        dtype=np.float32,
                    )
                    features = np.vstack([features, padding])
                elif features.shape[0] > target_length:
                    features = features[:target_length]

                features_list.append(features)
            else:
                features_list.append(
                    np.zeros((target_length, feat_dim), dtype=np.float32)
                )

        features_array = np.array(features_list, dtype=np.float32)
        np.save(output_file, features_array)
        print(f"Saved features to {output_file}, shape: {features_array.shape}")
        return features_array


def prepare_cv_data(config):
    """Prepare data splits for cross-validation"""
    data_base_path = os.path.expanduser("./data-nexar/")
    df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df["id"] = df["id"].astype(str).str.zfill(5)

    if config.DATASET_PERCENTAGE < 1.0:
        df = df.sample(
            frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

    cv_data, final_test_data = train_test_split(
        df,
        test_size=config.FINAL_TEST_SIZE,
        stratify=df["target"],
        random_state=config.RANDOM_STATE,
    )

    cv_data = cv_data.reset_index(drop=True)
    final_test_data = final_test_data.reset_index(drop=True)

    print(f"CV data: {len(cv_data)} samples")
    print(f"Final test data: {len(final_test_data)} samples")

    return cv_data, final_test_data


def load_or_extract_multimodal_features(
    config, df_split, split_name, is_final_test=False
):
    """Load features if they exist, otherwise extract them"""

    combined_features_file = os.path.join(
        config.RESULTS_DIR,
        f"X_{split_name}_combined_features_{config.percentage_str}pct.npy",
    )

    if os.path.exists(combined_features_file):
        print(f"Loading existing combined features from {combined_features_file}")
        return np.load(combined_features_file), df_split["target"].values
    else:
        print(f"Extracting {split_name} multimodal features...")
        X_combined, y = extract_multimodal_features(
            config, df_split, split_name, is_final_test
        )
        np.save(combined_features_file, X_combined)
        print(f"Saved combined features to {combined_features_file}")
        return X_combined, y


def extract_multimodal_features(config, df_split, split_name, is_final_test=False):
    """Extract features for a data split"""
    train_dir = os.path.expanduser("./data-nexar/train/")

    flow_features_file = os.path.join(
        config.FLOW_FEATURES_DIR,
        f"X_{split_name}_flow_sequences_{config.percentage_str}pct_31frames.npy",
    )
    rgb_features_file = os.path.join(
        config.TEMPORAL_RGB_FEATURES_DIR,
        f"X_{split_name}_crash_rgb_{config.percentage_str}pct_32frames.npy",
    )

    extractor = MultimodalFeatureExtractor(config)

    print(f"Extracting {split_name} frames...")

    extractor.extract_and_save_frames(
        df_split["id"],
        train_dir,
        df_split["time_of_event"].values,
        "uniform_flow",
        config.UNIFORM_FRAMES_DIR,
        is_final_test=is_final_test,
    )

    if is_final_test:
        # For final test, use uniform frames for RGB too
        extractor.extract_and_save_frames(
            df_split["id"],
            train_dir,
            df_split["time_of_event"].values,
            "crash_rgb",
            config.UNIFORM_FRAMES_DIR,
            is_final_test=True,
        )
        X_rgb = extractor.process_branch_to_features(
            df_split["id"], config.UNIFORM_FRAMES_DIR, rgb_features_file, is_flow=False
        )
    else:
        extractor.extract_and_save_frames(
            df_split["id"],
            train_dir,
            df_split["time_of_event"].values,
            "crash_rgb",
            config.TEMPORAL_FRAMES_DIR,
            is_final_test=False,
        )
        X_rgb = extractor.process_branch_to_features(
            df_split["id"], config.TEMPORAL_FRAMES_DIR, rgb_features_file, is_flow=False
        )

    print(f"Processing {split_name} features...")
    X_flow = extractor.process_branch_to_features(
        df_split["id"], config.UNIFORM_FRAMES_DIR, flow_features_file, is_flow=True
    )

    flow_padded = np.pad(
        X_flow, ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0
    )
    X_combined = np.concatenate([X_rgb, flow_padded], axis=2)

    print(f"{split_name} combined features shape: {X_combined.shape}")

    return X_combined, df_split["target"].values


def check_existing_models(config):
    """Check if all fold models already exist"""
    existing_models = []
    for fold in range(1, config.CV_FOLDS + 1):
        model_path = os.path.join(config.WEIGHTS_DIR, f"best_model_fold_{fold}.pth")
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
                        "final_train_loss": 0.0,
                        "final_val_loss": 0.0,
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

    if len(existing_models) == config.CV_FOLDS:
        print(f"Found all {config.CV_FOLDS} trained models. Skipping training.")
        return existing_models
    else:
        return None


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
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

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(val_loader)
    auc_score = roc_auc_score(all_labels, all_predictions)

    return avg_loss, auc_score, all_predictions


def run_cross_validation(config, X_cv, y_cv):
    """Run 5-fold cross-validation"""

    existing_models = check_existing_models(config)
    if existing_models is not None:
        return existing_models

    cv_results = []
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"\n=== FOLD {fold + 1}/{config.CV_FOLDS} ===")

        model_path = os.path.join(config.WEIGHTS_DIR, f"best_model_fold_{fold + 1}.pth")
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
                        "final_train_loss": 0.0,
                        "final_val_loss": 0.0,
                        "model_path": model_path,
                        "scaler": checkpoint.get("scaler", None),
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

        num_train_videos, num_frames_per_video, num_features = X_train_fold.shape
        X_train_reshaped = X_train_fold.reshape(-1, num_features)

        scaler = StandardScaler()
        X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_reshaped.reshape(
            num_train_videos, num_frames_per_video, num_features
        )

        num_val_videos = X_val_fold.shape[0]
        X_val_reshaped = X_val_fold.reshape(-1, num_features)
        X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled_reshaped.reshape(
            num_val_videos, num_frames_per_video, num_features
        )

        train_dataset = VideoSequenceDataset(X_train_scaled, y_train_fold)
        val_dataset = VideoSequenceDataset(X_val_scaled, y_val_fold)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        model = HierarchicalTransformer(
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            d_ff=config.D_FF,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
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
            anneal_strategy="cos",
        )

        best_auc = 0.0

        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )

            val_loss, val_auc, _ = validate(model, val_loader, criterion, device)

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "val_auc": val_auc,
                        "scaler": scaler,
                    },
                    model_path,
                )

            if (epoch + 1) % config.PRINT_FREQUENCY == 0:
                print(
                    f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"Best AUC: {best_auc:.4f}"
                )

        cv_results.append(
            {
                "fold": fold + 1,
                "best_val_auc": best_auc,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "model_path": model_path,
                "scaler": scaler,
            }
        )

        print(f"Fold {fold + 1} completed. Best AUC: {best_auc:.4f}")

    return cv_results


def evaluate_final_test(config, cv_results, X_final_test, y_final_test):
    """Evaluate on final test set using ensemble of CV models"""

    if X_final_test is None:
        print("Loading final test features for evaluation...")
        cv_data, final_test_data = prepare_cv_data(config)
        X_final_test, y_final_test = load_or_extract_multimodal_features(
            config, final_test_data, "final_test", is_final_test=True
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    all_predictions = []

    for fold_result in cv_results:
        model = HierarchicalTransformer(
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            d_ff=config.D_FF,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
        ).to(device)

        checkpoint = torch.load(
            fold_result["model_path"], map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = fold_result["scaler"]

        num_test_videos, num_frames_per_video, num_features = X_final_test.shape
        X_test_reshaped = X_final_test.reshape(-1, num_features)
        X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(
            num_test_videos, num_frames_per_video, num_features
        )

        test_dataset = VideoSequenceDataset(X_test_scaled, y_final_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

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


def save_results(config, cv_results, final_auc, final_predictions):
    """Save all results to files"""

    cv_df = pd.DataFrame(
        [
            {
                "fold": result["fold"],
                "best_val_auc": result["best_val_auc"],
                "final_train_loss": result.get("final_train_loss", 0.0),
                "final_val_loss": result.get("final_val_loss", 0.0),
            }
            for result in cv_results
        ]
    )

    cv_results_file = os.path.join(config.RESULTS_DIR, "cv_results.csv")
    cv_df.to_csv(cv_results_file, index=False)

    summary_file = os.path.join(config.RESULTS_DIR, "training_summary.txt")
    with open(summary_file, "w") as f:
        f.write("5-FOLD CROSS-VALIDATION RESULTS\n")
        f.write("=" * 30 + "\n\n")

        f.write(
            f"Dataset: {config.DATASET_PERCENTAGE * 100}% data, {config.CV_FOLDS} folds\n"
        )
        f.write(f"Backbone: {config.BACKBONE}\n")

        fold_aucs = [result["best_val_auc"] for result in cv_results]
        for i, result in enumerate(cv_results):
            f.write(f"Fold {result['fold']}: AUC = {result['best_val_auc']:.4f}\n")

        f.write(f"\nMean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n")
        f.write(f"Final Test AUC: {final_auc:.4f}\n")

    print(f"Results saved to: {summary_file}")
    print(f"\n=== FINAL RESULTS ===")
    print(f"CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Test AUC: {final_auc:.4f}")


def main():
    config = Config()

    print("Preparing cross-validation data...")
    cv_data, final_test_data = prepare_cv_data(config)

    existing_models = check_existing_models(config)
    if existing_models is not None:
        print("Using existing trained models...")
        cv_results = existing_models
        X_cv, y_cv = None, None
        X_final_test, y_final_test = None, None
    else:
        print("Loading/extracting CV features...")
        X_cv, y_cv = load_or_extract_multimodal_features(
            config, cv_data, "cv", is_final_test=False
        )

        print("Loading/extracting final test features...")
        X_final_test, y_final_test = load_or_extract_multimodal_features(
            config, final_test_data, "final_test", is_final_test=True
        )

        print("Running cross-validation...")
        cv_results = run_cross_validation(config, X_cv, y_cv)

    print("Evaluating on final test set...")
    final_auc, final_predictions = evaluate_final_test(
        config, cv_results, X_final_test, y_final_test
    )

    print("Saving results...")
    save_results(config, cv_results, final_auc, final_predictions)


if __name__ == "__main__":
    main()
