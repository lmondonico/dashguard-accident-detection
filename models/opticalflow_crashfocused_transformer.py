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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

current_dir = os.path.dirname(os.path.abspath(__file__))
codebase_dir = os.path.join(current_dir, "..")
sys.path.append(codebase_dir)

from module_hierarchical_transformer import HierarchicalTransformer
from utils.data_loader import create_data_loaders


# Configuration
class Config:
    # Data paths
    DATASET_PERCENTAGE = 1.0

    # Frame extraction parameters
    NUM_FRAMES = 32
    FRAME_SIZE = (299, 299)

    # Temporal sampling parameters (for RGB branch)
    CRASH_FOCUS_RATIO = 0.7  # 70% of frames around crash time
    TIME_WINDOW_SECONDS = 5.0  # Focus window around crash time

    # Feature dimensions
    RGB_FEAT_DIM = 2048
    FLOW_FEAT_DIM = 2048
    COMBINED_FEAT_DIM = RGB_FEAT_DIM + FLOW_FEAT_DIM  # 4096

    # Model architecture
    INPUT_DIM = COMBINED_FEAT_DIM
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 2
    D_FF = 512
    MAX_SEQ_LEN = 32
    DROPOUT = 0.3

    # Training parameters
    BATCH_SIZE = 16  # Reduced due to larger feature dimension
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # OneCycleLR parameters
    MAX_LR = 1e-6
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e3

    # Logging
    PRINT_FREQUENCY = 10

    # Directory structure
    RESULTS_DIR = "results"
    WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")

    # Intermediate directories (post-video-split, pre-V3)
    INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
    UNIFORM_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "uniform_flow_frames")
    TEMPORAL_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "crash_rgb_frames")

    # Feature directories (post-V3)
    FLOW_FEATURES_DIR = "features/optical_flow"
    TEMPORAL_RGB_FEATURES_DIR = "features/crash_rgb"

    def __init__(self):
        # Create directories
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

    # Output paths
    @property
    def percentage_str(self):
        return str(int(self.DATASET_PERCENTAGE * 100))

    # Flow feature files (post-V3)
    @property
    def TRAIN_FLOW_FEATURES_FILE(self):
        return os.path.join(
            self.FLOW_FEATURES_DIR,
            f"X_train_flow_sequences_{self.percentage_str}pct_31frames.npy",
        )

    @property
    def TEST_FLOW_FEATURES_FILE(self):
        return os.path.join(
            self.FLOW_FEATURES_DIR, "X_test_flow_sequences_31frames.npy"
        )

    # Temporal RGB feature files (post-V3)
    @property
    def TRAIN_RGB_FEATURES_FILE(self):
        return os.path.join(
            self.TEMPORAL_RGB_FEATURES_DIR,
            f"X_train_crash_rgb_{self.percentage_str}pct_32frames.npy",
        )

    @property
    def TEST_RGB_FEATURES_FILE(self):
        return os.path.join(
            self.TEMPORAL_RGB_FEATURES_DIR, "X_test_crash_rgb_32frames.npy"
        )

    @property
    def MODEL_SAVE_PATH(self):
        return os.path.join(
            self.WEIGHTS_DIR,
            f"best_multimodal_transformer_{self.percentage_str}pct.pth",
        )

    @property
    def PLOT_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR, f"training_curves_multimodal_{self.percentage_str}pct.png"
        )

    @property
    def SUBMISSION_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR, f"submission_multimodal_{self.percentage_str}pct.csv"
        )


def extract_uniform_frames(video_path, num_frames=32, frame_size=(299, 299)):
    """Extract uniformly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *frame_size, 3), dtype=np.uint8)

    # Uniform sampling
    frame_positions = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)

    frames_list = []
    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Pad if necessary
    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1])  # Repeat last frame
        else:
            frames_list.append(np.zeros((*frame_size, 3), dtype=np.uint8))

    return np.stack(frames_list[:num_frames])


def extract_crash_frames(
    video_path,
    crash_time,
    num_frames=32,
    crash_focus_ratio=0.7,
    time_window=5.0,
    frame_size=(299, 299),
):
    """Extract frames with higher density around crash time."""
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_video_frames / fps if fps > 0 else 10.0

    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *frame_size, 3), dtype=np.uint8)

    # Handle NaN crash times - fall back to uniform sampling
    if pd.isna(crash_time) or crash_time < 0:
        frame_positions = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)
    else:
        # Calculate frame positions with crash time focus
        crash_frame = int(crash_time * fps)
        crash_frame = max(0, min(crash_frame, total_video_frames - 1))

        # Frames around crash time
        crash_frames_count = int(num_frames * crash_focus_ratio)
        context_frames_count = num_frames - crash_frames_count

        # Window around crash time
        window_frames = int(time_window * fps)
        start_crash_window = max(0, crash_frame - window_frames // 2)
        end_crash_window = min(total_video_frames, crash_frame + window_frames // 2)

        # Generate frame positions
        frame_positions = []

        # High-density frames around crash
        if end_crash_window > start_crash_window:
            crash_positions = np.linspace(
                start_crash_window, end_crash_window, crash_frames_count, dtype=int
            )
            frame_positions.extend(crash_positions)

        # Context frames from rest of video
        if context_frames_count > 0:
            # Before crash window
            before_frames = context_frames_count // 2
            if start_crash_window > 0:
                before_positions = np.linspace(
                    0, start_crash_window, before_frames + 1, dtype=int
                )[:-1]
                frame_positions.extend(before_positions)

            # After crash window
            after_frames = context_frames_count - before_frames
            if end_crash_window < total_video_frames:
                after_positions = np.linspace(
                    end_crash_window,
                    total_video_frames - 1,
                    after_frames + 1,
                    dtype=int,
                )[1:]
                frame_positions.extend(after_positions)

        # Sort positions and ensure uniqueness
        frame_positions = sorted(list(set(frame_positions)))

    # Extract frames
    frames_list = []
    for pos in frame_positions[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Pad if necessary
    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1])
        else:
            frames_list.append(np.zeros((*frame_size, 3), dtype=np.uint8))

    return np.stack(frames_list[:num_frames])


def calculate_optical_flow(frames_sequence):
    """Calculate dense optical flow between consecutive frames."""
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
    """Normalize optical flow for CNN input."""
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

        # Load InceptionV3 model
        self.feature_extractor = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Preprocessing
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
        self, video_ids, video_folder, crash_times, branch_name, frames_dir
    ):
        """Extract and save frames for a specific branch."""
        for i, video_id in enumerate(
            tqdm(video_ids, desc=f"Extracting {branch_name} frames")
        ):
            frames_file = os.path.join(frames_dir, f"{video_id}_frames.npy")

            if os.path.exists(frames_file):
                continue  # Skip if already exists

            video_path = os.path.join(video_folder, f"{video_id}.mp4")

            if branch_name == "uniform_flow":
                frames = extract_uniform_frames(
                    video_path, self.config.NUM_FRAMES, self.config.FRAME_SIZE
                )
            elif branch_name == "crash_rgb":
                crash_time = crash_times[i] if crash_times is not None else None
                frames = extract_crash_frames(
                    video_path,
                    crash_time,
                    self.config.NUM_FRAMES,
                    self.config.CRASH_FOCUS_RATIO,
                    self.config.TIME_WINDOW_SECONDS,
                    self.config.FRAME_SIZE,
                )

            np.save(frames_file, frames)

    def process_flow_branch_to_features(self, video_ids, frames_dir, output_file):
        """Process optical flow branch and save features."""
        if os.path.exists(output_file):
            print(f"Loading existing flow features from {output_file}")
            return np.load(output_file)

        print(f"Processing optical flow features...")
        flow_features_list = []

        for video_id in tqdm(video_ids, desc="Processing optical flow"):
            frames_file = os.path.join(frames_dir, f"{video_id}_frames.npy")

            if not os.path.exists(frames_file):
                # Create zero features if frames not found
                flow_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES - 1, self.config.FLOW_FEAT_DIM),
                        dtype=np.float32,
                    )
                )
                continue

            frames = np.load(frames_file)

            if frames.shape[0] < 2:
                flow_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES - 1, self.config.FLOW_FEAT_DIM),
                        dtype=np.float32,
                    )
                )
                continue

            # Calculate optical flow
            flow_sequence = calculate_optical_flow(frames)

            if flow_sequence.shape[0] == 0:
                flow_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES - 1, self.config.FLOW_FEAT_DIM),
                        dtype=np.float32,
                    )
                )
                continue

            # Extract CNN features from flow
            batch_inputs = []
            for flow_field in flow_sequence:
                flow_img = normalize_flow_for_cnn(flow_field)
                processed_flow = self.preprocess(flow_img)
                batch_inputs.append(processed_flow)

            if batch_inputs:
                batch_tensor = torch.stack(batch_inputs).to(self.device)
                with torch.no_grad():
                    features = self.feature_extractor(batch_tensor).cpu().numpy()

                # Pad or truncate to match expected length (31 frames for flow)
                target_length = self.config.NUM_FRAMES - 1
                if features.shape[0] < target_length:
                    padding = np.zeros(
                        (target_length - features.shape[0], self.config.FLOW_FEAT_DIM),
                        dtype=np.float32,
                    )
                    features = np.vstack([features, padding])
                elif features.shape[0] > target_length:
                    features = features[:target_length]

                flow_features_list.append(features)
            else:
                flow_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES - 1, self.config.FLOW_FEAT_DIM),
                        dtype=np.float32,
                    )
                )

        flow_features = np.array(flow_features_list, dtype=np.float32)
        np.save(output_file, flow_features)
        print(f"Saved flow features to {output_file}, shape: {flow_features.shape}")
        return flow_features

    def process_rgb_branch_to_features(self, video_ids, frames_dir, output_file):
        """Process RGB branch and save features."""
        if os.path.exists(output_file):
            print(f"Loading existing RGB features from {output_file}")
            return np.load(output_file)

        print(f"Processing crash RGB features...")
        rgb_features_list = []

        for video_id in tqdm(video_ids, desc="Processing crash RGB"):
            frames_file = os.path.join(frames_dir, f"{video_id}_frames.npy")

            if not os.path.exists(frames_file):
                rgb_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES, self.config.RGB_FEAT_DIM),
                        dtype=np.float32,
                    )
                )
                continue

            frames = np.load(frames_file)

            if frames.shape[0] == 0:
                rgb_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES, self.config.RGB_FEAT_DIM),
                        dtype=np.float32,
                    )
                )
                continue

            # Extract CNN features from RGB frames
            batch_inputs = []
            for frame in frames:
                processed_frame = self.preprocess(frame)
                batch_inputs.append(processed_frame)

            if batch_inputs:
                batch_tensor = torch.stack(batch_inputs).to(self.device)
                with torch.no_grad():
                    features = self.feature_extractor(batch_tensor).cpu().numpy()

                # Pad or truncate to match expected length (32 frames for RGB)
                if features.shape[0] < self.config.NUM_FRAMES:
                    padding = np.zeros(
                        (
                            self.config.NUM_FRAMES - features.shape[0],
                            self.config.RGB_FEAT_DIM,
                        ),
                        dtype=np.float32,
                    )
                    features = np.vstack([features, padding])
                elif features.shape[0] > self.config.NUM_FRAMES:
                    features = features[: self.config.NUM_FRAMES]

                rgb_features_list.append(features)
            else:
                rgb_features_list.append(
                    np.zeros(
                        (self.config.NUM_FRAMES, self.config.RGB_FEAT_DIM),
                        dtype=np.float32,
                    )
                )

        rgb_features = np.array(rgb_features_list, dtype=np.float32)
        np.save(output_file, rgb_features)
        print(f"Saved RGB features to {output_file}, shape: {rgb_features.shape}")
        return rgb_features


def load_multimodal_features(config):
    """Load or generate multimodal features."""
    # Check if both branch features exist
    train_flow_exists = os.path.exists(config.TRAIN_FLOW_FEATURES_FILE)
    test_flow_exists = os.path.exists(config.TEST_FLOW_FEATURES_FILE)
    train_rgb_exists = os.path.exists(config.TRAIN_RGB_FEATURES_FILE)
    test_rgb_exists = os.path.exists(config.TEST_RGB_FEATURES_FILE)

    if train_flow_exists and test_flow_exists and train_rgb_exists and test_rgb_exists:
        print("Loading pre-computed branch features...")
        X_train_flow = np.load(config.TRAIN_FLOW_FEATURES_FILE)
        X_test_flow = np.load(config.TEST_FLOW_FEATURES_FILE)
        X_train_rgb = np.load(config.TRAIN_RGB_FEATURES_FILE)
        X_test_rgb = np.load(config.TEST_RGB_FEATURES_FILE)

        # Load labels
        data_base_path = os.path.expanduser("./data-nexar/")
        df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
        df["id"] = df["id"].astype(str).str.zfill(5)

        if config.DATASET_PERCENTAGE < 1.0:
            df = df.sample(
                frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)

        y_train = df["target"].values

        # Combine features in memory (don't save combined)
        train_flow_padded = np.pad(
            X_train_flow, ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0
        )
        test_flow_padded = np.pad(
            X_test_flow, ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0
        )
        X_train_combined = np.concatenate([X_train_rgb, train_flow_padded], axis=2)
        X_test_combined = np.concatenate([X_test_rgb, test_flow_padded], axis=2)

        return X_train_combined, X_test_combined, y_train

    print("Generating branch-specific features...")

    # Load data
    data_base_path = os.path.expanduser("./data-nexar/")
    df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_base_path, "test.csv"))
    df["id"] = df["id"].astype(str).str.zfill(5)
    df_test["id"] = df_test["id"].astype(str).str.zfill(5)

    if config.DATASET_PERCENTAGE < 1.0:
        df = df.sample(
            frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

    train_dir = os.path.join(data_base_path, "train/")
    test_dir = os.path.join(data_base_path, "test/")

    # Initialize feature extractor
    extractor = MultimodalFeatureExtractor(config)

    # Extract and save frames for both branches (intermediate post-split files)
    print("Extracting training frames...")
    extractor.extract_and_save_frames(
        df["id"],
        train_dir,
        df["time_of_event"].values,
        "uniform_flow",
        config.UNIFORM_FRAMES_DIR,
    )
    extractor.extract_and_save_frames(
        df["id"],
        train_dir,
        df["time_of_event"].values,
        "crash_rgb",
        config.TEMPORAL_FRAMES_DIR,
    )

    print("Extracting test frames...")
    extractor.extract_and_save_frames(
        df_test["id"], test_dir, None, "uniform_flow", config.UNIFORM_FRAMES_DIR
    )
    extractor.extract_and_save_frames(
        df_test["id"], test_dir, None, "crash_rgb", config.TEMPORAL_FRAMES_DIR
    )

    # Process both branches to features (post-V3)
    print("Processing training features...")
    X_train_flow = extractor.process_flow_branch_to_features(
        df["id"], config.UNIFORM_FRAMES_DIR, config.TRAIN_FLOW_FEATURES_FILE
    )
    X_train_rgb = extractor.process_rgb_branch_to_features(
        df["id"], config.TEMPORAL_FRAMES_DIR, config.TRAIN_RGB_FEATURES_FILE
    )

    print("Processing test features...")
    X_test_flow = extractor.process_flow_branch_to_features(
        df_test["id"], config.UNIFORM_FRAMES_DIR, config.TEST_FLOW_FEATURES_FILE
    )
    X_test_rgb = extractor.process_rgb_branch_to_features(
        df_test["id"], config.TEMPORAL_FRAMES_DIR, config.TEST_RGB_FEATURES_FILE
    )

    # Combine features in memory (don't save combined)
    print("Combining features in memory...")
    train_flow_padded = np.pad(
        X_train_flow, ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0
    )
    test_flow_padded = np.pad(
        X_test_flow, ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0
    )
    X_train_combined = np.concatenate([X_train_rgb, train_flow_padded], axis=2)
    X_test_combined = np.concatenate([X_test_rgb, test_flow_padded], axis=2)

    print(f"Combined training features shape: {X_train_combined.shape}")
    print(f"Combined test features shape: {X_test_combined.shape}")

    y_train = df["target"].values
    return X_train_combined, X_test_combined, y_train


class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("Multimodal Transformer Training", fontsize=16)

        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.learning_rates = []

        # Set up axes
        self.axes[0].set_title("Loss Curves")
        self.axes[0].set_xlabel("Epoch")
        self.axes[0].set_ylabel("Loss")

        self.axes[1].set_title("Validation AUC")
        self.axes[1].set_xlabel("Epoch")
        self.axes[1].set_ylabel("AUC")

        self.axes[2].set_title("Learning Rate")
        self.axes[2].set_xlabel("Epoch")
        self.axes[2].set_ylabel("Learning Rate")

        # Initialize lines
        (self.train_loss_line,) = self.axes[0].plot([], [], "b-", label="Train")
        (self.val_loss_line,) = self.axes[0].plot([], [], "r-", label="Val")
        self.axes[0].legend(frameon=False)

        (self.val_auc_line,) = self.axes[1].plot([], [], "g-", label="Val AUC")
        self.axes[1].legend(frameon=False)

        (self.lr_line,) = self.axes[2].plot([], [], "orange", label="LR")
        self.axes[2].legend(frameon=False)

        plt.tight_layout()
        plt.show()

    def update(self, train_loss, val_loss, val_auc, learning_rate):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_aucs.append(val_auc)
        self.learning_rates.append(learning_rate)

        epochs = list(range(1, len(self.train_losses) + 1))

        # Update loss curves
        self.train_loss_line.set_data(epochs, self.train_losses)
        self.val_loss_line.set_data(epochs, self.val_losses)
        self.axes[0].relim()
        self.axes[0].autoscale_view()

        # Update AUC curve
        self.val_auc_line.set_data(epochs, self.val_aucs)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

        # Update learning rate curve
        self.lr_line.set_data(epochs, self.learning_rates)
        self.axes[2].relim()
        self.axes[2].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save_final_plot(self, filename):
        plt.ioff()
        self.fig.savefig(filename, dpi=600, bbox_inches="tight")
        print(f"Final training curves saved to {filename}")

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for features, labels in tqdm(train_loader, desc="Training"):
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
        for features, labels in tqdm(val_loader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(val_loader)
    auc_score = roc_auc_score(all_labels, all_predictions)

    return avg_loss, auc_score


def main():
    config = Config()

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load multimodal features
    print("Loading multimodal data...")
    X_train, X_test, y_train = load_multimodal_features(config)

    # Scale features
    print("Scaling features...")
    num_train_videos, num_frames_per_video, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)

    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(
        num_train_videos, num_frames_per_video, num_features
    )

    num_test_videos = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, num_features)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(
        num_test_videos, num_frames_per_video, num_features
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_scaled,
        X_test_scaled,
        y_train,
        test_size=config.TEST_SIZE,
        batch_size=config.BATCH_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # Initialize model
    print("Initializing multimodal transformer...")
    model = HierarchicalTransformer(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR scheduler
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

    # Initialize live plotter
    live_plotter = LivePlotter()

    # Training loop
    print("Starting multimodal transformer training...")
    best_auc = 0.0

    try:
        for epoch in range(config.NUM_EPOCHS):
            # Training
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )

            # Validation
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            # Update live plot
            live_plotter.update(train_loss, val_loss, val_auc, current_lr)

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_auc": float(best_auc),
                        "epoch": int(epoch),
                        "config": {
                            "input_dim": config.INPUT_DIM,
                            "d_model": config.D_MODEL,
                            "num_heads": config.NUM_HEADS,
                            "num_layers": config.NUM_LAYERS,
                            "d_ff": config.D_FF,
                            "max_seq_len": config.MAX_SEQ_LEN,
                            "dropout": config.DROPOUT,
                        },
                    },
                    config.MODEL_SAVE_PATH,
                )
                print(
                    f"Epoch {epoch + 1}: New best model saved with Val AUC: {val_auc:.4f}"
                )

            # Print progress
            if (epoch + 1) % config.PRINT_FREQUENCY == 0:
                print(
                    f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | "
                    f"LR: {current_lr:.1e} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"Best Val AUC: {best_auc:.4f}"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        live_plotter.save_final_plot(config.PLOT_FILENAME)
        live_plotter.close()

    # Load best model for final evaluation
    print(f"Loading best model from {config.MODEL_SAVE_PATH}")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final validation
    _, final_auc = validate(model, val_loader, criterion, device)
    print(f"\nFinal Validation ROC-AUC: {final_auc:.4f}")

    # Test set inference
    print("Generating test predictions...")
    model.eval()
    test_predictions = []

    with torch.no_grad():
        for features in tqdm(test_loader, desc="Test inference"):
            features = features.to(device)
            outputs = model(features)
            test_predictions.extend(outputs.cpu().numpy().flatten())

    # Create submission file
    data_base_path = os.path.expanduser("./data-nexar/")
    df_test = pd.read_csv(os.path.join(data_base_path, "test.csv"))
    df_test["id"] = df_test["id"].astype(str).str.zfill(5)

    submission = pd.DataFrame({"id": df_test["id"], "score": test_predictions})
    submission.to_csv(config.SUBMISSION_FILENAME, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILENAME}")

    # Save training summary
    summary_file = os.path.join(
        config.RESULTS_DIR, f"training_summary_{config.percentage_str}pct.txt"
    )
    with open(summary_file, "w") as f:
        f.write(f"MULTIMODAL TRANSFORMER TRAINING SUMMARY\n")
        f.write(f"========================================\n\n")
        f.write(
            f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train)} samples)\n"
        )
        f.write(
            f"Feature dimensions: RGB={config.RGB_FEAT_DIM}, Flow={config.FLOW_FEAT_DIM}, Combined={config.COMBINED_FEAT_DIM}\n"
        )
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Training epochs: {config.NUM_EPOCHS}\n")
        f.write(f"Best validation AUC: {best_auc:.4f}\n")
        f.write(f"Final validation AUC: {final_auc:.4f}\n")
        f.write(f"Batch size: {config.BATCH_SIZE}\n")
        f.write(f"Learning rate: {config.LEARNING_RATE}\n")
        f.write(f"\nFile structure:\n")
        f.write(f"Intermediate frames (post-split, pre-V3):\n")
        f.write(f"  - Uniform flow frames: {config.UNIFORM_FRAMES_DIR}\n")
        f.write(f"  - Temporal RGB frames: {config.TEMPORAL_FRAMES_DIR}\n")
        f.write(f"Features (post-V3):\n")
        f.write(f"  - Flow features: {config.FLOW_FEATURES_DIR}\n")
        f.write(f"  - RGB features: {config.TEMPORAL_RGB_FEATURES_DIR}\n")
        f.write(f"Model weights: {config.WEIGHTS_DIR}\n")
        f.write(f"Results: {config.RESULTS_DIR}\n")

    print(f"\nMULTIMODAL TRANSFORMER SUMMARY")
    print(f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train)} samples)")
    print(
        f"Feature dimensions: RGB={config.RGB_FEAT_DIM}, Flow={config.FLOW_FEAT_DIM}, Combined={config.COMBINED_FEAT_DIM}"
    )
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
