import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import cv2
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

from hierarchical_transformer import HierarchicalTransformer
from data_loader import create_data_loaders


# Configuration
class Config:
    # Data paths
    FEATURES_DIR = "features/temporal/"
    DATASET_PERCENTAGE = 1.0

    # Temporal sampling parameters
    NUM_FRAMES = 16
    FRAME_SIZE = (299, 299)
    CRASH_FOCUS_RATIO = 0.7  # 70% of frames around crash time
    TIME_WINDOW_SECONDS = 5.0  # Focus window around crash time

    # Model architecture
    INPUT_DIM = 2048
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 2
    D_FF = 512
    MAX_SEQ_LEN = 16
    DROPOUT = 0.3

    # Training parameters
    BATCH_SIZE = 32
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

    # Output paths
    @property
    def MODEL_SAVE_PATH(self):
        return f"best_temporal_transformer_{int(self.DATASET_PERCENTAGE * 100)}pct.pth"

    @property
    def PLOT_FILENAME(self):
        return f"training_curves_temporal_{int(self.DATASET_PERCENTAGE * 100)}pct.png"

    @property
    def SUBMISSION_FILENAME(self):
        return f"submission_temporal_{int(self.DATASET_PERCENTAGE * 100)}pct.csv"


def extract_temporal_frames(
    video_path,
    crash_time,
    num_frames=16,
    crash_focus_ratio=0.7,
    time_window=5.0,
    frame_size=(299, 299),
):
    """
    Extract frames with higher density around crash time.

    Args:
        video_path: Path to video file
        crash_time: Time of crash event in seconds (can be NaN)
        num_frames: Total number of frames to extract
        crash_focus_ratio: Fraction of frames to concentrate around crash time
        time_window: Time window (seconds) around crash to focus on
        frame_size: Output frame size
    """
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_video_frames / fps if fps > 0 else 10.0

    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *frame_size, 3), dtype=np.uint8)

    # Handle NaN crash times - fall back to uniform sampling
    if pd.isna(crash_time) or crash_time < 0:
        # Use uniform sampling when crash time is unknown
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
    for pos in frame_positions[:num_frames]:  # Ensure we don't exceed num_frames
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


def load_temporal_features():
    """Load or generate temporal-aware features."""
    config = Config()

    # Create features directory
    os.makedirs(config.FEATURES_DIR, exist_ok=True)

    percentage_str = str(int(config.DATASET_PERCENTAGE * 100))
    train_features_file = os.path.join(
        config.FEATURES_DIR, f"X_train_temporal_{percentage_str}pct.npy"
    )
    test_features_file = os.path.join(config.FEATURES_DIR, "X_test_temporal.npy")

    # Load data
    data_base_path = os.path.expanduser("./data/nexar-collision-prediction/")
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

    if os.path.exists(train_features_file) and os.path.exists(test_features_file):
        print(f"Loading temporal features from {train_features_file}")
        X_train_sequences = np.load(train_features_file)
        X_test_sequences = np.load(test_features_file)
        y_train = df["target"].values
        return X_train_sequences, X_test_sequences, y_train

    # Set up feature extraction
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    base_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    base_model = base_model.to(device)
    base_model.eval()

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(config.FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_features_temporal(video_ids, video_folder, crash_times=None):
        """Extract features using temporal-aware sampling."""
        features_list = []

        for i, video_id in enumerate(
            tqdm(video_ids, desc=f"Extracting temporal features from {video_folder}")
        ):
            video_path = os.path.join(video_folder, f"{video_id}.mp4")

            # Get crash time for this video
            crash_time = (
                crash_times[i] if crash_times is not None else 5.0
            )  # Default for test

            try:
                frames_np = extract_temporal_frames(
                    video_path,
                    crash_time,
                    num_frames=config.NUM_FRAMES,
                    crash_focus_ratio=config.CRASH_FOCUS_RATIO,
                    time_window=config.TIME_WINDOW_SECONDS,
                    frame_size=config.FRAME_SIZE,
                )

                if frames_np.shape[0] == 0:
                    features_list.append(
                        np.zeros(
                            (config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32
                        )
                    )
                    continue

                # Process frames through CNN
                batch_frames = []
                for frame_idx in range(frames_np.shape[0]):
                    processed_frame = preprocess(frames_np[frame_idx])
                    batch_frames.append(processed_frame)

                batch_tensor = torch.stack(batch_frames).to(device)

                with torch.no_grad():
                    frame_features = base_model(batch_tensor)

                features_list.append(frame_features.cpu().numpy())

            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                features_list.append(
                    np.zeros((config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32)
                )

        return np.array(features_list, dtype=np.float32)

    # Extract training features
    print("Extracting temporal-aware training features...")
    X_train_sequences = extract_features_temporal(
        df["id"], train_dir, df["time_of_event"].values
    )
    np.save(train_features_file, X_train_sequences)

    # Extract test features
    print("Extracting temporal-aware test features...")
    X_test_sequences = extract_features_temporal(df_test["id"], test_dir)
    np.save(test_features_file, X_test_sequences)

    y_train = df["target"].values

    return X_train_sequences, X_test_sequences, y_train


class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("Temporal-Aware Transformer Training", fontsize=16)

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

    # Load temporal-aware features
    print("Loading temporal-aware data...")
    X_train, X_test, y_train = load_temporal_features()

    # Scale features
    from sklearn.preprocessing import StandardScaler

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
    print("Initializing temporal-aware transformer...")
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
    print("Starting temporal-aware transformer training...")
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
    data_base_path = os.path.expanduser("./data/nexar-collision-prediction/")
    df_test = pd.read_csv(os.path.join(data_base_path, "test.csv"))
    df_test["id"] = df_test["id"].astype(str).str.zfill(5)

    submission = pd.DataFrame({"id": df_test["id"], "score": test_predictions})
    submission.to_csv(config.SUBMISSION_FILENAME, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILENAME}")

    print(f"\nTEMPORAL-AWARE TRANSFORMER SUMMARY")
    print(f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train)} samples)")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
