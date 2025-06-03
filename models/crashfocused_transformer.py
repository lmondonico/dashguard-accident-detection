import os
import sys
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

current_dir = os.path.dirname(os.path.abspath(__file__))
codebase_dir = os.path.join(current_dir, "..")
sys.path.append(codebase_dir)

from module_hierarchical_transformer import HierarchicalTransformer
from utils.data_loader import create_data_loaders


class Config:
    FEATURES_DIR = "features/temporal/"
    DATASET_PERCENTAGE = 1.0

    NUM_FRAMES = 32
    FRAME_SIZE = (299, 299)
    CRASH_FOCUS_RATIO = 0.7
    TIME_WINDOW_SECONDS = 5.0

    INPUT_DIM = 2048
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 2
    D_FF = 512
    MAX_SEQ_LEN = 32
    DROPOUT = 0.3

    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MAX_LR = 1e-6
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e3

    PRINT_FREQUENCY = 10

    RESULTS_DIR = "results/crashfocused_transformer"
    WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")

    INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
    CRASH_RGB_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "crash_rgb_frames")

    def __init__(self):
        for directory in [
            self.RESULTS_DIR,
            self.WEIGHTS_DIR,
            self.FEATURES_DIR,
            self.INTERMEDIATE_FRAMES_DIR,
            self.CRASH_RGB_FRAMES_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    @property
    def percentage_str(self):
        return str(int(self.DATASET_PERCENTAGE * 100))

    @property
    def MODEL_SAVE_PATH(self):
        return os.path.join(
            self.WEIGHTS_DIR,
            f"best_temporal_transformer_{self.percentage_str}pct_N{self.NUM_FRAMES}.pth",
        )

    @property
    def PLOT_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR,
            f"training_curves_temporal_{self.percentage_str}pct_N{self.NUM_FRAMES}.png",
        )

    @property
    def SUBMISSION_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR,
            f"submission_temporal_{self.percentage_str}pct_N{self.NUM_FRAMES}.csv",
        )


def extract_temporal_frames(
    video_path,
    crash_time,
    num_frames,
    crash_focus_ratio,
    time_window,
    frame_size,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return np.array([np.zeros((*frame_size, 3), dtype=np.uint8)] * num_frames)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_video_frames <= 0:
        print(f"Warning: Video {video_path} has no frames or invalid metadata.")
        cap.release()
        return np.array([np.zeros((*frame_size, 3), dtype=np.uint8)] * num_frames)

    if pd.isna(crash_time) or crash_time < 0 or fps == 0 or fps is None:
        frame_positions = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)
    else:
        crash_frame = int(crash_time * fps)
        crash_frame = max(0, min(crash_frame, total_video_frames - 1))

        crash_frames_count = int(num_frames * crash_focus_ratio)
        context_frames_count = num_frames - crash_frames_count

        window_frames = int(time_window * fps)
        start_crash_window = max(0, crash_frame - window_frames // 2)
        end_crash_window = min(total_video_frames - 1, crash_frame + window_frames // 2)

        frame_positions = []

        if end_crash_window >= start_crash_window and crash_frames_count > 0:
            crash_positions = np.linspace(
                start_crash_window,
                end_crash_window,
                crash_frames_count,
                dtype=int,
                endpoint=True,
            )
            frame_positions.extend(crash_positions)

        if context_frames_count > 0:
            remaining_video_indices = list(
                set(range(total_video_frames)) - set(frame_positions)
            )

            context_candidate_indices = []
            if start_crash_window > 0:
                context_candidate_indices.extend(range(0, start_crash_window))
            if end_crash_window < total_video_frames - 1:
                context_candidate_indices.extend(
                    range(end_crash_window + 1, total_video_frames)
                )

            if len(context_candidate_indices) < context_frames_count:
                other_remaining = list(
                    set(remaining_video_indices) - set(context_candidate_indices)
                )
                np.random.shuffle(other_remaining)
                context_candidate_indices.extend(other_remaining)
            else:
                np.random.shuffle(context_candidate_indices)

            selected_context_frames = np.sort(
                np.array(context_candidate_indices[:context_frames_count], dtype=int)
            )
            frame_positions.extend(selected_context_frames)

        frame_positions = sorted(
            list(set(int(p) for p in frame_positions if 0 <= p < total_video_frames))
        )

    if len(frame_positions) < num_frames:
        if (
            len(frame_positions) == 0
        ):  # No valid frames could be sampled, use linspace over whole video
            frame_positions = np.linspace(
                0, total_video_frames - 1, num_frames, dtype=int
            ).tolist()
        else:  # Pad with existing frames or linspace
            additional_frames_needed = num_frames - len(frame_positions)
            # Try to fill by repeating last sampled frames, then resort to linspace if needed
            fill_positions = (
                np.random.choice(frame_positions, additional_frames_needed).tolist()
                if frame_positions
                else []
            )
            if len(fill_positions) < additional_frames_needed:
                fill_positions.extend(
                    np.linspace(
                        0,
                        total_video_frames - 1,
                        additional_frames_needed - len(fill_positions),
                        dtype=int,
                    ).tolist()
                )
            frame_positions.extend(fill_positions)
            frame_positions = sorted(list(set(frame_positions)))

    frame_positions = frame_positions[:num_frames]
    frame_positions = [
        min(max(0, int(p)), total_video_frames - 1) for p in frame_positions
    ]

    frames_list = []
    current_positions_len = len(frame_positions)
    if current_positions_len < num_frames:
        if current_positions_len > 0:
            frame_positions.extend(
                [frame_positions[-1]] * (num_frames - current_positions_len)
            )
        elif total_video_frames > 0:
            frame_positions.extend(
                [total_video_frames - 1] * (num_frames - current_positions_len)
            )
        else:
            frame_positions.extend([0] * (num_frames - current_positions_len))

    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            if frames_list:
                frames_list.append(frames_list[-1].copy())
            else:
                frames_list.append(np.zeros((*frame_size, 3), dtype=np.uint8))

    cap.release()

    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1].copy())
        else:
            frames_list.append(np.zeros((*frame_size, 3), dtype=np.uint8))

    return np.stack(frames_list[:num_frames])


def load_temporal_features():
    config = Config()

    train_cnn_features_file = os.path.join(
        config.FEATURES_DIR,
        f"X_train_temporal_{config.percentage_str}pct_N{config.NUM_FRAMES}.npy",
    )
    test_cnn_features_file = os.path.join(
        config.FEATURES_DIR, f"X_test_temporal_N{config.NUM_FRAMES}.npy"
    )

    data_base_path = os.path.expanduser("./data-nexar/")
    df_train_full = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df_train_full["id"] = df_train_full["id"].astype(str).str.zfill(5)

    if config.DATASET_PERCENTAGE < 1.0:
        df = df_train_full.sample(
            frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)
    else:
        df = df_train_full.copy()
    y_train = df["target"].values

    if os.path.exists(train_cnn_features_file) and os.path.exists(
        test_cnn_features_file
    ):
        print(
            f"Loading pre-computed temporal CNN features from {train_cnn_features_file}"
        )
        X_train_sequences = np.load(train_cnn_features_file)
        X_test_sequences = np.load(test_cnn_features_file)
        return X_train_sequences, X_test_sequences, y_train

    print(
        "CNN features not found. Attempting to generate them using intermediate frames..."
    )
    df_test = pd.read_csv(os.path.join(data_base_path, "test.csv"))
    df_test["id"] = df_test["id"].astype(str).str.zfill(5)
    train_dir = os.path.join(data_base_path, "train/")
    test_dir = os.path.join(data_base_path, "test/")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    cnn_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    cnn_model.fc = nn.Identity()
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    preprocess_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(config.FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_features_from_frames_or_video(
        video_ids, video_folder, crash_rgb_frames_storage_dir, crash_times_series=None
    ):
        cnn_features_list = []

        for i, video_id in enumerate(
            tqdm(
                video_ids,
                desc=f"Processing crash_rgb frames/videos from {video_folder} for CNN features",
            )
        ):
            intermediate_frames_file = os.path.join(
                crash_rgb_frames_storage_dir, f"{video_id}_frames.npy"
            )
            video_path = os.path.join(video_folder, f"{video_id}.mp4")
            frames_np = None
            current_crash_time = (
                crash_times_series.iloc[i]
                if crash_times_series is not None and i < len(crash_times_series)
                else None
            )

            try:
                if os.path.exists(intermediate_frames_file):
                    frames_np = np.load(intermediate_frames_file)
                    if frames_np.shape[0] < config.NUM_FRAMES:
                        print(
                            f"Padding loaded crash_rgb frames for {video_id} from {frames_np.shape[0]} to {config.NUM_FRAMES}"
                        )
                        padding_data = np.zeros(
                            (
                                config.NUM_FRAMES - frames_np.shape[0],
                                *config.FRAME_SIZE,
                                3,
                            ),
                            dtype=frames_np.dtype,
                        )
                        if frames_np.shape[0] > 0:
                            padding_data = np.repeat(
                                frames_np[-1:],
                                config.NUM_FRAMES - frames_np.shape[0],
                                axis=0,
                            )
                        frames_np = (
                            np.vstack((frames_np, padding_data))
                            if frames_np.shape[0] > 0
                            else padding_data
                        )
                    frames_np = frames_np[: config.NUM_FRAMES]
                else:
                    frames_np = extract_temporal_frames(
                        video_path,
                        current_crash_time,
                        num_frames=config.NUM_FRAMES,
                        crash_focus_ratio=config.CRASH_FOCUS_RATIO,
                        time_window=config.TIME_WINDOW_SECONDS,
                        frame_size=config.FRAME_SIZE,
                    )
                    np.save(intermediate_frames_file, frames_np)

                if frames_np.shape[0] == 0 or frames_np.shape[0] < config.NUM_FRAMES:
                    print(
                        f"Warning: Not enough crash_rgb frames for {video_id} ({frames_np.shape[0]}), using zeros."
                    )
                    cnn_features_list.append(
                        np.zeros(
                            (config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32
                        )
                    )
                    continue

                batch_frames_tensors = []
                for frame_idx in range(frames_np.shape[0]):
                    processed_frame = preprocess_transform(frames_np[frame_idx])
                    batch_frames_tensors.append(processed_frame)

                batch_tensor = torch.stack(batch_frames_tensors).to(device)

                with torch.no_grad():
                    output = cnn_model(batch_tensor)
                    frame_cnn_features = (
                        output[0]
                        if isinstance(output, tuple) and not cnn_model.training
                        else output
                    )
                cnn_features_list.append(frame_cnn_features.cpu().numpy())

            except Exception as e:
                print(
                    f"Error processing {video_id} for crash_rgb: {e}. Appending zeros."
                )
                cnn_features_list.append(
                    np.zeros((config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32)
                )

        processed_cnn_features_list = []
        for features_seq in cnn_features_list:
            if features_seq.shape[0] < config.NUM_FRAMES:
                padding_needed = config.NUM_FRAMES - features_seq.shape[0]
                padding_array = np.zeros(
                    (padding_needed, config.INPUT_DIM), dtype=np.float32
                )
                if features_seq.shape[0] > 0:
                    padding_array = np.repeat(features_seq[-1:], padding_needed, axis=0)
                features_seq = (
                    np.vstack((features_seq, padding_array))
                    if features_seq.shape[0] > 0
                    else padding_array
                )
            processed_cnn_features_list.append(features_seq[: config.NUM_FRAMES])

        return np.array(processed_cnn_features_list, dtype=np.float32)

    print("Generating/Loading training temporal (crash_rgb) CNN features...")
    X_train_sequences = extract_features_from_frames_or_video(
        df["id"], train_dir, config.CRASH_RGB_FRAMES_DIR, df["time_of_event"]
    )
    np.save(train_cnn_features_file, X_train_sequences)
    print(f"Saved training temporal CNN features to {train_cnn_features_file}")

    print("Generating/Loading test temporal (crash_rgb) CNN features...")
    X_test_sequences = extract_features_from_frames_or_video(
        df_test["id"],
        test_dir,
        config.CRASH_RGB_FRAMES_DIR,
        crash_times_series=None,  # No crash times for test
    )
    np.save(test_cnn_features_file, X_test_sequences)
    print(f"Saved test temporal CNN features to {test_cnn_features_file}")

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

        self.axes[0].set_title("Loss Curves")
        self.axes[0].set_xlabel("Epoch")
        self.axes[0].set_ylabel("Loss")

        self.axes[1].set_title("Validation AUC")
        self.axes[1].set_xlabel("Epoch")
        self.axes[1].set_ylabel("AUC")

        self.axes[2].set_title("Learning Rate")
        self.axes[2].set_xlabel("Epoch")
        self.axes[2].set_ylabel("Learning Rate")

        (self.train_loss_line,) = self.axes[0].plot([], [], "b-", label="Train")
        (self.val_loss_line,) = self.axes[0].plot([], [], "r-", label="Val")
        self.axes[0].legend(frameon=False)

        (self.val_auc_line,) = self.axes[1].plot([], [], "g-", label="Val AUC")
        self.axes[1].legend(frameon=False)

        (self.lr_line,) = self.axes[2].plot([], [], "orange", label="LR")
        self.axes[2].legend(frameon=False)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, train_loss, val_loss, val_auc, learning_rate):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_aucs.append(val_auc)
        self.learning_rates.append(learning_rate)

        epochs = list(range(1, len(self.train_losses) + 1))

        self.train_loss_line.set_data(epochs, self.train_losses)
        self.val_loss_line.set_data(epochs, self.val_losses)
        self.axes[0].relim()
        self.axes[0].autoscale_view()

        self.val_auc_line.set_data(epochs, self.val_aucs)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

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
        features, labels = features.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
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
            features, labels = (
                features.to(device),
                labels.to(device).float().unsqueeze(1),
            )
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(val_loader)
    try:
        auc_score = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        print(
            "Warning: ROC AUC score could not be calculated (possibly single class in batch). Setting to 0."
        )
        auc_score = 0.0
    return avg_loss, auc_score


def main():
    config = Config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading temporal-aware data (crash-focused)...")
    X_train, X_test, y_train = load_temporal_features()

    from sklearn.preprocessing import StandardScaler

    num_train_videos, num_frames_per_video, num_features_dim = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features_dim)

    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(
        num_train_videos, num_frames_per_video, num_features_dim
    )

    num_test_videos = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, num_features_dim)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(
        num_test_videos, num_frames_per_video, num_features_dim
    )

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_scaled,
        X_test_scaled,
        y_train,
        test_size=config.TEST_SIZE,
        batch_size=config.BATCH_SIZE,
        random_state=config.RANDOM_STATE,
    )

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
        anneal_strategy="cos",
    )

    live_plotter = LivePlotter()

    print("Starting temporal-aware transformer training...")
    best_auc = 0.0

    try:
        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )

            val_loss, val_auc = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            live_plotter.update(train_loss, val_loss, val_auc, current_lr)

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_auc": float(best_auc),
                        "epoch": int(epoch),
                        "config_params": {
                            "input_dim": config.INPUT_DIM,
                            "d_model": config.D_MODEL,
                            "num_heads": config.NUM_HEADS,
                            "num_layers": config.NUM_LAYERS,
                            "d_ff": config.D_FF,
                            "max_seq_len": config.MAX_SEQ_LEN,
                            "dropout": config.DROPOUT,
                            "num_frames": config.NUM_FRAMES,
                        },
                    },
                    config.MODEL_SAVE_PATH,
                )
                print(
                    f"Epoch {epoch + 1}: New best model saved with Val AUC: {val_auc:.4f}"
                )

            if (
                epoch + 1
            ) % config.PRINT_FREQUENCY == 0 or epoch == config.NUM_EPOCHS - 1:
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

    final_auc = 0.0
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading best model from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, final_auc = validate(model, val_loader, criterion, device)
        print(f"\nFinal Validation ROC-AUC: {final_auc:.4f}")

        print("Generating test predictions...")
        model.eval()
        test_predictions = []

        with torch.no_grad():
            for features_test_batch in tqdm(test_loader, desc="Test inference"):
                features_tensor = features_test_batch[0].to(device)
                outputs = model(features_tensor)
                test_predictions.extend(outputs.cpu().numpy().flatten())

        data_base_path = os.path.expanduser("./data-nexar/")
        df_test_info = pd.read_csv(os.path.join(data_base_path, "test.csv"))
        df_test_info["id"] = df_test_info["id"].astype(str).str.zfill(5)

        submission = pd.DataFrame({"id": df_test_info["id"], "score": test_predictions})
        submission.to_csv(config.SUBMISSION_FILENAME, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILENAME}")
    else:
        print(
            f"No best model found at {config.MODEL_SAVE_PATH}. Skipping final evaluation and submission."
        )

    summary_file = os.path.join(
        config.RESULTS_DIR,
        f"training_summary_temporal_{config.percentage_str}pct_N{config.NUM_FRAMES}.txt",
    )
    with open(summary_file, "w") as f:
        f.write(f"CRASH-FOCUSED (TEMPORAL) TRANSFORMER TRAINING SUMMARY\n")
        f.write(f"=====================================================\n\n")
        f.write(
            f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train) if 'X_train' in locals() else 'N/A'} samples)\n"
        )
        f.write(f"Input Dimension (CNN Features): {config.INPUT_DIM}\n")
        f.write(f"Transformer d_model: {config.D_MODEL}\n")
        f.write(f"Transformer num_layers: {config.NUM_LAYERS}\n")
        f.write(f"Transformer num_heads: {config.NUM_HEADS}\n")
        f.write(f"Number of frames per video: {config.NUM_FRAMES}\n")
        f.write(f"Crash focus ratio: {config.CRASH_FOCUS_RATIO}\n")
        f.write(f"Time window (seconds): {config.TIME_WINDOW_SECONDS}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Training epochs: {config.NUM_EPOCHS}\n")
        f.write(f"Best validation AUC: {best_auc:.4f}\n")
        f.write(f"Final validation AUC: {final_auc:.4f}\n")
        f.write(f"Batch size: {config.BATCH_SIZE}\n")
        f.write(f"Learning rate (initial): {config.LEARNING_RATE}\n")
        f.write(f"Max LR (OneCycle): {config.MAX_LR}\n")
        f.write(f"\nFile structure:\n")
        f.write(f"  - CNN Feature directory: {config.FEATURES_DIR}\n")
        f.write(
            f"  - Intermediate Crash RGB Frames directory: {config.CRASH_RGB_FRAMES_DIR}\n"
        )
        f.write(f"  - Model weights directory: {config.WEIGHTS_DIR}\n")
        f.write(f"  - Results directory: {config.RESULTS_DIR}\n")
        f.write(f"  - Plot file: {config.PLOT_FILENAME}\n")
        f.write(f"  - Submission file: {config.SUBMISSION_FILENAME}\n")
        if os.path.exists(config.MODEL_SAVE_PATH):
            f.write(f"  - Saved Model path: {config.MODEL_SAVE_PATH}\n")
        else:
            f.write(f"  - Saved Model path: Model not saved.\n")

    print(f"\nTEMPORAL-AWARE TRANSFORMER SUMMARY")
    print(
        f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train) if 'X_train' in locals() else 'N/A'} samples)"
    )
    print(f"Number of Frames: {config.NUM_FRAMES}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
