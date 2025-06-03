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
from sklearn.preprocessing import StandardScaler
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
    FEATURES_DIR = "features/attention/"
    DATASET_PERCENTAGE = 1.0
    NUM_FRAMES = 32
    FRAME_SIZE = (299, 299)

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

    RESULTS_DIR = "results/baseline_transformer"
    WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
    INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
    UNIFORM_FRAMES_DIR = os.path.join(INTERMEDIATE_FRAMES_DIR, "uniform_flow_frames")

    def __init__(self):
        for directory in [
            self.FEATURES_DIR,
            self.RESULTS_DIR,
            self.WEIGHTS_DIR,
            self.INTERMEDIATE_FRAMES_DIR,
            self.UNIFORM_FRAMES_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    @property
    def percentage_str(self):
        return str(int(self.DATASET_PERCENTAGE * 100))

    @property
    def MODEL_SAVE_PATH(self):
        return os.path.join(
            self.WEIGHTS_DIR,
            f"best_hierarchical_transformer_{self.percentage_str}pct_N{self.NUM_FRAMES}.pth",
        )

    @property
    def PLOT_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR,
            f"training_curves_hierarchical_{self.percentage_str}pct_N{self.NUM_FRAMES}.png",
        )

    @property
    def SUBMISSION_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR,
            f"submission_hierarchical_{self.percentage_str}pct_N{self.NUM_FRAMES}.csv",
        )

    @property
    def TRAIN_FEATURES_FILE_FINAL(self):
        return os.path.join(
            self.FEATURES_DIR,
            f"X_train_sequences_{self.percentage_str}pct_N{self.NUM_FRAMES}.npy",
        )

    @property
    def TEST_FEATURES_FILE_FINAL(self):
        return os.path.join(
            self.FEATURES_DIR, f"X_test_sequences_N{self.NUM_FRAMES}.npy"
        )


def extract_uniform_frames(video_path, num_frames, frame_size):
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        return np.zeros((num_frames, *frame_size, 3), dtype=np.uint8)

    frame_positions = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)
    frame_positions = np.clip(frame_positions, 0, total_video_frames - 1)

    frames_list = []
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


def load_or_generate_transformer_features(config, device):
    if os.path.exists(config.TRAIN_FEATURES_FILE_FINAL) and os.path.exists(
        config.TEST_FEATURES_FILE_FINAL
    ):
        print(
            f"Loading pre-computed final transformer sequence features from {config.FEATURES_DIR}"
        )
        X_train_sequences = np.load(config.TRAIN_FEATURES_FILE_FINAL)
        X_test_sequences = np.load(config.TEST_FEATURES_FILE_FINAL)

        data_base_path = os.path.expanduser("./data-nexar/")
        df_train_full = pd.read_csv(os.path.join(data_base_path, "train.csv"))
        if config.DATASET_PERCENTAGE < 1.0:
            df = df_train_full.sample(
                frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)
        else:
            df = df_train_full.copy()
        y_train = df["target"].values

    else:
        print(
            "Final transformer sequence features not found. Generating from intermediate frames or videos..."
        )
        data_base_path = os.path.expanduser("./data-nexar/")
        df_train_full = pd.read_csv(os.path.join(data_base_path, "train.csv"))
        df_test_full = pd.read_csv(os.path.join(data_base_path, "test.csv"))
        df_train_full["id"] = df_train_full["id"].astype(str).str.zfill(5)
        df_test_full["id"] = df_test_full["id"].astype(str).str.zfill(5)

        if config.DATASET_PERCENTAGE < 1.0:
            df_train = df_train_full.sample(
                frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)
        else:
            df_train = df_train_full.copy()

        y_train = df_train["target"].values
        train_video_dir = os.path.join(data_base_path, "train/")
        test_video_dir = os.path.join(data_base_path, "test/")

        cnn_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        cnn_model.fc = nn.Identity()
        cnn_model.aux_logits = (
            False  # Important for eval mode if model returns tuple otherwise
        )
        cnn_model = cnn_model.to(device)
        cnn_model.eval()

        preprocess_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(config.FRAME_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def get_cnn_sequence_features_for_videos(
            video_ids, video_folder, uniform_frames_storage_dir
        ):
            all_video_features_sequences = []
            for video_id in tqdm(
                video_ids, desc=f"Extracting CNN sequence features from {video_folder}"
            ):
                intermediate_frames_file = os.path.join(
                    uniform_frames_storage_dir, f"{video_id}_frames.npy"
                )
                frames_np = None
                if os.path.exists(intermediate_frames_file):
                    frames_np = np.load(intermediate_frames_file)
                    if frames_np.shape[0] != config.NUM_FRAMES:
                        print(
                            f"Frame count mismatch for {video_id} (loaded {frames_np.shape[0]}, expected {config.NUM_FRAMES}). Re-extracting."
                        )
                        frames_np = None

                if frames_np is None:
                    video_path = os.path.join(video_folder, f"{video_id}.mp4")
                    frames_np = extract_uniform_frames(
                        video_path, config.NUM_FRAMES, config.FRAME_SIZE
                    )
                    np.save(intermediate_frames_file, frames_np)

                if frames_np.shape[0] == 0 or frames_np.shape[0] < config.NUM_FRAMES:
                    print(
                        f"Warning: Insufficient frames for {video_id} ({frames_np.shape[0]}), using zeros for CNN features."
                    )
                    video_cnn_features = np.zeros(
                        (config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32
                    )
                else:
                    batch_tensors = torch.stack(
                        [preprocess_transform(frame) for frame in frames_np]
                    ).to(device)
                    with torch.no_grad():
                        video_cnn_features = cnn_model(batch_tensors)
                        video_cnn_features = video_cnn_features.cpu().numpy()

                if video_cnn_features.shape[0] < config.NUM_FRAMES:
                    padding = np.zeros(
                        (
                            config.NUM_FRAMES - video_cnn_features.shape[0],
                            config.INPUT_DIM,
                        ),
                        dtype=np.float32,
                    )
                    video_cnn_features = (
                        np.vstack((video_cnn_features, padding))
                        if video_cnn_features.shape[0] > 0
                        else padding
                    )
                all_video_features_sequences.append(
                    video_cnn_features[: config.NUM_FRAMES]
                )
            return np.array(all_video_features_sequences, dtype=np.float32)

        X_train_sequences = get_cnn_sequence_features_for_videos(
            df_train["id"], train_video_dir, config.UNIFORM_FRAMES_DIR
        )
        np.save(config.TRAIN_FEATURES_FILE_FINAL, X_train_sequences)
        print(
            f"Saved final training sequence features to {config.TRAIN_FEATURES_FILE_FINAL}"
        )

        X_test_sequences = get_cnn_sequence_features_for_videos(
            df_test_full["id"], test_video_dir, config.UNIFORM_FRAMES_DIR
        )
        np.save(config.TEST_FEATURES_FILE_FINAL, X_test_sequences)
        print(
            f"Saved final test sequence features to {config.TEST_FEATURES_FILE_FINAL}"
        )

    scaler = StandardScaler()
    X_train_flat = X_train_sequences.reshape(
        -1, X_train_sequences.shape[-1]
    )  # Reshape to (num_samples * NUM_FRAMES, INPUT_DIM)
    scaler.fit(X_train_flat)
    X_train_scaled_flat = scaler.transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(
        X_train_sequences.shape
    )  # Reshape back

    X_test_flat = X_test_sequences.reshape(-1, X_test_sequences.shape[-1])
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled_flat.reshape(X_test_sequences.shape)

    return X_train_scaled, X_test_scaled, y_train


class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("Hierarchical Transformer Training", fontsize=16)
        self.train_losses, self.val_losses, self.val_aucs, self.learning_rates = (
            [],
            [],
            [],
            [],
        )
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
    all_predictions, all_labels = [], []
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
        print("AUC calculation error")
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

    print("Loading/Generating transformer features...")
    X_train, X_test, y_train = load_or_generate_transformer_features(config, device)

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train,
        X_test,
        y_train,
        test_size=config.TEST_SIZE,
        batch_size=config.BATCH_SIZE,
        random_state=config.RANDOM_STATE,
    )

    print("Initializing hierarchical transformer...")
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
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        total_steps=config.NUM_EPOCHS * len(train_loader),
        pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
        anneal_strategy="cos",
    )
    live_plotter = LivePlotter()
    print("Starting hierarchical transformer training...")
    best_auc = 0.0
    final_auc = 0.0

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
                        "best_auc": best_auc,
                        "epoch": epoch,
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
                    f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Best Val AUC: {best_auc:.4f}"
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        live_plotter.save_final_plot(config.PLOT_FILENAME)
        live_plotter.close()

    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading best model from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, final_auc = validate(model, val_loader, criterion, device)
        print(f"\nFinal Validation ROC-AUC from best model: {final_auc:.4f}")

        print("Generating test predictions...")
        model.eval()
        test_predictions = []
        with torch.no_grad():
            for features_batch in tqdm(
                test_loader, desc="Test inference"
            ):  # features_batch is a tuple (features, labels) or just features
                features_tensor = (
                    features_batch[0]
                    if isinstance(features_batch, tuple)
                    else features_batch
                )
                features_tensor = features_tensor.to(device)
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
            f"No model found at {config.MODEL_SAVE_PATH}, skipping final evaluation and submission."
        )

    summary_path = os.path.join(
        config.RESULTS_DIR,
        f"training_summary_transformer_{config.percentage_str}pct_N{config.NUM_FRAMES}.txt",
    )
    with open(summary_path, "w") as f:
        f.write(
            "BASELINE TRANSFORMER TRAINING SUMMARY\n===================================\n"
        )
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Dataset Percentage: {config.DATASET_PERCENTAGE * 100}%\n")
        f.write(f"Number of Frames: {config.NUM_FRAMES}\n")
        f.write(
            f"Input Dimension: {config.INPUT_DIM}\nModel d_model: {config.D_MODEL}\n"
        )
        f.write(f"Model Layers: {config.NUM_LAYERS}, Heads: {config.NUM_HEADS}\n")
        f.write(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Initial LR: {config.LEARNING_RATE}, Max LR: {config.MAX_LR}\n")
        f.write(
            f"Best Validation AUC: {best_auc:.4f}\nFinal Validation AUC (loaded best model): {final_auc:.4f}\n\n"
        )
        f.write("PATHS:\n")
        f.write(f"  Final Features: {config.FEATURES_DIR}\n")
        f.write(f"  Intermediate Uniform Frames: {config.UNIFORM_FRAMES_DIR}\n")
        f.write(f"  Results: {config.RESULTS_DIR}\n")
        f.write(f"  Weights: {config.WEIGHTS_DIR}\n")
        f.write(f"  Model Saved: {config.MODEL_SAVE_PATH}\n")
        f.write(f"  Plot: {config.PLOT_FILENAME}\n")
        f.write(f"  Submission: {config.SUBMISSION_FILENAME}\n")
    print(f"Training summary saved to {summary_path}")
    print(
        f"\nHIERARCHICAL TRANSFORMER SUMMARY\nDataset used: {config.DATASET_PERCENTAGE * 100}% ({len(y_train) if 'y_train' in locals() else 'N/A'} samples)"
    )
    print(f"Best validation AUC: {best_auc:.4f}\nFinal validation AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
