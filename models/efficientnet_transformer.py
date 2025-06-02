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
import timm

# Add the transformer directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
transformer_dir = os.path.join(current_dir, "..", "transformer")
sys.path.append(transformer_dir)

from hierarchical_transformer import HierarchicalTransformer
from data_loader import create_data_loaders


# Configuration
class Config:
    # Data paths
    FEATURES_DIR = "features/efficientnet/"
    DATASET_PERCENTAGE = 1.0

    # Model architecture
    BACKBONE = "efficientnet_b3"  # More powerful backbone
    INPUT_DIM = 1536 + 512  # EfficientNet-B3 features + multi-scale features
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 3  # Slightly deeper due to better features
    D_FF = 1024
    MAX_SEQ_LEN = 16
    DROPOUT = 0.3

    # Frame extraction
    NUM_FRAMES = 16
    FRAME_SIZE = (300, 300)  # EfficientNet optimal size

    # Training parameters
    BATCH_SIZE = 24  # Smaller batch due to larger model
    NUM_EPOCHS = 45
    LEARNING_RATE = 8e-7
    WEIGHT_DECAY = 1e-3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # OneCycleLR parameters
    MAX_LR = 8e-7
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e3

    # Logging
    PRINT_FREQUENCY = 10

    # Output paths
    @property
    def MODEL_SAVE_PATH(self):
        return (
            f"best_efficientnet_transformer_{int(self.DATASET_PERCENTAGE * 100)}pct.pth"
        )

    @property
    def PLOT_FILENAME(self):
        return (
            f"training_curves_efficientnet_{int(self.DATASET_PERCENTAGE * 100)}pct.png"
        )

    @property
    def SUBMISSION_FILENAME(self):
        return f"submission_efficientnet_{int(self.DATASET_PERCENTAGE * 100)}pct.csv"


class MultiScaleFeatureExtractor(nn.Module):
    """Enhanced feature extractor using EfficientNet with multi-scale features."""

    def __init__(self, backbone_name="efficientnet_b3"):
        super(MultiScaleFeatureExtractor, self).__init__()

        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True
        )

        # Get feature dimensions from different scales
        dummy_input = torch.randn(1, 3, 300, 300)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        # Use features from multiple scales
        self.scale_dims = [f.shape[1] for f in features[-3:]]  # Last 3 scales

        # Adaptive pooling for different scales
        self.adaptive_pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(1) for _ in range(len(self.scale_dims))]
        )

        # Feature fusion
        total_dim = sum(self.scale_dims)
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, 1536),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
        )

        # Motion-sensitive features
        self.motion_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.backbone.eval()  # Keep backbone frozen initially

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        Returns:
            features: Combined features (batch_size, feature_dim)
        """
        # Multi-scale backbone features
        with torch.no_grad():  # Keep backbone frozen
            scale_features = self.backbone(x)

        # Pool and concatenate multi-scale features
        pooled_features = []
        for i, features in enumerate(scale_features[-3:]):
            pooled = self.adaptive_pools[i](features).flatten(1)
            pooled_features.append(pooled)

        combined_backbone = torch.cat(pooled_features, dim=1)
        backbone_features = self.feature_fusion(combined_backbone)

        # Motion-sensitive features
        motion_features = self.motion_extractor(x)

        # Combine all features
        final_features = torch.cat([backbone_features, motion_features], dim=1)

        return final_features


def extract_frames_enhanced(path, num_frames=16, size=(300, 300)):
    """Enhanced frame extraction with better temporal sampling."""
    cap = cv2.VideoCapture(path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *size, 3), dtype=np.uint8)

    # More sophisticated frame sampling
    if total_video_frames <= num_frames:
        # If video is short, use all frames and pad
        frame_indices = list(range(total_video_frames))
    else:
        # Use logarithmic spacing to get more frames from beginning and end
        # where accidents are more likely to be visible
        linear_indices = np.linspace(0, total_video_frames - 1, num_frames // 2)
        log_start = np.logspace(
            0, np.log10(total_video_frames / 2), num_frames // 4, dtype=int
        )
        log_end = (
            total_video_frames
            - np.logspace(
                0, np.log10(total_video_frames / 2), num_frames // 4, dtype=int
            )[::-1]
        )
        frame_indices = np.concatenate([log_start, linear_indices, log_end])
        frame_indices = np.unique(
            np.clip(frame_indices, 0, total_video_frames - 1).astype(int)
        )[:num_frames]

    frames_list = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Pad if necessary
    while len(frames_list) < num_frames:
        if frames_list:
            frames_list.append(frames_list[-1])  # Repeat last frame
        else:
            frames_list.append(np.zeros((*size, 3), dtype=np.uint8))

    return np.stack(frames_list[:num_frames])


def load_efficientnet_features():
    """Load or generate EfficientNet-based features using attention feature extraction."""
    config = Config()

    # Create features directory
    os.makedirs(config.FEATURES_DIR, exist_ok=True)

    percentage_str = str(int(config.DATASET_PERCENTAGE * 100))
    train_features_file = os.path.join(
        config.FEATURES_DIR, f"X_train_efficient_{percentage_str}pct.npy"
    )
    test_features_file = os.path.join(config.FEATURES_DIR, "X_test_efficient.npy")

    # Try to load existing attention features first
    attention_train_file = (
        f"features/attention/X_train_sequences_{percentage_str}pct.npy"
    )
    attention_test_file = "features/attention/X_test_sequences.npy"

    if os.path.exists(attention_train_file) and os.path.exists(attention_test_file):
        print(f"Loading existing attention features from {attention_train_file}")
        X_train_sequences = np.load(attention_train_file)
        X_test_sequences = np.load(attention_test_file)

        # Load labels
        data_base_path = os.path.expanduser("./data/nexar-collision-prediction/")
        df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
        df["id"] = df["id"].astype(str).str.zfill(5)

        if config.DATASET_PERCENTAGE < 1.0:
            df = df.sample(
                frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)

        y_train = df["target"].values
        return X_train_sequences, X_test_sequences, y_train

    # If attention features don't exist, create them
    if os.path.exists(train_features_file) and os.path.exists(test_features_file):
        print(f"Loading EfficientNet features from {train_features_file}")
        X_train_sequences = np.load(train_features_file)
        X_test_sequences = np.load(test_features_file)

        # Load labels
        data_base_path = os.path.expanduser("./data/nexar-collision-prediction/")
        df = pd.read_csv(os.path.join(data_base_path, "train.csv"))
        df["id"] = df["id"].astype(str).str.zfill(5)

        if config.DATASET_PERCENTAGE < 1.0:
            df = df.sample(
                frac=config.DATASET_PERCENTAGE, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)

        y_train = df["target"].values
        return X_train_sequences, X_test_sequences, y_train

    # Load data for feature extraction
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

    # Set up feature extraction
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Use enhanced feature extractor
    feature_extractor = MultiScaleFeatureExtractor(config.BACKBONE).to(device)
    feature_extractor.eval()

    # Enhanced preprocessing for EfficientNet
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(config.FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_features_enhanced(video_ids, video_folder):
        """Extract enhanced features using EfficientNet backbone."""
        features_list = []

        for video_id in tqdm(
            video_ids, desc=f"Extracting enhanced features from {video_folder}"
        ):
            video_path = os.path.join(video_folder, f"{video_id}.mp4")

            try:
                frames_np = extract_frames_enhanced(
                    video_path, num_frames=config.NUM_FRAMES, size=config.FRAME_SIZE
                )

                if frames_np.shape[0] == 0:
                    features_list.append(
                        np.zeros(
                            (config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32
                        )
                    )
                    continue

                # Process frames through enhanced CNN
                batch_frames = []
                for frame_idx in range(frames_np.shape[0]):
                    processed_frame = preprocess(frames_np[frame_idx])
                    batch_frames.append(processed_frame)

                batch_tensor = torch.stack(batch_frames).to(device)

                with torch.no_grad():
                    frame_features = feature_extractor(batch_tensor)

                features_list.append(frame_features.cpu().numpy())

            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                features_list.append(
                    np.zeros((config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32)
                )

        return np.array(features_list, dtype=np.float32)

    # Extract training features
    print("Extracting enhanced training features...")
    X_train_sequences = extract_features_enhanced(df["id"], train_dir)
    np.save(train_features_file, X_train_sequences)

    # Extract test features
    print("Extracting enhanced test features...")
    X_test_sequences = extract_features_enhanced(df_test["id"], test_dir)
    np.save(test_features_file, X_test_sequences)

    y_train = df["target"].values

    return X_train_sequences, X_test_sequences, y_train


class EnhancedHierarchicalTransformer(nn.Module):
    """
    Enhanced hierarchical transformer with improved architecture for crash detection.
    """

    def __init__(
        self,
        input_dim=2048,
        d_model=512,
        num_heads=8,
        num_layers=3,
        d_ff=1024,
        max_seq_len=16,
        dropout=0.3,
    ):
        super(EnhancedHierarchicalTransformer, self).__init__()

        self.d_model = d_model

        # Enhanced input projection with residual connection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

        # Enhanced transformer layers with different attention patterns
        self.local_transformer = nn.ModuleList()
        self.global_transformer = nn.ModuleList()

        for i in range(num_layers):
            # Local transformer with standard attention
            local_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.local_transformer.append(local_layer)

            # Global transformer with different head configuration for variety
            global_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.global_transformer.append(global_layer)

        # Enhanced feature fusion with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Temporal attention for sequence aggregation
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )

        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Enhanced weight initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "classifier" in name:
                    # Smaller initialization for classifier
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.size()

        # Enhanced input projection
        x = self.input_projection(x)

        # Add learnable positional encoding
        x = x + self.pos_embedding[:seq_len].unsqueeze(0)

        # Local processing (full sequence)
        local_features = x
        for layer in self.local_transformer:
            local_features = layer(local_features)

        # Global processing (downsampled sequence)
        global_indices = torch.arange(0, seq_len, 2, device=x.device)
        global_features = x[:, global_indices, :]

        for layer in self.global_transformer:
            global_features = layer(global_features)

        # Upsample global features with learned interpolation
        global_features_full = torch.zeros_like(local_features)
        global_features_full[:, global_indices, :] = global_features

        # Linear interpolation for missing frames
        for i in range(1, seq_len, 2):
            if i < seq_len:
                prev_idx = i - 1
                next_idx = min(i + 1, seq_len - 1)
                if prev_idx in global_indices and next_idx in global_indices:
                    alpha = 0.5
                    global_features_full[:, i, :] = (
                        alpha * global_features_full[:, prev_idx, :]
                        + (1 - alpha) * global_features_full[:, next_idx, :]
                    )
                elif prev_idx in global_indices:
                    global_features_full[:, i, :] = global_features_full[:, prev_idx, :]
                elif next_idx in global_indices:
                    global_features_full[:, i, :] = global_features_full[:, next_idx, :]

        # Enhanced fusion with cross-attention
        fused_features, _ = self.fusion_attention(
            local_features, global_features_full, global_features_full
        )

        # Combine with residual connection
        combined_features = torch.cat([local_features, fused_features], dim=-1)
        fused_features = self.fusion(combined_features) + local_features

        # Temporal attention for sequence aggregation
        attention_weights = self.temporal_attention(
            fused_features
        )  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum over sequence
        aggregated_features = torch.sum(
            attention_weights * fused_features, dim=1
        )  # (batch_size, d_model)

        # Classification
        output = self.classifier(aggregated_features)

        return output


class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("EfficientNet Transformer Training", fontsize=16)

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

        # Gradient clipping
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

    # Load EfficientNet features
    print("Loading EfficientNet-based data...")
    X_train, X_test, y_train = load_efficientnet_features()

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

    # Initialize enhanced model
    print("Initializing EfficientNet transformer...")
    model = EnhancedHierarchicalTransformer(
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
    print("Starting EfficientNet transformer training...")
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

    print(f"\nEFFICIENTNET TRANSFORMER SUMMARY")
    print(f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train)} samples)")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
