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

current_dir = os.path.dirname(os.path.abspath(__file__))
codebase_dir = os.path.join(current_dir, "..")
sys.path.append(codebase_dir)

from module_hierarchical_transformer import HierarchicalTransformer
from utils.data_loader import create_data_loaders


class Config:
    FEATURES_DIR = "features/efficientnet/"
    DATASET_PERCENTAGE = 1.0

    BACKBONE = "efficientnet_b3"
    INPUT_DIM = 1536 + 512
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 3
    D_FF = 1024
    MAX_SEQ_LEN = 32
    DROPOUT = 0.3

    NUM_FRAMES = 32
    FRAME_SIZE = (300, 300)

    BATCH_SIZE = 24
    NUM_EPOCHS = 45
    LEARNING_RATE = 8e-7
    WEIGHT_DECAY = 1e-3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MAX_LR = 8e-7
    PCT_START = 0.3
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1e3

    PRINT_FREQUENCY = 10

    RESULTS_DIR = "results/efficientnet"
    WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")

    INTERMEDIATE_FRAMES_DIR = "features/intermediate_post_split"
    UNIFORM_FRAMES_DIR = os.path.join(
        INTERMEDIATE_FRAMES_DIR, "uniform_efficientnet_frames"
    )

    def __init__(self):
        for directory in [
            self.RESULTS_DIR,
            self.WEIGHTS_DIR,
            self.FEATURES_DIR,
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
            f"best_efficientnet_transformer_{self.percentage_str}pct.pth",
        )

    @property
    def PLOT_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR,
            f"training_curves_efficientnet_{self.percentage_str}pct.png",
        )

    @property
    def SUBMISSION_FILENAME(self):
        return os.path.join(
            self.RESULTS_DIR, f"submission_efficientnet_{self.percentage_str}pct.csv"
        )


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, backbone_name="efficientnet_b3"):
        super(MultiScaleFeatureExtractor, self).__init__()

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True
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


def load_efficientnet_features():
    config = Config()

    train_cnn_features_file = os.path.join(
        config.FEATURES_DIR,
        f"X_train_efficient_{config.percentage_str}pct_N{config.NUM_FRAMES}.npy",
    )
    test_cnn_features_file = os.path.join(
        config.FEATURES_DIR, f"X_test_efficient_N{config.NUM_FRAMES}.npy"
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
            f"Loading pre-computed EfficientNet CNN features from {train_cnn_features_file}"
        )
        X_train_sequences = np.load(train_cnn_features_file)
        X_test_sequences = np.load(test_cnn_features_file)
        return X_train_sequences, X_test_sequences, y_train

    print("CNN features not found. Attempting to generate them...")
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

    cnn_feature_extractor = MultiScaleFeatureExtractor(config.BACKBONE).to(device)
    cnn_feature_extractor.eval()

    preprocess_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(config.FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_features_from_frames_or_video(
        video_ids, video_folder, uniform_frames_storage_dir
    ):
        cnn_features_list = []

        for video_id in tqdm(
            video_ids,
            desc=f"Processing frames/videos from {video_folder} for CNN features",
        ):
            intermediate_frames_file = os.path.join(
                uniform_frames_storage_dir, f"{video_id}_frames.npy"
            )
            video_path = os.path.join(video_folder, f"{video_id}.mp4")
            frames_np = None

            try:
                if os.path.exists(intermediate_frames_file):
                    frames_np = np.load(intermediate_frames_file)
                    if frames_np.shape[0] < config.NUM_FRAMES:
                        print(
                            f"Padding loaded frames for {video_id} from {frames_np.shape[0]} to {config.NUM_FRAMES}"
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
                    frames_np = extract_uniform_frames(
                        video_path,
                        num_frames=config.NUM_FRAMES,
                        frame_size=config.FRAME_SIZE,
                    )
                    np.save(intermediate_frames_file, frames_np)

                if frames_np.shape[0] == 0 or frames_np.shape[0] < config.NUM_FRAMES:
                    print(
                        f"Warning: Not enough frames for {video_id} after loading/extraction ({frames_np.shape[0]}), using zeros."
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
                    frame_cnn_features = cnn_feature_extractor(batch_tensor)

                cnn_features_list.append(frame_cnn_features.cpu().numpy())

            except Exception as e:
                print(
                    f"Error processing {video_id}: {e}. Appending zeros for this video."
                )
                cnn_features_list.append(
                    np.zeros((config.NUM_FRAMES, config.INPUT_DIM), dtype=np.float32)
                )

        processed_cnn_features_list = []
        for features_seq in cnn_features_list:
            if features_seq.shape[0] < config.NUM_FRAMES:
                print(
                    f"Final padding for a sequence from {features_seq.shape[0]} to {config.NUM_FRAMES}"
                )
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

    print("Generating/Loading training CNN features...")
    X_train_sequences = extract_features_from_frames_or_video(
        df["id"], train_dir, config.UNIFORM_FRAMES_DIR
    )
    np.save(train_cnn_features_file, X_train_sequences)
    print(f"Saved training CNN features to {train_cnn_features_file}")

    print("Generating/Loading test CNN features...")
    X_test_sequences = extract_features_from_frames_or_video(
        df_test["id"], test_dir, config.UNIFORM_FRAMES_DIR
    )
    np.save(test_cnn_features_file, X_test_sequences)
    print(f"Saved test CNN features to {test_cnn_features_file}")

    return X_train_sequences, X_test_sequences, y_train


class EnhancedHierarchicalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        d_model=512,
        num_heads=8,
        num_layers=3,
        d_ff=1024,
        max_seq_len=32,
        dropout=0.3,
    ):
        super(EnhancedHierarchicalTransformer, self).__init__()

        self.d_model = d_model

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

        self.local_transformer = nn.ModuleList()
        self.global_transformer = nn.ModuleList()

        for i in range(num_layers):
            local_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.local_transformer.append(local_layer)

            global_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.global_transformer.append(global_layer)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads // 2 if num_heads > 1 else 1,
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

        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )

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

        self._reset_parameters()

    def _reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "classifier" in name:
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

        x_projected = self.input_projection(x)

        x_embedded = x_projected + self.pos_embedding[:seq_len].unsqueeze(0)

        local_features = x_embedded
        for layer in self.local_transformer:
            local_features = layer(local_features)

        global_features_full = torch.zeros_like(local_features)

        if seq_len >= 2:
            global_indices = torch.arange(0, seq_len, 2, device=x.device)
            if len(global_indices) > 0:
                global_features_input = x_embedded[:, global_indices, :]

                temp_global_features = global_features_input
                for layer in self.global_transformer:
                    temp_global_features = layer(temp_global_features)
                global_features_processed = temp_global_features

                global_features_full[:, global_indices, :] = global_features_processed

                for i in range(seq_len):
                    if i not in global_indices:
                        prev_global_idx = max(
                            [idx for idx in global_indices if idx < i], default=-1
                        )
                        next_global_idx = min(
                            [idx for idx in global_indices if idx > i], default=-1
                        )

                        if (
                            prev_global_idx != -1
                            and next_global_idx != -1
                            and next_global_idx > prev_global_idx
                        ):
                            w_prev = (next_global_idx - i) / (
                                next_global_idx - prev_global_idx
                            )
                            w_next = (i - prev_global_idx) / (
                                next_global_idx - prev_global_idx
                            )
                            global_features_full[:, i, :] = (
                                w_prev * global_features_full[:, prev_global_idx, :]
                                + w_next * global_features_full[:, next_global_idx, :]
                            )
                        elif prev_global_idx != -1:
                            global_features_full[:, i, :] = global_features_full[
                                :, prev_global_idx, :
                            ]
                        elif next_global_idx != -1:
                            global_features_full[:, i, :] = global_features_full[
                                :, next_global_idx, :
                            ]
                        else:
                            global_features_full[:, i, :] = local_features[:, i, :]
        else:
            global_features_full = local_features.clone()

        fused_attention_out, _ = self.fusion_attention(
            local_features, global_features_full, global_features_full
        )

        combined_for_fusion = torch.cat([local_features, fused_attention_out], dim=-1)
        fused_processed = self.fusion(combined_for_fusion)

        final_fused_features = fused_processed + local_features

        attention_weights = self.temporal_attention(final_fused_features)
        attention_weights = torch.softmax(attention_weights, dim=1)

        aggregated_features = torch.sum(attention_weights * final_fused_features, dim=1)

        output = self.classifier(aggregated_features)

        if return_attention:
            return output, attention_weights
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
            "Warning: ROC AUC score could not be calculated (possibly due to single class in labels). Setting AUC to 0."
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

    print("Loading EfficientNet-based data...")
    X_train, X_test, y_train = load_efficientnet_features()

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

    print("Starting EfficientNet transformer training...")
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
                            "backbone": config.BACKBONE,
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
        f"training_summary_efficientnet_{config.percentage_str}pct_N{config.NUM_FRAMES}.txt",
    )
    with open(summary_file, "w") as f:
        f.write(f"EFFICIENTNET TRANSFORMER TRAINING SUMMARY\n")
        f.write(f"========================================\n\n")
        f.write(
            f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train) if 'X_train' in locals() else 'N/A'} samples)\n"
        )
        f.write(f"Backbone: {config.BACKBONE}\n")
        f.write(f"Number of Frames: {config.NUM_FRAMES}\n")
        f.write(f"Input Dimension: {config.INPUT_DIM}\n")
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
            f"  - Intermediate Uniform Frames directory: {config.UNIFORM_FRAMES_DIR}\n"
        )
        f.write(f"  - Model weights directory: {config.WEIGHTS_DIR}\n")
        f.write(f"  - Results directory: {config.RESULTS_DIR}\n")
        f.write(f"  - Plot file: {config.PLOT_FILENAME}\n")
        f.write(f"  - Submission file: {config.SUBMISSION_FILENAME}\n")
        if os.path.exists(config.MODEL_SAVE_PATH):
            f.write(f"  - Saved Model path: {config.MODEL_SAVE_PATH}\n")
        else:
            f.write(
                f"  - Saved Model path: Model not saved (e.g., due to interruption or no improvement).\n"
            )

    print(f"\nEFFICIENTNET TRANSFORMER SUMMARY")
    print(
        f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train) if 'X_train' in locals() else 'N/A'} samples)"
    )
    print(f"Number of Frames: {config.NUM_FRAMES}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
