"""
Optical Flow Feature Extractor for Video Sequences
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

os.environ["TORCH_HOME"] = "./cache"
os.makedirs("./cache", exist_ok=True)

DATASET_PERCENTAGE = 1.0
NUM_FRAMES = 32
FRAME_SIZE = (299, 299)
FLOW_FEAT_DIM = 2048

percentage_str = str(int(DATASET_PERCENTAGE * 100))
FLOW_FEATURES_BASE_DIR = "features/optical_flow/"
os.makedirs(FLOW_FEATURES_BASE_DIR, exist_ok=True)

X_TRAIN_FLOW_FEATURES_FILE = os.path.join(
    FLOW_FEATURES_BASE_DIR,
    f"X_train_flow_sequences_{percentage_str}pct_{NUM_FRAMES - 1}frames.npy",
)
X_TEST_FLOW_FEATURES_FILE = os.path.join(
    FLOW_FEATURES_BASE_DIR, f"X_test_flow_sequences_{NUM_FRAMES - 1}frames.npy"
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

print("Loading CSV data to get video IDs...")
data_base_path = os.path.expanduser("./data-nexar/")
try:
    df_train_full = pd.read_csv(os.path.join(data_base_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_base_path, "test.csv"))
except FileNotFoundError:
    print(f"ERROR: CSV files not found at {data_base_path}. Please check paths.")
    exit()

df_train_full["id"] = df_train_full["id"].astype(str).str.zfill(5)
df_test["id"] = df_test["id"].astype(str).str.zfill(5)

if DATASET_PERCENTAGE < 1.0:
    print(f"Sampling {DATASET_PERCENTAGE * 100}% of the training dataset video IDs...")
    df_train = df_train_full.sample(
        frac=DATASET_PERCENTAGE, random_state=42
    ).reset_index(drop=True)
else:
    df_train = df_train_full
print(f"Number of training videos to process: {len(df_train)}")
print(f"Number of test videos to process: {len(df_test)}")

train_video_dir = os.path.join(data_base_path, "train/")
test_video_dir = os.path.join(data_base_path, "test/")


def extract_rgb_frames(video_path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        return np.empty((0, *size, 3), dtype=np.uint8)

    frames_to_extract_indices = np.linspace(
        0, total_video_frames - 1, num_frames, dtype=int
    )

    extracted_frames = []
    for idx in frames_to_extract_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not extracted_frames:
        return np.empty((0, *size, 3), dtype=np.uint8)

    return np.stack(extracted_frames)


def calculate_optical_flow(frames_sequence):
    flow_sequence = []
    if len(frames_sequence) < 2:
        return np.empty((0, *frames_sequence.shape[1:3], 2), dtype=np.float32)

    prev_gray = cv2.cvtColor(frames_sequence[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames_sequence)):
        current_rgb = frames_sequence[i]
        current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)

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


print("Loading InceptionV3 base model for optical flow feature extraction...")
flow_feature_extractor = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
flow_feature_extractor.fc = nn.Identity()
flow_feature_extractor = flow_feature_extractor.to(device)
flow_feature_extractor.eval()

preprocess_for_cnn = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
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


def get_optical_flow_feature_sequences(
    video_ids, video_folder, target_num_rgb_frames=NUM_FRAMES
):
    video_flow_features_list = []

    target_num_flow_frames = target_num_rgb_frames - 1
    if target_num_flow_frames <= 0:
        raise ValueError("NUM_FRAMES must be at least 2 to compute optical flow.")

    for video_id in tqdm(
        video_ids, desc=f"Extracting optical flow features from {video_folder}"
    ):
        video_path = os.path.join(video_folder, f"{video_id}.mp4")

        rgb_frames_np = extract_rgb_frames(
            video_path, num_frames=target_num_rgb_frames, size=FRAME_SIZE
        )

        if rgb_frames_np.shape[0] < 2:
            print(
                f"Warning: Video {video_id} had < 2 RGB frames ({rgb_frames_np.shape[0]}). Skipping flow, appending zeros."
            )
            video_flow_features_list.append(
                np.zeros((target_num_flow_frames, FLOW_FEAT_DIM), dtype=np.float32)
            )
            continue

        optical_flow_sequence_np = calculate_optical_flow(rgb_frames_np)
        current_num_flow_frames = optical_flow_sequence_np.shape[0]

        if current_num_flow_frames == 0:
            print(
                f"Warning: Video {video_id} resulted in 0 flow frames. Appending zeros."
            )
            video_flow_features_list.append(
                np.zeros((target_num_flow_frames, FLOW_FEAT_DIM), dtype=np.float32)
            )
            continue

        batch_flow_cnn_inputs = []
        for i in range(current_num_flow_frames):
            flow_field = optical_flow_sequence_np[i]
            flow_img_3channel = normalize_flow_for_cnn(flow_field)
            processed_flow_img = preprocess_for_cnn(flow_img_3channel)
            batch_flow_cnn_inputs.append(processed_flow_img)

        if not batch_flow_cnn_inputs:
            print(
                f"Warning: Video {video_id} had no flow inputs for CNN. Appending zeros."
            )
            video_flow_features_list.append(
                np.zeros((target_num_flow_frames, FLOW_FEAT_DIM), dtype=np.float32)
            )
            continue

        batch_tensor = torch.stack(batch_flow_cnn_inputs).to(device)

        with torch.no_grad():
            flow_frame_features_tensor = flow_feature_extractor(batch_tensor)

        flow_frame_features_np = flow_frame_features_tensor.cpu().numpy()

        if current_num_flow_frames < target_num_flow_frames:
            print(
                f"Warning: Video {video_id} processed {current_num_flow_frames}/{target_num_flow_frames} flow frames. Padding features."
            )
            padding_shape = (
                target_num_flow_frames - current_num_flow_frames,
                FLOW_FEAT_DIM,
            )
            padding = np.zeros(padding_shape, dtype=np.float32)
            flow_frame_features_np = np.vstack((flow_frame_features_np, padding))
        elif current_num_flow_frames > target_num_flow_frames:
            flow_frame_features_np = flow_frame_features_np[:target_num_flow_frames, :]

        video_flow_features_list.append(flow_frame_features_np)

    return np.array(video_flow_features_list, dtype=np.float32)


if os.path.exists(X_TRAIN_FLOW_FEATURES_FILE) and os.path.exists(
    X_TEST_FLOW_FEATURES_FILE
):
    print(f"Loading pre-computed optical flow feature sequences...")
    print(f"Train features: {X_TRAIN_FLOW_FEATURES_FILE}")
    print(f"Test features: {X_TEST_FLOW_FEATURES_FILE}")
    print("Feature files already exist. To re-extract, please delete them first.")
else:
    print(
        "Pre-computed optical flow feature sequences not found. Starting extraction..."
    )

    print("\nExtracting for TRAINING videos:")
    X_train_flow_sequences = get_optical_flow_feature_sequences(
        df_train["id"], train_video_dir, target_num_rgb_frames=NUM_FRAMES
    )
    if X_train_flow_sequences.size > 0:
        np.save(X_TRAIN_FLOW_FEATURES_FILE, X_train_flow_sequences)
        print(
            f"Saved training optical flow feature sequences to {X_TRAIN_FLOW_FEATURES_FILE}"
        )
        print(f"Training flow sequences shape: {X_train_flow_sequences.shape}")
    else:
        print("No training flow features were extracted.")

    print("\nExtracting for TEST videos:")
    X_test_flow_sequences = get_optical_flow_feature_sequences(
        df_test["id"], test_video_dir, target_num_rgb_frames=NUM_FRAMES
    )
    if X_test_flow_sequences.size > 0:
        np.save(X_TEST_FLOW_FEATURES_FILE, X_test_flow_sequences)
        print(
            f"Saved test optical flow feature sequences to {X_TEST_FLOW_FEATURES_FILE}"
        )
        print(f"Test flow sequences shape: {X_test_flow_sequences.shape}")
    else:
        print("No test flow features were extracted.")

print("\nOptical flow feature extraction process finished.")
