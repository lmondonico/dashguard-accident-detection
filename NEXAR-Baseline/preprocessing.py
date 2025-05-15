import pandas as pd
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms.functional as F
from transformers import AutoFeatureExtractor

# Configure paths - modify these for your local setup
if os.path.exists("/kaggle/input/nexar-collision-prediction/"):
    # Kaggle environment
    BASE_PATH = "/kaggle/input/nexar-collision-prediction"
    OUTPUT_PATH = "/kaggle/working"
else:
    # Local environment - change these paths to match your setup
    BASE_PATH = "./nexar_collision_prediction"  # or wherever you extracted the data
    OUTPUT_PATH = "./preprocessed_data"

# Load TimeSformer feature extractor
model_name = "facebook/timesformer-base-finetuned-k400"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


# Dataset class for preprocessing
class DashcamDataset:
    def __init__(self, df, video_dir, mode="train", num_frames=8):
        self.df = df
        self.video_dir = video_dir
        self.mode = mode
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_id = self.df["id"].iloc[idx]
        video_path = f"{self.video_dir}/{str(video_id).zfill(5)}.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            return torch.zeros((self.num_frames, 3, 224, 224)), video_id

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames <= 0 or fps <= 0:
            cap.release()
            return torch.zeros((self.num_frames, 3, 224, 224)), video_id

        # Sampling strategy
        if (
            self.mode == "train"
            and "target" in self.df.columns
            and self.df["target"].iloc[idx] == 1
        ):
            alert_time = self.df["time_of_alert"].iloc[idx]
            event_time = self.df["time_of_event"].iloc[idx]
            alert_frame = int(alert_time * fps)
            event_frame = int(event_time * fps)
            start_frame = max(0, alert_frame - int(2 * fps))  # 2s before alert
            end_frame = min(total_frames, event_frame + int(2 * fps))  # 2s after event
        else:
            start_frame = max(0, total_frames - int(5 * fps))  # Last 5s
            end_frame = total_frames

        step = max(1, (end_frame - start_frame) // self.num_frames)
        frames = []
        for i in range(
            start_frame, min(start_frame + step * self.num_frames, end_frame), step
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )  # Keep as uint8 (0-255)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # [3, H, W]
                frame_tensor = F.resize(frame_tensor, (224, 224))  # [3, 224, 224]
                frames.append(
                    frame_tensor.numpy().transpose(1, 2, 0)
                )  # [224, 224, 3] for feature extractor
            if len(frames) == self.num_frames:
                break

        cap.release()
        if len(frames) < self.num_frames:
            frames.extend(
                [np.zeros((224, 224, 3), dtype=np.uint8)]
                * (self.num_frames - len(frames))
            )
        return frames, video_id  # Return list of [H, W, 3] arrays


# Preprocess and save function
def preprocess_and_save(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx in tqdm(range(len(dataset)), desc=f"Preprocessing {dataset.mode}"):
        frames, video_id = dataset[idx]
        # Apply feature extractor preprocessing
        inputs = feature_extractor(
            frames, return_tensors="np"
        )  # Pass list of [H, W, 3] frames
        save_path = os.path.join(save_dir, f"{video_id}.npy")
        np.save(
            save_path, inputs["pixel_values"][0]
        )  # Save as [num_frames, 3, 224, 224]


# Load and preprocess datasets
train_csv_path = os.path.join(BASE_PATH, "train.csv")
train_video_dir = os.path.join(BASE_PATH, "train")

test_csv_path = os.path.join(BASE_PATH, "test.csv")
test_video_dir = os.path.join(BASE_PATH, "test")

# Check if files exist before loading
if not os.path.exists(train_csv_path):
    print(f"Error: Could not find train.csv at {train_csv_path}")
    print(f"Please ensure the Nexar dataset is extracted to {BASE_PATH}")
    print("Expected directory structure:")
    print(f"{BASE_PATH}/")
    print("├── train.csv")
    print("├── test.csv")
    print("├── train/ (directory with .mp4 files)")
    print("└── test/ (directory with .mp4 files)")
    exit(1)

train_df = pd.read_csv(train_csv_path)
train_dataset = DashcamDataset(
    train_df,
    train_video_dir,
    mode="train",
    num_frames=16,
)
preprocess_and_save(
    train_dataset, os.path.join(OUTPUT_PATH, "preprocessed_train_16frames")
)

if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    test_dataset = DashcamDataset(test_df, test_video_dir, mode="test", num_frames=16)
    preprocess_and_save(
        test_dataset, os.path.join(OUTPUT_PATH, "preprocessed_test_16frames")
    )

print(f"Preprocessing complete. Preprocessed data saved to {OUTPUT_PATH}")
