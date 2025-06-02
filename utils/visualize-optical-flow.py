#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Path to your dataset
DATA_BASE_PATH = os.path.expanduser("./data-nexar/")
TRAIN_VIDEO_DIR = os.path.join(DATA_BASE_PATH, "train/")
TEST_VIDEO_DIR = os.path.join(DATA_BASE_PATH, "test/")

# Video ID to visualize
VIDEO_ID_TO_VISUALIZE = "00022"  # <--- CHANGE THIS
VIDEO_SET_DIRECTORY = TRAIN_VIDEO_DIR  # <--- CHANGE THIS if it's a test video

NUM_FRAMES_TO_EXTRACT = 16
FRAME_SIZE = (299, 299)
NUM_FLOW_FIELDS_TO_DISPLAY = 30  # Let's display one to see the quiver plot clearly


# --- HELPER FUNCTIONS (extract_rgb_frames, calculate_optical_flow - assumed to be the same as before) ---
def extract_rgb_frames(video_path, num_frames=NUM_FRAMES_TO_EXTRACT, size=FRAME_SIZE):
    """Extracts num_frames from a video, resized."""
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        print(f"Warning: Could not read video or video is empty: {video_path}")
        return np.empty((0, *size, 3), dtype=np.uint8)
    if total_video_frames < num_frames:
        frames_to_extract_indices = np.arange(0, total_video_frames)
    else:
        frames_to_extract_indices = np.linspace(
            0, total_video_frames - 1, num_frames, dtype=int
        )
    extracted_frames = []
    for idx in frames_to_extract_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at index {idx} from {video_path}")
            break
        frame = cv2.resize(frame, size)
        extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not extracted_frames:
        print(f"Warning: No frames extracted for video: {video_path}")
        return np.empty((0, *size, 3), dtype=np.uint8)
    return np.stack(extracted_frames)


def calculate_optical_flow(frames_sequence):
    """Calculates dense optical flow between consecutive frames in a sequence."""
    flow_sequence = []
    if len(frames_sequence) < 2:
        print("Warning: Need at least 2 frames to calculate optical flow.")
        return np.empty((0, *frames_sequence.shape[1:3], 2), dtype=np.float32)
    prev_gray = cv2.cvtColor(frames_sequence[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames_sequence)):
        current_gray = cv2.cvtColor(frames_sequence[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_sequence.append(flow)
        prev_gray = current_gray
    return (
        np.stack(flow_sequence)
        if flow_sequence
        else np.empty((0, *frames_sequence.shape[1:3], 2), dtype=np.float32)
    )


# --- NEW: Visualization with Arrows (Quiver Plot) ---
def visualize_optical_flow_arrows(rgb_frame, flow_field, frame_index_str="", step=16):
    """
    Visualizes dense optical flow using arrows (quiver plot) on a sparse grid.

    Args:
        rgb_frame (np.array): The original RGB frame (H, W, 3).
        flow_field (np.array): The 2-channel optical flow field (H, W, 2) (dx, dy).
        frame_index_str (str): String to identify the frame number in the title.
        step (int): Draw an arrow every 'step' pixels.
    """
    h, w = flow_field.shape[:2]

    # Create a grid of points to draw arrows
    y, x = (
        np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
    )
    fx, fy = flow_field[y, x].T  # dx, dy components at grid points

    plt.figure(figsize=(8, 8))  # Adjust figure size as needed
    plt.imshow(rgb_frame)

    # plt.quiver(x_coords, y_coords, u_flow_component, v_flow_component, **kwargs)
    # Note: quiver's y-axis is typically inverted compared to image y-axis,
    # but imshow handles the image display correctly, so using y directly is fine.
    # fx is dx (flow in x-direction), fy is dy (flow in y-direction)
    plt.quiver(
        x,
        y,
        fx,
        fy,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=3,
        headlength=4,
        width=0.002,
    )
    # Parameters for quiver:
    # angles='xy': Arrow GCs are `(x,y)` to `(x+u, y+v)`.
    # scale_units='xy': `u,v` are in data units. `scale` is number of data units per arrow length unit.
    # A scale of 1 means an arrow representing a displacement of (say) 10 pixels will be 10 pixels long on the plot.
    # You might need to adjust 'scale', 'headwidth', 'headlength', 'width' for best visual appearance.

    plt.title(
        f"Optical Flow (Arrows) - Frame {frame_index_str} to {int(frame_index_str) + 1 if frame_index_str.isdigit() else ''}"
    )
    plt.axis("off")
    plt.suptitle(f"Video ID: {VIDEO_ID_TO_VISUALIZE}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ========================================
# MAIN VISUALIZATION SCRIPT
# ========================================
def main_visualize_video_flow_arrows(video_id, video_dir):
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(video_dir, video_filename)

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return

    print(f"Processing video: {video_path}")
    rgb_frames = extract_rgb_frames(video_path)
    if rgb_frames.shape[0] < 2:
        print("Not enough RGB frames for optical flow.")
        return
    print(f"Extracted {rgb_frames.shape[0]} RGB frames.")

    optical_flow_fields = calculate_optical_flow(rgb_frames)
    if optical_flow_fields.shape[0] == 0:
        print("No optical flow fields generated.")
        return
    print(f"Calculated {optical_flow_fields.shape[0]} optical flow fields.")

    num_to_show = min(NUM_FLOW_FIELDS_TO_DISPLAY, optical_flow_fields.shape[0])
    if num_to_show == 0:
        print("Nothing to display.")
        return

    print(f"Displaying the first {num_to_show} optical flow field(s) with arrows...")
    for i in range(num_to_show):
        base_rgb_frame = rgb_frames[i]  # The first frame of the pair
        flow_field_to_show = optical_flow_fields[i]
        visualize_optical_flow_arrows(
            base_rgb_frame, flow_field_to_show, frame_index_str=str(i), step=10
        )  # Adjust step for density


if __name__ == "__main__":
    if not VIDEO_ID_TO_VISUALIZE:
        print("Please set the VIDEO_ID_TO_VISUALIZE variable in the script.")
    else:
        main_visualize_video_flow_arrows(VIDEO_ID_TO_VISUALIZE, VIDEO_SET_DIRECTORY)
