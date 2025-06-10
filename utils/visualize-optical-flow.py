"""
Visualize optical flow fields between consecutive frames in a video.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

DATA_BASE_PATH = os.path.expanduser("./data-nexar/")
TRAIN_VIDEO_DIR = os.path.join(DATA_BASE_PATH, "train/")
TEST_VIDEO_DIR = os.path.join(DATA_BASE_PATH, "test/")

VIDEO_ID_TO_VISUALIZE = "00021"
VIDEO_SET_DIRECTORY = TRAIN_VIDEO_DIR

NUM_FRAMES_TO_EXTRACT = 30
FRAME_SIZE = (1280, 720)
QUIVER_STEP = 35


def extract_rgb_frames(video_path, num_frames=NUM_FRAMES_TO_EXTRACT, size=FRAME_SIZE):
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


class OpticalFlowNavigator:
    def __init__(self, rgb_frames_all, flow_fields_all, video_id_str, quiver_step_val):
        self.rgb_frames = rgb_frames_all
        self.flow_fields = flow_fields_all
        self.video_id = video_id_str
        self.quiver_step = quiver_step_val
        self.current_flow_index = 0

        if self.flow_fields is None or len(self.flow_fields) == 0:
            print("Error: No flow fields provided to OpticalFlowNavigator.")
            self.fig = None
            return

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)

        self._draw_frame_and_flow()

        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")

        self.btn_prev.on_clicked(self._prev_flow)
        self.btn_next.on_clicked(self._next_flow)

    def _draw_frame_and_flow(self):
        if not (0 <= self.current_flow_index < len(self.flow_fields)):
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "Invalid flow index.",
                horizontalalignment="center",
                verticalalignment="center",
            )
            self.ax.axis("off")
            plt.draw()
            return

        self.ax.clear()

        base_rgb_frame = self.rgb_frames[self.current_flow_index]
        flow_field = self.flow_fields[self.current_flow_index]

        h, w = flow_field.shape[:2]
        y, x = (
            np.mgrid[
                self.quiver_step // 2 : h : self.quiver_step,
                self.quiver_step // 2 : w : self.quiver_step,
            ]
            .reshape(2, -1)
            .astype(int)
        )

        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)

        fx, fy = flow_field[y, x].T

        self.ax.imshow(base_rgb_frame, alpha=0.8)
        self.ax.quiver(
            x,
            y,
            fx,
            fy,
            color="palegreen",
            angles="xy",
            scale_units="xy",
            scale=0.7,
            headwidth=3.3,
            headlength=4.3,
            width=0.003,
            alpha=1.0,
        )

        frame_idx_str = self.current_flow_index
        self.ax.set_title(
            f"Optical Flow between Frame {frame_idx_str} and {frame_idx_str + 1}"
        )
        self.ax.axis("off")
        self.fig.suptitle(
            f"Video ID: {self.video_id} (Flow Field {self.current_flow_index + 1}/{len(self.flow_fields)})",
            fontsize=14,
        )
        plt.draw()

    def _next_flow(self, event):
        if self.current_flow_index < len(self.flow_fields) - 1:
            self.current_flow_index += 1
            self._draw_frame_and_flow()

    def _prev_flow(self, event):
        if self.current_flow_index > 0:
            self.current_flow_index -= 1
            self._draw_frame_and_flow()


def main_interactive_flow_visualization(video_id, video_dir, quiver_step):
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(video_dir, video_filename)

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return

    print(f"Processing video: {video_path}")
    rgb_frames = extract_rgb_frames(video_path)

    if rgb_frames.shape[0] < 2:
        print("Not enough RGB frames extracted for optical flow.")
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            f"Video {video_id} has < 2 frames.\nCannot compute optical flow.",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        plt.show()
        return
    print(f"Extracted {rgb_frames.shape[0]} RGB frames.")

    optical_flow_fields = calculate_optical_flow(rgb_frames)

    if optical_flow_fields is None or optical_flow_fields.shape[0] == 0:
        print("No optical flow fields generated.")
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            f"Could not generate optical flow for Video {video_id}.",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        plt.show()
        return
    print(f"Calculated {optical_flow_fields.shape[0]} optical flow fields.")

    navigator = OpticalFlowNavigator(
        rgb_frames, optical_flow_fields, video_id, quiver_step
    )

    if navigator.fig:
        plt.show()
    else:
        print("Could not initialize OpticalFlowNavigator.")


if __name__ == "__main__":
    if not VIDEO_ID_TO_VISUALIZE:
        print("Please set the VIDEO_ID_TO_VISUALIZE variable in the script.")
    else:
        main_interactive_flow_visualization(
            VIDEO_ID_TO_VISUALIZE, VIDEO_SET_DIRECTORY, QUIVER_STEP
        )
