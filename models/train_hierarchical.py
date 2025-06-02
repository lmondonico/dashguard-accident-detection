# train_hierarchical.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from hierarchical_transformer import HierarchicalTransformer
from data_loader import load_preprocessed_data, create_data_loaders


# Configuration
class Config:
    # Data paths
    FEATURES_DIR = "features/attention/"
    DATASET_PERCENTAGE = 1.0

    # Model architecture
    INPUT_DIM = 2048
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 2
    D_FF = 512
    MAX_SEQ_LEN = 16
    DROPOUT = 0.3

    # Training parameters
    BATCH_SIZE = 32  # Increased batch size
    NUM_EPOCHS = 40  # Reduced epochs for OneCycleLR
    LEARNING_RATE = 1e-6  # Higher starting LR for OneCycleLR
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
        return (
            f"best_hierarchical_transformer_{int(self.DATASET_PERCENTAGE * 100)}pct.pth"
        )

    @property
    def PLOT_FILENAME(self):
        return (
            f"training_curves_hierarchical_{int(self.DATASET_PERCENTAGE * 100)}pct.png"
        )

    @property
    def SUBMISSION_FILENAME(self):
        return f"submission_hierarchical_{int(self.DATASET_PERCENTAGE * 100)}pct.csv"


class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("Hierarchical Transformer Training", fontsize=16)

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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, scaler = load_preprocessed_data(
        config.FEATURES_DIR, config.DATASET_PERCENTAGE, config.RANDOM_STATE
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train,
        X_test,
        y_train,
        test_size=config.TEST_SIZE,
        batch_size=config.BATCH_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # Initialize model
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
    print("Starting hierarchical transformer training...")
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

    print(f"\nHIERARCHICAL TRANSFORMER SUMMARY")
    print(f"Dataset used: {config.DATASET_PERCENTAGE * 100}% ({len(X_train)} samples)")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
