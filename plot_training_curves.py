"""
Plot training loss and validation metrics across epochs from train_log.csv.

Usage:
  python plot_training_curves.py --log_file "train_log (1).csv" --out training_curves.png
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="train_log (1).csv" if os.path.exists("train_log (1).csv") else "train_log.csv")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()

    epochs, losses, mious, bf1s, qs = [], [], [], [], []

    with open(args.log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            mious.append(float(row["val_mIoU"]))
            bf1s.append(float(row["val_BF1"]))
            qs.append(float(row["val_Q"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Curve
    ax1.plot(epochs, losses, marker='o', markersize=3, color='#1f77b4', linewidth=2, label="Train Loss (Hybrid)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss vs. Epochs", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)

    # Validation Metrics Curve
    ax2.plot(epochs, mious, marker='s', markersize=3, color='#2ca02c', linewidth=2, label="Val mIoU")
    ax2.plot(epochs, bf1s, marker='^', markersize=3, color='#ff7f0e', linewidth=2, label="Val BF1")
    ax2.plot(epochs, qs, marker='d', markersize=3, color='#d62728', linewidth=2.5, label="Val Q (0.7*mIoU + 0.3*BF1)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Metric Score", fontsize=12)
    ax2.set_title("Validation Metrics vs. Epochs", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved training curves to {args.out}")


if __name__ == "__main__":
    main()

