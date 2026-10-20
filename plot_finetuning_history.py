import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Find the recorded fine-tuning history
history_path = Path("outputs") / "finetuning_history.csv"

if not history_path.exists():
    raise FileNotFoundError(
        f"Cannot find {history_path}. "
        "Run this script from your Individual Assignment folder."
    )

# Read the actual training history
epochs = []
training_loss = []
training_accuracy = []

with history_path.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        epochs.append(int(row["epoch"]))
        training_loss.append(float(row["train_loss"]))
        training_accuracy.append(float(row["train_accuracy"]) * 100)

if not epochs:
    raise ValueError("The training history CSV contains no results.")

# Create two plots with separate scales
fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# Plot training loss
axes[0].plot(epochs, training_loss, marker="o")
axes[0].set_title("Fine-tuned ResNet18: Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_xticks(epochs)
axes[0].grid(True, linestyle="--", alpha=0.4)

# Plot training accuracy
axes[1].plot(epochs, training_accuracy, marker="o")
axes[1].set_title("Fine-tuned ResNet18: Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Training Accuracy (%)")
axes[1].set_xticks(epochs)
axes[1].grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()

# Save the figure for your report
output_path = Path("outputs") / "finetuning_training_curves.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Figure saved to: {output_path}")
