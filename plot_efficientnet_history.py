import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Load the actual EfficientNet-B0 training history
history_path = Path("outputs") / "efficientnet_b0_history.csv"

if not history_path.exists():
    raise FileNotFoundError(
        f"Cannot find {history_path}. "
        "Run this script from your Individual Assignment folder."
    )

epochs = []
training_loss = []
validation_loss = []

with history_path.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        epochs.append(int(row["epoch"]))
        training_loss.append(float(row["train_loss"]))
        validation_loss.append(float(row["val_loss"]))

if not epochs:
    raise ValueError("The history CSV contains no results.")

# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    epochs, training_loss,
    marker="o", label="Training Loss"
)
plt.plot(
    epochs, validation_loss,
    marker="o", label="Validation Loss"
)

plt.title("EfficientNet-B0 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

# Save the chart
output_path = Path("outputs") / "efficientnet_b0_loss_curve.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Loss chart saved to: {output_path}")
