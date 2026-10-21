# Bird Species Classification
# Official Test-Set Results
# COS30082 Applied Machine Learning

from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------
# 1. OFFICIAL TEST-SET RESULTS
# --------------------------------------------------

results = [
    [
        "Custom CNN", "15", "121/1204",
        "10.05%", "26.50%", "9.58%",
        "9.35%", "9.58%", "7.57%"
    ],
    [
        "Frozen ResNet18", "5", "501/1204",
        "41.61%", "71.84%", "40.58%",
        "47.70%", "40.58%", "39.62%"
    ],
    [
        "Fine-tuned ResNet18", "5", "578/1204",
        "48.01%", "76.00%", "47.20%",
        "50.35%", "47.20%", "46.97%"
    ],
    [
        "Augmented ResNet18", "8", "577/1204",
        "47.92%", "75.33%", "47.14%",
        "51.61%", "47.14%", "46.27%"
    ],
    [
        "ResNet18 V2", "8", "580/1204",
        "48.17%", "75.58%", "47.44%",
        "51.43%", "47.44%", "47.05%"
    ],
    [
        "EfficientNet-B0", "8", "815/1204",
        "67.69%", "90.61%", "66.52%",
        "70.19%", "66.52%", "65.82%"
    ],
]

headers = [
    "Model",
    "Epochs",
    "Correct Predictions",
    "Top-1 Accuracy",
    "Top-5 Accuracy",
    "Avg Class Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

print("\n" + "=" * 130)
print("BIRD SPECIES CLASSIFICATION - OFFICIAL TEST RESULTS")
print("=" * 130)

print(tabulate(results, headers=headers, tablefmt="grid"))

# --------------------------------------------------
# 2. BEST OVERALL MODEL
# --------------------------------------------------

print("\nBEST OVERALL MODEL: EfficientNet-B0")
print("Top-1 Accuracy: 67.69%")
print("Top-5 Accuracy: 90.61%")
print("Average Class Accuracy: 66.52%")
print("Macro F1-score: 65.82%")
print("=" * 130)

# --------------------------------------------------
# 3. PREPARE DATA FOR THE COMBINED CHART
# --------------------------------------------------

models = [
    "Custom CNN",
    "Frozen ResNet18",
    "Fine-tuned ResNet18",
    "Augmented ResNet18",
    "ResNet18 V2",
    "EfficientNet-B0"
]

top1_accuracies = [10.05, 41.61, 48.01, 47.92, 48.17, 67.69]

class_accuracies = [9.58, 40.58, 47.20, 47.14, 47.44, 66.52]

# Create the outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 4. CREATE ONE COMBINED BAR CHART
# --------------------------------------------------

x = np.arange(len(models))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 7))

# Top-1 Accuracy bars
bars1 = ax.bar(
    x - width / 2,
    top1_accuracies,
    width,
    label="Top-1 Accuracy"
)

# Average Accuracy per Class bars
bars2 = ax.bar(
    x + width / 2,
    class_accuracies,
    width,
    label="Average Accuracy per Class"
)

# Chart title and axis labels
ax.set_title(
    "Bird Species Classification: Accuracy Comparison",
    fontsize=14
)

ax.set_xlabel("Model")
ax.set_ylabel("Accuracy (%)")

ax.set_ylim(0, 80)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right")

# Display legend
ax.legend()

# Display percentages above each bar
ax.bar_label(
    bars1,
    labels=[f"{value:.2f}%" for value in top1_accuracies],
    padding=3,
    fontsize=8
)

ax.bar_label(
    bars2,
    labels=[f"{value:.2f}%" for value in class_accuracies],
    padding=3,
    fontsize=8
)

fig.tight_layout()

# --------------------------------------------------
# 5. SAVE AND DISPLAY THE CHART
# --------------------------------------------------

output_path = "outputs/model_accuracy_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nCombined accuracy chart saved to: {output_path}")

plt.show()
