import matplotlib.pyplot as plt
import numpy as np

# Official test-set results
models = [
    "Frozen ResNet18",
    "Fine-tuned ResNet18",
    "Custom CNN"
]

top1_accuracy = [41.61, 48.01, 10.05]
average_per_class = [40.58, 47.20, 9.58]

# Set bar positions
x = np.arange(len(models))
width = 0.35

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(9, 5.5))

bars1 = ax.bar(
    x - width / 2,
    top1_accuracy,
    width,
    label="Top-1 Accuracy"
)

bars2 = ax.bar(
    x + width / 2,
    average_per_class,
    width,
    label="Average Per-Class Accuracy"
)

# Add percentage labels above each bar
ax.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.2f%%", padding=3, fontsize=9)

# Add titles and labels
ax.set_title(
    "Performance Comparison of Bird Classification Models"
)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Model")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 60)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()

# Save the chart for your report
plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Chart saved as model_comparison.png")
