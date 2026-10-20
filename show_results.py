
# Bird Species Classification
# Official Test-Set Results
# COS30082 Applied Machine Learning

from tabulate import tabulate

# Results from the actual model evaluation outputs
results = [
    ["Frozen ResNet18", "5", "501/1204", "41.61%", "40.58%"],
    ["Fine-tuned ResNet18", "5", "578/1204", "48.01%", "47.20%"],
    ["Custom CNN", "15", "121/1204", "10.05%", "9.58%"],
]

headers = [
    "Model",
    "Epochs",
    "Correct Predictions",
    "Top-1 Accuracy",
    "Average Per-Class Accuracy"
]

print("\n" + "=" * 100)
print("BIRD SPECIES CLASSIFICATION — OFFICIAL TEST RESULTS")
print("=" * 100)

print(tabulate(results, headers=headers, tablefmt="grid"))

print("\nBest Model: Fine-tuned ResNet18")
print("Top-1 Accuracy: 48.01%")
print("Average Per-Class Accuracy: 47.20%")
print("=" * 100)
