
from pathlib import Path
from collections import defaultdict
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix
)
from PIL import Image


# ------------------------------------------
# 1. Paths and configuration
# ------------------------------------------
DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "Test"
TEST_FILE = DATA_DIR / "test.txt"

OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = Path("outputs/resnet18_finetuned_augmented_best.pth")
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "resnet18_augmented_confusion_matrix.csv"

BATCH_SIZE = 8
device = torch.device("cpu")


# ------------------------------------------
# 2. Load test annotations
# ------------------------------------------
test_records = []

with TEST_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        test_records.append((filename, int(label)))

print("Test images:", len(test_records))


# ------------------------------------------
# 3. Load the saved model
# ------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

num_classes = checkpoint["num_classes"]
image_size = checkpoint["image_size"]
species_by_label = checkpoint["species_by_label"]

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# ------------------------------------------
# 4. Image preprocessing
# ------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------
# 5. Test dataset
# ------------------------------------------
class BirdTestDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        filename, label = self.records[index]
        image_path = TEST_DIR / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image_transform(image)

        return image, label


test_loader = DataLoader(
    BirdTestDataset(test_records),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------
# 6. Evaluate predictions
# ------------------------------------------
all_labels = []
all_predictions = []
all_top5_predictions = []

class_correct = defaultdict(int)
class_total = defaultdict(int)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Highest-scoring prediction
        predictions = outputs.argmax(dim=1)

        # Five highest-scoring predictions
        top5_predictions = outputs.topk(5, dim=1).indices

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())
        all_top5_predictions.extend(top5_predictions.tolist())

        for true_label, predicted_label in zip(
            labels.tolist(), predictions.tolist()
        ):
            class_total[true_label] += 1

            if true_label == predicted_label:
                class_correct[true_label] += 1


# ------------------------------------------
# 7. Calculate evaluation metrics
# ------------------------------------------
total = len(all_labels)

correct = sum(
    true_label == predicted_label
    for true_label, predicted_label in zip(
        all_labels, all_predictions
    )
)

top1_accuracy = correct / total

top5_correct = sum(
    true_label in top5
    for true_label, top5 in zip(
        all_labels, all_top5_predictions
    )
)

top5_accuracy = top5_correct / total

class_accuracies = [
    class_correct[label] / class_total[label]
    for label in sorted(class_total)
    if class_total[label] > 0
]

average_class_accuracy = (
    sum(class_accuracies) / len(class_accuracies)
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0
    )
)


# ------------------------------------------
# 8. Save confusion matrix
# Rows = true classes; columns = predicted classes
# ------------------------------------------
confusion = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with CONFUSION_MATRIX_PATH.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow(["True/Predicted"] + list(range(num_classes)))

    for label, row in enumerate(confusion):
        writer.writerow([label] + row.tolist())


# ------------------------------------------
# 9. Display overall results
# ------------------------------------------
print("\n" + "=" * 55)
print("RESNET18 — AUGMENTED FINE-TUNING TEST RESULTS")
print("=" * 55)
print(f"Correct Top-1 predictions: {correct}/{total}")
print(f"Top-1 accuracy: {top1_accuracy:.4%}")
print(f"Top-5 accuracy: {top5_accuracy:.4%}")
print(f"Average accuracy per class: {average_class_accuracy:.4%}")
print(f"Macro precision: {macro_precision:.4%}")
print(f"Macro recall: {macro_recall:.4%}")
print(f"Macro F1-score: {macro_f1:.4%}")
print(f"Classes represented: {len(class_accuracies)}/{num_classes}")
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


# ------------------------------------------
# 10. Show first 10 class-level results
# ------------------------------------------
print("\nFirst 10 class-level results:")

for label in sorted(class_total)[:10]:
    name = species_by_label.get(label, f"Class {label}")
    accuracy = class_correct[label] / class_total[label]

    print(
        f"Class {label:3d} | {name:30s} | "
        f"{class_correct[label]}/{class_total[label]} correct | "
        f"{accuracy:.2%}"
    )
