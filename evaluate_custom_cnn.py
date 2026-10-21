from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "Test"
TEST_ANNOTATION = DATA_DIR / "test.txt"

OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "custom_cnn_full.pth"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "custom_cnn_confusion_matrix.csv"

IMAGE_SIZE = 128
BATCH_SIZE = 16
NUM_CLASSES = 200

device = torch.device("cpu")


# ============================================================
# 2. IMAGE PREPROCESSING
# Must match the custom CNN training preprocessing
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. LOAD TEST ANNOTATIONS
# ============================================================

records = []

with TEST_ANNOTATION.open("r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        records.append((filename, int(label)))

print("=" * 55)
print("CUSTOM CNN — OFFICIAL TEST EVALUATION")
print("=" * 55)
print(f"Test images: {len(records)}")
print(f"Number of classes: {NUM_CLASSES}")


# ============================================================
# 4. TEST DATASET
# ============================================================

class BirdTestDataset(Dataset):
    def __init__(self, records, image_dir, transform):
        self.records = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        filename, label = self.records[index]
        image_path = self.image_dir / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label


test_dataset = BirdTestDataset(
    records,
    TEST_DIR,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. CUSTOM CNN ARCHITECTURE
# Must match the training architecture exactly
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 6. LOAD THE TRAINED MODEL
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model = CustomCNN(num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT_PATH}")


# ============================================================
# 7. COLLECT PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_top5_predictions = []

class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Highest-scoring class
        predictions = outputs.argmax(dim=1)

        # Five highest-scoring classes
        top5_predictions = outputs.topk(5, dim=1).indices

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())
        all_top5_predictions.extend(top5_predictions.tolist())

        for label, prediction in zip(
            labels.tolist(), predictions.tolist()
        ):
            class_total[label] += 1

            if prediction == label:
                class_correct[label] += 1


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

total_images = len(all_labels)

correct_total = sum(
    true_label == predicted_label
    for true_label, predicted_label in zip(
        all_labels, all_predictions
    )
)

top1_accuracy = 100 * correct_total / total_images

top5_correct = sum(
    true_label in top5
    for true_label, top5 in zip(
        all_labels, all_top5_predictions
    )
)

top5_accuracy = 100 * top5_correct / total_images

# Average accuracy across classes represented in the test set
class_accuracies = [
    class_correct[i] / class_total[i]
    for i in range(NUM_CLASSES)
    if class_total[i] > 0
]

average_class_accuracy = (
    100 * sum(class_accuracies) / len(class_accuracies)
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )
)

# Convert macro metrics to percentages
macro_precision *= 100
macro_recall *= 100
macro_f1 *= 100


# ============================================================
# 9. SAVE CONFUSION MATRIX
# Rows = true labels; columns = predicted labels
# ============================================================

confusion = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with CONFUSION_MATRIX_PATH.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow(["True/Predicted"] + list(range(NUM_CLASSES)))

    for label, row in enumerate(confusion):
        writer.writerow([label] + row.tolist())


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 55)
print("CUSTOM CNN OFFICIAL TEST RESULTS")
print("=" * 55)

print(f"Correct Top-1 predictions: {correct_total}/{total_images}")
print(f"Top-1 accuracy: {top1_accuracy:.4f}%")
print(f"Top-5 accuracy: {top5_accuracy:.4f}%")
print(f"Average accuracy per class: {average_class_accuracy:.4f}%")
print(f"Macro precision: {macro_precision:.4f}%")
print(f"Macro recall: {macro_recall:.4f}%")
print(f"Macro F1-score: {macro_f1:.4f}%")
print(f"Classes represented: {len(class_accuracies)}/{NUM_CLASSES}")
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

print("\nEvaluation completed.")
