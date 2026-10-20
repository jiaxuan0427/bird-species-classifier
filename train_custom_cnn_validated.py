from pathlib import Path
import random
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
NUM_CLASSES = 200
VALIDATION_RATIO = 0.20
BATCH_SIZE = 16
EPOCHS = 15
IMAGE_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Train"
ANNOTATION_FILE = DATA_DIR / "train.txt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "custom_cnn_validated_best.pth"
HISTORY_PATH = OUTPUT_DIR / "custom_cnn_validated_history.csv"

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


# ============================================================
# 2. READ ANNOTATIONS AND CREATE STRATIFIED SPLIT
# ============================================================

records = []

with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        records.append((filename, int(label)))

species_by_label = {}

for filename, label in records:
    species_by_label[label] = filename.rsplit("_", 2)[0].replace("_", " ")

labels = [label for _, label in records]

train_records, val_records = train_test_split(
    records,
    test_size=VALIDATION_RATIO,
    random_state=SEED,
    stratify=labels
)

print("=" * 60)
print("CUSTOM CNN — TRAINING WITH VALIDATION")
print("=" * 60)
print(f"Device: {device}")
print(f"Training images: {len(train_records)}")
print(f"Validation images: {len(val_records)}")
print(f"Classes: {NUM_CLASSES}")


# ============================================================
# 3. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. DATASET AND DATALOADERS
# ============================================================

class BirdDataset(Dataset):
    def __init__(self, records, image_dir, transform):
        self.records = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        filename, label = self.records[index]
        image = Image.open(self.image_dir / filename).convert("RGB")
        return self.transform(image), label


train_dataset = BirdDataset(
    train_records, TRAIN_DIR, train_transform
)

val_dataset = BirdDataset(
    val_records, TRAIN_DIR, val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. CUSTOM CNN ARCHITECTURE
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
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
        return self.classifier(self.features(x))


# ============================================================
# 6. MODEL, LOSS AND OPTIMISER
# ============================================================

model = CustomCNN(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

history = []
best_val_accuracy = -1.0


# ============================================================
# 7. TRAINING AND VALIDATION
# ============================================================

for epoch in range(EPOCHS):
    start_time = time.time()

    # Train
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} Training"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = 100.0 * train_correct / train_total

    # Validate without updating model weights
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} Validation"
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = 100.0 * val_correct / val_total
    elapsed = (time.time() - start_time) / 60

    row = {
        "epoch": epoch + 1,
        "training_loss": train_loss,
        "training_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "epoch_time_minutes": elapsed
    }
    history.append(row)

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    print("-" * 50)

    # Save the best checkpoint using validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "species_by_label": species_by_label,
            "epoch": epoch + 1,
            "best_val_accuracy": best_val_accuracy
        }, CHECKPOINT_PATH)

        print("Saved best validation checkpoint.")

    # Save history after every epoch
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(history)


print("\n" + "=" * 60)
print("CUSTOM CNN TRAINING WITH VALIDATION COMPLETED")
print("=" * 60)
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"History: {HISTORY_PATH}")
print("The official test set was not used.")
