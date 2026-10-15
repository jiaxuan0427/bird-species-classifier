from pathlib import Path
import random
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
NUM_CLASSES = 200
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

CHECKPOINT_PATH = OUTPUT_DIR / "custom_cnn_full.pth"
HISTORY_PATH = OUTPUT_DIR / "custom_cnn_full_history.csv"

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("=" * 60)
print("CUSTOM CNN — FULL TRAINING")
print("=" * 60)
print(f"Device: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")


# ============================================================
# 2. READ ALL TRAINING ANNOTATIONS
# ============================================================

records = []

with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        records.append((filename, int(label)))

# Obtain a readable species name from each filename.
species_by_label = {}

for filename, label in records:
    species_name = filename.rsplit("_", 2)[0].replace("_", " ")
    species_by_label[label] = species_name

labels_found = {label for _, label in records}

if len(labels_found) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(labels_found)}"
    )

if len(records) != 4829:
    print(
        f"Note: found {len(records)} training records, "
        "rather than the expected 4829."
    )

print(f"Training images: {len(records)}")
print(f"Classes: {len(labels_found)}")


# ============================================================
# 3. IMAGE PREPROCESSING AND AUGMENTATION
# ============================================================

# Augmentation creates varied training examples to help
# the model learn features that generalise to new images.
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


# ============================================================
# 4. DATASET
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
        image_path = self.image_dir / filename

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label


train_dataset = BirdDataset(
    records=records,
    image_dir=TRAIN_DIR,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# 5. CUSTOM CNN ARCHITECTURE
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            # Convolution block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Convolution block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Convolution block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Convolution block 4
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
        return self.classifier(x)


# ============================================================
# 6. INITIALISE MODEL AND OPTIMISER
# ============================================================

model = CustomCNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nModel architecture:")
print(model)


# ============================================================
# 7. TRAIN ON THE FULL TRAINING DATASET
# ============================================================

history = []

for epoch in range(EPOCHS):
    start_time = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and parameter update.
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.3f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    elapsed = (time.time() - start_time) / 60

    history.append({
        "epoch": epoch + 1,
        "training_loss": epoch_loss,
        "training_accuracy": epoch_accuracy,
        "epoch_time_minutes": elapsed
    })

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {epoch_loss:.4f}")
    print(f"Training accuracy: {epoch_accuracy:.2f}%")
    print(f"Epoch time: {elapsed:.2f} minutes")
    print("-" * 50)

    # Save the current model so the latest completed epoch
    # is available even if a later epoch is interrupted.
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "species_by_label": species_by_label,
        "epoch": epoch + 1,
        "training_accuracy": epoch_accuracy
    }, CHECKPOINT_PATH)

    # Save training history after each epoch.
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "training_loss",
                "training_accuracy",
                "epoch_time_minutes"
            ]
        )
        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 8. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("FULL CUSTOM CNN TRAINING COMPLETED")
print("=" * 60)
print(f"Training images used: {len(records)}")
print(f"Epochs completed: {EPOCHS}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Training history: {HISTORY_PATH}")
print("No validation split was used.")
print("The official test set was not used during training.")
