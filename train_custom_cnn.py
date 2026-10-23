from pathlib import Path
import random
import time
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 1. CONFIGURATION
SEED = 42
NUM_CLASSES = 200
VALIDATION_RATIO = 0.20

BATCH_SIZE = 16
EPOCHS = 8
IMAGE_SIZE = 128
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Train"
ANNOTATION_FILE = DATA_DIR / "train.txt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "custom_cnn_best.pth"
HISTORY_PATH = OUTPUT_DIR / "custom_cnn_history.csv"

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

print("=" * 55)
print("CUSTOM CNN BIRD CLASSIFIER")
print("=" * 55)
print("Device:", device)


# 2. READ ANNOTATIONS
records = []

with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        records.append((filename, int(label)))

species_by_label = {}

for filename, label in records:
    species = filename.rsplit("_", 2)[0].replace("_", " ")
    species_by_label[label] = species

train_records, val_records = train_test_split(
    records,
    test_size=VALIDATION_RATIO,
    random_state=SEED,
    stratify=[label for _, label in records]
)

print("Training images:", len(train_records))
print("Validation images:", len(val_records))
print("Classes:", len(species_by_label))


# 3. IMAGE TRANSFORMS
# Data augmentation is applied only to training images.
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


# 4. DATASET
class BirdDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]

        with Image.open(TRAIN_DIR / filename) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label


train_loader = DataLoader(
    BirdDataset(train_records, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    BirdDataset(val_records, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 5. DEFINE CUSTOM CNN ARCHITECTURE
class CustomCNN(nn.Module):
    """
    CNN trained from scratch.

    Each block uses convolution, batch normalization,
    ReLU activation and max pooling.
    Dropout helps reduce overfitting.
    """

    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 input channels -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Produce a fixed-size feature representation.
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


model = CustomCNN(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

print("\nCustom CNN architecture:")
print(model)


# 6. TRAIN AND VALIDATE
best_val_accuracy = -1.0
history = []

for epoch in range(EPOCHS):
    start_time = time.time()

    model.train()
    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} Training"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_total += loss.item() * batch_size
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    # Evaluate on validation data.
    model.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} Validation"
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss_total += loss.item() * batch_size
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += batch_size

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    elapsed = (time.time() - start_time) / 60

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Epoch time: {elapsed:.2f} minutes")
    print("-" * 50)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # Keep the checkpoint with the best validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "species_by_label": species_by_label,
            "best_val_accuracy": best_val_accuracy,
            "epoch": epoch + 1
        }, CHECKPOINT_PATH)

        print("Saved best custom CNN checkpoint.")


# 7. SAVE TRAINING HISTORY
with HISTORY_PATH.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ]
    )
    writer.writeheader()
    writer.writerows(history)

print("\nCustom CNN training completed.")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print("Checkpoint:", CHECKPOINT_PATH)
print("History:", HISTORY_PATH)
print("Official test set was not used.")
