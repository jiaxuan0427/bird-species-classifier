"""
Bird Species Classification
Model: ResNet18 with transfer learning

Dataset:
    data/Train/
    data/train.txt

This script trains the classification layer of a pretrained
ResNet18 model to classify 200 bird species.

The official test set is kept separate for final evaluation.
"""

from pathlib import Path
import random
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 1. CONFIGURATION
SEED = 42
VALIDATION_RATIO = 0.20

BATCH_SIZE = 8
EPOCHS = 5
IMAGE_SIZE = 160
LEARNING_RATE = 0.001

NUM_CLASSES = 200

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Train"
ANNOTATION_FILE = DATA_DIR / "train.txt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "resnet18_best.pth"

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

# Your installed PyTorch is CPU-only.
device = torch.device("cpu")

print("=" * 60)
print("BIRD SPECIES CLASSIFICATION")
print("=" * 60)
print("Device:", device)
print("Number of classes:", NUM_CLASSES)


# 2. READ ANNOTATIONS
records = []

with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        records.append((filename, int(label)))

print("Total training images:", len(records))

# Map numeric labels to readable species names.
species_by_label = {}

for filename, label in records:
    # Example:
    # Black_footed_Albatross_0004_2731401028.jpg
    # becomes Black footed Albatross.
    species_name = filename.rsplit("_", 2)[0]
    species_by_label[label] = species_name.replace("_", " ")

if len(species_by_label) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} species, "
        f"but found {len(species_by_label)}."
    )

print("Species identified:", len(species_by_label))


# 3. TRAIN / VALIDATION SPLIT
labels = [label for _, label in records]

train_records, val_records = train_test_split(
    records,
    test_size=VALIDATION_RATIO,
    random_state=SEED,
    stratify=labels
)

print("Training images:", len(train_records))
print("Validation images:", len(val_records))


# 4. IMAGE TRANSFORMS
# Normalize images using ImageNet statistics because the pretrained ResNet18 weights were trained on ImageNet.
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. CUSTOM DATASET
class BirdDataset(Dataset):
    """Load bird images and their numeric labels."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        image_path = TRAIN_DIR / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image_transform(image)

        return image, label


train_dataset = BirdDataset(train_records)
val_dataset = BirdDataset(val_records)

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


# 6. LOAD PRETRAINED RESNET18
print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze the pretrained feature extractor.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the original 1000-class layer with 200 outputs.
model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(device)

# Loss function for multi-class classification.
criterion = nn.CrossEntropyLoss()

# Only the new final layer is trained in this experiment.
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

print("Model ready for training.")


# 7. TRAINING AND VALIDATION
best_val_accuracy = -1.0
history = []

for epoch in range(EPOCHS):

    epoch_start = time.time()

    # The feature extractor is frozen, so keep the entire model in evaluation mode during training.
    # Gradients are still calculated for model.fc.
    model.eval()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    # ------------------------------
    # Training
    # ------------------------------
    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Calculate frozen image features without gradients.
        with torch.no_grad():
            features = model.conv1(images)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            features = model.layer4(features)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)

        # Train the new classification layer.
        outputs = model.fc(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_total += loss.item() * batch_size
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    # ------------------------------
    # Validation
    # ------------------------------
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss_total += loss.item() * batch_size
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += batch_size

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    epoch_time = (time.time() - epoch_start) / 60

    # Record metrics for later plotting.
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss:     {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation loss:   {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Time: {epoch_time:.2f} minutes")
    print("-" * 50)

    # Save the checkpoint with the best validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "species_by_label": species_by_label,
            "best_val_accuracy": best_val_accuracy,
            "epoch": epoch + 1,
        }, CHECKPOINT_PATH)

        print("Best checkpoint saved:", CHECKPOINT_PATH)


# 8. SAVE TRAINING HISTORY
import csv

history_path = OUTPUT_DIR / "training_history.csv"

with history_path.open(
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

print("\nTraining completed.")
print("Best validation accuracy:", f"{best_val_accuracy:.2%}")
print("Best model:", CHECKPOINT_PATH)
print("Training history:", history_path)
print("\nThe official test dataset has not been used.")
