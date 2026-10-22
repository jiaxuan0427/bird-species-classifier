from pathlib import Path
import random
import time
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 1. CONFIGURATION
SEED = 42
NUM_CLASSES = 200
VALIDATION_RATIO = 0.20

BATCH_SIZE = 8
EPOCHS = 8
IMAGE_SIZE = 224

FEATURE_LR = 0.00001
CLASSIFIER_LR = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 3

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Train"
ANNOTATION_FILE = DATA_DIR / "train.txt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "efficientnet_b0_best.pth"
HISTORY_PATH = OUTPUT_DIR / "efficientnet_b0_history.csv"

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("=" * 55)
print("EFFICIENTNET-B0 FINE-TUNING EXPERIMENT")
print("=" * 55)
print("Device:", device)


# 2. READ ANNOTATIONS AND CREATE SPLIT
records = []

with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        filename, label = line.strip().split()
        records.append((filename, int(label)))

species_by_label = {}

for filename, label in records:
    species = filename.rsplit("_", 2)[0].replace("_", " ")
    species_by_label[label] = species

labels = [label for _, label in records]

train_records, val_records = train_test_split(
    records,
    test_size=VALIDATION_RATIO,
    random_state=SEED,
    stratify=labels
)

print("Training images:", len(train_records))
print("Validation images:", len(val_records))
print("Number of species:", len(species_by_label))


# 3. IMAGE PREPROCESSING
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
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


# 4. DATASET AND DATALOADERS
class BirdDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        filename, label = self.records[index]
        image_path = TRAIN_DIR / filename

        with Image.open(image_path) as image:
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


# 5. LOAD PRETRAINED EFFICIENTNET-B0
# Load EfficientNet-B0 with ImageNet pretrained weights.
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Freeze all pretrained parameters first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the final classification layer for 200 bird species.
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# Unfreeze the final feature block.
for parameter in model.features[-1].parameters():
    parameter.requires_grad = True

# Unfreeze the new classification layer.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.features[-1].parameters(),
        "lr": FEATURE_LR
    },
    {
        "params": model.classifier.parameters(),
        "lr": CLASSIFIER_LR
    }
], weight_decay=WEIGHT_DECAY)

print("Loaded pretrained EfficientNet-B0.")
print("Unfrozen layers: final feature block and classifier")


# 6. TRAIN AND VALIDATE
best_val_accuracy = -1.0
epochs_without_improvement = 0
history = []

for epoch in range(EPOCHS):
    start_time = time.time()

    model.train()

    # Keep BatchNorm statistics fixed for this small dataset.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

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
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += batch_size

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    # Validation does not update model parameters.
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
            val_correct += (outputs.argmax(1) == labels).sum().item()
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

    # Save the best checkpoint using validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "species_by_label": species_by_label,
            "best_val_accuracy": best_val_accuracy,
            "epoch": epoch + 1,
            "architecture": "efficientnet_b0"
        }, CHECKPOINT_PATH)

        print("Saved best EfficientNet-B0 model.")

    else:
        epochs_without_improvement += 1
        print(
            "No validation improvement:",
            epochs_without_improvement,
            "epoch(s)"
        )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation accuracy has stopped improving.")
        break


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

print("\nEfficientNet-B0 training completed.")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print("Checkpoint:", CHECKPOINT_PATH)
print("History:", HISTORY_PATH)
print("Official test set was not used for training or selection.")
