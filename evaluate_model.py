from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image


# ------------------------------------------
# 1. Paths and configuration
# ------------------------------------------
DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "Test"
TEST_FILE = DATA_DIR / "test.txt"

CHECKPOINT_PATH = Path("outputs/resnet18_finetuned_best.pth")

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
correct = 0
total = 0

# Count correct predictions separately for each class.
class_correct = defaultdict(int)
class_total = defaultdict(int)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for true_label, predicted_label in zip(
            labels.tolist(), predictions.tolist()
        ):
            class_total[true_label] += 1

            if true_label == predicted_label:
                class_correct[true_label] += 1


# ------------------------------------------
# 7. Calculate required metrics
# ------------------------------------------
top1_accuracy = correct / total

class_accuracies = []

for label in sorted(class_total):
    accuracy = class_correct[label] / class_total[label]
    class_accuracies.append(accuracy)

average_class_accuracy = sum(class_accuracies) / len(class_accuracies)

print("\n" + "=" * 50)
print("OFFICIAL TEST SET RESULTS")
print("=" * 50)
print(f"Correct predictions: {correct}/{total}")
print(f"Top-1 accuracy: {top1_accuracy:.4%}")
print(f"Average accuracy per class: {average_class_accuracy:.4%}")
print(f"Number of classes evaluated: {len(class_accuracies)}")


# ------------------------------------------
# 8. Show some class-level results
# ------------------------------------------
print("\nFirst 10 class-level results:")

for label in sorted(class_total)[:10]:
    name = species_by_label.get(label, f"Class {label}")
    accuracy = class_correct[label] / class_total[label]

    print(
        f"Class {label:3d} | {name:30s} | "
        f"{class_correct[label]}/{class_total[label]} "
        f"correct | {accuracy:.2%}"
    )
