from pathlib import Path
from collections import Counter

# Project paths
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
TRAIN_FILE = DATA_DIR / "train.txt"
TEST_FILE = DATA_DIR / "test.txt"


def read_annotations(annotation_file):
    """Read image filenames and numeric class labels."""
    records = []

    with annotation_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()

            if len(parts) != 2:
                print(f"Invalid line {line_number}: {line.strip()}")
                continue

            filename, label = parts
            records.append((filename, int(label)))

    return records


def check_split(name, image_dir, annotation_file):
    """Check image existence, label range and class counts."""
    records = read_annotations(annotation_file)
    missing = [
        filename
        for filename, label in records
        if not (image_dir / filename).is_file()
    ]

    labels = [label for _, label in records]
    class_counts = Counter(labels)

    print(f"\n--- {name} dataset ---")
    print(f"Annotation entries: {len(records)}")
    print(f"Images found: {len(records) - len(missing)}")
    print(f"Missing images: {len(missing)}")
    print(f"Unique classes: {len(class_counts)}")

    if labels:
        print(f"Minimum label: {min(labels)}")
        print(f"Maximum label: {max(labels)}")

    if missing:
        print("First missing filenames:", missing[:5])

    return records


if __name__ == "__main__":
    train_records = check_split(
        "TRAIN", TRAIN_DIR, TRAIN_FILE
    )

    test_records = check_split(
        "TEST", TEST_DIR, TEST_FILE
    )

    # Identify species names from training filenames.
    # Expected format: Species_Name_0001_123456.jpg
    species_by_label = {}

    for filename, label in train_records:
        parts = filename.rsplit("_", 2)

        if len(parts) == 3:
            species_name = parts[0]
            species_by_label.setdefault(label, species_name)

    print("\n--- Overall checks ---")
    print(f"Training species names identified: {len(species_by_label)}")
    print("First five label-to-species mappings:")

    for label in sorted(species_by_label)[:5]:
        print(f"  {label}: {species_by_label[label]}")

    all_labels = {
        label for _, label in train_records
    } | {
        label for _, label in test_records
    }

    if all_labels:
        print(f"All labels are within 0-199: {all(0 <= x < 200 for x in all_labels)}")

    print("\nDataset check complete.")
