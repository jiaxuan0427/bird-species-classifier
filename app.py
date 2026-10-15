from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bird Species Classifier",
    page_icon="🐦",
    layout="centered"
)

CHECKPOINT_PATH = Path("outputs/resnet18_finetuned_best.pth")
NUM_CLASSES = 200
IMAGE_SIZE = 160

DEVICE = torch.device("cpu")

# Image preprocessing must match the ResNet18 evaluation setup.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Species names were saved during training.
    species_by_label = checkpoint.get("species_by_label", None)

    return model, species_by_label


# ============================================================
# 3. PREDICT BIRD SPECIES
# ============================================================

def predict_species(model, image):
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=3)

    return [
        (int(index), float(probability))
        for index, probability in zip(
            top_indices.cpu(), top_probabilities.cpu()
        )
    ]


def get_species_name(species_by_label, label):
    if species_by_label is not None:
        # Handle either integer or string dictionary keys.
        name = species_by_label.get(
            label,
            species_by_label.get(str(label))
        )

        if name is not None:
            return str(name)

    return f"Bird class {label}"


# ============================================================
# 4. WEBPAGE DESIGN
# ============================================================

st.title("🐦 Bird Species Classifier")

st.write(
    "Upload a bird photograph and let the AI model predict "
    "which of the 200 bird species it most closely matches."
)

st.info(
    "This is an educational image-classification model. "
    "Predictions may be incorrect, especially for unclear "
    "images or species that look similar."
)

uploaded_file = st.file_uploader(
    "Choose a bird image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    if st.button("Identify Bird", type="primary"):
        try:
            with st.spinner("Analysing your image..."):
                model, species_by_label = load_model()
                predictions = predict_species(model, image)

            best_label, best_probability = predictions[0]
            best_name = get_species_name(
                species_by_label, best_label
            )

            st.success("Prediction completed!")

            st.subheader("Predicted species")
            st.header(best_name.replace("_", " ").title())

            st.metric(
                "Model confidence",
                f"{best_probability * 100:.2f}%"
            )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the identification is correct."
            )

            st.subheader("Top 3 predictions")

            for rank, (label, probability) in enumerate(
                predictions, start=1
            ):
                species_name = get_species_name(
                    species_by_label, label
                )

                st.write(
                    f"**{rank}. {species_name.replace('_', ' ').title()}**"
                )
                st.progress(probability)
                st.caption(f"{probability * 100:.2f}%")

        except Exception as error:
            st.error(f"Could not make a prediction: {error}")

st.divider()

st.caption(
    "Bird Species Classification | Applied Machine Learning Project"
)
