from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Bird Species Classification · StreamLit",
    page_icon="🐦‍🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CHECKPOINT_PATH = Path("outputs/efficientnet_b0_best.pth")
DEVICE = torch.device("cpu")


# 2. CUSTOM WEBSITE DESIGN
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    .stApp {
        background: #f6f7f0;
        color: #203b2c;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 247, 240, 0.9);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #173e2b !important;
    }

    .brand {
        display: block !important;
        width: 100%;
        color: #24563b !important;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px !important;
        font-weight: 700;
        letter-spacing: 1px !important;
        line-height: 1.8 !important;
        text-transform: uppercase;
        margin: 0 0 18px 0;
        padding: 6px 0;
        white-space: normal;
        overflow: visible !important;
    }

    .hero {
        background: linear-gradient(130deg, #173e2b 0%, #2d6847 65%, #63865b 100%);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(23, 62, 43, 0.13);
    }

    .hero h1 {
        color: white !important;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(2.1rem, 5vw, 3.6rem);
        line-height: 1.15;
        margin-bottom: 1rem;
    }

    .hero p {
        color: #e5eee4;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 680px;
    }

    .hero-tag {
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.35);
        background: rgba(255,255,255,0.10);
        color: #f0f5e9;
        padding: 7px 13px;
        border-radius: 30px;
        font-size: 0.82rem;
        margin-bottom: 1.2rem;
    }

    .section-heading {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.7rem;
        color: #173e2b;
        margin-top: 1.1rem;
        margin-bottom: 0.4rem;
    }

    .section-description {
        color: #68796b;
        margin-bottom: 1rem;
    }

    .panel {
        background: white;
        border: 1px solid #e3e9dd;
        border-radius: 18px;
        padding: 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 18px rgba(32, 59, 44, 0.04);
    }

    .result-card {
        background: linear-gradient(135deg, #e7f0e3, #f7f7ed);
        border: 1px solid #d4e3d0;
        border-radius: 20px;
        padding: 1.6rem;
        margin: 0.8rem 0 1.3rem 0;
    }

    .result-label {
        color: #50735a;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .result-name {
        color: #173e2b;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(1.6rem, 3vw, 2.4rem);
        font-weight: 700;
        line-height: 1.25;
        margin: 0.5rem 0;
        overflow-wrap: anywhere;
    }

    .result-note {
        color: #5d7562;
        font-size: 0.92rem;
    }

    .stat-card {
        background: white;
        border: 1px solid #e3e9dd;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        min-height: 100px;
    }

    .stat-number {
        color: #24563b;
        font-size: 1.55rem;
        font-weight: 700;
    }

    .stat-label {
        color: #718073;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    .rank-number {
        display: inline-block;
        background: #e7f0e3;
        color: #24563b;
        border-radius: 9px;
        padding: 5px 10px;
        font-weight: 700;
        margin-right: 8px;
    }

    .footer {
        text-align: center;
        color: #748274;
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    div.stButton > button[kind="primary"] {
        background: #24563b;
        color: white;
        border: 1px solid #24563b;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #173e2b;
        border-color: #173e2b;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #a9c0a5;
        border-radius: 14px;
        padding: 0.6rem;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    
    /* Style the file uploader's inner area */
    [data-testid="stFileUploader"] section {
        background-color: #f5f8f1 !important;
        border: 1px dashed #8eaa8b !important;
        border-radius: 12px !important;
    }

    /* Change the Upload button */
    [data-testid="stFileUploader"] section button {
        background-color: #24563b !important;
        color: #ffffff !important;
        border: 1px solid #24563b !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Change the button when hovered */
    [data-testid="stFileUploader"] section button:hover {
        background-color: #173e2b !important;
        color: #ffffff !important;
        border-color: #173e2b !important;
    }

    /* Improve the uploader's text readability */
    [data-testid="stFileUploader"] section small {
        color: #526b57 !important;
    }

    /* Make the file uploader text readable */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #526b57 !important;
        opacity: 1 !important;
    }

    /* Keep the upload button text white */
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] section button span {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# 3. LOAD TRAINED EFFICIENTNET-B0
@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    num_classes = checkpoint["num_classes"]
    image_size = checkpoint["image_size"]
    species_by_label = checkpoint["species_by_label"]

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, species_by_label, image_transform


# 4. PREDICTION FUNCTIONS
def get_species_name(species_by_label, label):
    name = species_by_label.get(
        label,
        species_by_label.get(str(label))
    )

    if name is None:
        return f"Bird class {label}"

    return str(name).replace("_", " ").title()

def predict_species(model, image, image_transform):
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=5
    )

    return [
        (int(index), float(probability))
        for index, probability in zip(
            top_indices.cpu(),
            top_probabilities.cpu()
        )
    ]


# 5. HERO SECTION
st.markdown("""
<div class="brand">🐦 Bird Species Classification · Bird Explorer</div>

<div class="hero">
    <div class="hero-tag">POWERED BY EFFICIENTNET-B0</div>
    <h1>Every bird has<br>a story to discover.</h1>
    <p>
        Upload a bird photograph and explore the five species our
        image-classification model considers most likely.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. MODEL PERFORMANCE OVERVIEW
# ============================================================

st.markdown('<div class="section-heading">Explore bird species</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="section-description">'
    'Upload a clear photograph for the best chance of identification.'
    '</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">200</div>
        <div class="stat-label">Bird species supported</div>
    </div>
    """, unsafe_allow_html=True)

with metric2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">67.69%</div>
        <div class="stat-label">Test Top-1 accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">90.61%</div>
        <div class="stat-label">Test Top-5 accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.caption(
    "Performance measured on the assignment's 1204-image official test set. "
    "These are overall evaluation metrics, not the confidence for an uploaded image."
)

st.divider()


# 7. IMAGE UPLOAD AND PREDICTION
upload_col, result_col = st.columns([0.95, 1.05], gap="large")

with upload_col:
    st.markdown(
        '<div class="section-heading">01 · Upload a bird image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Choose a JPG, JPEG or PNG image.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose your bird image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image where the bird is visible."
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Your uploaded photograph",
                use_container_width=True
            )
        except Exception:
            image = None
            st.error("This image could not be opened. Please try another file.")
    else:
        image = None
        st.markdown("""
        <div class="panel">
            <div style="font-size:3rem;text-align:center;">🕊️🦅🦉🐦🦩</div>
            <p style="text-align:center;color:#718073;">
                Your bird photograph will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    identify_clicked = st.button(
        "Identify This Bird  →",
        type="primary",
        disabled=image is None,
        use_container_width=True
    )


with result_col:
    st.markdown(
        '<div class="section-heading">02 · AI identification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Your leading prediction and five most likely matches.'
        '</div>',
        unsafe_allow_html=True
    )

    if identify_clicked and image is not None:
        try:
            with st.spinner("Examining the image..."):
                model, species_by_label, image_transform = load_model()

                predictions = predict_species(
                    model,
                    image,
                    image_transform
                )

            best_label, best_probability = predictions[0]
            best_name = get_species_name(
                species_by_label,
                best_label
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Best match</div>
                <div class="result-name">{best_name}</div>
                <div class="result-note">
                    Model confidence: {best_probability * 100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Top 5 possible species")

            for rank, (label, probability) in enumerate(
                predictions,
                start=1
            ):
                species_name = get_species_name(
                    species_by_label,
                    label
                )

                row_col1, row_col2 = st.columns([4, 1])

                with row_col1:
                    st.markdown(
                        f'<span class="rank-number">{rank:02}</span>'
                        f'**{species_name}**',
                        unsafe_allow_html=True
                    )

                with row_col2:
                    st.markdown(
                        f"<div style='text-align:right;font-weight:700;"
                        f"color:#24563b;padding-top:5px;'>"
                        f"{probability * 100:.1f}%</div>",
                        unsafe_allow_html=True
                    )

                st.progress(probability)

            st.info(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the species identification is correct."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.markdown("""
        <div class="panel">
            <div style="font-size:2.4rem;">🔎</div>
            <h3 style="margin-top:0.5rem;">Ready to discover?</h3>
            <p style="color:#718073;line-height:1.7;">
                Upload a photograph on the left and select
                <b>Identify This Bird</b>. Your prediction results
                will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 8. ABOUT THE MODEL
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">Behind the feathers</div>',
    unsafe_allow_html=True
)

about_col1, about_col2 = st.columns(2, gap="large")

with about_col1:
    st.markdown("""
    <div class="panel">
        <h3>🧠 The AI model</h3>
        <p style="color:#68796b;line-height:1.8;">
            Bird Species Classification uses EfficientNet-B0, a convolutional neural
            network fine-tuned for a 200-class bird image-classification
            task. It learns visual patterns to rank possible species.
        </p>
    </div>
    """, unsafe_allow_html=True)

with about_col2:
    st.markdown("""
    <div class="panel">
        <h3>📊 Understanding the results</h3>
        <p style="color:#68796b;line-height:1.8;">
            Top-1 accuracy measures how often the first prediction is
            correct. Top-5 accuracy measures how often the correct
            species appears among the five highest-ranked predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    BIRD SPECIES CLASSIFICATION · BIRD EXPLORER<br>
    Applied Machine Learning Project · EfficientNet-B0
</div>
""", unsafe_allow_html=True)
